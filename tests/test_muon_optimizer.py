import copy
import io
import importlib.util
import unittest

import torch
from torch import nn

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.policy_value_network_exp__by import MatrixConv1x1, PolicyValueNetwork
from dlshogi.optim import build_optimizer


class MuonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def model(self):
        return PolicyValueNetwork(1, 8, nhead=2, fcl=8, attn_backend='math')

    def inputs(self):
        return (torch.randn(2, FEATURES1_NUM, 9, 9),
                torch.randn(2, FEATURES2_NUM, 9, 9))

    def optimizer(self, model):
        def make(spec, groups):
            cls, kwargs = spec
            return cls(groups, **kwargs)

        return build_optimizer(
            model, make,
            (torch.optim.AdamW, dict(lr=0.001, weight_decay=0.01)),
            {"matrix": (torch.optim.Muon, dict(lr=0.001, weight_decay=0.01,
                                             adjust_lr_fn='match_rms_adamw'))},
        )

    def test_partition_and_legacy_checkpoint_fusion(self):
        model = self.model()
        optimizer = build_optimizer(
            model, lambda cls, groups: cls(groups, lr=0.001),
            torch.optim.AdamW, {"matrix": torch.optim.SGD},
        )
        matrices = [p for group in optimizer.optimizers['matrix'].param_groups
                    for p in group['params']]
        other = optimizer.optimizers['default'].param_groups
        self.assertEqual(len(matrices), 8)  # Q, K, V, gate, proj, FFN gate/value/out
        self.assertTrue(all(p.ndim == 2 for p in matrices))
        ids = {id(p) for p in matrices}
        self.assertEqual(ids, {id(p) for name, p in model.named_parameters()
                               if name in model.matrix_parameters()})
        others = {id(p) for group in other for p in group['params']}
        self.assertFalse(ids & others)
        self.assertEqual(ids | others, {id(p) for p in model.parameters()})
        self.assertNotIn(id(model.blocks[0].relative_position_bias_table), ids)
        self.assertNotIn(id(model.value_fc1.weight), ids)

        # Emulate old model structure/checkpoints to verify Conv->matrix migration.
        legacy = copy.deepcopy(model)
        for block in legacy.blocks:
            for name in ('qkv_gate', 'proj', 'ffn_in', 'ffn_out'):
                setattr(block, name, getattr(block, name).to_conv2d())
        restored = self.model()
        restored.load_state_dict(legacy.state_dict(), strict=True)
        inputs = self.inputs()
        for m in (model, legacy, restored):
            m.eval()
        with torch.no_grad():
            expected = legacy(*inputs)
            for output in (model(*inputs), restored(*inputs), restored.fuse()(*inputs)):
                for actual, target in zip(output, expected):
                    torch.testing.assert_close(actual, target, atol=2e-6, rtol=2e-5)
        self.assertFalse(any(isinstance(m, MatrixConv1x1) for m in restored.modules()))
        restored.fuse()  # idempotent

    @unittest.skipUnless(hasattr(torch.optim, 'Muon'), 'requires torch.optim.Muon')
    def test_resume_and_scheduler(self):
        model = self.model()
        optimizer = self.optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        inputs = self.inputs()

        def update(m, opt):
            opt.zero_grad()
            sum(y.square().mean() for y in m(*inputs)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        before = [p.detach().clone() for p in model.parameters()]
        update(model, optimizer)
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        scheduler.step()
        self.assertTrue(all(g['lr'] == 0.0005 for g in optimizer.param_groups))
        self.assertTrue(all(g['lr'] == 0.0005 for o in optimizer.optimizers.values() for g in o.param_groups))
        restored = self.model()
        restored.load_state_dict(model.state_dict())
        resumed = self.optimizer(restored)
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        update(model, optimizer)
        update(restored, resumed)
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch.optim, 'Muon'), 'requires torch.optim.Muon')
    def test_amp_skips_both_optimizers_on_overflow(self):
        model = self.model()
        optimizer = self.optimizer(model)
        scaler = torch.amp.GradScaler('cpu')
        for overflow in (False, True):
            optimizer.zero_grad()
            before = [p.detach().clone() for p in model.parameters()]
            loss = sum(y.square().mean() for y in model(*self.inputs()))
            scaler.scale(loss).backward()
            if overflow:
                next(model.parameters()).grad.fill_(float('inf'))
            scaler.unscale_(optimizer)
            if not overflow:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            changed = [not torch.equal(a, b) for a, b in zip(before, model.parameters())]
            self.assertEqual(any(changed), not overflow)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "requires onnx")
    def test_onnx_has_single_conv_per_projection(self):
        import onnx
        model = self.model().eval().fuse()
        data = io.BytesIO()
        torch.onnx.export(model, self.inputs(), data, opset_version=17,
                          dynamo=False, do_constant_folding=True)
        graph = onnx.load_from_string(data.getvalue())
        onnx.checker.check_model(graph)
        self.assertEqual(sum(n.op_type == 'Conv' for n in graph.graph.node),
                         4 * len(model.blocks) + 4)


if __name__ == '__main__':
    unittest.main()
