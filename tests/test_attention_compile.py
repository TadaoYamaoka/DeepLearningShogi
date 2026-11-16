"""Compiled attention, CUDA backward, and exp__bv equivalence regressions."""
from unittest.mock import Mock, patch

import pytest
import torch

from dlshogi.network.policy_value_network import _default_backend, compile_model
from dlshogi.network.policy_value_network_exp__by import AttentionBlock
from dlshogi.network.policy_value_network_exp__bz import AttentionBlock as BZAttentionBlock

@pytest.mark.parametrize('name, expected', [('nt', 'aot_eager'), ('posix', None)])
def test_default_backend(monkeypatch, name, expected):
    with monkeypatch.context() as context:
        context.setattr('dlshogi.network.policy_value_network.os.name', name)
        actual = _default_backend()
    assert actual == expected


@pytest.mark.parametrize('default_backend, backend, options, expected', [
    (None, None, {}, {}),
    ('aot_eager', None, {}, {'backend': 'aot_eager'}),
    ('aot_eager', 'inductor', {}, {'backend': 'inductor'}),
    (None, 'aot_eager', {'mode': 'default', 'fullgraph': True, 'dynamic': True},
     {'backend': 'aot_eager', 'mode': 'default', 'fullgraph': True, 'dynamic': True}),
])
def test_compile_model_options_and_notification(default_backend, backend, options, expected):
    block = AttentionBlock(8, 2)
    logger = Mock()
    block.set_compile_backend = Mock()

    def capture(model, **kwargs):
        assert model is block
        assert kwargs == expected
        block.set_compile_backend.assert_called_once_with(expected.get('backend'))
        logger.info.assert_called_once()
        return compiled

    compiled = object()
    with patch('dlshogi.network.policy_value_network._default_backend', return_value=default_backend), \
         patch('torch.compile', side_effect=capture):
        assert compile_model(block, backend=backend, logger=logger, **options) is compiled


def test_compile_model_requires_torch_compile(monkeypatch):
    monkeypatch.delattr(torch, "compile")
    with pytest.raises(RuntimeError, match='torch.compile is not available'):
        compile_model(torch.nn.Identity())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('backend, fullgraph', [
    ('aot_eager', False), ('cudagraphs', False), ('inductor', True),
])
@pytest.mark.parametrize('block_class', [AttentionBlock, BZAttentionBlock], ids=['by', 'bz'])
def test_cuda_backward(backend, fullgraph, block_class):
    block = block_class(192, 4).cuda().train()
    compiled = compile_model(block, backend=backend, fullgraph=fullgraph)
    x = torch.randn(2, 192, 9, 9, device='cuda', requires_grad=True)
    compiled(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_lightning_aot_eager_cuda_backward():
    pytest.importorskip('lightning')
    from dlshogi.ptl import Model
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

    model = Model(network='exp__by1x8', use_compile=True, compile_backend='aot_eager').cuda()
    x1 = torch.randn(2, FEATURES1_NUM, 9, 9, device='cuda')
    x2 = torch.randn(2, FEATURES2_NUM, 9, 9, device='cuda')
    sum(output.square().mean() for output in model(x1, x2)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('block_class', [AttentionBlock, BZAttentionBlock], ids=['by', 'bz'])
def test_dynamo_onnx_unfused_uses_math(tmp_path, block_class):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    block = block_class(192, 4).cuda().eval()
    x = torch.randn(1, 192, 9, 9, device='cuda')
    program = torch.onnx.export(block, (x,), tmp_path / 'attention.onnx', dynamo=True)
    graph = program.model_proto
    onnx.checker.check_model(graph)
    ops = [node.op_type for node in graph.graph.node]
    assert ops.count('MatMul') == 2
    assert ops.count('Softmax') == 1
    assert not block._fused
    with torch.no_grad():
        torch.testing.assert_close(program.exported_program.module()(x), block(x),
                                   atol=2e-5, rtol=2e-5)

@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
@pytest.mark.parametrize('amp', [False, True])
@pytest.mark.parametrize('block_class', [AttentionBlock, BZAttentionBlock], ids=['by', 'bz'])
def test_default_attention_matches_bv_outputs_and_gradients(device, amp, block_class):
    from dlshogi.network.policy_value_network_exp__bv import AttentionBlock as BVAttentionBlock

    torch.manual_seed(42)
    reference = BVAttentionBlock(32, 4).to(device).train()
    block = block_class(32, 4).to(device).train()
    block.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 32, 9, 9, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        expected = reference(reference_x)
        actual = block(x)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).mean().backward()
    (expected * probe).mean().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in block.named_parameters():
        if '.weights.' in name:
            module_name, index = name.split('.weights.')
            module = getattr(block, module_name)
            expected_grad = reference_parameters[module_name + '.weight'].grad[:, :, 0, 0]
            expected_grad = expected_grad.chunk(len(module.weights), dim=0)[int(index)]
        else:
            expected_grad = reference_parameters[name].grad
        torch.testing.assert_close(parameter.grad, expected_grad)
