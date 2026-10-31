"""CPU tests: python -m unittest discover -s tests -p test_weight_decay.py.

The native shogi decoder is not exercised; if unavailable, only its import-time
feature-count function is stubbed. No training data or GPU is required.
"""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dlshogi.optim import build_weight_decay_param_groups


def setUpModule():
    global native_stub
    native_stub = None
    try:
        importlib.import_module("dlshogi.cppshogi")
    except ImportError:
        stub = types.ModuleType("dlshogi.cppshogi")
        stub.get_max_features2_nyugyoku_num = lambda: 0
        native_stub = patch.dict(sys.modules, {"dlshogi.cppshogi": stub})
        native_stub.start()


def tearDownModule():
    if native_stub is not None:
        native_stub.stop()


class WeightDecayTests(unittest.TestCase):
    def test_target_model_groups(self):
        from dlshogi.network.policy_value_network_resnet import PolicyValueNetwork

        model = PolicyValueNetwork(9, 16)
        model.value_fc2.weight.requires_grad_(False)
        groups = build_weight_decay_param_groups(model, 0.01)
        decay = set(map(id, groups[0]["params"]))
        no_decay = set()
        self.assertEqual(len(groups), 1)
        self.assertFalse(decay & no_decay)
        self.assertEqual(decay | no_decay, {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(decay) + len(no_decay), sum(len(g["params"]) for g in groups))
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertIn(id(module.weight), decay)
                self.assertIn(id(module.bias), decay)
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad:
                self.assertIn(id(module.weight), decay)
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                self.assertIn(id(param), decay)
        again = build_weight_decay_param_groups(model, 0.01)
        self.assertEqual(
            [[id(p) for p in g["params"]] for g in groups],
            [[id(p) for p in g["params"]] for g in again],
        )

    def test_generic_tied_parameters_and_default_decay(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model[1].weight = model[0].weight
        groups = build_weight_decay_param_groups(model, None)
        optimizer = torch.optim.AdamW(groups)
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.01])
        self.assertEqual(len(groups[0]["params"]), 3)
        # Zero gradients isolate decoupled weight decay from gradient updates.
        before = {id(p): p.detach().clone() for p in model.parameters()}
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
        for p in model.parameters():
            self.assertFalse(torch.equal(p, before[id(p)]))
        self.assertFalse(torch.equal(model[0].weight, before[id(model[0].weight)]))

    def test_model_hooks(self):
        class HookModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))
                self.absolute_pos_embed = nn.Parameter(torch.ones(2, 2))
                self.attention = nn.Module()
                self.attention.relative_position_bias_table = nn.Parameter(torch.ones(2, 2))

            def no_weight_decay(self):
                return {"absolute_pos_embed", "attention.relative_position_bias_table"}

        model = HookModel()
        groups = build_weight_decay_param_groups(model)
        self.assertEqual([id(p) for p in groups[0]["params"]], [id(model.weight)])
        self.assertEqual(
            {id(p) for p in groups[1]["params"]},
            {id(model.absolute_pos_embed), id(model.attention.relative_position_bias_table)},
        )

    def make_cli(self, network="resnet2x16_relu", interval="step", scheduler=True,
                 use_no_weight_decay=True):
        from dlshogi.ptl import CustomLightningCLI, DataModule, Model

        config = {
            "model": {
                "network": network, "use_ema": True,
                "use_no_weight_decay": use_no_weight_decay,
                "lr_scheduler_interval": interval,
            },
            "data": {"train_files": [], "val_files": []},
            "trainer": {"logger": False, "enable_checkpointing": False, "enable_progress_bar": False},
            "optimizer": {"class_path": "torch.optim.AdamW", "init_args": {"lr": 0.01, "weight_decay": 0.02}},
        }
        if scheduler:
            config["lr_scheduler"] = {
                "class_path": "dlshogi.lr_scheduler.CosineLRScheduler",
                "init_args": {"t_initial": 10},
            }
        return CustomLightningCLI(Model, DataModule, args=config, run=False, save_config_callback=None)

    def test_lightning_groups_and_scheduler(self):
        for exclude in (False, True):
            for interval in ("step", "epoch"):
                with self.subTest(exclude=exclude, interval=interval):
                    network = "exp__bm_fuse9x16_relu" if exclude else "resnet2x16_relu"
                    cli = self.make_cli(network, interval)
                    result = cli.model.configure_optimizers()
                    if interval == "step":
                        optimizer = result["optimizer"]
                        self.assertEqual(result["lr_scheduler"]["interval"], "step")
                        scheduler = result["lr_scheduler"]["scheduler"]
                    else:
                        optimizer, scheduler = result[0][0], result[1][0]
                    self.assertEqual(len(optimizer.param_groups), 2 if exclude else 1)
                    self.assertEqual(len(scheduler.base_lrs), len(optimizer.param_groups))
                    if exclude:
                        ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
                        self.assertEqual(ids, {id(p) for p in cli.model.model.parameters()})
                        self.assertFalse(ids & {id(p) for p in cli.model.ema_model.parameters()})
                        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
                    optimizer.step()
                    scheduler.step()
                    self.assertEqual(len(scheduler.get_last_lr()), len(optimizer.param_groups))

    def test_lightning_optimizer_resume(self):
        first = self.make_cli("exp__bm_fuse9x16_relu", scheduler=False)
        optimizer = first.model.configure_optimizers()
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.grad = torch.zeros_like(param)
        optimizer.step()
        second = self.make_cli("exp__bm_fuse9x16_relu", scheduler=False)
        restored = second.model.configure_optimizers()
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual([g["weight_decay"] for g in restored.param_groups], [0.02, 0.0])
        self.assertEqual(len(restored.state), len(optimizer.state))

    def test_explicit_model_parameters_only(self):
        from dlshogi.network.policy_value_network_exp__bm_fuse import PolicyValueNetwork

        model = PolicyValueNetwork(9, 16)
        groups = build_weight_decay_param_groups(model)
        expected = {id(model.absolute_pos_embed), id(model.blocks[8].relative_position_bias_table)}
        self.assertEqual({id(p) for p in groups[1]["params"]}, expected)
        self.assertEqual(
            {id(p) for p in groups[0]["params"]},
            {id(p) for p in model.parameters()} - expected,
        )
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertNotIn(id(module.weight), expected)
                self.assertNotIn(id(module.bias), expected)
        plain = build_weight_decay_param_groups(nn.Linear(2, 2))
        self.assertEqual(len(plain), 1)

    def test_model_hook_disabled(self):
        # A hook must not even be called when the option is disabled.
        from dlshogi.network.policy_value_network_exp__bm_fuse import PolicyValueNetwork

        with patch.object(PolicyValueNetwork, "no_weight_decay", side_effect=AssertionError("hook called")):
            cli = self.make_cli("exp__bm_fuse9x16_relu", scheduler=False, use_no_weight_decay=False)
            optimizer = cli.model.configure_optimizers()
            self.assertEqual(len(optimizer.param_groups), 1)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.02)
            optimizer, _ = self.run_train_setup(("--network", "exp__bm_fuse9x16_relu"))
            self.assertEqual(len(optimizer.param_groups), 1)

    def test_model_only_learning_entrypoints(self):
        cli = self.make_cli("exp__bm_fuse9x16_relu")
        result = cli.model.configure_optimizers()
        optimizer = result["optimizer"]
        expected = build_weight_decay_param_groups(cli.model.model)
        self.assertEqual(
            [id(p) for p in optimizer.param_groups[1]["params"]],
            [id(p) for p in expected[1]["params"]],
        )
        self.assertEqual(len(result["lr_scheduler"]["scheduler"].base_lrs), 2)
        trained, _ = self.run_train_setup(("--use_no_weight_decay", "--network", "exp__bm_fuse9x16_relu"))
        self.assertEqual(len(trained.param_groups[1]["params"]), 2)
        self.assertEqual(trained.param_groups[1]["weight_decay"], 0.0)

    def run_train_setup(self, extra_args=()):
        from dlshogi import train

        optimizers, schedulers = [], []
        adamw, step_lr = torch.optim.AdamW, torch.optim.lr_scheduler.StepLR

        def optimizer_factory(*args, **kwargs):
            optimizer = adamw(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        def scheduler_factory(*args, **kwargs):
            scheduler = step_lr(*args, **kwargs)
            schedulers.append(scheduler)
            return scheduler

        with patch.object(torch.optim, "AdamW", side_effect=optimizer_factory), \
             patch.object(torch.optim.lr_scheduler, "StepLR", side_effect=scheduler_factory), \
             patch.object(train.Hcpe3DataLoader, "load_files", side_effect=StopIteration("setup complete")):
            with self.assertRaisesRegex(StopIteration, "setup complete"):
                train.main("unused-train", "unused-test", "--gpu", "-1",
                           "--network", "resnet2x16_relu", "--optimizer", "AdamW()", *extra_args)
        return optimizers[0], schedulers

    def test_train_default_and_enabled_scheduler(self):
        optimizer, _ = self.run_train_setup()
        self.assertEqual(len(optimizer.param_groups), 1)
        optimizer, schedulers = self.run_train_setup(("--use_no_weight_decay", "--network", "exp__bm_fuse9x16_relu", "--lr_scheduler", "StepLR(step_size=1)"))
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.0001, 0.0])
        self.assertEqual(len(schedulers[0].base_lrs), 2)
        optimizer, _ = self.run_train_setup(("--use_no_weight_decay", "--network", "exp__bm_fuse9x16_relu", "--weight_decay", "-1"))
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.01, 0.0])

    def test_train_resume_and_reset_from_old_checkpoint(self):
        from dlshogi.network.policy_value_network_exp__bm_fuse import PolicyValueNetwork

        for exclude in (False, True):
            with self.subTest(exclude=exclude), tempfile.TemporaryDirectory() as directory:
                model = PolicyValueNetwork(9, 16)
                params = build_weight_decay_param_groups(model, 0.03) if exclude else model.parameters()
                optimizer = torch.optim.AdamW(params, weight_decay=0.03)
                checkpoint = Path(directory) / "checkpoint.pth"
                torch.save({"epoch": 2, "t": 10, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
                args = ("--use_no_weight_decay", "--network", "exp__bm_fuse9x16_relu", "--resume", str(checkpoint))
                if not exclude:
                    args += ("--reset_optimizer", "--reset_scheduler")
                restored, _ = self.run_train_setup(args)
                self.assertEqual([g["weight_decay"] for g in restored.param_groups], [0.0001, 0.0])


if __name__ == "__main__":
    unittest.main()
