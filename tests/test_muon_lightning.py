"""Optional Lightning integration test (requires lightning and jsonargparse)."""
import importlib.util
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM


@unittest.skipUnless(importlib.util.find_spec('lightning') and hasattr(torch.optim, 'Muon'),
                     'requires Lightning CLI and torch.optim.Muon')
class LightningMuonTests(unittest.TestCase):
    def test_automatic_optimization_accumulation_and_resume(self):
        from dlshogi.ptl import CustomLightningCLI, Model, DataModule
        from dlshogi.optim import MultiOptimizer

        torch.set_num_threads(1)
        dataset = TensorDataset(
            torch.randn(4, FEATURES1_NUM, 9, 9),
            torch.randn(4, FEATURES2_NUM, 9, 9),
            torch.randn(4, 81 * MAX_MOVE_LABEL_NUM).softmax(-1),
            torch.rand(4, 1), torch.rand(4, 1),
        )
        loader = DataLoader(dataset, batch_size=2)

        def make_cli(epochs):
            return CustomLightningCLI(Model, DataModule, run=False, save_config_callback=None, args={
                'model': {'network': 'exp__by1x8', 'lr_scheduler_interval': 'step'},
                'data': {'train_files': [], 'val_files': []},
                'optimizer': {'class_path': 'torch.optim.AdamW',
                              'init_args': {'lr': 0.001, 'weight_decay': 0.01}},
                'param_optimizers': {'matrix': {'class_path': 'torch.optim.Muon', 'init_args': {
                    'lr': 0.001, 'weight_decay': 0.01, 'adjust_lr_fn': 'match_rms_adamw'}}},
                'lr_scheduler': {'class_path': 'torch.optim.lr_scheduler.StepLR',
                                 'init_args': {'step_size': 1, 'gamma': 0.5}},
                'trainer': {'accelerator': 'cpu', 'logger': False,
                            'enable_checkpointing': False, 'enable_progress_bar': False,
                            'enable_model_summary': False, 'max_epochs': epochs,
                            'accumulate_grad_batches': 2, 'gradient_clip_val': 1.0,
                            'limit_val_batches': 0, 'num_sanity_val_steps': 0},
            })

        with tempfile.TemporaryDirectory() as directory:
            cli = make_cli(1)
            cli.trainer.fit(cli.model, train_dataloaders=loader)
            self.assertTrue(cli.model.automatic_optimization)
            self.assertEqual(cli.trainer.global_step, 1)
            opt = cli.trainer.optimizers[0]
            self.assertIsInstance(opt, MultiOptimizer)
            self.assertTrue(all(g['lr'] == 0.0005 for g in opt.param_groups))
            checkpoint = str(Path(directory) / 'resume.ckpt')
            cli.trainer.save_checkpoint(checkpoint)
            resumed = make_cli(2)
            resumed.trainer.fit(resumed.model, train_dataloaders=loader, ckpt_path=checkpoint)
            self.assertEqual(resumed.trainer.global_step, 2)
            opt = resumed.trainer.optimizers[0]
            self.assertTrue(all(g['lr'] == 0.00025 for g in opt.param_groups))
            self.assertTrue(all(p in opt.state for child in opt.optimizers.values()
                                for group in child.param_groups for p in group["params"]))


@unittest.skipUnless(importlib.util.find_spec('lightning'), 'requires Lightning CLI')
class LightningPlainOptimizerTests(unittest.TestCase):
    def test_single_group_without_routes_retains_ema_checkpoint_layout(self):
        from dlshogi.ptl import CustomLightningCLI, Model, DataModule
        from dlshogi.network.policy_value_network import policy_value_network

        def make_network(*args, **kwargs):
            network = policy_value_network(*args, **kwargs)
            network.absolute_pos_embed.requires_grad_(False)
            return network

        for skip in (set(), {"absolute_pos_embed"}):
            with self.subTest(no_weight_decay=skip), \
                 patch('dlshogi.ptl.policy_value_network', side_effect=make_network), \
                 patch('dlshogi.network.policy_value_network_exp__by.PolicyValueNetwork.no_weight_decay',
                       return_value=skip), \
                 patch('dlshogi.ptl.build_optimizer',
                       side_effect=AssertionError('fallback must not build a temporary optimizer')):
                cli = CustomLightningCLI(Model, DataModule, run=False, save_config_callback=None, args={
                    'model': {'network': 'exp__by1x8', 'use_ema': True},
                    'data': {'train_files': [], 'val_files': []},
                    'optimizer': {'class_path': 'torch.optim.AdamW',
                                  'init_args': {'lr': 0.001, 'weight_decay': 0.01}},
                    'trainer': {'accelerator': 'cpu', 'logger': False,
                                'enable_checkpointing': False},
                })
                optimizer = cli.model.configure_optimizers()
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(len(optimizer.param_groups), 1)
                self.assertEqual([id(p) for p in optimizer.param_groups[0]['params']],
                                 [id(p) for p in cli.model.parameters()])
                self.assertTrue(any(not p.requires_grad
                                    for p in optimizer.param_groups[0]['params']))

                # The original Lightning path included frozen network and EMA parameters.
                legacy = torch.optim.AdamW(cli.model.parameters(), lr=0.001, weight_decay=0.01)
                for parameter in cli.model.model.parameters():
                    if parameter.requires_grad:
                        parameter.grad = torch.ones_like(parameter)
                legacy.step()
                optimizer.load_state_dict(copy.deepcopy(legacy.state_dict()))
                optimizer.step()


if __name__ == '__main__':
    unittest.main()
