"""CLI spec resolution preserves scheduler references and train.py globals."""
import pytest
import torch

pytest.importorskip("dlshogi.cppshogi", reason="requires the built cppshogi extension")

from dlshogi.train import resolve_spec


def test_optimizer_with_omitted_parentheses():
    cls, kwargs = resolve_spec('AdamW', torch.optim)
    assert cls is torch.optim.AdamW
    assert kwargs == {}


def test_scheduler_with_omitted_parentheses():
    cls, kwargs = resolve_spec('ExponentialLR', torch.optim.lr_scheduler)
    assert cls is torch.optim.lr_scheduler.ExponentialLR
    assert kwargs == {}


def test_nested_scheduler_uses_optimizer_and_module_globals():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    cls, kwargs = resolve_spec(
        'torch.optim.lr_scheduler.SequentialLR('
        'schedulers=[optim.lr_scheduler.LinearLR(optimizer, total_iters=2), '
        'torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)], milestones=[2])',
        torch.optim.lr_scheduler, {"optimizer": optimizer},
    )
    scheduler = cls(optimizer, **kwargs)
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert all(child.optimizer is optimizer for child in kwargs['schedulers'])
