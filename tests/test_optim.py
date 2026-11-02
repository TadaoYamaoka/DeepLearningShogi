"""pytest test_optim.py  (CPU only, no dlshogi data needed)"""
import copy
import pickle

import pytest
import torch
import torch.nn as nn

from dlshogi.optim import MultiOptimizer, build_optimizer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))

    def no_weight_decay(self):
        return {"a.bias", "b.bias"}

    def matrix_parameters(self):
        return {"a.weight"}

    def alpha_parameters(self):
        return {"a.weight", "a.bias"}

    def beta_parameters(self):
        return {"b.weight", "b.bias"}

    def all_parameters(self):
        return {name for name, _ in self.named_parameters()}


ADAMW = (torch.optim.AdamW, dict(lr=1e-2, weight_decay=0.1))
SGD = (torch.optim.SGD, dict(lr=1e-1, momentum=0.9, weight_decay=0.01))
ROUTES = {"matrix": SGD}


def make(spec, groups):
    cls, kwargs = spec
    return cls(groups, **kwargs)


def train(model, optimizers, steps=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for opt in optimizers:
            opt.zero_grad()
        model(torch.randn(16, 4, generator=gen)).square().mean().backward()
        for opt in optimizers:
            opt.step()


def test_plain_optimizer_without_routes():
    opt = build_optimizer(Net(), make, ADAMW)
    assert type(opt) is torch.optim.AdamW
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]


def test_matches_separate_optimizers():
    m1 = Net()
    m2 = copy.deepcopy(m1)
    opt = build_optimizer(m1, make, ADAMW, ROUTES)
    assert isinstance(opt, MultiOptimizer) and list(opt.optimizers) == ["default", "matrix"]
    ref = [
        torch.optim.AdamW([{"params": [m2.b.weight]},
                           {"params": [m2.a.bias, m2.b.bias], "weight_decay": 0.0}],
                          lr=1e-2, weight_decay=0.1),
        torch.optim.SGD([m2.a.weight], lr=1e-1, momentum=0.9, weight_decay=0.01),
    ]
    train(m1, [opt])
    train(m2, ref)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_state_dict_roundtrip():
    m1 = Net()
    opt1 = build_optimizer(m1, make, ADAMW, ROUTES)
    train(m1, [opt1])
    m2 = copy.deepcopy(m1)
    opt2 = build_optimizer(m2, make, ADAMW, ROUTES)
    opt2.load_state_dict(copy.deepcopy(opt1.state_dict()))
    assert all(child.state is opt2.state for child in opt2.optimizers.values())
    assert all(g in opt2.param_groups for c in opt2.optimizers.values() for g in c.param_groups)
    train(m1, [opt1], seed=1)
    train(m2, [opt2], seed=1)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_layout_mismatch_is_rejected():
    m = Net()
    plain = build_optimizer(m, make, ADAMW)
    multi = build_optimizer(m, make, ADAMW, ROUTES)
    with pytest.raises(ValueError, match="layout"):
        multi.load_state_dict(plain.state_dict())


def test_route_order_change_is_rejected():
    # Same classes and group counts, different parameters per position.
    m = Net()
    ab = build_optimizer(m, make, ADAMW, {"alpha": ADAMW, "beta": ADAMW})
    train(m, [ab])
    ba = build_optimizer(m, make, ADAMW, {"beta": ADAMW, "alpha": ADAMW})
    with pytest.raises(ValueError, match="layout"):
        ba.load_state_dict(ab.state_dict())


def test_default_is_omitted_when_everything_is_routed():
    m = Net()
    opt = build_optimizer(m, make, ADAMW, {"all": SGD})
    assert list(opt.optimizers) == ["all"]
    train(m, [opt])


def test_route_without_trainable_parameters_is_rejected():
    m = Net()
    m.a.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable"):
        build_optimizer(m, make, ADAMW, ROUTES)


def test_invalid_routes():
    with pytest.raises(ValueError, match="does not define"):
        build_optimizer(Net(), make, ADAMW, {"embedding": ADAMW})
    with pytest.raises(ValueError, match="overlaps"):
        build_optimizer(Net(), make, ADAMW, {"matrix": SGD, "alpha": ADAMW})
    with pytest.raises(ValueError, match="reserved"):
        build_optimizer(Net(), make, ADAMW, {"default": SGD})


def test_closure_required_optimizer_is_rejected():
    with pytest.raises(TypeError, match="closure"):
        build_optimizer(Net(), make, ADAMW, {"matrix": (torch.optim.LBFGS, dict(lr=1.0))})


def test_scheduler_updates_every_child():
    opt = build_optimizer(Net(), make, ADAMW, ROUTES)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    opt.step()
    sched.step()
    assert opt.optimizers["default"].param_groups[0]["lr"] == pytest.approx(5e-3)
    assert opt.optimizers["matrix"].param_groups[0]["lr"] == pytest.approx(5e-2)


def test_grad_scaler():
    m = Net()
    opt = build_optimizer(m, make, ADAMW, ROUTES)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    before = [p.detach().clone() for p in m.parameters()]
    scaler.scale(m(torch.randn(16, 4)).square().mean()).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    assert all(not torch.equal(b, p) for b, p in zip(before, m.parameters()))


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon not available")
def test_muon_route():
    m = Net()
    opt = build_optimizer(m, make, ADAMW,
                          {"matrix": (torch.optim.Muon, dict(lr=1e-2, adjust_lr_fn="match_rms_adamw"))})
    train(m, [opt])


@pytest.mark.parametrize("resume", [False, True])
def test_add_param_group_after_construction_is_rejected(resume):
    model = Net()
    opt = build_optimizer(model, make, ADAMW, ROUTES)
    if resume:
        train(model, [opt])
        opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    groups_before = list(opt.param_groups)
    layout_before = copy.deepcopy(opt.state_dict()["layout"])
    extra = nn.Parameter(torch.ones(4))
    with pytest.raises(RuntimeError, match="adding parameter groups"):
        opt.add_param_group({"params": [extra], "lr": 0.1})
    assert len(opt.param_groups) == len(groups_before)
    assert all(a is b for a, b in zip(opt.param_groups, groups_before))
    assert opt.state_dict()["layout"] == layout_before
    train(model, [opt])


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda opt: pickle.loads(pickle.dumps(opt))],
                         ids=["deepcopy", "pickle"])
def test_optimizer_copy_preserves_children_and_updates(clone):
    model = Net()
    opt = build_optimizer(model, make, ADAMW, ROUTES)
    train(model, [opt])
    copied = clone(opt)
    assert copied.state_dict()["layout"] == opt.state_dict()["layout"]
    assert all(child.state is copied.state for child in copied.optimizers.values())
    assert all(any(group is parent for parent in copied.param_groups)
               for child in copied.optimizers.values() for group in child.param_groups)
    original_params = [p for group in opt.param_groups for p in group["params"]]
    copied_params = [p for group in copied.param_groups for p in group["params"]]
    for original, duplicate in zip(original_params, copied_params):
        assert original is not duplicate
        original.grad = torch.ones_like(original)
        duplicate.grad = torch.ones_like(duplicate)
    opt.step()
    copied.step()
    for original, duplicate in zip(original_params, copied_params):
        torch.testing.assert_close(original, duplicate)
    with pytest.raises(RuntimeError, match="adding parameter groups"):
        copied.add_param_group({"params": [nn.Parameter(torch.ones(4))]})


@pytest.mark.parametrize("case", ["frozen_without_hook", "all_no_decay"])
def test_plain_optimizer_resumes_original_checkpoint_layout(case):
    model = nn.Linear(4, 2)
    if case == "frozen_without_hook":
        model.weight.requires_grad_(False)
        original = torch.optim.AdamW(model.parameters(), **ADAMW[1])
    else:
        model.no_weight_decay = lambda: {"weight", "bias"}
        original = torch.optim.AdamW([
            {"params": []},
            {"params": list(model.parameters()), "weight_decay": 0.0},
        ], **ADAMW[1])

    model(torch.ones(2, 4)).square().mean().backward()
    original.step()
    restored_model = copy.deepcopy(model)
    resumed = build_optimizer(restored_model, make, ADAMW)
    assert [len(g["params"]) for g in resumed.param_groups] == [
        len(g["params"]) for g in original.param_groups
    ]
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    for net, opt in ((model, original), (restored_model, resumed)):
        opt.zero_grad()
        net(torch.ones(2, 4)).square().mean().backward()
        opt.step()
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(expected, actual)


def test_route_preserves_model_parameter_order():
    model = Net()
    opt = build_optimizer(model, make, ADAMW, {"all": SGD})
    groups = opt.state_dict()["layout"][0][2]
    assert groups == [["a.weight", "b.weight"], ["a.bias", "b.bias"]]
