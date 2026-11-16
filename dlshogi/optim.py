"""Optimizer helpers driven by model hooks.

A model may define these hooks, each returning full parameter names
(as in ``model.named_parameters()``):

- ``no_weight_decay()``: parameters trained with ``weight_decay=0`` (timm convention).
- ``<tag>_parameters()``: parameters routed to a dedicated optimizer, selected by
  ``build_optimizer(..., routes={tag: spec})``. For example ``matrix_parameters()``
  for matrix-aware optimizers such as Muon.

Every optimizer, including a routed one, gets decay/no-decay groups from
``no_weight_decay()``. Without routes a plain optimizer is returned; with routes
the optimizers are combined into one ``MultiOptimizer``.
"""
import inspect

import torch

DEFAULT_TAG = "default"


def _hook_names(model, hook, named_parameters, required=False):
    fn = getattr(model, hook, None)
    if fn is None:
        if required:
            raise ValueError(f"{type(model).__name__} does not define {hook}()")
        return set()
    names = set(fn())
    unknown = names - named_parameters.keys()
    if unknown:
        raise ValueError(f"{hook}() returned unknown parameters: {sorted(unknown)}")
    return names


def _split_decay(named, skip):
    decay, no_decay = [], []
    for name, param in named:
        (no_decay if name in skip else decay).append(param)

    groups = []
    if decay:
        groups.append({"params": decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _legacy_param_groups(named, skip):
    """Retain the empty decay group used by older hook-aware optimizers.

    Kept separate from _split_decay on purpose: this layout exists only for
    checkpoint compatibility and should not leak into the routed path.
    """
    trainable = [(n, p) for n, p in named.items() if p.requires_grad]
    groups = [{"params": [p for n, p in trainable if n not in skip]}]
    no_decay = [p for n, p in trainable if n in skip]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(model, make, default, routes=None):
    """Build the optimizer for ``model``.

    make(spec, param_groups) -> torch.optim.Optimizer
        Creates an optimizer from a spec (a CLI string, a Lightning init dict, ...).
    default
        Spec for every trainable parameter not claimed by a route.
    routes
        ``{tag: spec}``; ``model.<tag>_parameters()`` is optimized with ``spec``.

    Without routes a plain optimizer is returned (checkpoint compatible with
    earlier versions). With routes a MultiOptimizer is returned:
    - a route must select at least one trainable parameter, otherwise ValueError;
    - the default optimizer is omitted when every trainable parameter is routed.
    """
    named = dict(model.named_parameters())
    skip = _hook_names(model, "no_weight_decay", named)
    routes = dict(routes or {})
    if not routes:
        # Preserve the original optimizer layout for ordinary checkpoint resumes.
        # Without the hook, model.parameters() included frozen parameters too.
        if not hasattr(model, "no_weight_decay"):
            return make(default, [{"params": list(named.values())}])
        return make(default, _legacy_param_groups(named, skip))

    trainable = {n: p for n, p in named.items() if p.requires_grad}

    def groups_for(names):
        return _split_decay(((n, p) for n, p in trainable.items() if n in names), skip)

    selected, claimed = {}, set()
    for tag in routes:
        if tag == DEFAULT_TAG:
            raise ValueError(f"'{DEFAULT_TAG}' is reserved for the default optimizer")
        names = _hook_names(model, f"{tag}_parameters", named, required=True)
        if names & claimed:
            raise ValueError(f"{tag}_parameters() overlaps another route")
        if names.isdisjoint(trainable):
            raise ValueError(f"{tag}_parameters() selects no trainable parameters")
        selected[tag] = names
        claimed |= names

    optimizers = {}
    rest = groups_for(trainable.keys() - claimed)
    if rest:
        optimizers[DEFAULT_TAG] = make(default, rest)
    for tag, spec in routes.items():
        optimizers[tag] = make(spec, groups_for(selected[tag]))
    return MultiOptimizer(optimizers, named.items())


class MultiOptimizer(torch.optim.Optimizer):
    """Tagged optimizers over disjoint parameters, exposed as one Optimizer.

    ``param_groups`` holds the children's group dicts (the same objects) and all
    children share this optimizer's ``state`` dict, so LR schedulers, GradScaler,
    gradient clipping, Lightning and checkpoints treat it as a plain Optimizer.

    ``state_dict()`` records the layout (tag, class, parameter names of every
    group); ``load_state_dict()`` rejects a checkpoint whose layout differs, so
    state is never restored into the wrong parameters.

    Supported children: optimizers whose ``step()`` works without a closure
    (SGD, Adam(W), Muon, ...). The closure, if any, is evaluated once here.
    Optimizers that require a closure (e.g. LBFGS) are rejected.

    Notes:
    - Parameter groups are fixed at construction; later add_param_group() calls
      are rejected because each group must belong to a tagged child optimizer.
    - ``defaults`` is empty, so schedulers that read ``optimizer.defaults``
      (OneCycleLR / CyclicLR with cycle_momentum=True) are not supported.
    - Do not call a child's ``state_dict()``; the shared state also holds the
      other children's parameters.
    """

    def __init__(self, optimizers, named_parameters):
        optimizers = dict(optimizers)
        if not optimizers:
            raise ValueError("MultiOptimizer needs at least one optimizer")
        for tag, opt in optimizers.items():
            closure = inspect.signature(opt.step).parameters.get("closure")
            if closure is not None and closure.default is inspect.Parameter.empty:
                raise TypeError(f"{type(opt).__name__} ({tag}) requires a closure "
                                "and cannot be used in MultiOptimizer")
        self.optimizers = optimizers
        # Raises if a parameter appears in more than one group.
        self._initializing = True
        super().__init__([g for opt in optimizers.values() for g in opt.param_groups], {})
        self._initializing = False

        names = {id(p): name for name, p in named_parameters}
        if any(id(p) not in names for g in self.param_groups for p in g["params"]):
            raise ValueError("named_parameters must name every optimized parameter")
        self._layout = [
            [tag, f"{type(opt).__module__}.{type(opt).__qualname__}",
             [[names[id(p)] for p in g["params"]] for g in opt.param_groups]]
            for tag, opt in optimizers.items()
        ]

        for opt in optimizers.values():
            self.state.update(opt.state)
        self._link()

    def add_param_group(self, param_group):
        if not self._initializing:
            raise RuntimeError("MultiOptimizer does not support adding parameter groups after construction")
        super().add_param_group(param_group)

    def _link(self):
        offset = 0
        for opt in self.optimizers.values():
            count = len(opt.param_groups)
            opt.param_groups = self.param_groups[offset:offset + count]
            opt.state = self.state
            offset += count

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers.values():
            opt.step()
        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["layout"] = self._layout
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        if state_dict.pop("layout", None) != self._layout:
            raise ValueError("Optimizer layout (tags, classes or parameters) changed; "
                             "resume with --reset_optimizer")
        super().load_state_dict(state_dict)  # replaces param_groups and state
        self._link()

    def __getstate__(self):
        return {**super().__getstate__(), "optimizers": self.optimizers,
                "_layout": self._layout, "_initializing": False}

    def __repr__(self):
        children = ", ".join(f"{tag}={opt!r}" for tag, opt in self.optimizers.items())
        return f"{type(self).__name__}({children})"
