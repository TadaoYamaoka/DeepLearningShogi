def build_weight_decay_param_groups(model, weight_decay=None):
    """Split trainable parameters using the model's optional no_weight_decay().

    A weight_decay of None leaves the decay group's optimizer default intact.
    Parameter order is stable within each group, including for tied parameters.
    Only full parameter names returned by the model's hook are excluded.
    """
    skip = set(getattr(model, "no_weight_decay", lambda: ())())
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in skip:
            no_decay.append(param)
        else:
            decay.append(param)

    decay_group = {"params": decay}
    if weight_decay is not None:
        decay_group["weight_decay"] = weight_decay
    if no_decay:
        return [decay_group, {"params": no_decay, "weight_decay": 0.0}]
    return [decay_group]
