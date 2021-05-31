import torch
import torch.nn as nn

def policy_value_network(network, add_sigmoid=False):
    if network == 'wideresnet10':
        from dlshogi.network.policy_value_network_wideresnet10 import PolicyValueNetwork
    elif network == 'wideresnet15':
        from dlshogi.network.policy_value_network_wideresnet15 import PolicyValueNetwork
    elif network == 'resnet10_swish':
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
    elif network == 'resnet15_swish':
        from dlshogi.network.policy_value_network_resnet15_swish import PolicyValueNetwork
    elif network == 'resnet20_swish':
        from dlshogi.network.policy_value_network_resnet20_swish import PolicyValueNetwork
    elif network[:5] == 'senet':
        from dlshogi.network.policy_value_network_senet import PolicyValueNetwork
    else:
        names = network.split('.')
        if len(names) == 1:
            PolicyValueNetwork = globals()[names[0]]
        else:
            from importlib import import_module
            PolicyValueNetwork = getattr(import_module('.'.join(names[:-1])), names[-1])

    if add_sigmoid:
        class PolicyValueNetworkAddSigmoid(PolicyValueNetwork):
            def __init__(self, *args, **kwargs):
                super(PolicyValueNetworkAddSigmoid, self).__init__(*args, **kwargs)

            def __call__(self, x1, x2):
                y1, y2 = super(PolicyValueNetworkAddSigmoid, self).__call__(x1, x2)
                return y1, torch.sigmoid(y2)

        PolicyValueNetwork = PolicyValueNetworkAddSigmoid

    if network == 'senet10':
        return PolicyValueNetwork(blocks=10, channels=192)
    elif network == 'senet10_swish':
        return PolicyValueNetwork(blocks=10, channels=192, activation=nn.SiLU())
    elif network == 'senet15_swish':
        return PolicyValueNetwork(blocks=15, channels=224, activation=nn.SiLU())
    elif network == 'senet20_swish':
        return PolicyValueNetwork(blocks=20, channels=256, activation=nn.SiLU())
    else:
        return PolicyValueNetwork()
