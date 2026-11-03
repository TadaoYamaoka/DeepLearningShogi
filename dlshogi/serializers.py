import numpy as np
from collections import OrderedDict
import torch

def load_npz(file, obj, remove_aux=False):
    state_dict = OrderedDict()
    with np.load(file) as f:
        for key in f.keys():
            if remove_aux and '_aux' in key:
                continue
            # Chainer形式で保存されたパラメータ名との互換性を維持するために変換する。
            names = key.split('/')
            if names[-1] == 'W':
                names[-1] = 'weight'
            elif names[-1] == 'b':
                names[-1] = 'bias'
            elif names[-1] == 'avg_mean':
                names[-1] = 'running_mean'
            elif names[-1] == 'avg_var':
                names[-1] = 'running_var'
            elif names[-1] == 'gamma':
                names[-1] = 'weight'
            elif names[-1] == 'beta':
                names[-1] = 'bias'
            elif names[-1] == 'N':
                names[-1] = 'num_batches_tracked'
            state_dict['.'.join(names)] = torch.tensor(f[key])
    obj.load_state_dict(state_dict)

def save_npz(file, obj):
    state_dict = {key: value.cpu().numpy() for key, value in obj.state_dict().items()}
    with open(file, 'wb') as f:
        np.savez_compressed(f, **state_dict)
