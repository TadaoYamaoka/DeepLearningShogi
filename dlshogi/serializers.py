import numpy as np
from collections import OrderedDict
import torch

def load_npz(file, obj, remove_aux=False):
    state_dict = OrderedDict()
    with np.load(file) as f:
        for key in f.keys():
            if remove_aux and '_aux' in key:
                continue
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
    state_dict = obj.state_dict()
    state_dict2 = OrderedDict()
    for key in state_dict.keys():
        names = key.split('.')
        if key.find('norm') < 0 and key.find('bn') < 0:
            if names[-1] == 'weight':
                names[-1] = 'W'
            elif names[-1] == 'bias':
                names[-1] = 'b'
        else:
            if names[-1] == 'running_mean':
                names[-1] = 'avg_mean'
            elif names[-1] == 'running_var':
                names[-1] = 'avg_var'
            elif names[-1] == 'weight':
                names[-1] = 'gamma'
            elif names[-1] == 'bias':
                names[-1] = 'beta'
            elif names[-1] == 'num_batches_tracked':
                names[-1] = 'N'
        state_dict2['/'.join(names)] = state_dict[key].cpu().numpy()
    with open(file, 'wb') as f:
        np.savez_compressed(f, **state_dict2)