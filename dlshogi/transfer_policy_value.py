from chainer import serializers

from dlshogi.policy_value_network_old import *
from dlshogi.policy_value_network import *

import re
import argparse

ptn = re.compile('^.*(\d\d).*$')

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('dst', type=str)
args = parser.parse_args()

src = PolicyValueNetworkOld()
dst = PolicyValueNetwork()

print('Load src model from', args.src)
serializers.load_npz(args.src, src)

print('dst model params')
dst_dict = {}
for path, param in dst.namedparams():
    print(path, param.data.shape)
    dst_dict[path] = param

print('policy model params')
for path, param in src.namedparams():
    if path in dst_dict:
    	m = ptn.match(path)
    	if m is not None and int(m.group(1)) > 21:
    		pass
    	else:
		    print(path, param.data.shape)
		    dst_dict[path].data = param.data

print('save the model')
serializers.save_npz(args.dst, dst)
