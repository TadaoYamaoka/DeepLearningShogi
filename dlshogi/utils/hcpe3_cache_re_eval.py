import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
from dlshogi import serializers
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.cppshogi import (
    hcpe3_cache_re_eval,
    hcpe3_cache_write,
    hcpe3_cache_write_end,
    hcpe3_cache_write_start,
    hcpe3_reserve_train_data,
)

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('cache')
parser.add_argument('out_cache')
parser.add_argument('--alpha_p', type=float, default=0.95)
parser.add_argument('--alpha_v', type=float, default=0.95)
parser.add_argument('--alpha_r', type=float, default=0.95)
parser.add_argument('--dropoff', type=float, default=0.5)
parser.add_argument('--limit_candidates', type=int, default=10)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--batch_size', '-b', type=int, default=1024)
parser.add_argument('--network')
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--use_amp', action='store_true')
parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'bfloat16'])
parser.add_argument('--use_compile', action='store_true', help='Use torch.compile')
parser.add_argument('--compile_backend', type=str, help='Backend for torch.compile')
parser.add_argument('--compile_mode', type=str, help='Mode for torch.compile')
parser.add_argument('--compile_fullgraph', action='store_true', help='Use fullgraph=True for torch.compile')
parser.add_argument('--compile_dynamic', action='store_true', help='Use dynamic=True for torch.compile')
args = parser.parse_args()

alpha_p = args.alpha_p
alpha_v = args.alpha_v
alpha_r = args.alpha_r
dropoff = args.dropoff
limit_candidates = args.limit_candidates
temperature = args.temperature
batch_size = args.batch_size
use_amp = args.use_amp
amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

model = policy_value_network(args.network)
model.to(device)
serializers.load_npz(args.model, model)
model.eval()

compiled_model = model
if args.use_compile:
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile is not available. Please use PyTorch 2.0 or later.')

    compile_kwargs = {}
    compile_backend = args.compile_backend
    if compile_backend is None and os.name == 'nt':
        compile_backend = 'aot_eager'
    if compile_backend:
        compile_kwargs['backend'] = compile_backend
    if args.compile_mode:
        compile_kwargs['mode'] = args.compile_mode
    if args.compile_fullgraph:
        compile_kwargs['fullgraph'] = True
    if args.compile_dynamic:
        compile_kwargs['dynamic'] = True

    compiled_model = torch.compile(model, **compile_kwargs)

data_len, actual_len = Hcpe3DataLoader.load_files([], cache=args.cache)
indexes = np.arange(data_len, dtype=np.uint64)
dataloader = Hcpe3DataLoader(indexes, batch_size, device)

hcpe3_reserve_train_data(batch_size)
hcpe3_cache_write_start(args.out_cache, data_len)

for i in tqdm(range(0, len(indexes), batch_size)):
    chunk = indexes[i:i + batch_size]
    chunk_size = len(chunk)
    if chunk_size < batch_size:
        chunk_tmp = chunk
        chunk = np.zeros(batch_size, dtype=np.uint64)
        chunk[:chunk_size] = chunk_tmp

    x1, x2, t1, t2, value = dataloader.mini_batch(chunk)
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
        with torch.no_grad():
            y1, y2 = compiled_model(x1, x2)

            y1 = y1.float().cpu().numpy()
            y2 = y2.float().sigmoid().cpu().numpy()
    if chunk_size < batch_size:
        hcpe3_cache_re_eval(chunk_tmp, y1[:chunk_size], y2[:chunk_size], alpha_p, alpha_v, alpha_r, dropoff, limit_candidates, temperature)
    else:
        hcpe3_cache_re_eval(chunk, y1, y2, alpha_p, alpha_v, alpha_r, dropoff, limit_candidates, temperature)
    hcpe3_cache_write()

hcpe3_cache_write_end()
