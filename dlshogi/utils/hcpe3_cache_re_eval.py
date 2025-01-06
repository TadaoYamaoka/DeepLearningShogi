import torch
import argparse
from tqdm import tqdm
import numpy as np
from dlshogi import serializers
from dlshogi.network.policy_value_network import policy_value_network
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.cppshogi import hcpe3_cache_re_eval, hcpe3_create_cache, hcpe3_reserve_train_data

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('cache')
parser.add_argument('out_cache')
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--dropoff', type=float, default=0.5)
parser.add_argument('--limit_candidates', type=int, default=10)
parser.add_argument('--batch_size', '-b', type=int, default=1024)
parser.add_argument('--network')
parser.add_argument('--gpu', '-g', type=int, default=0)
args = parser.parse_args()

alpha = args.alpha
dropoff = args.dropoff
limit_candidates = args.limit_candidates
batch_size = args.batch_size

if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

model = policy_value_network(args.network)
model.to(device)
serializers.load_npz(args.model, model)
model.eval()

data_len, actual_len = Hcpe3DataLoader.load_files([], cache=args.cache)
indexes = np.arange(data_len, dtype=np.uint32)
dataloader = Hcpe3DataLoader(indexes, batch_size, device)

hcpe3_reserve_train_data(data_len)

for i in tqdm(range(0, len(indexes), batch_size)):
    chunk = indexes[i:i + batch_size]
    chunk_size = len(chunk)
    if chunk_size < batch_size:
        chunk_tmp = chunk
        chunk = np.zeros(batch_size, dtype=np.uint32)
        chunk[:chunk_size] = chunk_tmp

    x1, x2, t1, t2, value = dataloader.mini_batch(chunk)
    with torch.no_grad():
        y1, y2 = model(x1, x2)

    y1 = y1.cpu().numpy()
    y2 = torch.sigmoid(y2).cpu().numpy()
    if chunk_size < batch_size:
        hcpe3_cache_re_eval(chunk_tmp, y1[:chunk_size], y2[:chunk_size], alpha, dropoff, limit_candidates)
    else:
        hcpe3_cache_re_eval(chunk, y1, y2, alpha, dropoff, limit_candidates)

hcpe3_create_cache(args.out_cache)
