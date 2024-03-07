import argparse

import torch

from dlshogi.common import *
from dlshogi.network.policy_value_network import policy_value_network
from torch.nn.utils import prune

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("output_checkpoint")
parser.add_argument("--network", default="resnet10_swish", help="network type")
parser.add_argument("--amount", type=float, default=0.05)
args = parser.parse_args()

device = torch.device("cpu")
model = policy_value_network(args.network)

checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint["model"])

# prune
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", n=1, dim=0, amount=args.amount)
        prune.remove(module, "weight")

checkpoint["model"] = model.state_dict()
torch.save(checkpoint, args.output_checkpoint)
