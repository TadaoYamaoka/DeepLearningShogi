import os

from dlshogi import serializers

import argparse
from dlshogi.ptl import Model

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("model")
parser.add_argument('--network', default='resnet10_swish')
args = parser.parse_args()

model = Model.load_from_checkpoint(args.ckpt)

serializers.save_npz(
    os.path.join(args.model),
    model.model,
)
