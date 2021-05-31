from cshogi.usi import Engine
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('engine')
parser.add_argument('model')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--nodelimit', type=int, default=10000000)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--byoyomi', type=int, default=5000)
parser.add_argument('--options')
args = parser.parse_args()

engine = Engine(args.engine, debug=True)
engine.setoption('USI_Ponder', 'false')
engine.setoption('Resign_Threshold', '0')
engine.setoption('PV_Interval', '0')
engine.setoption('DNN_Model', args.model)
engine.setoption('Byoyomi_Margin', '0')
engine.setoption('UCT_NodeLimit', str(args.nodelimit))
engine.setoption('DNN_Batch_Size', str(args.batch))
engine.setoption('ReuseSubtree', 'false')
engine.setoption('UCT_Threads', str(args.threads))
for i in range(2, args.gpus + 1):
    engine.setoption('UCT_Threads' + str(i), str(args.threads))
if args.options:
    for option in args.options.split(','):
        name, value = option.split(':')
        engine.setoption(name, value)
engine.isready()

positions = [
    '',
    '7g7f 7a6b 2g2f 4a3b 2f2e 8c8d 6i7h 5a4a 2e2d 2c2d 2h2d P*2c 2d2h 8d8e 3i3h 3c3d 3h2g 8e8f 8g8f 8b8f 2g3f 8f8d 3f4e 4a5b P*8e 8d8e 4e3d 8e3e 8h2b+ 3a2b B*5f 2c2d P*8b B*5e 8b8a+ 5e9i+ 8i7g L*5d 3g3f 3e3f 3d4e 5d5f 4e3f 5f5g+ N*6i 5g5f 8a9a P*3e 3f3e P*8h 7i8h 9i8i 5i6h B*6d 2h2f',
    '2g2f 8c8d 2f2e 4a3b 7g7f 8d8e 2e2d 2c2d 2h2d P*2c 2d2f 3c3d 6i7h 7a7b P*2d 2c2d 2f2d 5a4b 2d3d 2b3c 3d3f 3a2b 5i5h 3c8h+ 7i8h B*2g 3f2f 2g5d+ 8h7g 6c6d 3i3h 8e8f 8g8f 7c7d 3g3f 7d7e 7f7e P*7f 7g8h 8b8f P*8g 8f8d P*2d P*2c 2d2c+ 2b2c 3f3e P*2d',
    '7g7f 8b4b 2g2f 3c3d 5i6h 2b8h+ 7i8h 3a2b 3i4h 5a6b 4g4f 6b7b 4h4g 7b8b 9g9f 9c9d 4i5h 2b3c 6h7h 4b2b 4g5f 7a7b 8g8f 3c4d B*7g 2b5b 2f2e 5c5d 2e2d 5d5e 5f4g 2c2d 2h2d P*2b 8h8g 8c8d 7g6f 7b8c 3g3f 4a3b 2i3g 2a3c 8i7g 6c6d 6i6h 6a7b 8f8e 8d8e P*8d 8c7d 4f4e 3c4e 3g4e 4d4e 2d2e 4c4d 5g5f 9d9e 9f9e P*9h 9i9h N*8f 7h6i 8f9h+ 8g9h 9a9e P*9g B*4c N*8g 3d3e 2e2g L*2e P*2f 4c7f 7g8e 7d8e N*7e 7c7d 8d8c+ 7b8c',
    '7g7f 3c3d 2g2f 4c4d 3i4h 3a4b 5i6h 4b4c 4i5h 8c8d 2f2e 4a3b 2e2d 2c2d 2h2d P*2c 2d2f 8d8e',
    '7g7f 3c3d 6g6f 8c8d 2h6h 7a6b 1g1f 5a4b 1f1e 4b3b 3i3h 5c5d 7i7h 2b3c 8h7g 6b5c 7h6g 6a5b 4g4f 4c4d 6i5h 5b4c 5i4h 3b2b 4h3i 1a1b 3i2h 2b1a 3g3f 3a2b 5h4g 4a3a 2i3g 7c7d 6f6e 8d8e 2g2f 3c5a 3h2g 9c9d 9g9f 5c4b 6g5f 5a7c 4i3h 4b3c 6h6i 8b5b 5f6g 5b6b 6g5f 6b5b 4g4h 7c4f 6e6d 4f6d 5f6e 5b6b 6e6d 6c6d 3g2e 3c4b 5g5f S*7h 6i5i 7h6g+ B*7a 6b5b 7g4d 4b5c 4d5c+ 4c5c S*6a B*4f 4h3g 4f3g+ 2h3g',
    '7g7f 3c3d 6g6f 6c6d 2h6h 7a7b 6f6e 6d6e 6h6e 6a5b P*6d P*6c 5i4h 6c6d 6e6d 7b6c 6d6h 8b6b 4h3h 5a4b 3i4h 4b3b 4i3i 1c1d 8h2b+ 3a2b 7i7h 1d1e 3h2h 2b3c 7h6g 6c5d B*7g 7c7d 6g5f P*6e 5f5e 5d5e 7g5e S*6d 5e8h 5c5d 9g9f 6d5e P*6g 3c4d 6i7h 2a3c 6h6i 2c2d 2h3h 2d2e 6i6h 4a4b S*2d 6b6a 8h5e 5d5e 4g4f 6a2a S*2c 3b4a 2c3d+ 2e2f 2g2f 5e5f 5g5f B*1b 3d4d 4c4d 2d3e 1b5f S*4g 5f2c 3e4d P*5f P*5c 5b6b 6h6i 4a3b 8i7g 6b5c 4d5c 4b5c 6i5i S*5e P*5g 5f5g+ 4h5g P*5h 5i4i 6e6f P*5f 6f6g+ 7h6g 5e4d 3g3f 2c3d 7g6e 5c4c G*3e 4d3e',
    '7g7f 3c3d 2h7h 1c1d 7i6h 7a6b 5i4h 1d1e 4h3h 5a4b 3h2h 2b8h+ 7h8h 6c6d 6g6f 6b6c 6h6g 4b3b 8h6h 6c5d 6g5f 8b6b 6i5h 6a5b 8g8f 7c7d 8f8e B*3c 3i3h 8a7c 9i9h 4a4b 4g4f 6d6e 6f6e 7c8e 6e6d 8e7g+ 8i7g 3c7g+ B*7c 6b6a N*6f N*4d 6f5d 5c5d 6h6e 4d5f S*7b N*6f 6e6f 6a7a 6d6c+ 7g6f',
    '2g2f 8c8d 7g7f 4a3b 6i7h 8d8e 8h7g 3c3d 7i8h 2b7g+ 8h7g 3a4b 3i3h 7a7b 4g4f 6c6d 3h4g 7b6c 5i6h 5a4a 1g1f 1c1d 4g5f 4a3a',
    '7g7f 8c8d 2g2f 3c3d 2f2e 4a3b 5i5h 5a4a 2e2d 2c2d 2h2d 5c5d 3g3f 7a6b 2i3g 5d5e 3i3h 6b5c 2d2e P*2c 2e5e 2b5e 8h5e 5c4d 5e7g 3a4b 4g4f 4b5c 7i7h R*2h P*2g 5c6d 4i3i 2h3h+ 3i3h S*6e 4f4e 4d5e',
    '7g7f 3c3d 2g2f 5c5d 2f2e 8b5b 4i5h 5a6b 8h6f 6b7b 5i6h 4c4d 2e2d 2c2d 2h2d 5b3b 2d2c+ 3a4b 2c2h 4b5c 6h7h 7a6b P*2d 4a5a 2d2c+ 5a4a 2c3b 4a3b 5g5f 5c6d 3i4h 9c9d 9g9f 6d6e 6f8h 6e7f 4h5g P*2c 5g6f 7c7d 7i6h 8a7c 6h7g 7f7g+ 8h7g 7c8e 7g6h S*7f 7h8h 4d4e',
    '7g7f 8c8d 7i6h 3c3d 6g6f',
    '7g7f 3c3d 6g6f 8c8d 1g1f 8d8e 8h7g 7a6b 1f1e 5a4b 2h6h 5c5d 3i3h 4b3b 7i7h 2b4d 7h6g 4d5c 6h8h 7c7d 5i4h 6a5b 4h3i 2a3c 4g4f',
    '2g2f 8c8d 7g7f 5a4b 2f2e 3c3d 2e2d 2c2d 2h2d 4a3b 6g6f 8d8e 8h7g 7c7d 6i7h 7a7b 7i6h 7d7e 6h6g 7e7f 6g7f 1c1d 2d2h P*2c 4i5h 6c6d 3i4h 6a5b 3g3f 7b8c',
    '7g7f 3c3d 2g2f 4c4d 2f2e 2b3c 3i4h 8b2b 5i6h 5a6b 6h7h 6b7b 8h7g',
    '2g2f 8c8d 7g7f 3c3d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 3d3f 3a2b P*8g 8f8d 3f2f 8d3d 3i3h 5a6b 5i6h 7a7b 2f2h 6b7a 9g9f 5c5d 3g3f 7a8b 3h3g 3d2d P*2e 2d3d 4i4h 9c9d 8h6f 3c6f 6g6f 2a3c B*6g 3d4d 3g4f 6c6d 7h7g P*2f 4h3h 2b2c 4f3e B*5e 2i3g 6d6e 2h2f 6e6f 6g5f P*3d 3e4d 5e4d 2f2i 3d3e 3f3e 5d5e 5f7h P*3f 3g4e 3f3g+ 3h3g S*6g 7h6g 6f6g+ 6h6g 3c4e 3g4f 4e5g+ 6g5g N*6e 5g6g 6e7g+ 6g7g 5e5f S*5e 5f5g+ 5e4d 4c4d N*8d G*6g 7g8f 7b8c N*7e 8c7d P*6b 6a6b P*6c 9d9e B*9d P*8e 8f9g 9e9f 9g9f 9a9d P*9e 9d9e 9f9e P*9d',
    '7g7f 5a4b 3i4h 3c3d 5g5f 8c8d 4h5g 8d8e 5g4f 4a3b 6i7h 8e8f 8g8f 8b8f 5i6i',
    '7g7f 8c8d 7i6h 3c3d 6g6f 7a6b 5g5f 5c5d 3i4h 3a4b 4i5h 6c6d 6i7h 6b6c',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d',
    '7g7f 3c3d 2g2f 4c4d 3i4h 8b4b 5g5f 5a6b 5i6h 7a7b 4i5h 9c9d 6h7h 6b7a 4h5g 7a8b 9g9f 4d4e 3g3f 4a3b 2f2e 2b8h+ 7h8h 3a2b 7i7h 2b3c 8i7g 3c4d 2e2d 2c2d 2h2d P*2c 2d2h 3d3e 3f3e 4d3e B*6f 3b2b 8g8f B*6d 5f5e P*3f 5g5f 4e4f 4g4f 3e4f P*3h 5c5d 5e5d 4f5g 5h5g 6d2h+ 5d5c+ 4b4h+ 5g5h 4h4i 6f2b+ P*5b 5h5i 4i2i 5c5b 6a5b 2b4d N*7a P*5c 5b5a G*5b P*5h 6i5h 5a6a S*6b 6a6b 5b6b R*7d 4d3e S*4d 5f6e 4d3e 6e7d',
    '7g7f 3c3d 2g2f 8c8d 6i7h 8d8e 8h2b+ 3a2b 7i8h 7a7b 3i4h 7c7d 4g4f 8a7c 4h4g 2b3c 8h7g 6c6d 4g5f 6a5b 5i6h 9c9d 9g9f 7b6c 1g1f 8b8a 1f1e 5a6b 4i5h 5c5d 6g6f 4a3b 3g3f 6b7b 6h7i 3c4d 2f2e 3d3e 3f3e 4d3e 5h4g P*3f 4f4e 5b6b 7i8h 8e8f 7g8f 1c1d 1e1d 3b4b B*2b B*3c 2b3c+ 2a3c P*3d 3c4e 5f4e 4b5b 4e5f 5d5e 5f6g B*6i 2h4h 3e4d 1d1c+ 6d6e 6f6e 7c6e P*6f 3f3g+ 2i3g P*3f 6f6e 3f3g+ 4g3g 5e5f 6g5f 6i7h+ 8h7h N*4e 5f4e 4d4e 4h4e G*5d 4e4h P*5f 5g5f 5d6e B*3f P*6f P*6h S*5g 4h2h 1a1c 1i1c+ 5b5c N*5e 6e5e 5f5e P*8e 8f9g N*5f S*7g 5g6h+ 7g6h 5f6h+ 7h6h',
    '7g7f 3c3d 2g2f 4c4d 3i4h 8b4b 5i6h 5a6b 6h7h 3a3b 5g5f 6b7b 4i5h 9a9b 7i6h 7b8b 9g9f 8b9a 2f2e 2b3c 5f5e 7a8b 8h9g 4a5b 5h5g 6a7a 5g5f 5b6b 1g1f 3b4c 4h5g 4b2b 6i7i 2c2d 2e2d 3c2d 4g4f 4d4e 5e5d P*2g 2h2g 4e4f 5d5c+ 4f4g+ 5c6b 4g5g 5f5g 7a6b 9g3a+ 2b2c P*4f 2a3c',
    '7g7f 3c3d 2g2f 4a3b 2f2e 2b8h+ 7i8h 3a2b 3i4h 2b3c 3g3f 7a6b 4h3g 7c7d 8h7g 6b7c 4i5h 7c6d 6g6f 5a4b 5i6h 8c8d 3g4f 8d8e 3f3e 3d3e 4f3e 7d7e P*3d 3c2b 7f7e 8e8f 8g8f P*7f 7g8h 6d7e B*4f 6c6d 6i7h 8b8f 2e2d 2c2d',
    '7g7f 3c3d 2g2f 2b8h+ 7i8h 3a4b 8h7g 4b3c 3i4h 8b2b B*6e B*7d 6e4c+ 6a5b 4c5b 4a5b G*7e 2b4b 4i5h 6c6d 7e6d 7d8e 5g5f 5b4c 6g6f 4b6b 6f6e 5a4b 4h5g 7a7b 9g9f 8e4a 5i6h 7c7d 6h7h 7d7e 7f7e 7b7c 6i6h 7c6d 6e6d 6b6d 2f2e 4b3a 3g3f 4c4b 2i3g 8a7c S*7f',
    '6g6f 8c8d 7g7f 3c3d 7i7h 7a6b 7h6g 5c5d 3i3h 3a4b 4g4f 4b3c 3g3f 2b3a 3h4g 8d8e 6i7h 8e8f 8g8f 3a8f 5i6i P*8g 7h8g 8f5i+ 4i5i 8b8g+ 2h6h 7c7d 2i3g 8a7c 6i7i 7d7e 7f7e 8g8d 5i6i G*8g 6i7h 8g8h 7h8h 8d7e P*7f 7e8d P*8g B*5i 6h3h P*8f 8g8f 5i8f+ 8h7h 8f5i P*8g 7c8e G*5h 5i4i 3h2h 4i3i 2h1h P*7g 7h8h 3d3e 8g8f 3e3f 4g3f 3i2i B*5i 2i1h 1i1h R*2h 3f4g 2h1h+ 8f8e 8d8e P*8g 8e3e P*3f 3e2d 8h7g 2d2g 7g7h',
    '2g2f 1c1d 2f2e 4a3b 6i7h 7a6b 7g7f 8c8d 3i4h 8d8e 8h7g',
    '5i6h 8c8d 6h7h 8d8e 3i4h 7c7d 2g2f 7a6b 2f2e 4a3b 2e2d 2c2d 2h2d 3c3d 7g7f 6b7c P*2c 2b8h+ 7i8h 7d7e 7f7e 8e8f 8g8f B*3c 2d2f 3c4d',
    '7g7f 8c8d 7i6h 3c3d 6h7g 7a6b 3i4h 8d8e 6i7h 5a4b 2g2f 6a5b 2f2e 2b3c 5g5f 3a2b 8h7i 7c7d 3g3f 8a7c 7i6h 4a3b 4i5h 4b3a 5i6i 3c4b 6g6f 2b3c 4h3g 5c5d 5h6g 6c6d 9g9f 3a2b 3f3e 3d3e 6h3e 6b5c 6i7i 5c4d 3e6h 1c1d 3g3f 6d6e 6f6e 7c6e 7g6f 8e8f 8g8f 4b6d 2h1h 6d8f 6h8f 8b8f P*8g 8f8a 4g4f B*4i 1h6h P*6d 4f4e P*3e 3f4g 4d4e P*4f P*8f 8g8f 4e3d 4g3h 4i6g+ 6h6g 5d5e B*1h 8a8f P*8g 8f8e B*7c 5e5f P*5h 5b5c 7c6b+ 3c4b 2e2d 2c2d 7i8h P*8f 8i9g 8e8b',
    '5i4h 3a4b 9g9f 8c8d 4h5i 3c3d 2h4h 7a7b 3i2h 4a3b 7i6h 7b8c 4i3h 8c7d 6g6f 2b6f 8h9g 5a4a 9g7e 7d7e 2h3i 6a5b 4h5h 5c5d 1g1f 7c7d 3i4h 5b5a 2g2f 5a5b 3h2h 5b5a 9f9e',
    '2g2f 8c8d 7g7f 8d8e 2f2e 4a3b 8h7g 3c3d 7i6h 3a4b 7g2b+ 3b2b 6h7g 4b3c 3i4h 7c7d B*4f',
    '2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 3i4h 3c3d 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f P*8g P*2c 2d2h 8f8b 4g4f 7c7d 7g7f 6c6d 8h2b+ 3a2b B*6c 8b8d 6c4e+ 7a6b 4e3d 8a7c 3d5f 6b6c 7i6h 5a4b 6g6f 6c5d 4h4g 2b3c 4g3f 6a5b 3f4e',
    '7g7f 3c3d 2g2f 8c8d 6i7h 4c4d 2f2e 2b3c 3i4h 3a3b 5g5f 3b4c 7i6h 7a7b 5i6i 9c9d 4i5h 7b8c 4h5g 9d9e 4g4f 5a6b 6g6f 6b7a 6h6g 8b4b 6f6e 4c5d 5g6f 8c7d 7f7e 7d6e 5f5e 6e6f 6g6f 5d4c 6f6e 6a7b 7e7d 7c7d 6e7d S*6d S*6f 4b6b 8i7g 4a5b 2e2d 3c2d P*2b P*7c 2b2a+ 7c7d 2a1a S*7e 6f5g 6d5e N*1f 5e4f 1f2d 4f5g 5h5g 2c2d 2h2d 8a7c 2d2a+ 6b6a 2a2b 7e7f S*6f S*8c L*4i 6a6b 2b2a',
    '5g5f 3c3d 2h5h 4a4b 5i4h 7a6b 1g1f 8c8d 7g7f 1c1d 5f5e 5a4a 4h3h 4a3b 7i6h 5c5d 6h5g 5d5e 5g4f 8d8e 8h7g 8e8f 8g8f 7c7d 5h8h 8a7c 7g5e 2b5e 4f5e B*2b P*5f 6c6d 8h8g 6d6e B*8h 6a7b 3h2h 7b6c 3i3h P*5d 5e4f 2b8h+ 8g8h B*3c B*7g 3c7g+ 8i7g 6e6f 6i7h 6f6g+ 7h6g',
    '7g7f 3c3d 2g2f 5c5d 2f2e 8b5b 8h2b+ 5b2b B*7g B*3c 7g3c+ 2a3c B*5f 7a7b 5f3d 3a4b 2e2d 2c2d P*2c 2b2a 2h2d 4a3b 5i6h 5a6b 6g6f 6b7a 3d6g 7a8b 3i3h P*2e 4i3i 4b5c 9g9f 5c4d 6i7h 5d5e',
    '7g7f 8c8d 2g2f 4a3b 2f2e 8d8e 8h7g',
    '2h7h 8c8d 7g7f 7a6b 7f7e 6a7b 1g1f 1c1d 5i4h 5a4b 3i3h',
    '2g2f 8c8d 7g7f 4a3b 2f2e 8d8e 6i7h 8e8f 8g8f 8b8f 2e2d 2c2d 2h2d P*2c 2d2f 8f8b 9g9f 7a6b 7f7e 3c3d 8i7g 2b4d 2f3f 5a4a 5i4h 3a4b 1g1f 4b3c 1f1e 4a3a 3i3h 3a2b P*8e 5c5d 3f7f 9c9d 4h3i 6b5c 7i6h 8a9c 9f9e 9d9e 9i9e 6a5b P*9b P*8g 7h8g 9a9b 7f8f 5c6d 8g7f',
    '7g7f 3c3d 2g2f 4c4d 2f2e 2b3c 3i4h 3a3b 5g5f 8b4b 5i6h 5a6b 6h7h 6b7b 8h7g 3b4c 4h5g 7b8b 7h8h 9a9b 4i5h 8b9a 9i9h 4c5d 6g6f 7a8b 8h9i 6c6d 7i8h 4b6b 6i7i 6a7a 5h6h 4a5b 6h7h 6b6a 9g9f 7c7d 3g3f 4d4e 9f9e 5b6b 5g6h 6a4a 2e2d 2c2d 3f3e',
    '5i6h 4a3b 6h7h 8b5b 2g2f 7c7d 3i4h 7a7b 2f2e 7b7c 2e2d 2c2d 2h2d P*2c 2d2f 8c8d P*2d 2c2d 2f2d P*2c 2d2h 8d8e 5g5f 5b8b 6g6f 3c3d 4i5h 7c8d 7g7f 8e8f 8g8f 8d9e 8h7g 9e8f 7g8f 8b8f 7i8h 8f8b 5h6g P*8f 6i6h 6c6d 8i7g 8a7c 4h5g B*3i',
    '7g7f 3c3d 2g2f 5c5d 2f2e 8b5b 3i4h 5d5e 5i6h 2b3c 6h7h 5a6b 7i6h 3a4b 3g3f 6b7b 4h3g 4b5c 3g4f 5c4d 4i5h 7b8b 6h7g 6a7b 7g6f 9c9d 9g9f 4a3b 2i3g 1c1d 3g4e 3c2b 2e2d 2c2d 2h2d P*2c 2d3d 2a3c 6f5e 4d4e 4f4e 3c4e S*6a 5b6b 6a7b 6b7b',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 3d3f 8f8d 3f2f 3a2b P*8g 5a5b 5i5h 7a7b 4i3h 9c9d 9g9f 2b2c 3i4h 3c8h+ 7i8h 2a3c 8h7g 2c2d 2f5f 8d3d 7f7e 2d3e 5f8f 3d2d P*2h P*8d 4h3i 7b8c B*8b 9a9b 8b9a+ 6a7a 3g3f 3e2f 7e7d 5b6b 3f3e 2f3e 7d7c+ 8a7c 8f7f 7a7b P*7d 8c7d 9a9b P*7e 7f5f 7d8c 9b8a B*4e 5f8f 8d8e 8f6f 6c6d 9f9e 7c6e 7g6h 2d2a 8a7b 8c7b 9e9d P*2g 9d9c+ 2g2h+ P*2b 2h3i 2b2a+ 3i3h 9c8b 7b6c 2a3a 3b3a R*7a S*8h L*4i 3h4i 5h4i 4e2g+ G*3h G*6a 3h2g 6a7a 8b7a R*6i G*5i',
    '7g7f 3c3d 5i6h 4a3b 3i4h 8c8d 8h7g',
    '7g7f 8c8d 7i6h 3c3d 6g6f 5a4b 2g2f 6a5b 3i4h',
    '2g2f 8c8d 2f2e 4a3b 7g7f 8d8e 8h7g 3c3d 7i8h 2b7g+ 8h7g 3a2b 3i3h 7a7b 3g3f 2b3c 6i7h 7c7d 5i6h 7b7c',
    '7g7f 3c3d 2g2f 4c4d 5i6h 8b4b 4i5h 5a6b 3i4h 6b7b 6h7h 7b8b 8h7g 4d4e 5g5f 4e4f 4g4f 4b4f 7i6h 2b7g+ 8i7g 2a3c 6h5g 4f4b 9g9f 9c9d 6i6h 7a7b P*4f 4a5b 6g6f 6c6d 4h4g 4b2b',
    '2g2f 5a5b 2f2e',
    '2g2f 8c8d 7g7f 8d8e 8h7g 3c3d 7i8h 4a3b 6i7h 6c6d 2f2e 2b7g+ 8h7g 3a2b B*5e 6d6e 2e2d 2c2d 2h2d P*2c 2d2g 7a7b 3g3f 7b6c 2i3g 5a4b 3i4h 6a5b 5g5f 5c5d 5e4f 4c4d P*2d 2c2d 4f2d 4b3a 4i3h P*2c 2d4f 3b4c 1g1f 3a3b 5i5h 5b5c P*2d 2c2d 4f2d 8b5b 2d4f P*2c 2g2i 4d4e 3g4e 5c4d 2i2e 5d5e 4f5e 4d5e 5f5e 5b5e 5h6i 5e5d G*5c 4c5c P*5e 5d5e 4e3c+ 2b3c 2e5e B*4d 5e6e B*7d 6e7e 6c5b R*7b 7d6c 7b8b+ 8e8f 7e6e P*6d 6e2e 8f8g+ 8b8g P*5f P*5h N*6e 7g6f G*2f 2e2f 4d2f G*3i 7c7d 6f5e 3c4d 5e4f 4d4e 4f5e 4e5d 5e5d 5c5d 6g6f P*8f 8g6g S*5g 6f6e 5g4h 3h4h S*2h 6e6d 6c7b 3i3h',
    '7g7f 7c7d 2g2f 4a3b 2f2e 7a6b 3i4h 6b7c 5i6h 8c8d 6h7h 8d8e 7i6h 7c6d 8h6f 3c3d 6h7g 6d5e 5g5f 5e6f 6g6f 2b3c 4i5h 5a4a S*4e 3c5a 4e3d 6a5b 5h6g 6c6d 6i6h 3a2b 4h5g 5a8d 2e2d 2c2d 2h2d P*2c 2d2h',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h',
    '7g7f 3c3d 2g2f 4c4d 3i4h 3a4b 5g5f 5c5d 4i5h 7a6b 7i6h 6a5b 6g6f 4b3c 6h7g 5b4c 6i7h 5a4b 5i6i 4b3b 8h7i 2b3a 1g1f 7c7d 1f1e 7d7e 7f7e 3a7e 3g3f 6c6d 5h6g 6b6c 2i3g 7e8d 7i6h 8a7c 6i7i 6c7d 3g2e 3c2d 7i8h 8b6b 2e1c+ 2d1c 1e1d 1c2b 1d1c+ 1a1c 1i1c+ 2b1c P*7e 7d7e P*7f 7e7f 7g7f 6d6e S*7e 6e6f 6g6f L*7d P*6d 6b6d 4h5g 7d7e 7f7e 6d6f 5g6f N*7f 8h9h 7f6h+ 2h6h P*6g R*8b 4a4b 6h6g S*7f 6g6i 8d7e 6f7e P*7g 8i7g P*6h 6i6h P*6g 6h1h P*1g 1h3h S*6h L*8i B*1f 3h2h',
    '2g2f 3c3d 3i4h 8c8d 2f2e 3a3b 2e2d 2c2d 2h2d 2b3c 2d2h P*2d 7g7f 4c4d 5i6h 8d8e 6h7h 8e8f 8g8f 8b8f 7i6h 8f7f 6h7g 7f7e 5g5f 7e2e 2h2e',
    '2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 3i3h 7a7b 5i5h 5a5b 9g9f 9c9d 3g3f 8e8f 8g8f 8b8f 2i3g 8f3f P*8b 8a9c 8b8a+ 7b8a 9f9e P*8g 8h9g 9c8e 9g7e 3c3d 2e2d 2c2d 2h2d 8a7b P*2c 2b5e 3g4e 6c6d 7e6f 3f6f 6g6f 5e1i+ 2d3d L*3c 4e3c+ 2a3c P*8c 8g8h+ 7h8h N*5e 8h7h B*4e R*3e P*3g',
    '7g7f 3c3d 2g2f 5c5d 2f2e 8b5b 5g5f 5a6b 3i4h 5d5e 5f5e 6b7b 5i6h 3a3b 6h7h 7b8b 7i6h 2b5e 8h5e 5b5e 2e2d 2c2d P*2b 2a3c 2b2a+ 3b2a 2h2d 2a3b 2d2b+ 5e2e 2b2e 3c2e R*2h B*4d B*7g P*2g 2h2g P*2f 2g2h P*5e 4h5g R*2g 2h5h',
    '2g2f 8c8d 7g7f 4a3b 6i7h 8d8e 8h7g 3c3d 7i8h 3a4b 7g2b+ 3b2b 8h7g 2b3b 9g9f 9c9d 3i3h 7a7b 4g4f 6c6d 1g1f 1c1d 3h4g 7b6c 5i6h 4b3c 3g3f 7c7d 2i3g 8a7c 2h2i 5a4b 4i4h 4c4d 4g5f 6a5b 4f4e 4d4e 5f4e 7c6e B*7c 6e7g+ 8i7g 8b8a 3f3e P*4g 4h3h 7d7e 3e3d 3c2d P*4c 4b3a N*4d 7e7f 4d5b+ 6c5b 7c6d+ 8a6a 6d5e 7f7g+ 7h7g 5b4c P*4d 4c5b 2f2e P*7f 7g7f 2d3e 2e2d N*8d 7f6f 2c2d',
    '7g7f 3c3d 2g2f 4c4d 2f2e',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b',
    '7g7f 8c8d 2g2f 8d8e 8h7g 3c3d 7i8h 4a3b 6i7h 2b7g+ 8h7g 3a2b 3i3h 7a7b 4g4f 7b8c 3h4g 8c8d 9g9f',
    '2g2f 8c8d 7g7f 8d8e 8h7g 3c3d 7i6h 4c4d 3i4h 7a6b 6i7h 6a5b 4g4f 5c5d 3g3f 3a3b 4h4g 7c7d 4f4e 6b5c 4e4d 2b4d 5i6i 5a4b 2h4h 4d7g+ 6h7g 4b3a 4i5h 6c6d 4g5f 8a7c 6g6f 8e8f 7g8f B*2b 8f7g 7d7e 7f7e P*7f 7g7f 2b6f B*7g 6f7g+ 8i7g P*8f 7e7d 7c6e 7g6e 8f8g+ 7h8g 6d6e 7d7c+ 5d5e 5f5e B*8h N*4d 8h5e+ 4d3b+ 3a3b 7c8b N*6f 5h6h P*4g 4h4g S*4f 4g4i N*4e S*4h 4f5g 4h5g 4e5g 6h5g S*7h 6i5i P*5f 5g6f 5e6f N*2d 2c2d B*2c 3b2b 4i4a+ G*5h 5i5h 5f5g+ 5h4i',
    '7g7f 3c3d 2g2f 4c4d 2f2e 2b3c 3i4h 8b4b 5i6h 5a6b 6h7h 7a7b 4i5h 6b7a 5g5f 3a3b 9g9f 9c9d 7i6h 7a8b 3g3f 3b4c 4h3g 1a1b 6h7g 4a5b 8h7i 6c6d 2e2d 2c2d 7i2d 4b2b P*2e 4c3b 6g6f 7c7d 3f3e 3d3e 2d3e 3b2c 3e4f 5b6c 3g3f P*3d',
    '9g9f 3c3d 2g2f 2b3c 7g7f 3a3b 3i4h 8c8d 6i7h 8d8e 8h3c+ 3b3c 7i8h 7c7d 8h7g 7a7b 4g4f 8a7c 3g3f 5a4b 5i6h 7c6e 7g8h 8e8f 8g8f 8b8f 6g6f B*5e P*8g 8f7f 6f6e 7f7h+ 6h7h G*6h 7h6h 5e8h+ 6h5h 8h9i 8i9g 9i5e 4h4g 5e6e B*8b 6e8g 8b9a+ 8g9g 5h4h 9g7e 4i5h L*5d N*4i N*6e 2f2e S*6f L*5i 4b3a 1g1f 4a3b 2i3g 5d5g+ 5h5g 6e5g+ 4i5g G*6h 5i5h 6h5h 4g5h L*5f L*5i 5f5g+ 5h5g 6f5g 5i5g N*6e L*5h 6e5g+ 5h5g L*5f N*6i 5f5g+ 6i5g L*5f G*4g 5f5g+ 4g5g S*5f S*6f 5f5g 6f5g G*5f G*4g 5f5g 4g5g S*5f S*6f 5f5g 6f5g N*6e G*6f 6e5g+ 4h5g 7e8d N*5h',
    '7g7f 8c8d 6i7h 8d8e 8h7g 3c3d 7i6h 7a6b 3g3f 2b7g+ 6h7g 3a2b 3i4h 9c9d 9g9f 1c1d 1g1f 6c6d 2g2f 4a3b 4g4f 7c7d 2i3g 2b3c 2f2e 5a4b 5i6h 6b6c 4h4g 8a7c 4i4h 6a6b 4g5f 8b8a 2h2i 6c5d 6g6f 4b5b 6h7i 4c4d 2i6i 3c4b 6i2i 4b3c 7i8h 8a4a 2i4i 4a8a 4i5i 5b4b 5i6i 4b3a 8h7i 8a4a 6i2i 3a4b 7i8h 4b3a 2i6i 3a2b 6i2i 6d6e 6f6e 7d7e 7f7e 7c6e 7g6f 4a8a 6f6e 5d6e 5f6e 9d9e 2e2d 3c2d P*6f 9e9f N*2f S*2e 3g2e 2d2e 8h7g 9f9g+ 8i9g P*9h 9i9h 2e2f P*9d P*9f 9g8e 8a8e P*2d 8e8a 2d2c+ 3b2c P*2d 2c2d S*4c N*8e 7g6g P*2h B*4b N*3c S*3a 2b2c 2i3i N*2g 3i4i 2g1i+ P*2b 2d2e 2b2a+ L*4a 2a2b 2c2d 4c3b 2e3f 4b3c+ 2d2e 4i1i',
    '7g7f 4a3b 7i6h 6a5b 6g6f 8c8d 6h6g 3a4b 2h7h 8d8e 8h7g 5a4a 5i4h 1c1d 4h3h 1d1e 3h2h 4a3a',
    '2g2f 3c3d 2f2e 2b3c 7g7f 3a2b 3g3f 8b4b 3i4h 5a6b 5g5f 6b7b 5i6h 9a9b 6g6f 7b8b 8h7g 8b9a 6h7h 7a8b 7h8h 7c7d 4h5g 4b3b 9i9h 4a5b 8h9i 3c5a',
    '7g7f 8c8d 7i6h 3c3d 6h7g 7a6b 2g2f 3a4b 3i4h 5c5d 6i7h 4a3b 5i6i 5a4a 5g5f 6a5b 4i5h 4c4d 8h7i 1c1d 6g6f 6b5c 2f2e',
    '7g7f 8c8d 2h6h 3c3d 5i4h 5a4b 4h3h 8d8e 8h2b+ 3a2b 7i8h 1c1d 1g1f 5c5d 8h7g 4b3b 6h8h',
    '7g7f 8c8d 2g2f 3c3d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 3d3f 8f8d',
    '2g2f 8c8d 2f2e 4a3b 2e2d 2c2d 2h2d P*2c 2d2h 8d8e 6i7h 3c3d 7g7f 8e8f 8g8f 8b8f P*8g 8f7f 8h2b+ 3a2b B*8e 7f7h+ 2h7h 5a5b 5i4h B*5d 3i3h 5d8g+ 8e9f 8g5d 7h8h P*8b 7i6h G*2h 8h8d 2h1i 8i7g 6a7b 8d5d 5c5d 7g6e 7a6b B*8h L*4d 8h7g R*1h 4i3i 2b3c 7g8f 3c4b',
    '2g2f 3c3d 2f2e 2b3c',
    '2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 5i6h 8e8f 8g8f 8b8f P*8g 8f8d 7g7f 5a6b 3i3h 3c3d',
    '2g2f 8b6b 2f2e 3a3b 2e2d 2c2d 2h2d P*2c 2d2h 6b8b',
    '7g7f 3c3d 2g2f 5c5d 5g5f 7a6b 2f2e 6b5c 3i4h 4a3b 5i6h 8c8d 6g6f 8d8e 7i7h 8e8f 8g8f 8b8f 8h7g 8f8b 6h7i 5a4a 4i5h 7c7d 5h6g 5c6d P*8f 3a4b 7h8g 7d7e 7f7e 6d7e P*7f 7e6d 4h5g 4a3a 6i7h 8a7c 3g3f 5d5e 5f5e 2b5e 5g4f 5e4d P*5f 4b3c 7g6h 4d5c 7i8h 3c4d 2e2d 2c2d 2h2d P*2c 2d2h 6a5b 9g9f 8b8d 2i3g 5c4b 3f3e 3d3e 4f4e P*7e',
    '7g7f 3c3d 7f7e 5a4b 2h7h 2b8h+ 7i8h B*4e 7e7d 7c7d B*5e 4b3b 5e1a+ 3a2b L*4f 4e6g+ 4f4c+ 3b3a 1a1b 6g4e 4c5c 7a6b 5c6b 8b6b 7h7d P*7c 7d7h 4a3b 4g4f 4e5e 6i5h L*5a 5i4h 5e4f 4h3h 4f4e 7h7i 4e5e 8h7g 2b1a 1b1a 5e1a 3h2h 1a3c 1i1h P*4f 2h1i 6b4b P*4h 2c2d 4i3h 2d2e 5g5f',
    '2h6h 8c8d 7g7f 3c3d 8h2b+ 3a2b 7i8h 5a4b 5i4h 7a6b 4h3h 4b3b 3h2h 6a5b 4i3h 5c5d 8h7g 6b5c 6i5i 1c1d 3g3f 7c7d 5i4i 1d1e 2i3g 2c2d 2h2i 2b2c 3i2h 3b2b 4i3i 4c4d 6h7h 5b4c 7g6f B*6i 7h8h 4a3b',
    '2g2f 3c3d 2f2e 2b3c 7g7f 4c4d 3i4h 3a3b 5g5f 8b4b 5i6h 5a6b 9g9f 9c9d 6h7h 7a7b 4i5h 7c7d 3g3f 6c6d 2h3h 3b4c 4h5g 4a5b 3f3e 3d3e 5g4f 6b7a 4f3e 4b1b P*3d 3c5a 6i5i 7a8b 2e2d',
    '7g7f 8c8d 5g5f 8d8e 8h7g 5c5d 2h8h 7a6b 7i6h 3c3d 5i4h 6c6d 4h3h 5a4b 3h2h 4b3b 3i3h 6a5b 6g6f 7c7d 6i5h 1c1d 1g1f 2c2d 4g4f 2b3c 6h5g 3a2b 3g3f 2b2c 2i3g 3b2b 2g2f',
    '7g7f 8c8d 7i6h 3c3d 8h7g 5a4b 2g2f 7a6b 2f2e 4b3b 2e2d 2c2d 2h2d 2b7g+',
    '7g7f 3c3d 6g6f 8b3b 2h6h 3a4b 5i4h 5a6b 3i3h 5c5d 4g4f 6b7b 3h4g 7b8b 7i7h 4b5c 9g9f 9a9b 4h3h 4a5b 8g8f 8b9a 4i4h 7a8b 1g1f 2b3c 8f8e 2c2d 8h7g 6a7a 9f9e 5b6b 7h6g 2d2e 3g3f 1c1d 2i3g 3c2d 6f6e 4c4d 6h8h 7a7b 6i5h 1d1e 1f1e',
    '2g2f 8c8d 2f2e 8d8e 9g9f 4a3b 3i3h 5a5b 6i7h 1c1d 5i5h 8e8f',
    '7g7f 8c8d 7i6h 3c3d 6g6f 7a6b 5g5f 5c5d 3i4h 3a4b 4i5h 4a3b 5h6g 5a4a 6h7g 7c7d 6i7h 6a5b 5i6i 4b3c 8h7i 2b3a 7i4f 3a6d 6i7i',
    '2g2f 3c3d 7g7f 4a3b 2f2e 2b3c 8h3c+ 3b3c 3i3h 8b2b 5i6h 5a6b 6h7h 3a4b 7i8h 6b7b 3g3f 7b8b 4g4f 9a9b 8h7g 8b9a 2i3g',
    '9g9f 3c3d 8h9g 3a4b 9f9e 9c9d 9e9d 9a9d 9g8h 9d9i+ 8h9i 8b9b L*9h P*9g 9h9g 9b9g+ 8i9g L*7d R*9a 7a7b 9a9b+ 7d7g+ 2h9h 7g8g 9i2b+ 8g9h 2b2a R*9i B*2b 9i8i+ 2b1a+ L*7d P*7g 9h8h 7i8h 8i8h 7g7f 8h8i L*7i S*8h P*8b 8h7i+ 6i5h 7i7h 5i4h P*9a 9b9a 7d7f 8b8a+ L*4d 8a7a 4d4g+',
    '7g7f 9c9d 2g2f 3c3d 2f2e 2b8h+ 7i8h',
    '7g7f 3c3d 3i4h 4c4d 2g2f 3a4b 5g5f 8b5b 5i6h 5a6b 6h7h 6b7b 9g9f 5c5d 4i5h 9c9d 4h5g 7b8b 6g6f 7a7b 8h7g 4b5c 7h8h 4d4e 5h6g 6c6d 9i9h 5d5e 5f5e 2b5e 8h9i 5c5d 7i8h 4a3b 6i7i 3b4c 2f2e 5e3c 3g3f 7c7d 7g5i 3c5e 5i3g 5e3g+ 2i3g 3d3e B*4a 5b5a 4a7d+ 7b6c P*5b 5a5b P*5c 4c5c 7d5f 5c4d 4g4f 3e3f 3g4e 3f3g+ 2h5h P*5e 5f2i B*4g 5h5i 4g3f+ 2e2d P*4g 5i4i 5e5f',
    '2g2f 8c8d 2f2e 8d8e 7g7f 4a3b 6i7h 8e8f 8g8f 8b8f 2e2d 2c2d 2h2d P*2c 2d2f 3c3d P*8g 8f8b 3i3h 7a7b 5i5h 4c4d 3g3f 3a4b 2i3g 7b8c 7i6h 4b4c 4g4f 8c8d 4i4h 8d8e 6h7g 4c5d P*2d 2c2d 2f2d P*2c 2d3d 3b3c 3d3e 3c2d',
    '7i6h 3c3d 8h7i 8c8d 6i7h 7a7b 5g5f 3a3b 6h5g 8d8e 2g2f 3b3c 2f2e 1c1d 3g3f 4a3b 5i6i 7c7d 5g4f 5a6b',
    '2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d P*2c 2d2h 8e8f 8g8f 8b8f P*8g 8f8d 3i3h 3c3d 3h2g 7a7b 2g3f 5a4a 7g7f 2b4d 8h7g 4d7g+ 8i7g P*8f B*6f 8d8b 8g8f 3a2b 8f8e 2b3c 8e8d 6a5b 3f4e 3d3e 7i8h 1c1d 8h8g 1d1e 5i6h',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 3d3f 3a2b 4i3h 5a4a 5i5h 6a5a 8h3c+ 2a3c 7i8h',
    '2g2f 3c3d 7g7f 8c8d 2f2e 8d8e 6i7h 4a3b 5i5h 5c5d 2e2d 2c2d 2h2d 7a6b 3i3h 6b5c 2d2h P*2c 8h2b+ 3a2b 7i8h 2a3c 8h7g 7c7d 3g3f 6a6b 4i4h 1c1d 3f3e 8a7c 3e3d 3c4e 6g6f B*6d 2h1h 5a4a 4g4f 6d4f 7f7e 8e8f 8g8f P*8h',
    '2g2f 8c8d 6i7h 8d8e 2f2e 4a3b 3i4h 7a6b 3g3f 3c3d 7g7f 8e8f 8g8f 8b8f 2e2d 2c2d 2h2d 7c7d 2d3d 5a4a P*2c 2b8h+ 7i8h P*2b 3d7d 8f8e 5i5h 2b2c 4h3g 6b7c 7d7e 8e8b B*5e P*7d 7e7d 6c6d 7d7c+ 8a7c 5e1a+ 2a3c P*8c 8b6b P*2h P*8g 8h8g B*4d 8i7g 3c4e 1a4d 4c4d 3g4f R*8i B*9f B*5b S*7d 8i9i+ L*6c 7c8e 9f8e L*8d 6c6b+ 6a6b R*7a L*5a N*6i 9i8i 7d7c 6b7c 8e5b+ 4a5b 7a7c+ 8d8g+ B*9f 5b4c 9f8g S*8f G*8h 8i9i 7c6d 8f8g 8h8g S*5d L*3e 4e5g+',
    '7g7f 2c2d 2g2f 6a5b 2f2e 8b9b 2e2d 7a7b 2d2c+ 9b8b 2c2b 8c8d 2b3a 9a9b',
    '7g7f 8c8d 5g5f 3c3d 6g6f 6a5b 4i5h 3a3b 5h6g 8d8e 5i6h 5a4b 2g2f 6c6d 3i4h 8e8f 8g8f 8b8f 6h7h 8f8b P*8g 7a6b 7i6h 6b6c 2f2e 7c7d 6h7g 8a7c',
    '2h7h 8c8d 7g7f 8d8e 8h7g 7a6b 6g6f 1c1d 1g1f 5a4b 7i6h 4b3b 5i4h 6a5b 4h3h 3c3d 6h6g 5c5d 6i5h 7c7d 3h2h 6c6d 6g5f 8a7c 7h8h 3a4b 5f4e 4b3c 4e5d 3c4d 3i3h 9c9d 8h6h 8b8d 5h6g 8e8f 8g8f 2b3a 4g4f 6d6e 5d4e 3a8f 4e4d 4c4d 4f4e 8f7g+ 8i7g 8d8i+ 4e4d P*4b 6h4h 6b5c 6f6e 8i9i B*5e L*4e 4d4c+ 4b4c P*4f B*3c 5e7c+ P*6f 6g5f 6f6g+ 4f4e 3c7g+ L*3f N*2b S*6c S*5h 6c5b 4a5b 7c7d S*4a G*3i 5h4i+ 4h4i 9i8h N*5e 6g5h 4e4d 4c4d S*4c 5b4c 7d4a 3b4a 5e4c+ G*5b P*4b 5b4b G*3b 4b3b S*5b 4a3a 4c3b 3a3b G*4c 3b3a 4c5c S*4b S*5d G*3b P*4c 4b3c 5c4b 3c4b',
    '7g7f 3c3d 3i4h 4c4d 5i6h 8b4b 6h7h 5a6b 9i9h 3a3b 5g5f 3b4c 4i5h 4c5d 6g6f 6b7b 8h7g 7b8b 7h8h 6c6d 5h6h 9a9b 8h9i 8b9a 4h5g 4b6b 7i8h 4a5b 2g2f 7a8b 2f2e 2b3c 6h7h 5b6c 6i7i 6c7d 3g3f 7d8e 7h6g 7c7d 3f3e 3d3e 2h3h 7d7e 7f7e P*7f',
    '2g2f 8c8d 7g7f 8d8e 8h7g 3c3d 7i6h 4c4d 6i7h 4a3b 4g4f 3a4b 3i4h 7a6b 4h4g 6c6d 5i6i 6b6c 4g5f 6a5b 2f2e 2b3c 4f4e 4d4e 5f4e 8e8f 8g8f P*8e P*4d 8e8f P*8h P*4c 4i5h 6c5d 2e2d 2c2d 4e3d 4c4d 3g3f 4b4c 3f3e 3c4b P*4e 4c3d 3e3d 5d4e 7g4d P*3c 2h2d P*2c 2d2f',
    '7g7f 3c3d 2g2f 8c8d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 5i5h 5a5b 3g3f 8f7f 8h7g 3c7g+ 8i7g B*5e P*2b 2a3c 2b2a+ 3a4b P*2c 3b2c 3d8d P*8b 7i6h 7c7d 6g6f 5e1i+ B*6g 7f7e',
    '2g2f 3c3d 7g7f 4c4d 3i3h 3a3b 4g4f 8b4b 3h4g 3b4c 4g5f 5a6b 5i6h 6b7b 4i5h 7b8b 7i7h 7a7b 6h7i 4a5b 9g9f 9c9d 3g3f 6c6d 2f2e 2b3c 2i3g 7c7d 4f4e 4d4e 8h3c+ 2a3c 2e2d 2c2d 2h2d B*4f 3f3e 4b4a 2d2c+ 4f3g+ 3e3d P*3b 3d3c+ 3b3c N*8f 7b7c P*3d 3c3d P*3c 6a7b 3c3b+ 4a7a 3b3c 4c5d 3c3d 7b6b P*4d N*5a 2c3b 3g1i 3d3c 1i4f 3c4b 5b6c 4b5a 7a5a 4d4c+ L*5e 5f5e 4f5e B*7g',
    '7g7f 8c8d 7i6h 3c3d 6g6f 7a6b 5g5f 5c5d 3i4h 6a5b 4i5h 3a3b 5h6g 5a4b 6h7g 6b5c',
    '2g2f 8c8d 7g7f 3c3d 2f2e 8d8e 6i7h 4a3b 2e2d 2c2d 2h2d 8e8f 8g8f 8b8f 2d3d 2b3c 3d3f 3a2b 4i3h 5a4a 5i5h 7a6b 8h3c+ 2a3c 7i8h 8f8d P*2c 3b2c P*8b 8d8b P*8c 8b9b 7f7e P*8e 8i7g 5c5d 7g8e 4a3b 8e9c 8a9c B*8a 9c8e 8a9b+ 9a9b 8c8b+ 9b9g+ 9i9g P*8g 8h7i 8e9g+ R*9a 6a5b 9a9g+ P*3e 3f8f 3c4e N*4i 6b5a 9g8g 5d5e 8g9f N*6d 7i6h 5e5f L*2h P*2d P*2e 2d2e 3i4h L*7f 8f7f 5f5g+ 4h5g 6d7f 9f7f B*9h 7f4f 4e5g+ 4f5g 9h6e+ N*1e 2c2d L*5f P*5d P*2c 2b3c 5f5d P*5f 5d5b+ 5a5b 5g4f S*5e 4f1f B*9f 2c2b+ 3c2b 7h7i L*5g 5h4h 1c1d 2h2e 1d1e 1f2g 6e5d N*3f 3e3f 2e2d P*2c 2d2c+ 2b2c P*2d 3f3g+ 2g3g N*3f',
    '7g7f 8c8d 2g2f 4a3b 6i7h 8d8e 8h7g 3c3d 7i8h 2b7g+ 8h7g 3a4b 3i3h 7a6b 9g9f 6c6d 1g1f 7c7d 6g6f 8a7c 4g4f 6a5b 2f2e B*3c 4i5h 6d6e 5h6g 6b6c 3h4g 5a4a 4g5f 6e6f 7g6f 6c6d 5i6h 8e8f 8g8f 8b8f P*8g 8f8b P*6e 7c6e B*4g',
    '3i4h 5a6b 2h1h 6b7b 7i6h 3c3d 8g8f 8c8d 6i7i 8d8e',
]

for moves in positions:
    engine.position(moves=moves.split(' '))
    engine.go(byoyomi=args.byoyomi)
engine.quit()
