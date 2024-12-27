import onnxruntime
import argparse
from tqdm import tqdm
import numpy as np
from cshogi import Board, HuffmanCodedPosAndEval
from cshogi.dlshogi import make_input_features, FEATURES1_NUM, FEATURES2_NUM

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('hcpe')
parser.add_argument('out_hcpe')
parser.add_argument('--a', type=float, default=756.0864962951762)
parser.add_argument('--batch_size', '-b', type=int, default=1024)
parser.add_argument('--tensorrt', action='store_true')
args = parser.parse_args()

a = args.a
batch_size = args.batch_size
hcpes = np.memmap(args.hcpe, HuffmanCodedPosAndEval, mode='r')
hcpes_size = len(hcpes)
f_out = open(args.out_hcpe, 'wb')

session = onnxruntime.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if args.tensorrt else ['CUDAExecutionProvider', 'CPUExecutionProvider'])

x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)

def value_to_score(values, a):
    scores = np.empty_like(values)
    scores[values == 1] = 30000
    scores[values == 0] = -30000
    mask = (values != 1) & (values != 0)
    scores[mask] = -a * np.log(1 / values[mask] - 1)
    return scores

board = Board()
j = 0
for i in tqdm(range(hcpes_size)):
    board.set_hcp(hcpes[i]['hcp'])
    make_input_features(board, x1[j], x2[j])

    if j == batch_size - 1 or i == hcpes_size - 1:
        io_binding = session.io_binding()
        io_binding.bind_cpu_input('input1', x1)
        io_binding.bind_cpu_input('input2', x2)
        io_binding.bind_output('output_policy')
        io_binding.bind_output('output_value')
        session.run_with_iobinding(io_binding)
        _, values = io_binding.copy_outputs_to_cpu()

        scores = value_to_score(values.reshape(-1), a)
        out_hcpes = hcpes[i - j:i + 1].copy()
        for out_hcpe, score in zip(out_hcpes, scores):
            out_hcpe['eval'] = score
        out_hcpes.tofile(f_out)

        j = 0
    else:
        j += 1
