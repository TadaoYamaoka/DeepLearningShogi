import sys
import random
import numpy as np
from cshogi import Board, move_to_usi
from cshogi import DfPn
from cshogi.dlshogi import (
    make_input_features,
    make_move_label,
    FEATURES1_NUM,
    FEATURES2_NUM,
)
import onnxruntime as ort  # Ensure the 'onnxruntime' library is installed

temperature1 = 1.0
temperature2 = 0
temperature_threshold = 16
model_path = None
session = None
use_value_network = False
use_mate = False
use_dfpn = False
use_cuda = False
onnx_threads = 0

dfpn = DfPn()

input_features = [
    np.empty((593, FEATURES1_NUM, 9, 9), dtype=np.float32),
    np.empty((593, FEATURES2_NUM, 9, 9), dtype=np.float32),
]


def load_model(path):
    global session
    if use_cuda:
        session = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
    elif onnx_threads:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = onnx_threads
        session = ort.InferenceSession(path, sess_options)
    else:
        session = ort.InferenceSession(path)


def predict_logits(board):
    make_input_features(board, input_features[0][0], input_features[1][0])
    io_binding = session.io_binding()
    io_binding.bind_cpu_input("input1", input_features[0][0:1])
    io_binding.bind_cpu_input("input2", input_features[1][0:1])
    io_binding.bind_output("output_policy")
    io_binding.bind_output("output_value")
    session.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()

    logits = outputs[0][0]
    legal_moves = list(board.legal_moves)
    legal_logtis = np.empty(len(legal_moves), dtype=np.float32)
    for i, move in enumerate(legal_moves):
        move_label = make_move_label(move, board.turn)
        legal_logtis[i] = logits[move_label]

    return legal_moves, legal_logtis


def predict_values(board):
    legal_moves = list(board.legal_moves)
    for i, move in enumerate(legal_moves):
        board.push(move)
        make_input_features(board, input_features[0][i], input_features[1][i])
        board.pop()
    io_binding = session.io_binding()
    io_binding.bind_cpu_input("input1", input_features[0][:i + 1])
    io_binding.bind_cpu_input("input2", input_features[1][:i + 1])
    io_binding.bind_output("output_policy")
    io_binding.bind_output("output_value")
    session.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()

    values = 1 - outputs[1].reshape(-1)
    return legal_moves, values


def softmax(logits, temperature):
    if temperature == 0:
        temperature = 1e-10
    max_logit = np.max(logits)
    exp_logits = np.exp((logits - max_logit) / temperature)
    sum_exp = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp
    return probabilities


def select_move(board):
    if board.is_game_over():
        return "resign"
    if board.is_nyugyoku():
        return "win"
    if use_mate:
        mate_move = board.mate_move(3)
        if mate_move:
            return move_to_usi(mate_move)
    if use_dfpn:
        if dfpn.search(board):
            return move_to_usi(dfpn.get_move(board))
    temperature = (
        temperature1 if board.move_number <= temperature_threshold else temperature2
    )
    if use_value_network:
        legal_moves, values = predict_values(board)
        probabilities = softmax(values, temperature)
    else:
        legal_moves, logits = predict_logits(board)
        probabilities = softmax(logits, temperature)
    move = random.choices(legal_moves, weights=probabilities)[0]
    return move_to_usi(move)


def main():
    global temperature1, temperature2, temperature_threshold, model_path, use_value_network, use_mate, use_dfpn, use_cuda, onnx_threads

    board = Board()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command, *args = line.strip().split(" ", 1)
        args = args[0] if args else ""
        if command == "usi":
            print("id name PolicyOnly")
            print("option name Temperature1 type spin default 100 min 0 max 1000")
            print("option name Temperature2 type spin default 0 min 0 max 1000")
            print("option name TemperatureThreshold type spin default 24 min 0 max 512")
            print("option name ModelPath type string default model.onnx")
            print("option name UseValueNetwork type check default false")
            print("option name UseMate type check default false")
            print("option name UseDfPn type check default false")
            print("option name UseCUDA type check default false")
            print("option name OnnxThreads type spin default 0 min 0 max 1024")
            print("usiok", flush=True)
        elif command == "isready":
            load_model(model_path)
            print("readyok", flush=True)
        elif command == "setoption":
            tokens = args.split()
            option_name = tokens[1]
            option_value = tokens[3]
            if option_name == "Temperature1":
                temperature1 = float(option_value) / 100
            elif option_name == "Temperature2":
                temperature2 = float(option_value) / 100
            elif option_name == "TemperatureThreshold":
                temperature_threshold = int(option_value)
            elif option_name == "ModelPath":
                model_path = option_value
            elif option_name == "UseValueNetwork":
                use_value_network = option_value.lower() == "true"
            elif option_name == "UseMate":
                use_mate = option_value.lower() == "true"
            elif option_name == "UseDfPn":
                use_dfpn = option_value.lower() == "true"
            elif option_name == "UseCUDA":
                use_cuda = option_value.lower() == "true"
            elif option_name == "OnnxThreads":
                onnx_threads = int(option_value)
        elif command == "position":
            board.set_position(args)
        elif command == "go":
            move = select_move(board)
            print(f"bestmove {move}", flush=True)
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
