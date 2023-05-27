from cshogi import *
from cshogi.dlshogi import make_input_features, FEATURES1_NUM, FEATURES2_NUM

import numpy as np
import onnxruntime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, default='model', help='model file name')
parser.add_argument('sfen', type=str, help='position')
parser.add_argument('--svg', type=str)
args = parser.parse_args()

session = onnxruntime.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

board = Board(sfen=args.sfen)
features1 = np.zeros((41, FEATURES1_NUM, 9, 9), dtype=np.float32)
features2 = np.zeros((41, FEATURES2_NUM, 9, 9), dtype=np.float32)
make_input_features(board, features1, features2)

pos = []
i = 1
rank = 0
file = 0
j = 0

pieces_src = board.pieces
pieces_in_hand_src = board.pieces_in_hand

for sq in SQUARES:
    if pieces_src[sq] == NONE:
        continue

    file, rank = divmod(sq, 9)
    pieces_dst = pieces_src.copy()
    pieces_dst[sq] = NONE

    board_dst = board.copy()
    board_dst.set_pieces(pieces_dst, pieces_in_hand_src)
    make_input_features(board_dst, features1[i], features2[i])
    pos.append((file, rank, pieces_src[sq]))
    i += 1

hand = []
for c in COLORS:
    for hp in HAND_PIECES:
        if pieces_in_hand_src[c][hp] == 0:
            continue

        pieces_in_hand_dst = (pieces_in_hand_src[0].copy(), pieces_in_hand_src[1].copy())
        pieces_in_hand_dst[c][hp] = 0

        board_dst = board.copy()
        board_dst.set_pieces(pieces_src, pieces_in_hand_dst)
        make_input_features(board_dst, features1[i], features2[i])
        hand.append((c, hp, pieces_in_hand_src[c][hp]))
        i += 1

io_binding = session.io_binding()
io_binding.bind_cpu_input('input1', features1)
io_binding.bind_cpu_input('input2', features2)
io_binding.bind_output('output_policy')
io_binding.bind_output('output_value')
session.run_with_iobinding(io_binding)
y1, y2 = io_binding.copy_outputs_to_cpu()

importance = y2 - y2[0]

output = [['' for _ in range(9)] for _ in range(9)]
for i in range(len(pos)):
    file, rank, pc = pos[i]
    output[rank][8 - file] = format(float(importance[i + 1]), '.5f')
print('\n'.join(['\t'.join(row) for row in output]))

for i in range(len(hand)):
    c, hp, n = hand[i]
    symbol = HAND_PIECE_SYMBOLS[hp]
    if c == BLACK:
        symbol = symbol.upper()
    print(symbol, format(float(importance[len(pos) + 1 + i]), '.5f'), sep='\t')


def value_to_rgb(value):
	if value < 0 :
		r = 252 + int(value * 4)
		g = 252 + int(value * 147)
		b = 255 + int(value * 148)
	else:
		r = 252 - int(value * 162)
		g = 252 - int(value * 114)
		b = 255 - int(value * 57)
	return f"rgb({r},{g},{b})"

def to_svg(pos, hand, importance, scale=2.5):
	import xml.etree.ElementTree as ET

	width = 230
	height = 192

	svg = ET.Element("svg", {
		"xmlns": "http://www.w3.org/2000/svg",
		"version": "1.1",
		"xmlns:xlink": "http://www.w3.org/1999/xlink",
		"width": str(width * scale),
		"height": str(height * scale),
		"viewBox": "0 0 {} {}".format(width, height),
	})

	defs = ET.SubElement(svg, "defs")
	for piece_def in SVG_PIECE_DEFS:
		defs.append(ET.fromstring(piece_def))

	for i in range(len(pos)):
		file, rank, pc = pos[i]
		value = float(importance[i + 1])
		ET.SubElement(svg, "rect", {
			"x": str(20.5 + (8 - file) * 20),
			"y": str(10.5 + rank * 20),
			"width": str(20),
			"height": str(20),
			"fill": value_to_rgb(value)
		})

	svg.append(ET.fromstring(SVG_SQUARES))
	svg.append(ET.fromstring(SVG_COORDINATES))

	for i in range(len(pos)):
		file, rank, pc = pos[i]
		x = 20.5 + (8 - file) * 20
		y = 10.5 + rank * 20
		value = float(importance[i + 1])

		ET.SubElement(svg, "use", {
			"xlink:href": "#{}".format(SVG_PIECE_DEF_IDS[pc]),
			"x": str(x),
			"y": str(y),
		})
		e = ET.SubElement(svg, "text", {
			"font-family": "serif",
			"font-size": "5",
			"stroke-width": "1",
			"stroke": "#fff",
			"fill": "#000",
			"paint-order": "stroke",
			"x": str(x + 12),
			"y": str(y + 19)
		})
		e.text = format(abs(value), '.2f')[1:]

	hand_by_color = [[], []]
	for i in range(len(hand)):
		c, hp, n = hand[i]
		hand_by_color[c].append((i, hp, n))

	hand_pieces = [[], []]
	for c in COLORS:
		i = 0
		for index, hp, n in hand_by_color[c]:
			if n >= 11:
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[n % 10], None))
				i += 1
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[10], None))
				i += 1
			elif n >= 2:
				hand_pieces[c].append((i, NUMBER_JAPANESE_KANJI_SYMBOLS[n], None))
				i += 1
			if n >= 1:
				hand_pieces[c].append((i, HAND_PIECE_JAPANESE_SYMBOLS[hp], index))
				i += 1
		i += 1
		hand_pieces[c].append((i, "手", None))
		i += 1
		hand_pieces[c].append((i, "先" if c == BLACK else "後", None))
		i += 1
		hand_pieces[c].append(( i, "☗" if c == BLACK else "☖", None))

	for c in COLORS:
		if c == BLACK:
			x = 214
			y = 190
			x_rect = 214
			y_rect = 178
		else:
			x = -16
			y = -10
			x_rect = 2
			y_rect = 8
		scale = 1
		if len(hand_pieces[c]) + 1 > 13:
			scale = 13.0 / (len(hand_pieces[c]) + 1)
		for i, text, index in hand_pieces[c]:
			if index is not None:
				value = float(importance[len(pos) + 1 + index])
				ET.SubElement(svg, "rect", {
					"x": str(x_rect),
					"y": str(y_rect + 14 * scale * i * (-1 if c == BLACK else 1)),
					"width": str(14),
					"height": str(14 * scale),
					"fill": value_to_rgb(value)
				})

			e = ET.SubElement(svg, "text", {
				"font-family": "serif",
				"font-size": str(14 * scale),
			})
			e.set("x", str(x))
			e.set("y", str(y - 14 * scale * i))
			if c == WHITE:
				e.set("transform", "rotate(180)")
			e.text = text

			if index is not None:
				e = ET.SubElement(svg, "text", {
					"font-family": "serif",
					"font-size": "5",
					"stroke-width": "1",
					"stroke": "#fff",
					"fill": "#000",
					"paint-order": "stroke",
					"x": str(x_rect + 7),
					"y": str(y_rect + 14 * scale * i * (-1 if c == BLACK else 1) + 13.5 * scale)
				})
				e.text = format(abs(value), '.2f')[1:]

	return ET.ElementTree(svg)

if args.svg:
	with open(args.svg, 'wb') as f:
		svg = to_svg(pos, hand, importance)
		svg.write(f)
