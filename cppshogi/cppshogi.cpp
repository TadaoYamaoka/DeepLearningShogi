﻿#include <numeric>
#include <algorithm>

#include "cppshogi.h"

// make input features
void make_input_features(const Position& position, features1_t* features1, features2_t* features2) {
	DType(*features2_hand)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum] = reinterpret_cast<DType(*)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum]>(features2);

	const Bitboard occupied_bb = position.occupiedBB();

	// 駒の利き(駒種でマージ)
	Bitboard attacks[ColorNum][PieceTypeNum] = {
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
	};
	for (Square sq = SQ11; sq < SquareNum; sq++) {
		Piece p = position.piece(sq);
		if (p != Empty) {
			Color pc = pieceToColor(p);
			PieceType pt = pieceToPieceType(p);
			Bitboard bb = position.attacksFrom(pt, pc, sq, occupied_bb);
			attacks[pc][pt] |= bb;
		}
	}

	for (Color c = Black; c < ColorNum; ++c) {
		// 白の場合、色を反転
		Color c2 = c;
		if (position.turn() == White) {
			c2 = oppositeColor(c);
		}

		// 駒の配置
		Bitboard bb[PieceTypeNum];
		for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
			bb[pt] = position.bbOf(pt, c);
		}

		for (Square sq = SQ11; sq < SquareNum; ++sq) {
			// 白の場合、盤面を180度回転
			Square sq2 = sq;
			if (position.turn() == White) {
				sq2 = SQ99 - sq;
			}

			for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
				// 駒の配置
				if (bb[pt].isSet(sq)) {
					(*features1)[c2][pt - 1][sq2] = _one;
				}

				// 駒の利き
				if (attacks[c][pt].isSet(sq)) {
					(*features1)[c2][PIECETYPE_NUM + pt - 1][sq2] = _one;
				}
			}

			// 利き数
			const int num = std::min(MAX_ATTACK_NUM, position.attackersTo(c, sq, occupied_bb).popCount());
			for (int k = 0; k < num; k++) {
				(*features1)[c2][PIECETYPE_NUM + PIECETYPE_NUM + k][sq2] = _one;
			}
		}

		// hand
		Hand hand = position.hand(c);
		int p = 0;
		for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
			u32 num = hand.numOf(hp);
			if (num >= MAX_PIECES_IN_HAND[hp]) {
				num = MAX_PIECES_IN_HAND[hp];
			}
			std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, _one);
			p += MAX_PIECES_IN_HAND[hp];
		}
	}

	// is check
	if (position.inCheck()) {
		std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SquareNum, _one);
	}
}

// make move
int make_move_label(const u16 move16, const Position& position) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;

	// move direction
	int move_direction_label;
	if (from_sq < SquareNum) {
		// 白の場合、盤面を180度回転
		if (position.turn() == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;
		const int to_y = to_d.rem;
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
		const int dir_x = from_x - to_x;
		const int dir_y = to_y - from_y;

		MOVE_DIRECTION move_direction;
		if (dir_y < 0 && dir_x == 0) {
			move_direction = UP;
		}
		else if (dir_y == -2 && dir_x == -1) {
			move_direction = UP2_LEFT;
		}
		else if (dir_y == -2 && dir_x == 1) {
			move_direction = UP2_RIGHT;
		}
		else if (dir_y < 0 && dir_x < 0) {
			move_direction = UP_LEFT;
		}
		else if (dir_y < 0 && dir_x > 0) {
			move_direction = UP_RIGHT;
		}
		else if (dir_y == 0 && dir_x < 0) {
			move_direction = LEFT;
		}
		else if (dir_y == 0 && dir_x > 0) {
			move_direction = RIGHT;
		}
		else if (dir_y > 0 && dir_x == 0) {
			move_direction = DOWN;
		}
		else if (dir_y > 0 && dir_x < 0) {
			move_direction = DOWN_LEFT;
		}
		else if (dir_y > 0 && dir_x > 0) {
			move_direction = DOWN_RIGHT;
		}

		// promote
		if ((move16 & 0b100000000000000) > 0) {
			move_direction = MOVE_DIRECTION_PROMOTED[move_direction];
		}
		move_direction_label = move_direction;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (position.turn() == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
	}

	return 9 * 9 * move_direction_label + to_sq;
}

int make_move_label(const u16 move16, const Color color) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;

	// move direction
	int move_direction_label;
	if (from_sq < SquareNum) {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;
		const int to_y = to_d.rem;
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
		const int dir_x = from_x - to_x;
		const int dir_y = to_y - from_y;

		MOVE_DIRECTION move_direction;
		if (dir_y < 0 && dir_x == 0) {
			move_direction = UP;
		}
		else if (dir_y == -2 && dir_x == -1) {
			move_direction = UP2_LEFT;
		}
		else if (dir_y == -2 && dir_x == 1) {
			move_direction = UP2_RIGHT;
		}
		else if (dir_y < 0 && dir_x < 0) {
			move_direction = UP_LEFT;
		}
		else if (dir_y < 0 && dir_x > 0) {
			move_direction = UP_RIGHT;
		}
		else if (dir_y == 0 && dir_x < 0) {
			move_direction = LEFT;
		}
		else if (dir_y == 0 && dir_x > 0) {
			move_direction = RIGHT;
		}
		else if (dir_y > 0 && dir_x == 0) {
			move_direction = DOWN;
		}
		else if (dir_y > 0 && dir_x < 0) {
			move_direction = DOWN_LEFT;
		}
		else if (dir_y > 0 && dir_x > 0) {
			move_direction = DOWN_RIGHT;
		}

		// promote
		if ((move16 & 0b100000000000000) > 0) {
			move_direction = MOVE_DIRECTION_PROMOTED[move_direction];
		}
		move_direction_label = move_direction;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
	}

	return 9 * 9 * move_direction_label + to_sq;
}

// Boltzmann distribution
// see: Reinforcement Learning : An Introduction 2.3.SOFTMAX ACTION SELECTION
constexpr float default_softmax_temperature = 1.0f;
float beta = 1.0f / default_softmax_temperature; 
void set_softmax_temperature(const float temperature) {
	beta = 1.0f / temperature;
}
void softmax_temperature(std::vector<float> &log_probabilities) {
	// apply beta exponent to probabilities(in log space)
	float max = 0.0f;
	for (float& x : log_probabilities) {
		x *= beta;
		if (x > max) {
			max = x;
		}
	}
	// オーバーフローを防止するため最大値で引く
	for (float& x : log_probabilities) {
		x = expf(x - max);
	}
}

void softmax_temperature_with_normalize(std::vector<float> &log_probabilities) {
	// apply beta exponent to probabilities(in log space)
	float max = 0.0f;
	for (float& x : log_probabilities) {
		x *= beta;
		if (x > max) {
			max = x;
		}
	}
	// オーバーフローを防止するため最大値で引く
	float sum = 0.0f;
	for (float& x : log_probabilities) {
		x = expf(x - max);
		sum += x;
	}
	// normalize
	for (float& x : log_probabilities) {
		x /= sum;
	}
}
