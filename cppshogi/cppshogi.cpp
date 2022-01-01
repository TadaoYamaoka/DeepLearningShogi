#include <numeric>
#include <algorithm>

#include "cppshogi.h"

// make input features
template <Color turn>
inline void make_input_features(const Position& position, features1_t* features1, features2_t* features2) {
	DType(* const features2_hand)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum] = reinterpret_cast<DType(* const)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum]>(features2);

	const Bitboard occupied_bb = position.occupiedBB();

	// 歩と歩以外に分ける
	Bitboard pawns_bb = position.bbOf(Pawn);
	Bitboard without_pawns_bb = occupied_bb & ~pawns_bb;
	// 利き数集計用
	int attack_num[ColorNum][SquareNum] = {};

	// 歩以外
	FOREACH_BB(without_pawns_bb, Square sq, {
		const Piece pc = position.piece(sq);
		const PieceType pt = pieceToPieceType(pc);
		Color c = pieceToColor(pc);
		Bitboard attacks = Position::attacksFrom(pt, c, sq, occupied_bb);

		// 後手の場合、色を反転し、盤面を180度回転
		if (turn == White) {
			c = oppositeColor(c);
			sq = SQ99 - sq;
		}

		// 駒の配置
		(*features1)[c][pt - 1][sq] = 1.0f;

		FOREACH_BB(attacks, Square to, {
			// 後手の場合、盤面を180度回転
			if (turn == White) to = SQ99 - to;

			// 駒の利き
			(*features1)[c][PIECETYPE_NUM + pt - 1][to] = 1.0f;

			// 利き数
			auto& num = attack_num[c][to];
			if (num < MAX_ATTACK_NUM) {
				(*features1)[c][PIECETYPE_NUM + PIECETYPE_NUM + num][to] = 1.0f;
				num++;
			}
		});
	});

	for (Color c = Black; c < ColorNum; ++c) {
		// 後手の場合、色を反転
		const Color c2 = turn == Black ? c : oppositeColor(c);

		// 歩
		Bitboard pawns_bb2 = pawns_bb & position.bbOf(c2);
		const SquareDelta pawnDelta = c == Black ? DeltaN : DeltaS;
		FOREACH_BB(pawns_bb2, Square sq, {
			// 後手の場合、盤面を180度回転
			if (turn == White) sq = SQ99 - sq;

			// 駒の配置
			(*features1)[c][Pawn - 1][sq] = 1.0f;

			// 駒の利き
			const Square to = sq + pawnDelta; // 1マス先
			(*features1)[c][PIECETYPE_NUM + Pawn - 1][to] = 1.0f;

			// 利き数
			auto& num = attack_num[c][to];
			if (num < MAX_ATTACK_NUM) {
				(*features1)[c][PIECETYPE_NUM + PIECETYPE_NUM + num][to] = 1.0f;
				num++;
			}
		});

		// 持ち駒
		const Hand hand = position.hand(c);
		int p = 0;
		for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
			u32 num = hand.numOf(hp);
			if (num >= MAX_PIECES_IN_HAND[hp]) {
				num = MAX_PIECES_IN_HAND[hp];
			}
			std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, 1.0f);
			p += MAX_PIECES_IN_HAND[hp];
		}
	}

	// is check
	if (position.inCheck()) {
		std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SquareNum, 1.0f);
	}
}

void make_input_features(const Position& position, features1_t* features1, features2_t* features2) {
	position.turn() == Black ? make_input_features<Black>(position, features1, features2) : make_input_features<White>(position, features1, features2);
}

inline MOVE_DIRECTION get_move_direction(const int dir_x, const int dir_y) {
	if (dir_y < 0 && dir_x == 0) {
		return UP;
	}
	else if (dir_y == -2 && dir_x == -1) {
		return UP2_LEFT;
	}
	else if (dir_y == -2 && dir_x == 1) {
		return UP2_RIGHT;
	}
	else if (dir_y < 0 && dir_x < 0) {
		return UP_LEFT;
	}
	else if (dir_y < 0 && dir_x > 0) {
		return UP_RIGHT;
	}
	else if (dir_y == 0 && dir_x < 0) {
		return LEFT;
	}
	else if (dir_y == 0 && dir_x > 0) {
		return RIGHT;
	}
	else if (dir_y > 0 && dir_x == 0) {
		return DOWN;
	}
	else if (dir_y > 0 && dir_x < 0) {
		return DOWN_LEFT;
	}
	else /* if (dir_y > 0 && dir_x > 0) */ {
		return DOWN_RIGHT;
	}
}

// make move
int make_move_label(const u16 move16, const Color color) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;

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

		MOVE_DIRECTION move_direction = get_move_direction(dir_x, dir_y);

		// promote
		if ((move16 & 0b100000000000000) > 0) {
			move_direction = (MOVE_DIRECTION)(move_direction + 10);
		}
		return 9 * 9 * move_direction + to_sq;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		const int move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
		return 9 * 9 * move_direction_label + to_sq;
	}
}
