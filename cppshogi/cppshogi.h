#pragma once

#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>

#include "init.hpp"
#include "position.hpp"
#include "search.hpp"
#include "generateMoves.hpp"

#define LEN(array) (sizeof(array) / sizeof(array[0]))

const int MAX_HPAWN_NUM = 8; // 歩の持ち駒の上限
const int MAX_HLANCE_NUM = 4;
const int MAX_HKNIGHT_NUM = 4;
const int MAX_HSILVER_NUM = 4;
const int MAX_HGOLD_NUM = 4;
const int MAX_HBISHOP_NUM = 2;
const int MAX_HROOK_NUM = 2;

const u32 MAX_PIECES_IN_HAND[] = {
	MAX_HPAWN_NUM, // PAWN
	MAX_HLANCE_NUM, // LANCE
	MAX_HKNIGHT_NUM, // KNIGHT
	MAX_HSILVER_NUM, // SILVER
	MAX_HGOLD_NUM, // GOLD
	MAX_HBISHOP_NUM, // BISHOP
	MAX_HROOK_NUM, // ROOK
};
const u32 MAX_PIECES_IN_HAND_SUM = MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;
const u32 MAX_FEATURES2_HAND_NUM = (int)ColorNum * MAX_PIECES_IN_HAND_SUM;

const int PIECETYPE_NUM = 14; // 駒の種類
const int MAX_ATTACK_NUM = 3; // 利き数の最大値
const u32 MAX_FEATURES1_NUM = PIECETYPE_NUM/*駒の配置*/ + PIECETYPE_NUM/*駒の利き*/ + MAX_ATTACK_NUM/*利き数*/;
const u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/;

// 移動の定数
enum MOVE_DIRECTION {
	UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
	UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
	MOVE_DIRECTION_NUM
};

const MOVE_DIRECTION MOVE_DIRECTION_PROMOTED[] = {
	UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
};

// 指し手を表すラベルの数
const int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HandPieceNum;

void make_input_features(const Position& position, float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum], float(*features2)[MAX_FEATURES2_NUM][SquareNum]);
int make_move_label(const u16 move16, const Position& position);
int make_move_label(const u16 move16, const Color color);
void softmax_tempature_with_normalize(std::vector<float> &log_probabilities);
void set_softmax_tempature(const float tempature);

// 評価値から価値(勝率)に変換
// スケールパラメータは、elmo_for_learnの勝率から調査した値
inline float score_to_value(const Score score) {
	return 1.0f / (1.0f + expf(-(float)score * 0.0013226f));
}
