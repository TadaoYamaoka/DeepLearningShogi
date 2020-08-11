#pragma once

#include "init.hpp"
#include "position.hpp"
#include "search.hpp"
#include "generateMoves.hpp"
#include "../usi/cudnn_dtype.h"

#define LEN(array) (sizeof(array) / sizeof(array[0]))

constexpr int MAX_HPAWN_NUM = 8; // 歩の持ち駒の上限
constexpr int MAX_HLANCE_NUM = 4;
constexpr int MAX_HKNIGHT_NUM = 4;
constexpr int MAX_HSILVER_NUM = 4;
constexpr int MAX_HGOLD_NUM = 4;
constexpr int MAX_HBISHOP_NUM = 2;
constexpr int MAX_HROOK_NUM = 2;

const u32 MAX_PIECES_IN_HAND[] = {
	MAX_HPAWN_NUM, // PAWN
	MAX_HLANCE_NUM, // LANCE
	MAX_HKNIGHT_NUM, // KNIGHT
	MAX_HSILVER_NUM, // SILVER
	MAX_HGOLD_NUM, // GOLD
	MAX_HBISHOP_NUM, // BISHOP
	MAX_HROOK_NUM, // ROOK
};
constexpr u32 MAX_PIECES_IN_HAND_SUM = MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;
constexpr u32 MAX_FEATURES2_HAND_NUM = (int)ColorNum * MAX_PIECES_IN_HAND_SUM;

constexpr int PIECETYPE_NUM = 14; // 駒の種類
constexpr int MAX_ATTACK_NUM = 3; // 利き数の最大値
constexpr u32 MAX_FEATURES1_NUM = PIECETYPE_NUM/*駒の配置*/ + PIECETYPE_NUM/*駒の利き*/ + MAX_ATTACK_NUM/*利き数*/;
constexpr u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/;

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
constexpr int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HandPieceNum;

typedef DType features1_t[ColorNum][MAX_FEATURES1_NUM][SquareNum];
typedef DType features2_t[MAX_FEATURES2_NUM][SquareNum];

void make_input_features(const Position& position, features1_t* features1, features2_t* features2);
int make_move_label(const u16 move16, const Position& position);
int make_move_label(const u16 move16, const Color color);
void softmax_temperature(std::vector<float> &log_probabilities);
void softmax_temperature_with_normalize(std::vector<float> &log_probabilities);
void set_softmax_temperature(const float temperature);

// 評価値から価値(勝率)に変換
// スケールパラメータは、elmo_for_learnの勝率から調査した値
inline float score_to_value(const Score score) {
	return 1.0f / (1.0f + expf(-(float)score * 0.0013226f));
}
