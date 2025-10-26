#pragma once

#include "init.hpp"
#include "position.hpp"
#include "generateMoves.hpp"
#include "dtype.h"

#define LEN(array) (sizeof(array) / sizeof(array[0]))

constexpr int MAX_HPAWN_NUM = 8; // 歩の持ち駒の上限
constexpr int MAX_HLANCE_NUM = 4;
constexpr int MAX_HKNIGHT_NUM = 4;
constexpr int MAX_HSILVER_NUM = 4;
constexpr int MAX_HGOLD_NUM = 4;
constexpr int MAX_HBISHOP_NUM = 2;
constexpr int MAX_HROOK_NUM = 2;

constexpr u32 MAX_PIECES_IN_HAND[] = {
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

#ifdef NYUGYOKU_FEATURES
constexpr u32 MAX_NYUGYOKU_OPP_FIELD = 10; // 敵陣三段目以内の駒(10枚までの残り枚数)
constexpr u32 MAX_NYUGYOKU_SCORE = 20; // 点数(先手28点、後手27点までの残り枚数)
constexpr u32 MAX_FEATURES2_NYUGYOKU_NUM = 1/*入玉*/ + MAX_NYUGYOKU_OPP_FIELD + MAX_NYUGYOKU_SCORE;
#endif

constexpr int PIECETYPE_NUM = 14; // 駒の種類
constexpr int MAX_ATTACK_NUM = 3; // 利き数の最大値
constexpr u32 MAX_FEATURES1_NUM = PIECETYPE_NUM/*駒の配置*/ + PIECETYPE_NUM/*駒の利き*/ + MAX_ATTACK_NUM/*利き数*/;
constexpr u32 MAX_FEATURES2_NUM = MAX_FEATURES2_HAND_NUM + 1/*王手*/
#ifdef NYUGYOKU_FEATURES
    + (int)ColorNum * MAX_FEATURES2_NYUGYOKU_NUM
#endif
;

// 移動の定数
enum MOVE_DIRECTION {
	UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
	UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
	MOVE_DIRECTION_NUM
};

// 指し手を表すラベルの数
constexpr int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HandPieceNum;

typedef char packed_features1_t[((size_t)ColorNum * MAX_FEATURES1_NUM * (size_t)SquareNum + 7) / 8];
typedef char packed_features2_t[((size_t)MAX_FEATURES2_NUM + 7) / 8];

typedef DType features1_t[ColorNum][MAX_FEATURES1_NUM][SquareNum];
typedef DType features2_t[MAX_FEATURES2_NUM][SquareNum];

void make_input_features(const Position& position, features1_t features1, features2_t features2);
void make_input_features(const Position& position, packed_features1_t packed_features1, packed_features2_t packed_features2);
int make_move_label(const u16 move16, const Color color);

// 評価値から価値(勝率)に変換
// スケールパラメータは、elmo_for_learnの勝率から調査した値
inline float score_to_value(const Score score) {
	return 1.0f / (1.0f + expf(-(float)score * 0.0013226f));
}

struct HuffmanCodedPosAndEval2 {
	HuffmanCodedPos hcp;
	s16 eval;
	u16 bestMove16;
	uint8_t result; // xxxxxx11 : 勝敗、xxxxx1xx : 千日手、xxxx1xxx : 入玉宣言、xxx1xxxx : 最大手数
};
static_assert(sizeof(HuffmanCodedPosAndEval2) == 38, "");

struct HuffmanCodedPosAndEval3 {
	HuffmanCodedPos hcp; // 開始局面
	u16 moveNum; // 手数
	u8 result; // xxxxxx11 : 勝敗、xxxxx1xx : 千日手、xxxx1xxx : 入玉宣言、xxx1xxxx : 最大手数
	u8 opponent; // 対戦相手（0:自己対局、1:先手usi、2:後手usi）
};
static_assert(sizeof(HuffmanCodedPosAndEval3) == 36, "");

struct MoveInfo {
	u16 selectedMove16; // 指し手
	s16 eval; // 評価値
	u16 candidateNum; // 候補手の数
};
static_assert(sizeof(MoveInfo) == 6, "");

struct MoveVisits {
	MoveVisits() {}
	MoveVisits(const u16 move16, const u16 visitNum) : move16(move16), visitNum(visitNum) {}

	u16 move16;
	u16 visitNum;
};
static_assert(sizeof(MoveVisits) == 4, "");

struct Hcpe3CacheBody {
	HuffmanCodedPos hcp; // 局面
	float value;
	float result;
	int count; // 重複カウント
};

struct Hcpe3CacheCandidate {
	u16 move16;
	float prob;
};

struct TrainingData {
    TrainingData() = default;
	TrainingData(const HuffmanCodedPos& hcp, const float value, const float result)
		: hcp(hcp), value(value), result(result), count(1) {};
	TrainingData(const Hcpe3CacheBody& body, const Hcpe3CacheCandidate* candidates, const size_t candidateNum)
		: hcp(body.hcp), value(body.value), result(body.result), count(body.count), candidates(candidateNum) {
		for (size_t i = 0; i < candidateNum; i++) {
			this->candidates.emplace(candidates[i].move16, candidates[i].prob);
		}
	};

	HuffmanCodedPos hcp;
	float value;
	float result;
	std::unordered_map<u16, float> candidates;
	int count; // 重複カウント
};

constexpr u8 GAMERESULT_SENNICHITE = 0x4;
constexpr u8 GAMERESULT_NYUGYOKU = 0x8;
constexpr u8 GAMERESULT_MAXMOVE = 0x16;
