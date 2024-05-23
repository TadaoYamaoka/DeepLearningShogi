#pragma once

#include "init.hpp"
#include "position.hpp"
#include "search.hpp"
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
constexpr int MAX_LEGAL_MOVEL_LABL_NUM = 1496;

typedef char packed_features1_t[((size_t)ColorNum * MAX_FEATURES1_NUM * (size_t)SquareNum + 7) / 8];
typedef char packed_features2_t[((size_t)MAX_FEATURES2_NUM + 7) / 8];

typedef DType features1_t[ColorNum][MAX_FEATURES1_NUM][SquareNum];
typedef DType features2_t[MAX_FEATURES2_NUM][SquareNum];

// トークン
enum class Token : int64_t {
	None,
	BPawn, BLance, BKnight, BSilver, BBishop, BRook, BGold, BKing,
	BProPawn, BProLance, BProKnight, BProSilver, BHorse, BDragon,
	WPawn, WLance, WKnight, WSilver, WBishop, WRook, WGold, WKing,
	WProPawn, WProLance, WProKnight, WProSilver, WHorse, WDragon,
	// 駒ごとの利き
	NoAttack,
	BPawnAttack, BLanceAttack, BKnightAttack, BSilverAttack, BBishopAttack, BRookAttack, BGoldAttack, BKingAttack,
	BProPawnAttack, BProLanceAttack, BProKnightAttack, BProSilverAttack, BHorseAttack, BDragonAttack,
	AWPawn, EWLance, EWKnight, EWSilver, EWBishop, EWRook, EWGold, EWKing,
	WProPawnAttack, WProLanceAttack, WProKnightAttack, WProSilverAttack, WHorseAttack, WDragonAttack,
	// 利き数
	BAttackNum, WAttackNum,
	// 持ち駒
	BHPawn, BHLance, BHKnight, BHSilver, BHGold, BHBishop, BHRook,
	WHPawn, WHLance, WHKnight, WHSilver, WHGold, WHBishop, WHRook,
};

// トークン列(盤上の駒,駒の利き,利き数のBag + 持ち駒×先後,王手)
constexpr auto MAX_TOKEN_LEN = (size_t)SquareNum + (size_t)HandPieceNum * (size_t)ColorNum + 1;
// EmbeddingBagの入力サイズ(駒,駒の利き,利き数,持ち駒,王手)
constexpr size_t BOARD_BAG_SIZE = 1 + 14 * (size_t)ColorNum + 3 * (size_t)ColorNum;
constexpr size_t HAND_WORD_SIZE = 8 + 4 * 4 + 2 * 2;
constexpr size_t MAX_WORD_SIZE = BOARD_BAG_SIZE * (size_t)SquareNum + HAND_WORD_SIZE * (size_t)ColorNum + 1;
constexpr size_t MAX_BAG_SIZE = BOARD_BAG_SIZE * MAX_TOKEN_LEN;
typedef int8_t bags_t[MAX_BAG_SIZE];
constexpr int64_t offsets[MAX_TOKEN_LEN] = {
	BOARD_BAG_SIZE * 0, BOARD_BAG_SIZE * 1, BOARD_BAG_SIZE * 2, BOARD_BAG_SIZE * 3, BOARD_BAG_SIZE * 4, BOARD_BAG_SIZE * 5, BOARD_BAG_SIZE * 6, BOARD_BAG_SIZE * 7, BOARD_BAG_SIZE * 8,
	BOARD_BAG_SIZE * 9, BOARD_BAG_SIZE * 10, BOARD_BAG_SIZE * 11, BOARD_BAG_SIZE * 12, BOARD_BAG_SIZE * 13, BOARD_BAG_SIZE * 14, BOARD_BAG_SIZE * 15, BOARD_BAG_SIZE * 16, BOARD_BAG_SIZE * 17,
	BOARD_BAG_SIZE * 18, BOARD_BAG_SIZE * 19, BOARD_BAG_SIZE * 20, BOARD_BAG_SIZE * 21, BOARD_BAG_SIZE * 22, BOARD_BAG_SIZE * 23, BOARD_BAG_SIZE * 24, BOARD_BAG_SIZE * 25, BOARD_BAG_SIZE * 26,
	BOARD_BAG_SIZE * 27, BOARD_BAG_SIZE * 28, BOARD_BAG_SIZE * 29, BOARD_BAG_SIZE * 30, BOARD_BAG_SIZE * 31, BOARD_BAG_SIZE * 32, BOARD_BAG_SIZE * 33, BOARD_BAG_SIZE * 34, BOARD_BAG_SIZE * 35,
	BOARD_BAG_SIZE * 36, BOARD_BAG_SIZE * 37, BOARD_BAG_SIZE * 38, BOARD_BAG_SIZE * 39, BOARD_BAG_SIZE * 40, BOARD_BAG_SIZE * 41, BOARD_BAG_SIZE * 42, BOARD_BAG_SIZE * 43, BOARD_BAG_SIZE * 44,
	BOARD_BAG_SIZE * 45, BOARD_BAG_SIZE * 46, BOARD_BAG_SIZE * 47, BOARD_BAG_SIZE * 48, BOARD_BAG_SIZE * 49, BOARD_BAG_SIZE * 50, BOARD_BAG_SIZE * 51, BOARD_BAG_SIZE * 52, BOARD_BAG_SIZE * 53,
	BOARD_BAG_SIZE * 54, BOARD_BAG_SIZE * 55, BOARD_BAG_SIZE * 56, BOARD_BAG_SIZE * 57, BOARD_BAG_SIZE * 58, BOARD_BAG_SIZE * 59, BOARD_BAG_SIZE * 60, BOARD_BAG_SIZE * 61, BOARD_BAG_SIZE * 62,
	BOARD_BAG_SIZE * 63, BOARD_BAG_SIZE * 64, BOARD_BAG_SIZE * 65, BOARD_BAG_SIZE * 66, BOARD_BAG_SIZE * 67, BOARD_BAG_SIZE * 68, BOARD_BAG_SIZE * 69, BOARD_BAG_SIZE * 70, BOARD_BAG_SIZE * 71,
	BOARD_BAG_SIZE * 72, BOARD_BAG_SIZE * 73, BOARD_BAG_SIZE * 74, BOARD_BAG_SIZE * 75, BOARD_BAG_SIZE * 76, BOARD_BAG_SIZE * 77, BOARD_BAG_SIZE * 78, BOARD_BAG_SIZE * 79, BOARD_BAG_SIZE * 80,
	BOARD_BAG_SIZE * 81, BOARD_BAG_SIZE * 82, BOARD_BAG_SIZE * 83, BOARD_BAG_SIZE * 84, BOARD_BAG_SIZE * 85, BOARD_BAG_SIZE * 86, BOARD_BAG_SIZE * 87,
	BOARD_BAG_SIZE * 88, BOARD_BAG_SIZE * 89, BOARD_BAG_SIZE * 90, BOARD_BAG_SIZE * 91, BOARD_BAG_SIZE * 92, BOARD_BAG_SIZE * 93, BOARD_BAG_SIZE * 94,
	BOARD_BAG_SIZE * 95
};

void make_input_features(const Position& position, bags_t tokens);
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
