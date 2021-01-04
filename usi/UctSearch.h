#pragma once

#include <atomic>
#include <random>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "Node.h"

// 候補手の最大数(盤上全体)
// http://www.nara-wu.ac.jp/math/personal/shinoda/bunki.html
// 篠田 正人、将棋における最大分岐数、コンピュータ将棋協会誌Vol.12 (1999), 57-58.
constexpr int UCT_CHILD_MAX = 593;

// Virtual Loss (Best Parameter)
constexpr int VIRTUAL_LOSS = 1;

extern float c_init;
extern float c_base;
extern float c_fpu_reduction;
extern float c_init_root;
extern float c_base_root;
extern float c_fpu_reduction_root;

struct po_info_t {
	int halt;  // 探索を打ち切る回数
	std::atomic<int> count;       // 現在の探索回数
};

void SetLimits(const LimitsType& limits);
void SetLimits(const Position* pos, const LimitsType& limits);
void SetConstPlayout(const int playout);

// 残り時間
extern int remaining_time[ColorNum];

// 現在のルートのインデックス
extern unsigned int current_root;

// ノード数の上限
extern unsigned int po_max;

// 予測読みを止める
void StopUctSearch(void);

// 予測読みのモードの設定
void SetPonderingMode(bool flag);


// 使用するスレッド数の指定
constexpr int max_gpu = 8;
void SetThread(const int new_thread[max_gpu], const int new_policy_value_batch_maxsize[max_gpu]);


// 投了の閾値設定（1000分率）
void SetResignThreshold(const int resign_threshold);

// 千日手の価値設定（1000分率）
void SetDrawValue(const int value_black, const int value_white);

// UCT探索の初期設定
void InitializeUctSearch(const unsigned int node_limit);
// UCT探索の終了処理
void TerminateUctSearch();

// 探索設定の初期化
// void InitializeSearchSetting(void);

// UCT探索の終了処理
void FinalizeUctSearch(void);

// UCT探索による着手生成
Move UctSearchGenmove(Position *pos, const Key starting_pos_key, const std::vector<Move>& moves, Move &ponderMove, bool ponder = false);

// 探索の再利用の設定
void SetReuseSubtree(bool flag);

// PV表示間隔設定
void SetPvInterval(const int interval);

// MultiPV設定
void SetMultiPV(const int multipv);

// モデルパスの設定
void SetModelPath(const std::string path[max_gpu]);

// 新規ゲーム
void NewGame();

// ゲーム終了
void GameOver();

// 1手にかける時間取得（ms）
int GetTimeLimit();

// 引き分けとする手数の設定
void SetDrawPly(const int ply);
