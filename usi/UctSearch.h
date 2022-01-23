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
bool IsUctSearchStoped();

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
Move UctSearchGenmove(Position* pos, const Key starting_pos_key, const std::vector<Move>& moves, Move& ponderMove, bool ponder = false);

// 探索の再利用の設定
void SetReuseSubtree(bool flag);

// PV表示間隔設定
void SetPvInterval(const int interval);

// MultiPV設定
void SetMultiPV(const int multipv);

// 勝率から評価値に変換する際の係数設定
void SetEvalCoef(const int eval_coef);

// ランダムムーブ設定（1000分率）
void SetRandomMove(const int ply, const int temperature, const int temperature_drop, const int cutoff, const int cutoff_drop);

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

// PVの詰み探索の設定
void SetPvMateSearch(const int threads, const int depth, const int nodes);

// 衝突回数の上限の設定
void SetCollisionLimit(const int limit);

// 訪問回数が最大の子ノードを選択
inline unsigned int select_max_child_node(const uct_node_t* uct_node)
{
	const child_node_t* uct_child = uct_node->child.get();

	unsigned int select_index = 0;
	int max_count = 0;
	const int child_num = uct_node->child_num;
	int child_win_count = 0;
	int child_lose_count = 0;

	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].IsWin()) {
			// 負けが確定しているノードは選択しない
			if (child_win_count == i && uct_child[i].move_count > max_count) {
				// すべて負けの場合は、探索回数が最大の手を選択する
				select_index = i;
				max_count = uct_child[i].move_count;
			}
			child_win_count++;
			continue;
		}
		else if (uct_child[i].IsLose()) {
			// 子ノードに一つでも負けがあれば、勝ちなので選択する
			if (child_lose_count == 0 || uct_child[i].move_count > max_count) {
				// すべて勝ちの場合は、探索回数が最大の手を選択する
				select_index = i;
				max_count = uct_child[i].move_count;
			}
			child_lose_count++;
			continue;
		}

		if (child_lose_count == 0 && uct_child[i].move_count > max_count) {
			select_index = i;
			max_count = uct_child[i].move_count;
		}
	}

	return select_index;
}
