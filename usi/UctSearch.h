#pragma once

#include <atomic>
#include <random>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "ZobristHash.h"

constexpr double ALL_THINKING_TIME = 1.0;   // 持ち時間(デフォルト)
constexpr int CONST_PLAYOUT = 10000;        // 1手あたりのプレイアウト回数(デフォルト)
constexpr double CONST_TIME = 10.0;         // 1手あたりの思考時間(デフォルト)
constexpr int PLAYOUT_SPEED = 5000;         // 初期盤面におけるプレイアウト速度

// 候補手の最大数(盤上全体)
// http://www.nara-wu.ac.jp/math/personal/shinoda/bunki.html
// 篠田 正人、将棋における最大分岐数、コンピュータ将棋協会誌Vol.12 (1999), 57-58.
constexpr int UCT_CHILD_MAX = 593;

// 未展開のノードのインデックス
constexpr unsigned int NOT_EXPANDED = -1;

// Virtual Loss (Best Parameter)
constexpr int VIRTUAL_LOSS = 1;

extern float c_init;
extern float c_base;
extern float c_fpu;

enum SEARCH_MODE {
	CONST_PLAYOUT_MODE,             // 1手のプレイアウト回数を固定したモード
	CONST_TIME_MODE,                // 1手の思考時間を固定したモード
	TIME_SETTING_MODE,              // 持ち時間ありのモード(秒読みなし)
	TIME_SETTING_WITH_BYOYOMI_MODE, // 持ち時間ありのモード(秒読みあり)
};


struct child_node_t {
	Move move;  // 着手する座標
	std::atomic<int> move_count;  // 探索回数
	std::atomic<float> win;         // 勝った回数
	unsigned int index;   // インデックス
	float nnrate; // ニューラルネットワークでのレート
};

struct uct_node_t {
	std::atomic<int> move_count;
	std::atomic<float> win;
	std::atomic<bool> evaled;      // 評価済か
	std::atomic<bool> draw;        // 千日手の可能性あり
	std::atomic<float> value_win;
	std::atomic<float> visited_nnrate;
	std::atomic<int> child_num;         // 子ノードの数
	child_node_t child[UCT_CHILD_MAX];  // 子ノードの情報
};

struct po_info_t {
	int num;   // 次の手の探索回数
	int halt;  // 探索を打ち切る回数
	std::atomic<int> count;       // 現在の探索回数
};


// 残り時間
extern double remaining_time[ColorNum];
// UCTのノード
extern uct_node_t *uct_node;

// 現在のルートのインデックス
extern unsigned int current_root;


// 予測読みを止める
void StopUctSearch(void);

// 予測読みのモードの設定
void SetPonderingMode(bool flag);

// 探索のモードの指定
void SetMode(enum SEARCH_MODE mode);
SEARCH_MODE GetMode();

// 1手あたりのプレイアウト回数の指定
void SetPlayout(int po);

// 1手あたりの思考時間の指定
void SetConstTime(double time);

// 使用するスレッド数の指定
constexpr int max_gpu = 4;
void SetThread(const int new_thread[max_gpu], const int new_policy_value_batch_maxsize[max_gpu]);

// 持ち時間の指定
void SetTime(double time);
void SetRemainingTime(double time, Color c);
void SetIncTime(double time, Color c);

// time_settingsコマンドによる設定
void SetTimeSettings(int main_time, int byoyomi, int stones);

// 投了の閾値設定（1000分率）
void SetResignThreshold(const int resign_threshold);

// 千日手の価値設定（1000分率）
void SetDrawValue(const int value_black, const int value_white);

// UCT探索の初期設定
void InitializeUctSearch(const unsigned int hash_size);
// UCT探索の終了処理
void TerminateUctSearch();

// 探索設定の初期化
void InitializeSearchSetting(void);

// UCT探索の終了処理
void FinalizeUctSearch(void);

// UCT探索による着手生成
Move UctSearchGenmove(Position *pos, Move &ponderMove, bool ponder = false);
inline Move UctSearchGenmoveNoPonder(Position *pos) {
	Move move = Move::moveNone();
	return UctSearchGenmove(pos, move);
}
// 探索の再利用の設定
void SetReuseSubtree(bool flag);

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