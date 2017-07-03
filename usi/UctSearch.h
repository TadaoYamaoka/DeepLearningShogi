#pragma once

#include <atomic>
#include <random>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "ZobristHash.h"

//#define USE_VALUENET

const int THREAD_MAX = 20;              // 使用するスレッド数の最大値
const int MAX_NODES = 1000000;          // UCTのノードの配列のサイズ
const double ALL_THINKING_TIME = 1.0;  // 持ち時間(デフォルト)
const int CONST_PLAYOUT = 2000;        // 1手あたりのプレイアウト回数(デフォルト)
const double CONST_TIME = 1.0;         // 1手あたりの思考時間(デフォルト)
const int PLAYOUT_SPEED = 1000;         // 初期盤面におけるプレイアウト速度

										// 思考時間の割り振り
const int TIME_RATE_9 = 150;
const int TIME_C_13 = 30;
const int TIME_MAXPLY_13 = 30;
const int TIME_C_19 = 60;
const int TIME_MAXPLY_19 = 80;

// 先頭打着緊急度
const double FPU = 5.0;

// 候補手の最大数(盤上全体)
// http://www.nara-wu.ac.jp/math/personal/shinoda/bunki.html
// 篠田 正人、将棋における最大分岐数、コンピュータ将棋協会誌Vol.12 (1999), 57-58.
const int UCT_CHILD_MAX = 593;

// 未展開のノードのインデックス
const int NOT_EXPANDED = -1;

// 投了する勝率の閾値
const double RESIGN_THRESHOLD = 0.01;

// Virtual Loss (Best Parameter)
const int VIRTUAL_LOSS = 1;

const float c_puct = 1.0f;

enum SEARCH_MODE {
	CONST_PLAYOUT_MODE,             // 1手のプレイアウト回数を固定したモード
	CONST_TIME_MODE,                // 1手の思考時間を固定したモード
	TIME_SETTING_MODE,              // 持ち時間ありのモード(秒読みなし)
	TIME_SETTING_WITH_BYOYOMI_MODE, // 持ち時間ありのモード(秒読みあり)
};


struct thread_arg_t {
	const Position *pos; // 探索対象の局面
	int thread_id;   // スレッド識別番号
};

struct statistic_t {
	std::atomic<int> colors[3];  // その箇所を領地にした回数
};

struct child_node_t {
	Move move;  // 着手する座標
	std::atomic<int> move_count;  // 探索回数
	std::atomic<float> win;         // 勝った回数
	int index;   // インデックス
	float nnrate; // ニューラルネットワークでのレート
};

//  9x9  : 1828bytes
// 13x13 : 3764bytes
// 19x19 : 7988bytes
struct uct_node_t {
	std::atomic<int> move_count;
	std::atomic<float> win;
	int child_num;                      // 子ノードの数
	child_node_t child[UCT_CHILD_MAX];  // 子ノードの情報
	bool evaled;
	std::atomic<float> value_win;
};

struct po_info_t {
	int num;   // 次の手の探索回数
	int halt;  // 探索を打ち切る回数
	std::atomic<int> count;       // 現在の探索回数
};

struct rate_order_t {
	int index;    // ノードのインデックス
	double rate;  // その手のレート
};


// 残り時間
extern double remaining_time[ColorNum];
// UCTのノード
extern uct_node_t *uct_node;

// 現在のルートのインデックス
extern int current_root;


// 予測読みを止める
void StopPondering(void);

// 予測読みのモードの設定
void SetPonderingMode(bool flag);

// 探索のモードの指定
void SetMode(enum SEARCH_MODE mode);

// 1手あたりのプレイアウト回数の指定
void SetPlayout(int po);

// 1手あたりの思考時間の指定
void SetConstTime(double time);

// 使用するスレッド数の指定
void SetThread(int new_thread);

// 持ち時間の指定
void SetTime(double time);
void SetRemainingTime(double time, Color c);
void SetIncTime(double time, Color c);

// 相手がパスしたらパスする
void SetEarlyPass(bool pass);

// ノード展開の有無指定
void SetNoExpand(bool flag);

//
void ToggleLiveBestSequence();

// パラメータの設定
void SetParameter(void);

// time_settingsコマンドによる設定
void SetTimeSettings(int main_time, int byoyomi, int stones);

// UCT探索の初期設定
void InitializeUctSearch(void);

// 探索設定の初期化
void InitializeSearchSetting(void);

// UCT探索の終了処理
void FinalizeUctSearch(void);

// UCT探索による着手生成
Move UctSearchGenmove(const Position *pos);

// 探索の再利用の設定
void SetReuseSubtree(bool flag);

// モデルパスの設定
void SetModelPath(const char* path);