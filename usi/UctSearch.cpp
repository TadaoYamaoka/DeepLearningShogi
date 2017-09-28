#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <thread>
#include <random>
#include <queue>

#include "Message.h"
#include "UctSearch.h"
#include "Utility.h"
#include "mate.h"

#if defined (_WIN32)
#define NOMINMAX
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#endif

#include "cppshogi.h"
namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace std;

#define LOCK_NODE(var) mutex_nodes[(var)].lock()
#define UNLOCK_NODE(var) mutex_nodes[(var)].unlock()
#define LOCK_EXPAND mutex_expand.lock();
#define UNLOCK_EXPAND mutex_expand.unlock();

void ReadWeights();
void EvalNode();

////////////////
//  大域変数  //
////////////////

// 持ち時間
double remaining_time[ColorNum];
double inc_time[ColorNum];
double po_per_sec = PLAYOUT_SPEED;

// UCTのノード
uct_node_t *uct_node;

// プレイアウト情報
static po_info_t po_info;

// 試行時間を延長するかどうかのフラグ
static bool extend_time = false;

unsigned int current_root; // 現在のルートのインデックス
mutex mutex_nodes[MAX_NODES];
mutex mutex_expand;       // ノード展開を排他処理するためのmutex

// 探索の設定
enum SEARCH_MODE mode = TIME_SETTING_WITH_BYOYOMI_MODE;
// 使用するスレッド数
int threads = 16;
// 1手あたりの試行時間
double const_thinking_time = CONST_TIME;
// 1手当たりのプレイアウト数
int playout = CONST_PLAYOUT;
// デフォルトの持ち時間
double default_remaining_time = ALL_THINKING_TIME;

// 各スレッドに渡す引数
thread_arg_t t_arg[THREAD_MAX];

double time_limit;

std::thread *handle[THREAD_MAX];    // スレッドのハンドル

// 乱数生成器
std::mt19937_64 *mt[THREAD_MAX];

// 
bool reuse_subtree = true;

// 自分の手番の色
int my_color;

//
static bool live_best_sequence = false;

ray_clock::time_point begin_time;

// 2つのキューを交互に使用する
const int policy_value_batch_maxsize = THREAD_MAX * 10; // スレッド数以上確保する
static float features1[2][policy_value_batch_maxsize][ColorNum][MAX_FEATURES1_NUM][SquareNum];
static float features2[2][policy_value_batch_maxsize][MAX_FEATURES2_NUM][SquareNum];
static unsigned int policy_value_hash_index[2][policy_value_batch_maxsize];
static int current_policy_value_queue_index = 0;
static int current_policy_value_batch_index = 0;

// 予測関数
py::object dlshogi_predict;

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
const int MATE_SEARCH_DEPTH = 7;

//template<float>
double atomic_fetch_add(std::atomic<float> *obj, float arg) {
	float expected = obj->load();
	while (!atomic_compare_exchange_weak(obj, &expected, expected + arg))
		;
	return expected;
}

static void
ClearEvalQueue()
{
	current_policy_value_queue_index = 0;
	current_policy_value_batch_index = 0;
}

////////////
//  関数  //
////////////

// Virtual Lossを加算
static void AddVirtualLoss(child_node_t *child, unsigned int current);

// 次のプレイアウト回数の設定
static void CalculatePlayoutPerSec(double finish_time);
static void CalculateNextPlayouts(const Position *pos);

// ノードの展開
static unsigned int ExpandNode(Position *pos, unsigned int current, const std::vector<int>& path);

// ルートの展開
static unsigned int ExpandRoot(const Position *pos);

// 思考時間を延長する処理
static bool ExtendTime(void);

// 候補手の初期化
static void InitializeCandidate(child_node_t *uct_child, Move move);

// 探索打ち切りの確認
static bool InterruptionCheck(void);

// UCT探索
static void ParallelUctSearch(thread_arg_t *arg);

// ノードのレーティング
static void QueuingNode(const Position *pos, unsigned int index);

// UCB値が最大の子ノードを返す
static int SelectMaxUcbChild(const Position *pos, unsigned int current, mt19937_64 *mt);

// UCT探索(1回の呼び出しにつき, 1回の探索)
static float UctSearch(Position *pos, mt19937_64 *mt, unsigned int current, std::vector<int>& path);

// 結果の更新
static void UpdateResult(child_node_t *child, float result, unsigned int current);



////////////////////////
//  探索モードの指定  //
////////////////////////
void
SetMode(enum SEARCH_MODE new_mode)
{
	mode = new_mode;
}
SEARCH_MODE GetMode()
{
	return mode;
}

///////////////////////////////////////
//  1手あたりのプレイアウト数の指定  //
///////////////////////////////////////
void
SetPlayout(int po)
{
	playout = po;
}


/////////////////////////////////
//  1手にかける試行時間の設定  //
/////////////////////////////////
void
SetConstTime(double time)
{
	const_thinking_time = time;
}


////////////////////////////////
//  使用するスレッド数の指定  //
////////////////////////////////
void
SetThread(int new_thread)
{
	threads = new_thread;
}


//////////////////////
//  持ち時間の設定  //
//////////////////////
void
SetTime(double time)
{
	default_remaining_time = time;
}
void
SetRemainingTime(double time, Color c)
{
	remaining_time[c] = time;
}
void
SetIncTime(double time, Color c)
{
	inc_time[c] = time;
}

//////////////////////////
//  ノード再利用の設定  //
//////////////////////////
void
SetReuseSubtree(bool flag)
{
	reuse_subtree = flag;
}

//////////////////////////////
// Toggle Live Best Sequece //
//////////////////////////////
void
ToggleLiveBestSequence()
{
	live_best_sequence = !live_best_sequence;
}

//////////////////////////////////////
//  time_settingsコマンドによる設定  //
//////////////////////////////////////
void
SetTimeSettings(int main_time, int byoyomi, int stone)
{
	if (main_time == 0) {
		const_thinking_time = (double)byoyomi * 0.85;
		mode = CONST_TIME_MODE;
		cerr << "Const Thinking Time Mode" << endl;
	}
	else {
		if (byoyomi == 0) {
			default_remaining_time = main_time;
			mode = TIME_SETTING_MODE;
			cerr << "Time Setting Mode" << endl;
		}
		else {
			default_remaining_time = main_time;
			const_thinking_time = ((double)byoyomi) / stone;
			mode = TIME_SETTING_WITH_BYOYOMI_MODE;
			cerr << "Time Setting Mode (byoyomi)" << endl;
		}
	}
}

/////////////////////////
//  UCT探索の初期設定  //
/////////////////////////
void
InitializeUctSearch(void)
{
	// UCTのノードのメモリを確保
	uct_node = (uct_node_t *)malloc(sizeof(uct_node_t) * uct_hash_size);

	if (uct_node == NULL) {
		cerr << "Cannot allocate memory !!" << endl;
		cerr << "You must reduce tree size !!" << endl;
		exit(1);
	}

	// use_nn && !nn_model
	ReadWeights();
}


////////////////////////
//  探索設定の初期化  //
////////////////////////
void
InitializeSearchSetting(void)
{
	// 乱数の初期化
	for (int i = 0; i < THREAD_MAX; i++) {
		if (mt[i]) {
			delete mt[i];
		}
		mt[i] = new mt19937_64((unsigned int)(time(NULL) + i));
	}

	// 持ち時間の初期化
	for (int i = 0; i < 3; i++) {
		remaining_time[i] = default_remaining_time;
	}

	// 制限時間を設定
	// プレイアウト回数の初期化
	if (mode == CONST_PLAYOUT_MODE) {
		time_limit = 100000.0;
		po_info.num = playout;
		extend_time = false;
	}
	else if (mode == CONST_TIME_MODE) {
		time_limit = const_thinking_time;
		po_info.num = 100000000;
		extend_time = false;
	}
	else if (mode == TIME_SETTING_MODE ||
		mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
		time_limit = remaining_time[0];
		po_info.num = (int)(PLAYOUT_SPEED * time_limit);
		extend_time = true;
	}
	po_per_sec = PLAYOUT_SPEED;
}


////////////
//  終了  //
////////////
void
FinalizeUctSearch(void)
{

}


/////////////////////////////////////
//  UCTアルゴリズムによる着手生成  //
/////////////////////////////////////
Move
UctSearchGenmove(Position *pos)
{
	Move move;
	double finish_time;
	child_node_t *uct_child;

	// 探索情報をクリア
	po_info.count = 0;

	if (reuse_subtree) {
		DeleteOldHash(pos);
	}
	else {
		ClearUctHash();
	}

	ClearEvalQueue();

	// 探索開始時刻の記録
	begin_time = ray_clock::now();

	// UCTの初期化
	current_root = ExpandRoot(pos);

#ifndef USE_VALUENET
	// 従来の評価関数を使う
	// evaluate() の差分計算を無効化する。
	LimitsType limits;
	limits.depth = static_cast<Depth>(1);
	pos->searcher()->alpha = -ScoreMaxEvaluate;
	pos->searcher()->beta = ScoreMaxEvaluate;
	pos->searcher()->threads.startThinking(*pos, limits, pos->searcher()->states);
	pos->searcher()->threads.main()->waitForSearchFinished();
	Score score = pos->searcher()->threads.main()->rootMoves[0].score;
	uct_node[current_root].value_win = score_to_value(score);
	//std::cout << score << std::endl;
#endif // !USE_VALUENET

	// 詰みのチェック
	if (uct_node[current_root].child_num == 0) {
		return Move::moveNone();
	}
	else if (uct_node[current_root].value_win == FLT_MAX) {
		// 詰み
		return mateMoveInOddPlyReturnMove(*pos, MATE_SEARCH_DEPTH);
	}
	else if (uct_node[current_root].value_win == FLT_MIN) {
		// 自玉の詰み
		return Move::moveNone();
	}

	// 前回から持ち込んだ探索回数を記録
	int pre_simulated = uct_node[current_root].move_count;

	// 探索回数の閾値を設定
	CalculateNextPlayouts(pos);
	po_info.halt = po_info.num;

	// 自分の手番を設定
	my_color = pos->turn();

	// 探索時間とプレイアウト回数の予定値を出力
	PrintPlayoutLimits(time_limit, po_info.halt);

	for (int i = 0; i < threads; i++) {
		t_arg[i].thread_id = i;
		t_arg[i].pos = pos;
		handle[i] = new thread(ParallelUctSearch, &t_arg[i]);
	}

	// use_nn
	handle[threads] = new thread(EvalNode);

	for (int i = 0; i < threads; i++) {
		handle[i]->join();
		delete handle[i];
		handle[i] = nullptr;
	}
	// use_nn
	handle[threads]->join();
	delete handle[threads];
	handle[threads] = nullptr;

	// 探索にかかった時間を求める
	finish_time = GetSpendTime(begin_time);

	uct_child = uct_node[current_root].child;

	int max_count = 0;
	unsigned int select_index;

	// 探索回数最大の手を見つける
	for (int i = 0; i < uct_node[current_root].child_num; i++) {
		if (uct_child[i].move_count > max_count) {
			select_index = i;
			max_count = uct_child[i].move_count;
		}
		cout << "info string " << i << ":" << uct_child[i].move.toUSI() << " move_count:" << uct_child[i].move_count << " win_rate:" << uct_child[i].win / (uct_child[i].move_count + 0.0001f) << endl;
	}

	// 選択した着手の勝率の算出
	float best_wp = uct_child[select_index].win / uct_child[select_index].move_count;

	if (best_wp <= RESIGN_THRESHOLD) {
		move = Move::moveNone();
	}
	else {
		move = uct_child[select_index].move;

		// 歩、角、飛が成らない場合、強制的に成る
		if (!move.isDrop() && !move.isPromotion() &&
			(move.pieceTypeTo() == Pawn || move.pieceTypeTo() == Bishop || move.pieceTypeTo() == Rook)) {
			// 合法手に成る手があるか
			for (int i = 0; i < uct_node[current_root].child_num; i++) {
				if (uct_child[i].move.isPromotion() && uct_child[i].move.fromAndTo() == move.fromAndTo()) {
					// 強制的に成る
					move = uct_child[i].move;
					break;
				}
			}
		}

		int cp;
		if (best_wp == 1.0f) {
			cp = 30000;
		}
		else {
			cp = int(-logf(1.0f / best_wp - 1.0f) * 754.3f);
		}

		// PV表示
		string pv = move.toUSI();
		{
			unsigned int best_index = select_index;
			child_node_t *best_node = uct_child;

			while (best_node[best_index].index != -1) {
				const int best_node_index = best_node[best_index].index;

				best_node = uct_node[best_node_index].child;
				max_count = 0;
				for (int i = 0; i < uct_node[best_node_index].child_num; i++) {
					if (best_node[i].move_count > max_count) {
						best_index = i;
						max_count = best_node[i].move_count;
					}
				}

				if (max_count < 20)
					break;

				pv += " " + best_node[best_index].move.toUSI();
			}
		}

		cout << "info nps " << int(uct_node[current_root].move_count / finish_time) << " time " << int(finish_time * 1000) << " nodes " << uct_node[current_root].move_count << " hashfull " << GetUctHashUsageRate() << " score cp " << cp << " pv " << pv << endl;

		// 次の探索でのプレイアウト回数の算出
		CalculatePlayoutPerSec(finish_time);
	}

	// 最善応手列を出力
	//PrintBestSequence(pos, uct_node, current_root);
	// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
	PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, pre_simulated);

	ClearEvalQueue();

	return move;
}


/////////////////////
//  候補手の初期化  //
/////////////////////
static void
InitializeCandidate(child_node_t *uct_child, Move move)
{
	uct_child->move = move;
	uct_child->move_count = 0;
	uct_child->win = 0;
	uct_child->index = NOT_EXPANDED;
	uct_child->nnrate = 0;
}


/////////////////////////
//  ルートノードの展開  //
/////////////////////////
static unsigned int
ExpandRoot(const Position *pos)
{
	unsigned int index = FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly());
	child_node_t *uct_child;
	int child_num = 0;

	std::vector<int> path;

	// 既に展開されていた時は, 探索結果を再利用する
	if (index != uct_hash_size) {
		path.push_back(index);

		PrintReuseCount(uct_node[index].move_count);

		return index;
	}
	else {
		// 空のインデックスを探す
		index = SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly());

		assert(index != uct_hash_size);

		// ルートノードの初期化
		uct_node[index].move_count = 0;
		uct_node[index].win = 0;
		uct_node[index].child_num = 0;
		uct_node[index].evaled = false;

		uct_child = uct_node[index].child;

		// 候補手の展開
		for (MoveList<Legal> ml(*pos); !ml.end(); ++ml) {
			InitializeCandidate(&uct_child[child_num], ml.move());
			child_num++;
		}

		path.push_back(index);

		// 子ノード個数の設定
		uct_node[index].child_num = child_num;

		// 候補手のレーティング
		QueuingNode(pos, index);

	}

	return index;
}



///////////////////
//  ノードの展開  //
///////////////////
static unsigned int
ExpandNode(Position *pos, unsigned int current, const std::vector<int>& path)
{
	unsigned int index = FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly() + path.size());
	child_node_t *uct_child;

	// 合流先が検知できれば, それを返す
	if (index != uct_hash_size) {
		return index;
	}

	// 空のインデックスを探す
	index = SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly() + path.size());

	assert(index != uct_hash_size);

	// 現在のノードの初期化
	uct_node[index].move_count = 0;
	uct_node[index].win = 0;
	uct_node[index].child_num = 0;
	uct_node[index].evaled = false;
	uct_child = uct_node[index].child;

	// 候補手の展開
	int child_num = 0;
	for (MoveList<Legal> ml(*pos); !ml.end(); ++ml) {
		InitializeCandidate(&uct_child[child_num], ml.move());
		child_num++;
	}

	// 子ノードの個数を設定
	uct_node[index].child_num = child_num;

	// 候補手のレーティング
	if (child_num > 0) {
		QueuingNode(pos, index);
	}
	else {
		uct_node[index].value_win = 0.0f;
		uct_node[index].evaled = true;
	}

	return index;
}


//////////////////////////////////////
//  ノードをキューに追加            //
//////////////////////////////////////
static void
QueuingNode(const Position *pos, unsigned int index)
{
	//cout << "QueuingNode:" << index << ":" << current_policy_value_queue_index << ":" << current_policy_value_batch_index << endl;
	//cout << pos->toSFEN() << endl;

	if (current_policy_value_batch_index >= policy_value_batch_maxsize) {
		std::cout << "error" << std::endl;
	}
	// set all zero
	std::fill_n((float*)features1[current_policy_value_queue_index][current_policy_value_batch_index], (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
	std::fill_n((float*)features2[current_policy_value_queue_index][current_policy_value_batch_index], MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

	make_input_features(*pos, &features1[current_policy_value_queue_index][current_policy_value_batch_index], &features2[current_policy_value_queue_index][current_policy_value_batch_index]);
	policy_value_hash_index[current_policy_value_queue_index][current_policy_value_batch_index] = index;
	current_policy_value_batch_index++;
}


//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
static bool
InterruptionCheck(void)
{
	int max = 0, second = 0;
	const int child_num = uct_node[current_root].child_num;
	const int rest = po_info.halt - po_info.count;
	child_node_t *uct_child = uct_node[current_root].child;

	if (mode != CONST_PLAYOUT_MODE &&
		GetSpendTime(begin_time) * 10.0 < time_limit) {
		return false;
	}

	// 探索回数が最も多い手と次に多い手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max) {
			second = max;
			max = uct_child[i].move_count;
		}
		else if (uct_child[i].move_count > second) {
			second = uct_child[i].move_count;
		}
	}

	// 残りの探索を全て次善手に費やしても
	// 最善手を超えられない場合は探索を打ち切る
	if (max - second > rest) {
		return true;
	}
	else {
		return false;
	}
}


///////////////////////////
//  思考時間延長の確認   //
///////////////////////////
static bool
ExtendTime(void)
{
	int max = 0, second = 0;
	const int child_num = uct_node[current_root].child_num;
	child_node_t *uct_child = uct_node[current_root].child;

	// 探索回数が最も多い手と次に多い手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max) {
			second = max;
			max = uct_child[i].move_count;
		}
		else if (uct_child[i].move_count > second) {
			second = uct_child[i].move_count;
		}
	}

	// 最善手の探索回数がが次善手の探索回数の
	// 1.2倍未満なら探索延長
	if (max < second * 1.2) {
		return true;
	}
	else {
		return false;
	}
}



/////////////////////////////////
//  並列処理で呼び出す関数     //
//  UCTアルゴリズムを反復する  //
/////////////////////////////////
static void
ParallelUctSearch(thread_arg_t *arg)
{
	thread_arg_t *targ = (thread_arg_t *)arg;
	bool interruption = false;
	bool enough_size = true;

	// policyが計算されるのを待つ
	//cout << "wait policy:" << current_root << ":" << uct_node[current_root].evaled << endl;
	while (!uct_node[current_root].evaled)
		this_thread::sleep_for(chrono::milliseconds(0));

	// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
	do {
		// 探索回数を1回増やす	
		atomic_fetch_add(&po_info.count, 1);
		// 盤面のコピー
		Position pos(*targ->pos);
		//cout << pos.toSFEN() << ":" << pos.getKey() << endl;
		// 1回プレイアウトする
		std::vector<int> path;
		UctSearch(&pos, mt[targ->thread_id], current_root, path);
		//cout << "root:" << current_root << " move_count:" << uct_node[current_root].move_count << endl;
		// 探索を打ち切るか確認
		interruption = InterruptionCheck();
		// ハッシュに余裕があるか確認
		enough_size = CheckRemainingHashSize();
		if (GetSpendTime(begin_time) > time_limit) break;
	} while (po_info.count < po_info.halt && !interruption && enough_size);

	return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
static float
UctSearch(Position *pos, mt19937_64 *mt, unsigned int current, std::vector<int>& path)
{
	// 詰みのチェック
	if (uct_node[current].child_num == 0) {
		return 1.0f; // 反転して値を返すため1を返す
	}
	else if (uct_node[current].value_win == FLT_MAX) {
		// 詰み
		return 0.0f;  // 反転して値を返すため0を返す
	}
	else if (uct_node[current].value_win == FLT_MIN) {
		// 自玉の詰み
		return 1.0f; // 反転して値を返すため1を返す
	}


#ifndef USE_VALUENET
	// policyが計算されるのを待つ
	//cout << "wait policy:" << current_root << ":" << uct_node[current_root].evaled << endl;
	while (!uct_node[current].evaled)
		this_thread::sleep_for(chrono::milliseconds(0));
#endif // !USE_VALUENET

	float result;
	unsigned int next_index;
	double score;
	child_node_t *uct_child = uct_node[current].child;

	// 現在見ているノードをロック
	LOCK_NODE(current);
	// UCB値最大の手を求める
	next_index = SelectMaxUcbChild(pos, current, mt);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	path.push_back(current);
	// Virtual Lossを加算
	AddVirtualLoss(&uct_child[next_index], current);
	// ノードの展開の確認
	if (uct_child[next_index].index == NOT_EXPANDED ) {
#ifndef USE_VALUENET
		// キューがいっぱいの場合待機する
		while (current_policy_value_batch_index >= policy_value_batch_maxsize - THREAD_MAX)
			this_thread::sleep_for(chrono::milliseconds(0));
#endif // !USE_VALUENET

		// ノードの展開中はロック
		LOCK_EXPAND;
		// ノードの展開
		// ノード展開処理の中でvalueを計算する
		unsigned int child_index = ExpandNode(pos, current, path);
		uct_child[next_index].index = child_index;
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;
		// ノード展開のロックの解除
		UNLOCK_EXPAND;

#ifndef USE_VALUENET
		// 従来の評価関数を使う
		// evaluate() の差分計算を無効化する。
		LimitsType limits;
		limits.depth = static_cast<Depth>(1);
		pos->searcher()->alpha = -ScoreMaxEvaluate;
		pos->searcher()->beta = ScoreMaxEvaluate;
		pos->searcher()->threads.startThinking(*pos, limits, pos->searcher()->states);
		pos->searcher()->threads.main()->waitForSearchFinished();
		Score score = pos->searcher()->threads.main()->rootMoves[0].score;
		uct_node[child_index].value_win = score_to_value(score);
		//std::cout << score << std::endl;
#endif // !USE_VALUENET

		// 現在見ているノードのロックを解除
		UNLOCK_NODE(current);

		// 詰みチェック(ValueNet計算中にチェック)
		int isMate = 0;
		if (!pos->inCheck()) {
			if (mateMoveInOddPly(*pos, MATE_SEARCH_DEPTH)) {
				isMate = 1;
			}
		}
		else {
			if (mateMoveInEvenPly(*pos, MATE_SEARCH_DEPTH - 1)) {
				isMate = -1;
			}
		}

#ifdef USE_VALUENET
		// valueが計算されるのを待つ
		//cout << "wait value:" << child_index << ":" << uct_node[child_index].evaled << endl;
		while (!uct_node[child_index].evaled)
			this_thread::sleep_for(chrono::milliseconds(0));
#endif // !NOUSE_VALUENET

		// 詰みの場合、ValueNetの値を上書き
		if (isMate == 1) {
			uct_node[child_index].value_win = FLT_MAX;
			result = 0.0f;
		}
		else if (isMate == -1) {
			uct_node[child_index].value_win = FLT_MIN;
			result = 1.0f;
		}
		else {
			// valueを勝敗として返す
			result = 1 - uct_node[child_index].value_win;
		}
	}
	else {
		// 現在見ているノードのロックを解除
		UNLOCK_NODE(current);

		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, mt, uct_child[next_index].index, path);
	}

	// 探索結果の反映
	UpdateResult(&uct_child[next_index], result, current);

	return 1 - result;
}


//////////////////////////
//  Virtual Lossの加算  //
//////////////////////////
static void
AddVirtualLoss(child_node_t *child, unsigned int current)
{
	atomic_fetch_add(&uct_node[current].move_count, VIRTUAL_LOSS);
	atomic_fetch_add(&child->move_count, VIRTUAL_LOSS);
}


//////////////////////
//  探索結果の更新  //
/////////////////////
static void
UpdateResult(child_node_t *child, float result, unsigned int current)
{
	atomic_fetch_add(&uct_node[current].win, result);
	atomic_fetch_add(&uct_node[current].move_count, 1 - VIRTUAL_LOSS);
	atomic_fetch_add(&child->win, result);
	atomic_fetch_add(&child->move_count, 1 - VIRTUAL_LOSS);
}


/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
static int
SelectMaxUcbChild(const Position *pos, unsigned int current, mt19937_64 *mt)
{
	child_node_t *uct_child = uct_node[current].child;
	const int child_num = uct_node[current].child_num;
	int max_child = 0;
	const int sum = uct_node[current].move_count;
	float q, u, max_value;
	float ucb_value;
	unsigned int max_index;
	//const bool debug = GetDebugMessageMode() && current == current_root && sum % 100 == 0;

	max_value = -1;

	const float p_p = uct_node[current].win / uct_node[current].move_count;
	const float p_v = uct_node[current].value_win;

	// UCB値最大の手を求める  
	for (int i = 0; i < child_num; i++) {
		float win = uct_child[i].win;
		int move_count = uct_child[i].move_count;

		// evaled
		/*if (debug) {
			cerr << i << ":";
			cerr << uct_node[current].move_count << " ";
			cerr << setw(3) << uct_child[i].move.toUSI();
			cerr << ": move " << setw(5) << move_count << " policy "
				<< setw(10) << uct_child[i].nnrate << " ";
		}*/
		if (move_count == 0) {
			q = p_v;
			u = 1.0f;
		}
		else {
			q = win / move_count;
			u = sqrtf(sum) / (1 + move_count);
		}

		float rate = max(uct_child[i].nnrate, 0.01f);
		// ランダムに確率を上げる
		if (pos->turn() == my_color && rnd(*mt) == 0) {
			rate *= 1.5f;
		}

		ucb_value = q + c_puct * u * rate;

		/*if (debug) {
			cerr << " Q:" << q << " U:" << c_puct * u * rate << " UCB:" << ucb_value << endl;
		}*/

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	/*if (debug) {
		cerr << "select node:" << current << " child:" << max_child << endl;
	}*/

	return max_child;
}


/////////////////////////////////
//  次のプレイアウト回数の設定  //
/////////////////////////////////
static void
CalculatePlayoutPerSec(double finish_time)
{
	if (finish_time != 0.0) {
		po_per_sec = po_info.count / finish_time;
	}
	else {
		po_per_sec = PLAYOUT_SPEED * threads;
	}
}

static void
CalculateNextPlayouts(const Position *pos)
{
	int color = pos->turn();

	// 探索の時の探索回数を求める
	if (mode == CONST_TIME_MODE) {
		po_info.num = (int)(po_per_sec * const_thinking_time);
	}
	else if (mode == TIME_SETTING_MODE ||
		mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
		time_limit = remaining_time[color] / max(10, TIME_RATE_9 - pos->gamePly()) + inc_time[color];
		if (mode == TIME_SETTING_WITH_BYOYOMI_MODE &&
			time_limit < (const_thinking_time)) {
			time_limit = const_thinking_time;
		}
		po_info.num = (int)(po_per_sec * time_limit);
	}
}

string model_path;
void SetModelPath(const char* path)
{
	model_path = path;
}

void
ReadWeights()
{
	// Boost.PythonとBoost.Numpyの初期化
	Py_Initialize();
	np::initialize();

	// Pythonモジュール読み込み
	py::object dlshogi_ns = py::import("dlshogi.predict").attr("__dict__");

	// modelロード
	py::object dlshogi_load_model = dlshogi_ns["load_model"];
	dlshogi_load_model(model_path.c_str());

	// 予測関数取得
	dlshogi_predict = dlshogi_ns["predict"];
}

void EvalNode() {
	while (true) {
		LOCK_EXPAND;
		bool running = handle[threads - 1] != nullptr;
		if (!running
			&& (!reuse_subtree || current_policy_value_batch_index == 0)) {
			UNLOCK_EXPAND;
			break;
		}

		if (current_policy_value_batch_index == 0) {
			UNLOCK_EXPAND;
			this_thread::sleep_for(chrono::milliseconds(1));
			//cerr << "EMPTY QUEUE" << endl;
			continue;
		}

		if (running && current_policy_value_batch_index == 0) {
			UNLOCK_EXPAND;
		}
		else {
			int policy_value_batch_size = current_policy_value_batch_index;
			int policy_value_queue_index = current_policy_value_queue_index;
			current_policy_value_batch_index = 0;
			current_policy_value_queue_index = current_policy_value_queue_index ^ 1;
			UNLOCK_EXPAND;

			// predict
			np::ndarray ndfeatures1 = np::from_data(
				features1[policy_value_queue_index],
				np::dtype::get_builtin<float>(),
				py::make_tuple(policy_value_batch_size, (int)ColorNum * MAX_FEATURES1_NUM, 9, 9),
				py::make_tuple(sizeof(float)*(int)ColorNum*MAX_FEATURES1_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
				py::object());

			np::ndarray ndfeatures2 = np::from_data(
				features2[policy_value_queue_index],
				np::dtype::get_builtin<float>(),
				py::make_tuple(policy_value_batch_size, MAX_FEATURES2_NUM, 9, 9),
				py::make_tuple(sizeof(float)*MAX_FEATURES2_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
				py::object());

			auto ret = dlshogi_predict(ndfeatures1, ndfeatures2);
			py::tuple ret_list = py::extract<py::tuple>(ret);
			np::ndarray y1_data = py::extract<np::ndarray>(ret_list[0]);
			np::ndarray y2_data = py::extract<np::ndarray>(ret_list[1]);

			float(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1_data.get_data());
			float *value = reinterpret_cast<float*>(y2_data.get_data());

			for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
				const unsigned int index = policy_value_hash_index[policy_value_queue_index][i];

				/*if (index == current_root) {
					string str;
					for (int sq = 0; sq < SquareNum; sq++) {
						str += to_string((int)features1[policy_value_queue_index][i][0][0][sq]);
						str += " ";
					}
					cout << str << endl;
				}*/

				LOCK_NODE(index);

				const int child_num = uct_node[index].child_num;
				child_node_t *uct_child = uct_node[index].child;
				Color color = (Color)node_hash[index].color;

				// 合法手一覧
				std::vector<float> legal_move_probabilities;
				for (int j = 0; j < child_num; j++) {
					Move move = uct_child[j].move;
					const int move_label = make_move_label((u16)move.proFromAndTo(), move.pieceTypeFrom(), color);
					legal_move_probabilities.emplace_back((*logits)[move_label]);
				}

				// Boltzmann distribution
				softmax_tempature_with_normalize(legal_move_probabilities);

				for (int j = 0; j < child_num; j++) {
					uct_child[j].nnrate = legal_move_probabilities[j];
				}

#ifdef USE_VALUENET
				uct_node[index].value_win = *value;
#endif // USE_VALUENET
				uct_node[index].evaled = true;
				UNLOCK_NODE(index);
			}
		}
	}
}
