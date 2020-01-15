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

#include "fastmath.h"
#include "Message.h"
#include "UctSearch.h"
#include "Utility.h"
#include "mate.h"
#include "nn_wideresnet10.h"
#include "nn_fused_wideresnet10.h"
#include "nn_wideresnet15.h"
#include "nn_senet10.h"

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

using namespace std;

#define LOCK_NODE(var) mutex_nodes[(var)].lock()
#define UNLOCK_NODE(var) mutex_nodes[(var)].unlock()
#define LOCK_EXPAND mutex_expand.lock();
#define UNLOCK_EXPAND mutex_expand.unlock();


////////////////
//  大域変数  //
////////////////

// 持ち時間
double remaining_time[ColorNum];
double inc_time[ColorNum];
double po_per_sec = PLAYOUT_SPEED;

// UCTハッシュ
UctHash uct_hash;

// UCTのノード
uct_node_t *uct_node;

// プレイアウト情報
static po_info_t po_info;

// 試行時間を延長するかどうかのフラグ
static bool extend_time = false;
// 探索対象の局面
const Position *pos_root;
// 現在のルートのインデックス
unsigned int current_root;

unsigned int uct_hash_size;
mutex* mutex_nodes;
mutex mutex_expand;       // ノード展開を排他処理するためのmutex

// 探索の設定
enum SEARCH_MODE mode = SEARCH_MODE::TIME_SETTING_WITH_BYOYOMI_MODE;
// 1手あたりの試行時間
double const_thinking_time = CONST_TIME;
// 1手当たりのプレイアウト数
int playout = CONST_PLAYOUT;
// デフォルトの持ち時間
double default_remaining_time = ALL_THINKING_TIME;

bool pondering_mode = false;

bool pondering = false;

atomic<bool> uct_search_stop(false);

double time_limit;

// ハッシュの再利用
bool reuse_subtree = true;

ray_clock::time_point begin_time;

// 投了する勝率の閾値
float RESIGN_THRESHOLD = 0.01f;

// PUCTの定数
float c_init;
float c_base;
float c_fpu;

// モデルのパス
string model_path[max_gpu];

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
constexpr int MATE_SEARCH_DEPTH = 5;

// 詰み探索で詰みの場合のvalue_winの定数
constexpr float VALUE_WIN = FLT_MAX;
constexpr float VALUE_LOSE = -FLT_MAX;

// 探索の結果を評価のキューに追加したか、破棄したか
constexpr float QUEUING = FLT_MAX;
constexpr float DISCARDED = -FLT_MAX;

//template<float>
double atomic_fetch_add(std::atomic<float> *obj, float arg) {
	float expected = obj->load();
	while (!atomic_compare_exchange_weak(obj, &expected, expected + arg))
		;
	return expected;
}

////////////
//  関数  //
////////////

// Virtual Lossを加算
static void AddVirtualLoss(child_node_t *child, unsigned int current);
// Virtual Lossを減算
static void SubVirtualLoss(child_node_t *child, unsigned int current);

// 次のプレイアウト回数の設定
static void CalculatePlayoutPerSec(double finish_time);
static void CalculateNextPlayouts(const Position *pos);

// ルートの展開
static unsigned int ExpandRoot(const Position *pos);

// 思考時間を延長する処理
static bool ExtendTime(void);

// 候補手の初期化
static void InitializeCandidate(child_node_t *uct_child, Move move);

// 探索打ち切りの確認
static bool InterruptionCheck(void);

// 結果の更新
static void UpdateResult(child_node_t *child, float result, unsigned int current);

// 入玉宣言勝ち
bool nyugyoku(const Position& pos);

class UCTSearcher;
class UCTSearcherGroup {
public:
	UCTSearcherGroup() : threads(0), nn(nullptr) {}
	~UCTSearcherGroup() {
		delete nn;
	}

	void Initialize(const int new_thread, const int gpu_id, const int policy_value_batch_maxsize);
	void InitGPU() {
		mutex_gpu.lock();
		if (nn == nullptr) {
			if (model_path[gpu_id].find("wideresnet15") != string::npos)
				nn = (NN*)new NNWideResnet15(policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("fused_wideresnet10") != string::npos)
				nn = (NN*)new NNFusedWideResnet10(policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("senet10") != string::npos)
				nn = (NN*)new NNSENet10(policy_value_batch_maxsize);
			else
				nn = (NN*)new NNWideResnet10(policy_value_batch_maxsize);
			nn->load_model(model_path[gpu_id].c_str());
		
		}
		mutex_gpu.unlock();
	}
	void nn_foward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2) {
		mutex_gpu.lock();
		nn->foward(batch_size, x1, x2, y1, y2);
		mutex_gpu.unlock();
	}
	void Run();
	void Join();

	// GPUID
	int gpu_id;
private:
	// 使用するスレッド数
	int threads;

	// UCTSearcher
	vector<UCTSearcher> searchers;

	// neural network
	NN* nn;
	int policy_value_batch_maxsize;

	// mutex for gpu
	mutex mutex_gpu;
};
UCTSearcherGroup* search_groups;

class UCTSearcher {
public:
	UCTSearcher(UCTSearcherGroup* grp, const int thread_id, const int policy_value_batch_maxsize) :
		grp(grp),
		thread_id(thread_id),
		mt(new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + thread_id)),
		policy_value_batch_maxsize(policy_value_batch_maxsize) {
		// キューを動的に確保する
		checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		policy_value_hash_index = new unsigned int[policy_value_batch_maxsize];

		checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
	}
	UCTSearcher(UCTSearcher&& o) :
		grp(grp),
		thread_id(thread_id),
		mt(move(o.mt)) {}
	~UCTSearcher() {
		checkCudaErrors(cudaFreeHost(features1));
		checkCudaErrors(cudaFreeHost(features2));
		delete[] policy_value_hash_index;
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
	}

	void Run() {
		handle = new thread([this]() { this->ParallelUctSearch(); });
	}
	// スレッド終了待機
	void Join() {
		handle->join();
		delete handle;
	}

private:
	// UCT探索
	void ParallelUctSearch();
	//  UCT探索(1回の呼び出しにつき, 1回の探索)
	float UctSearch(Position* pos, const unsigned int current, const int depth, vector<pair<unsigned int, unsigned int>>& trajectories);
	// ノードの展開
	unsigned int ExpandNode(Position* pos, const int depth);
	// UCB値が最大の子ノードを返す
	int SelectMaxUcbChild(const Position* pos, const unsigned int current, const int depth);
	// ノードをキューに追加
	void QueuingNode(const Position* pos, unsigned int index);
	// ノードを評価
	void EvalNode();
	// スレッド開始

	UCTSearcherGroup* grp;
	// スレッド識別番号
	int thread_id;
	// 乱数生成器
	unique_ptr<std::mt19937_64> mt;
	// スレッドのハンドル
	thread *handle;

	int policy_value_batch_maxsize;
	features1_t* features1;
	features2_t* features2;
	DType* y1;
	DType* y2;
	unsigned int* policy_value_hash_index;
	int current_policy_value_batch_index;
};

/////////////////////
//  予測読みの設定  //
/////////////////////
void
SetPonderingMode(bool flag)
{
	pondering_mode = flag;
}

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
void SetThread(const int new_thread[max_gpu], const int new_policy_value_batch_maxsize[max_gpu])
{
	for (int i = 0; i < max_gpu; i++) {
		if (new_thread[i] > 0) {
			int policy_value_batch_maxsize = new_policy_value_batch_maxsize[i];
			if (policy_value_batch_maxsize == 0)
				policy_value_batch_maxsize = new_policy_value_batch_maxsize[0];
			search_groups[i].Initialize(new_thread[i], i, policy_value_batch_maxsize);
		}
	}
}

void NewGame()
{
	uct_hash.ClearUctHash();
}

void GameOver()
{
}

// 投了の閾値設定（1000分率）
void SetResignThreshold(const int resign_threshold)
{
	RESIGN_THRESHOLD = (float)resign_threshold / 1000.0f;
}

void
UCTSearcherGroup::Initialize(const int new_thread, const int gpu_id, const int policy_value_batch_maxsize)
{
	this->gpu_id = gpu_id;
	if (threads != new_thread) {
		threads = new_thread;

		// UCTSearcher
		searchers.clear();
		searchers.reserve(threads);
		for (int i = 0; i < threads; i++) {
			searchers.emplace_back(this, i, policy_value_batch_maxsize);
		}
	}
	this->policy_value_batch_maxsize = policy_value_batch_maxsize;
}

// スレッド開始
void
UCTSearcherGroup::Run()
{
	if (threads > 0) {
		// 探索用スレッド
		for (int i = 0; i < threads; i++) {
			searchers[i].Run();
		}
	}
}

// スレッド終了待機
void
UCTSearcherGroup::Join()
{
	if (threads > 0) {
		// 探索用スレッド
		for (int i = 0; i < threads; i++) {
			searchers[i].Join();
		}
	}
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
	uct_search_stop = false;
}

//////////////////////////
//  ノード再利用の設定  //
//////////////////////////
void
SetReuseSubtree(bool flag)
{
	reuse_subtree = flag;
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
InitializeUctSearch(const unsigned int hash_size)
{
	uct_hash_size = hash_size;

	// ミューテックスを初期化
	mutex_nodes = new mutex[uct_hash_size];

	// UCTのノードのメモリを確保
	uct_hash.Init(uct_hash_size);
	uct_node = new uct_node_t[uct_hash_size];

	if (uct_node == nullptr) {
		cerr << "Cannot allocate memory !!" << endl;
		cerr << "You must reduce tree size !!" << endl;
		exit(1);
	}

	search_groups = new UCTSearcherGroup[max_gpu];
}

//  UCT探索の終了処理
void TerminateUctSearch()
{
	delete[] search_groups;
	delete[] mutex_nodes;
}

////////////////////////
//  探索設定の初期化  //
////////////////////////
void
InitializeSearchSetting(void)
{
	// 持ち時間の初期化
	for (int i = 0; i < ColorNum; i++) {
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

void
StopUctSearch(void)
{
	uct_search_stop = true;
}

/////////////////////////////////////
//  UCTアルゴリズムによる着手生成  //
/////////////////////////////////////
Move
UctSearchGenmove(Position *pos, Move &ponderMove, bool ponder)
{
	Move move;
	double finish_time;

	// ルート局面をグローバル変数に保存
	pos_root = pos;

	pondering = ponder;

	// 探索情報をクリア
	po_info.count = 0;

	if (reuse_subtree) {
		uct_hash.DeleteOldHash(pos);
	}
	else {
		uct_hash.ClearUctHash();
	}

	// 探索開始時刻の記録
	begin_time = ray_clock::now();

	// UCTの初期化
	current_root = ExpandRoot(pos);

	// 詰みのチェック
	if (uct_node[current_root].child_num == 0) {
		return Move::moveNone();
	}
	if (uct_node[current_root].value_win == VALUE_WIN) {
		// 詰み
		Move move;
		if (pos->inCheck())
			move = mateMoveInOddPlyReturnMove<true>(*pos, MATE_SEARCH_DEPTH);
		else
			move = mateMoveInOddPlyReturnMove<false>(*pos, MATE_SEARCH_DEPTH);
		// 伝播したVALUE_WINの場合、詰みが見つからない場合がある
		if (move != Move::moveNone())
			return move;
	}
	else if (uct_node[current_root].value_win == VALUE_LOSE) {
		// 自玉の詰み
		return Move::moveNone();
	}

	// 前回から持ち込んだ探索回数を記録
	int pre_simulated = uct_node[current_root].move_count;

	// 探索回数の閾値を設定
	CalculateNextPlayouts(pos);
	const int rest_uct_hash = uct_hash.GetRestUctHash();
	if (po_info.num > rest_uct_hash) {
		po_info.halt = rest_uct_hash;
	}
	else {
		po_info.halt = po_info.num;
	}

	// 探索時間とプレイアウト回数の予定値を出力
	PrintPlayoutLimits(time_limit, po_info.halt);

	// 探索スレッド開始
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Run();

	// 探索スレッド終了待機
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Join();

	// 着手が21手以降で,
	// 時間延長を行う設定になっていて,
	// 探索時間延長をすべきときは
	// 探索回数を1.5倍に増やす
	if (pos->gamePly() > 20 &&
		extend_time &&
		time_limit > const_thinking_time * 1.5 &&
		ExtendTime()) {
		if (debug_message) cout << "ExtendTime" << endl;
		po_info.halt = (int)(1.5 * po_info.halt);
		time_limit *= 1.5;
		// 探索スレッド開始
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Run();

		// 探索スレッド終了待機
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Join();
	}

	// 探索にかかった時間を求める
	finish_time = GetSpendTime(begin_time);

	const child_node_t* uct_child = uct_node[current_root].child;

	int max_count = 0;
	unsigned int select_index = 0;
	int child_win_count = 0;
	int child_lose_count = 0;

	// 探索回数最大の手を見つける
	const int child_num = uct_node[current_root].child_num;
	for (int i = 0; i < child_num; i++) {
		if (debug_message) {
			cout << i << ":" << uct_child[i].move.toUSI() << " move_count:" << uct_child[i].move_count << " nnrate:" << uct_child[i].nnrate << " value_win:";
			if (uct_child[i].index != NOT_EXPANDED) cout << uct_node[uct_child[i].index].value_win;
			cout << " win_rate:" << uct_child[i].win / (uct_child[i].move_count + 0.0001f) << endl;
		}

		if (uct_child[i].index != NOT_EXPANDED) {
			uct_node_t& child_node = uct_node[uct_child[i].index];
			// 詰みの場合evaledは更新しないためevaledはチェックしない
			const float child_value_win = child_node.value_win;
			if (child_value_win == VALUE_WIN) {
				// 負けが確定しているノードは選択しない
				if (child_win_count == i || uct_child[i].move_count > max_count) {
					// すべて負けの場合は、探索回数が最大の手を選択する
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				child_win_count++;
				continue;
			}
			else if (child_value_win == VALUE_LOSE) {
				// 子ノードに一つでも負けがあれば、勝ちなので選択する
				if (child_lose_count == 0 || uct_child[i].move_count > max_count) {
					// すべて勝ちの場合は、探索回数が最大の手を選択する
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				child_lose_count++;
				continue;
			}
		}
		if (child_lose_count == 0 && uct_child[i].move_count > max_count) {
			select_index = i;
			max_count = uct_child[i].move_count;
		}
	}

	// 選択した着手の勝率の算出
	float best_wp = uct_child[select_index].win / uct_child[select_index].move_count;

	// 勝ちの場合
	if (child_lose_count > 0) {
		best_wp = 1.0f;
	}
	// すべて負けの場合
	else if (child_win_count == child_num) {
		best_wp = 0.0f;
	}

	if (best_wp < RESIGN_THRESHOLD) {
		move = Move::moveNone();
	}
	else {
		move = uct_child[select_index].move;

		int cp;
		if (best_wp == 1.0f) {
			cp = 30000;
		}
		else if (best_wp == 0.0f) {
			cp = -30000;
		}
		else {
			cp = int(-logf(1.0f / best_wp - 1.0f) * 756.0864962951762f);
		}

		// PV表示
		string pv = move.toUSI();
		int depth = 1;
		{
			unsigned int best_index = select_index;
			const child_node_t *best_node = uct_child;

			while (best_node[best_index].index != NOT_EXPANDED) {
				const int best_node_index = best_node[best_index].index;

				best_node = uct_node[best_node_index].child;
				max_count = 0;
				best_index = 0;
				for (int i = 0; i < uct_node[best_node_index].child_num; i++) {
					if (best_node[i].move_count > max_count) {
						best_index = i;
						max_count = best_node[i].move_count;
					}
				}

				// ponderの着手
				if (pondering_mode && ponderMove == Move::moveNone())
					ponderMove = best_node[best_index].move;

				if (max_count < 1)
					break;

				pv += " " + best_node[best_index].move.toUSI();
				depth++;
			}
		}

		if (!pondering)
			cout << "info nps " << int((uct_node[current_root].move_count - pre_simulated) / finish_time) << " time " << int(finish_time * 1000) << " nodes " << uct_node[current_root].move_count << " hashfull " << uct_hash.GetUctHashUsageRate() << " score cp " << cp << " depth " << depth << " pv " << pv << endl;

		// 次の探索でのプレイアウト回数の算出
		CalculatePlayoutPerSec(finish_time);

		if (!pondering)
			remaining_time[pos->turn()] -= finish_time;
	}

	// 最善応手列を出力
	//PrintBestSequence(pos, uct_node, current_root);
	// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
	if (debug_message) PrintPlayoutInformation(&uct_node[current_root], &po_info, finish_time, pre_simulated);

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
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->gamePly());
	child_node_t *uct_child;
	int child_num = 0;

	// 既に展開されていた時は, 探索結果を再利用する
	if (index != NOT_FOUND) {
		PrintReuseCount(uct_node[index].move_count);

		return index;
	}
	else {
		// 空のインデックスを探す
		index = uct_hash.SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly());

		assert(index != uct_hash_size);

		// ルートノードの初期化
		uct_node[index].move_count = 0;
		uct_node[index].win = 0;
		uct_node[index].child_num = 0;
		uct_node[index].evaled = false;
		uct_node[index].draw = false;
		uct_node[index].value_win = 0.0f;
		uct_node[index].visited_nnrate = 0.0f;

		uct_child = uct_node[index].child;

		// 候補手の展開
		for (MoveList<Legal> ml(*pos); !ml.end(); ++ml) {
			InitializeCandidate(&uct_child[child_num], ml.move());
			child_num++;
		}

		// 子ノード個数の設定
		uct_node[index].child_num = child_num;

	}

	return index;
}



///////////////////
//  ノードの展開  //
///////////////////
unsigned int
UCTSearcher::ExpandNode(Position *pos, const int depth)
{
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->gamePly() + depth);
	child_node_t *uct_child;

	// 合流先が検知できれば, それを返す
	if (index != NOT_FOUND) {
		return index;
	}

	// 空のインデックスを探す
	index = uct_hash.SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth);

	assert(index != NOT_FOUND);

	// 現在のノードの初期化
	uct_node[index].move_count = 0;
	uct_node[index].win = 0;
	uct_node[index].child_num = 0;
	uct_node[index].evaled = false;
	uct_node[index].draw = false;
	uct_node[index].value_win = 0.0f;
	uct_node[index].visited_nnrate = 0.0f;
	uct_child = uct_node[index].child;

	// 候補手の展開
	int child_num = 0;
	for (MoveList<Legal> ml(*pos); !ml.end(); ++ml) {
		InitializeCandidate(&uct_child[child_num], ml.move());
		child_num++;
	}

	// 子ノードの個数を設定
	uct_node[index].child_num = child_num;

	return index;
}


//////////////////////////////////////
//  ノードをキューに追加            //
//////////////////////////////////////
void
UCTSearcher::QueuingNode(const Position *pos, unsigned int index)
{
	//cout << "QueuingNode:" << index << ":" << current_policy_value_queue_index << ":" << current_policy_value_batch_index << endl;
	//cout << pos->toSFEN() << endl;

	/* if (current_policy_value_batch_index >= policy_value_batch_maxsize) {
		std::cout << "error" << std::endl;
	}*/
	// set all zero
	std::fill_n((DType*)features1[current_policy_value_batch_index], sizeof(features1_t) / sizeof(DType), _zero);
	std::fill_n((DType*)features2[current_policy_value_batch_index], sizeof(features2_t) / sizeof(DType), _zero);

	make_input_features(*pos, &features1[current_policy_value_batch_index], &features2[current_policy_value_batch_index]);
	policy_value_hash_index[current_policy_value_batch_index] = index;
	current_policy_value_batch_index++;
}


//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
static bool
InterruptionCheck(void)
{
	if (pondering)
		return uct_search_stop;

	if (uct_search_stop)
		return true;

	int max = 0, second = 0;
	const int child_num = uct_node[current_root].child_num;
	const int rest = po_info.halt - po_info.count;
	const child_node_t *uct_child = uct_node[current_root].child;

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
	const child_node_t *uct_child = uct_node[current_root].child;

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
void
UCTSearcher::ParallelUctSearch()
{
	// スレッドにGPUIDを関連付けてから初期化する
	cudaSetDevice(grp->gpu_id);
	grp->InitGPU();

	// ルートノードを評価
	LOCK_EXPAND;
	if (!uct_node[current_root].evaled) {
		current_policy_value_batch_index = 0;
		QueuingNode(pos_root, current_root);
		EvalNode();
	}
	UNLOCK_EXPAND;

	bool interruption = false;
	bool enough_size = true;

	// 探索経路のバッチ
	vector<vector<pair<unsigned int, unsigned int>>> trajectories_batch;

	// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
	do {
		trajectories_batch.clear();
		current_policy_value_batch_index = 0;

		// バッチサイズ分探索を繰り返す
		for (int i = 0; i < policy_value_batch_maxsize; i++) {
			// 盤面のコピー
			Position pos(*pos_root);
			
			// 1回プレイアウトする
			trajectories_batch.emplace_back();
			float result = UctSearch(&pos, current_root, 0, trajectories_batch.back());

			if (result != DISCARDED) {
				// 探索回数を1回増やす
				atomic_fetch_add(&po_info.count, 1);
			}

			// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
			if (result == DISCARDED || result != QUEUING) {
				trajectories_batch.pop_back();
			}
		}

		// 評価
		EvalNode();

		// バックアップ
		float result = 0.0f;
		for (auto& trajectories : trajectories_batch) {
			for (int i = trajectories.size() - 1; i >= 0; i--) {
				pair<unsigned int, unsigned int>& current_next = trajectories[i];
				const unsigned int current = current_next.first;
				const unsigned int next_index = current_next.second;
				child_node_t* uct_child = uct_node[current].child;
				if ((size_t)i == trajectories.size() - 1) {
					const unsigned int child_index = uct_child[next_index].index;
					result = 1.0f - uct_node[child_index].value_win;
				}
				UpdateResult(&uct_child[next_index], result, current);
				result = 1.0f - result;
			}
		}

		// 探索を打ち切るか確認
		interruption = InterruptionCheck();
		// ハッシュに余裕があるか確認
		enough_size = uct_hash.CheckRemainingHashSize();
		if (!pondering && GetSpendTime(begin_time) > time_limit) break;
	} while (po_info.count < po_info.halt && !interruption && enough_size);

	return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position *pos, const unsigned int current, const int depth, vector<pair<unsigned int, unsigned int>>& trajectories)
{
	if (current != current_root) {
		// 詰みのチェック
		if (uct_node[current].child_num == 0) {
			return 1.0f; // 反転して値を返すため1を返す
		}
		else if (uct_node[current].value_win == VALUE_WIN) {
			// 詰み
			return 0.0f;  // 反転して値を返すため0を返す
		}
		else if (uct_node[current].value_win == VALUE_LOSE) {
			// 自玉の詰み
			return 1.0f; // 反転して値を返すため1を返す
		}

		// 千日手チェック
		if (uct_node[current].draw) {
			switch (pos->isDraw(16)) {
			case NotRepetition: break;
			case RepetitionDraw: return 0.5f;
			case RepetitionWin: return 0.0f;
			case RepetitionLose: return 1.0f;
			case RepetitionSuperior: return 0.0f;
			case RepetitionInferior: return 1.0f;
			default: UNREACHABLE;
			}
		}
	}

	// policy計算中のため破棄する(他のスレッドが同じノードを先に展開した場合)
	if (!uct_node[current].evaled)
		return DISCARDED;

	float result;
	unsigned int next_index;
	double score;
	child_node_t *uct_child = uct_node[current].child;

	// 現在見ているノードをロック
	LOCK_NODE(current);
	// UCB値最大の手を求める
	next_index = SelectMaxUcbChild(pos, current, depth);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	// 経路を記録
	trajectories.emplace_back(current, next_index);

	// Virtual Lossを加算
	AddVirtualLoss(&uct_child[next_index], current);
	// ノードの展開の確認
	if (uct_child[next_index].index == NOT_EXPANDED ) {
		// ノードの展開中はロック
		LOCK_EXPAND;
		// ノードの展開
		unsigned int child_index = ExpandNode(pos, depth + 1);
		uct_child[next_index].index = child_index;
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;
		// ノード展開のロックの解除
		UNLOCK_EXPAND;

		// 現在見ているノードのロックを解除
		UNLOCK_NODE(current);

		// 合流検知
		if (uct_node[child_index].evaled) {
			// 手番を入れ替えて1手深く読む
			result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories);
		}
		else if (uct_node[child_index].child_num == 0) {
			// 詰み
			uct_node[child_index].value_win = VALUE_LOSE;
			result = 1.0f;
		}
		else {
			// 千日手チェック
			int isDraw = 0;
			switch (pos->isDraw(16)) {
			case NotRepetition: break;
			case RepetitionDraw: isDraw = 2; break; // Draw
			case RepetitionWin: isDraw = 1; break;
			case RepetitionLose: isDraw = -1; break;
			case RepetitionSuperior: isDraw = 1; break;
			case RepetitionInferior: isDraw = -1; break;
			default: UNREACHABLE;
			}

			// 千日手の場合、ValueNetの値を使用しない（経路によって判定が異なるため上書きはしない）
			if (isDraw != 0) {
				uct_node[child_index].draw = true;
				if (isDraw == 1) {
					result = 0.0f;
				}
				else if (isDraw == -1) {
					result = 1.0f;
				}
				else {
					result = 0.5f;
				}
				// 経路が異なる場合にNNの計算が必要なためキューに追加する
				QueuingNode(pos, child_index);
			}
			else {
				// 詰みチェック
				int isMate = 0;
				if (!pos->inCheck()) {
					if (mateMoveInOddPly<false>(*pos, MATE_SEARCH_DEPTH)) {
						isMate = 1;
					}
				}
				else {
					if (mateMoveInOddPly<true>(*pos, MATE_SEARCH_DEPTH)) {
						isMate = 1;
					}
					// 偶数手詰めは親のノードの奇数手詰めでチェックされているためチェックしない
					/*else if (mateMoveInEvenPly(*pos, MATE_SEARCH_DEPTH - 1)) {
						isMate = -1;
					}*/
				}

				// 入玉勝ちかどうかを判定
				if (nyugyoku(*pos)) {
					isMate = 1;
				}

				// 詰みの場合、ValueNetの値を上書き
				if (isMate == 1) {
					uct_node[child_index].value_win = VALUE_WIN;
					result = 0.0f;
				}
				/*else if (isMate == -1) {
					uct_node[child_index].value_win = VALUE_LOSE;
					// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
					uct_node[current].value_win = VALUE_WIN;
					result = 1.0f;
				}*/
				else {
					// ノードをキューに追加
					QueuingNode(pos, child_index);
					return QUEUING;
				}
			}
		}
	}
	else {
		// 現在見ているノードのロックを解除
		UNLOCK_NODE(current);

		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories);
	}

	if (result == QUEUING)
		return result;
	else if (result == DISCARDED) {
		// Virtual Lossを戻す
		SubVirtualLoss(&uct_child[next_index], current);
		return result;
	}

	// 探索結果の反映
	UpdateResult(&uct_child[next_index], result, current);

	return 1.0f - result;
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

// Virtual Lossを減算
static void
SubVirtualLoss(child_node_t *child, unsigned int current)
{
	atomic_fetch_add(&uct_node[current].move_count, -VIRTUAL_LOSS);
	atomic_fetch_add(&child->move_count, -VIRTUAL_LOSS);
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

// ディリクレ分布
void random_dirichlet(std::mt19937_64 &mt, float *x, const int size) {
	const float dirichlet_alpha = 0.15f;
	static std::gamma_distribution<float> gamma(dirichlet_alpha, 1.0f);

	float sum_y = 0;
	for (int i = 0; i < size; i++) {
		float y = gamma(mt);
		sum_y += y;
		x[i] = y;
	}
	std::for_each(x, x + size, [sum_y](float &v) mutable { v /= sum_y; });
}

/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
int
UCTSearcher::SelectMaxUcbChild(const Position *pos, const unsigned int current, const int depth)
{
	const child_node_t *uct_child = uct_node[current].child;
	const int child_num = uct_node[current].child_num;
	int max_child = 0;
	const int sum = uct_node[current].move_count;
	float q, u, max_value;
	float ucb_value;
	int child_win_count = 0;

	max_value = -1;

	float fpu_reduction = 0.0f;
	if (depth > 0)
		fpu_reduction = c_fpu * sqrtf(uct_node[current].visited_nnrate);

	// UCB値最大の手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].index != NOT_EXPANDED) {
			uct_node_t& child_node = uct_node[uct_child[i].index];
			if (child_node.evaled) {
				const float child_value_win = child_node.value_win;
				if (child_value_win == VALUE_WIN) {
					child_win_count++;
					// 負けが確定しているノードは選択しない
					continue;
				}
				else if (child_value_win == VALUE_LOSE) {
					// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
					uct_node[current].value_win = VALUE_WIN;
				}
			}
		}
		float win = uct_child[i].win;
		int move_count = uct_child[i].move_count;

		if (move_count == 0) {
			// 未探索のノードの価値に、親ノードの価値を使用する
			if (uct_node[current].win > 0)
				q = uct_node[current].win / uct_node[current].move_count - fpu_reduction;
			else
				q = 0.0f;
			u = sum == 0 ? 1.0f : sqrtf(sum);
		}
		else {
			q = win / move_count;
			u = sqrtf(sum) / (1 + move_count);
		}

		const float rate = uct_child[i].nnrate;

		const float c = FastLog((sum + c_base + 1.0f) / c_base) + c_init;
		ucb_value = FastLogit(0.9999999f * q) + c * u * rate;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	if (child_win_count == child_num) {
		// 子ノードがすべて勝ちのため、自ノードを負けにする
		uct_node[current].value_win = VALUE_LOSE;
	}

	// for FPU reduction
	if (uct_child[max_child].index == NOT_EXPANDED) {
		atomic_fetch_add(&uct_node[current].visited_nnrate, uct_child[max_child].nnrate);
	}

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
		po_per_sec = PLAYOUT_SPEED;
	}
}

static void
CalculateNextPlayouts(const Position *pos)
{
	if (pondering) {
		po_info.num = uct_hash_size;
		return;
	}

	int color = pos->turn();

	// 探索の時の探索回数を求める
	if (mode == CONST_TIME_MODE) {
		po_info.num = (int)(po_per_sec * const_thinking_time);
	}
	else if (mode == TIME_SETTING_MODE ||
		mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
		time_limit = remaining_time[color] / (14 + max(0, 30 - pos->gamePly())) + inc_time[color];
		if (mode == TIME_SETTING_WITH_BYOYOMI_MODE &&
			time_limit < (const_thinking_time)) {
			time_limit = const_thinking_time;
		}
		po_info.num = (int)(po_per_sec * time_limit);
	}
}

void SetModelPath(const std::string path[max_gpu])
{
	for (int i = 0; i < max_gpu; i++) {
		if (path[i] == "")
			model_path[i] = path[0];
		else
			model_path[i] = path[i];
	}
}

void UCTSearcher::EvalNode() {
	if (current_policy_value_batch_index == 0)
		return;

	const int policy_value_batch_size = current_policy_value_batch_index;

	// predict
	grp->nn_foward(policy_value_batch_size, features1, features2, y1, y2);

	const DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
	const DType *value = y2;

	for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
		const unsigned int index = policy_value_hash_index[i];

		LOCK_NODE(index);

		const int child_num = uct_node[index].child_num;
		child_node_t *uct_child = uct_node[index].child;
		Color color = uct_hash[index].color;

		// 合法手一覧
		std::vector<float> legal_move_probabilities;
		legal_move_probabilities.reserve(child_num);
		for (int j = 0; j < child_num; j++) {
			Move move = uct_child[j].move;
			const int move_label = make_move_label((u16)move.proFromAndTo(), color);
#ifdef FP16
			const float logit = __half2float((*logits)[move_label]);
#else
			const float logit = (*logits)[move_label];
#endif
			legal_move_probabilities.emplace_back(logit);
		}

		// Boltzmann distribution
		softmax_temperature_with_normalize(legal_move_probabilities);

		for (int j = 0; j < child_num; j++) {
			uct_child[j].nnrate = legal_move_probabilities[j];
		}

#ifdef FP16
		uct_node[index].value_win = __half2float(*value);
#else
		uct_node[index].value_win = *value;
#endif
		uct_node[index].evaled = true;
		UNLOCK_NODE(index);
	}
}
