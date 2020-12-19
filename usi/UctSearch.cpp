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
#include "Node.h"
#include "mate.h"

#ifdef ONNXRUNTIME
#include "nn_onnxruntime.h"
#else
#include "nn_wideresnet10.h"
#include "nn_fused_wideresnet10.h"
#include "nn_wideresnet15.h"
#include "nn_senet10.h"
#include "nn_tensorrt.h"
#endif

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

#define LOCK_EXPAND mutex_expand.lock();
#define UNLOCK_EXPAND mutex_expand.unlock();


////////////////
//  大域変数  //
////////////////

#ifdef MAKE_BOOK
#include "book.hpp"
extern std::map<Key, std::vector<BookEntry> > bookMap;
#endif

// 持ち時間
int remaining_time[ColorNum];
int inc_time[ColorNum];
int const_playout = 0;

// ゲーム木
std::unique_ptr<NodeTree> tree;

// プレイアウト情報
static po_info_t po_info;

// 試行時間を延長するかどうかのフラグ
static bool extend_time = false;
// 探索対象の局面
const Position *pos_root;

unsigned int uct_node_limit; // UCTノードの上限
mutex mutex_expand;       // ノード展開を排他処理するためのmutex

bool pondering_mode = false;

bool pondering = false;

atomic<bool> uct_search_stop(false);

int time_limit;
int minimum_time = 0;

int last_pv_print; // 最後にpvが表示された時刻
int pv_interval = 500; // pvを表示する周期(ms)

// ハッシュの再利用
bool reuse_subtree = true;

// 思考開始時刻
Timer begin_time;
// 探索スレッドの思考開始時刻
Timer search_begin_time;
atomic<bool> init_search_begin_time;
atomic<bool> interruption;

// 投了する勝率の閾値
float RESIGN_THRESHOLD = 0.01f;

// PUCTの定数
float c_init;
float c_base;
float c_fpu_reduction;
float c_init_root;
float c_base_root;
float c_fpu_reduction_root;

// モデルのパス
string model_path[max_gpu];

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
constexpr int MATE_SEARCH_DEPTH = 5;

// 詰み探索で詰みの場合のvalue_winの定数
constexpr float VALUE_WIN = FLT_MAX;
constexpr float VALUE_LOSE = -FLT_MAX;
// 千日手の場合のvalue_winの定数
constexpr float VALUE_DRAW = FLT_MAX / 2;

// 探索の結果を評価のキューに追加したか、破棄したか
constexpr float QUEUING = FLT_MAX;
constexpr float DISCARDED = -FLT_MAX;
constexpr float DISCARDED2 = -FLT_MAX / 2;

// 千日手の価値
float draw_value_black = 0.5f;
float draw_value_white = 0.5f;

// 引き分けとする手数（0以外の場合、この手数に達した場合引き分けとする）
int draw_ply = 0;


////////////
//  関数  //
////////////

// ルートの展開
static void ExpandRoot(const Position *pos);

// 思考時間を延長する処理
static bool ExtendTime(void);

// 探索打ち切りの確認
static bool InterruptionCheck(void);

template <typename T>
inline void atomic_fetch_add(std::atomic<T>* obj, T arg) {
	T expected = obj->load();
	while (!atomic_compare_exchange_weak(obj, &expected, expected + arg))
		;
}


// バッチの要素
struct batch_element_t {
	uct_node_t* node;
	Color color;
};

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
#ifdef ONNXRUNTIME
			nn = (NN*)new NNOnnxRuntime(model_path[gpu_id].c_str(), gpu_id, policy_value_batch_maxsize);
#else
			if (model_path[gpu_id].find("onnx") != string::npos)
				nn = (NN*)new NNTensorRT(model_path[gpu_id].c_str(), gpu_id, policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("wideresnet15") != string::npos)
				nn = (NN*)new NNWideResnet15(model_path[gpu_id].c_str(), policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("fused_wideresnet10") != string::npos)
				nn = (NN*)new NNFusedWideResnet10(model_path[gpu_id].c_str(), policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("senet10") != string::npos)
				nn = (NN*)new NNSENet10(model_path[gpu_id].c_str(), policy_value_batch_maxsize);
			else
				nn = (NN*)new NNWideResnet10(model_path[gpu_id].c_str(), policy_value_batch_maxsize);
#endif
		}
		mutex_gpu.unlock();
	}
	void nn_forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2) {
		mutex_gpu.lock();
		nn->forward(batch_size, x1, x2, y1, y2);
		mutex_gpu.unlock();
	}
	void Run();
	void Join();
#ifdef THREAD_POOL
	void Term();
#endif

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
		handle(nullptr),
#ifdef THREAD_POOL
		ready_th(true),
		term_th(false),
#endif
		policy_value_batch_maxsize(policy_value_batch_maxsize) {
		// キューを動的に確保する
#ifdef ONNXRUNTIME
		features1 = new features1_t[policy_value_batch_maxsize];
		features2 = new features2_t[policy_value_batch_maxsize];
		y1 = new DType[MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize];
		y2 = new DType[policy_value_batch_maxsize];
#else
		checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
#endif
		policy_value_batch = new batch_element_t[policy_value_batch_maxsize];
#ifdef MAKE_BOOK
		policy_value_book_key = new Key[policy_value_batch_maxsize];
#endif

	}
	UCTSearcher(UCTSearcher&& o) :
		grp(grp),
		thread_id(thread_id),
		mt(move(o.mt)) {}
	~UCTSearcher() {
#ifdef ONNXRUNTIME
		delete[] features1;
		delete[] features2;
		delete[] y1;
		delete[] y2;
#else
		checkCudaErrors(cudaFreeHost(features1));
		checkCudaErrors(cudaFreeHost(features2));
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
#endif
		delete[] policy_value_batch;
	}

	void Run() {
#ifdef THREAD_POOL
		if (handle == nullptr) {
			handle = new thread([this]() {
				// スレッドにGPUIDを関連付けてから初期化する
				cudaSetDevice(grp->gpu_id);
				grp->InitGPU();

				while (!term_th) {
					this->ParallelUctSearch();

					std::unique_lock<std::mutex> lk(mtx_th);
					ready_th = false;
					cond_th.notify_all();

					// スレッドを停止しないで待機する
					cond_th.wait(lk, [this] { return ready_th || term_th; });
				}
			});
		}
		else {
			// スレッドを再開する
			std::unique_lock<std::mutex> lk(mtx_th);
			ready_th = true;
			cond_th.notify_all();
		}
#else
		handle = new thread([this]() {
#ifndef ONNXRUNTIME
			// スレッドにGPUIDを関連付けてから初期化する
			cudaSetDevice(grp->gpu_id);
#endif
			grp->InitGPU();

			this->ParallelUctSearch();
		});
#endif
	}
	// スレッド終了待機
	void Join() {
#ifdef THREAD_POOL
		std::unique_lock<std::mutex> lk(mtx_th);
		if (ready_th && !term_th)
			cond_th.wait(lk, [this] { return !ready_th || term_th; });
#else
		handle->join();
		delete handle;
#endif
	}
#ifdef THREAD_POOL
	// スレッドを終了
	void Term() {
		{
			std::unique_lock<std::mutex> lk(mtx_th);
			term_th = true;
			ready_th = false;
			cond_th.notify_all();
		}
		handle->join();
		delete handle;
	}
#endif

private:
	// UCT探索
	void ParallelUctSearch();
	//  UCT探索(1回の呼び出しにつき, 1回の探索)
	float UctSearch(Position* pos, uct_node_t* current, const int depth, vector<uct_node_t*>& trajectories);
	// ノードの展開
	unsigned int ExpandNode(Position* pos, const int depth);
	// UCB値が最大の子ノードを返す
	uct_node_t* SelectMaxUcbChild(const Position* pos, uct_node_t* current, const int depth);
	// ノードをキューに追加
	void QueuingNode(const Position* pos, uct_node_t* node);
	// ノードを評価
	void EvalNode();
	// Virtual Lossの加算
	int AddVirtualLoss(uct_node_t* child)
	{
		return child->move_count.fetch_add(virtual_loss);
	}
	// Virtual Lossを減算
	void SubVirtualLoss(uct_node_t* child)
	{
		child->move_count -= VIRTUAL_LOSS;
	}
	// 探索結果の更新
	void UpdateResult(uct_node_t* child, float result)
	{
		atomic_fetch_add(&child->win, result);
		if constexpr (VIRTUAL_LOSS != 1) child->move_count += 1 - VIRTUAL_LOSS;
	}

	UCTSearcherGroup* grp;
	// スレッド識別番号
	int thread_id;
	// 乱数生成器
	unique_ptr<std::mt19937_64> mt;
	// スレッドのハンドル
	thread *handle;
#ifdef THREAD_POOL
	// スレッドプール用
	std::mutex mtx_th;
	std::condition_variable cond_th;
	bool ready_th;
	bool term_th;
#endif

	int policy_value_batch_maxsize;
	features1_t* features1;
	features2_t* features2;
	DType* y1;
	DType* y2;
	batch_element_t* policy_value_batch;
#ifdef MAKE_BOOK
	Key* policy_value_book_key;
#endif
	int current_policy_value_batch_index;

	// 通常探索時のvirtual loss
	int virtual_loss;
	// 再探索時に設定する
	bool collision;
	// 仮展開したノードに到達した
	bool cache_hit;
};

/////////////////////
//  予測読みの設定  //
/////////////////////
void
SetPonderingMode(bool flag)
{
	pondering_mode = flag;
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
}

void GameOver()
{
}

// 投了の閾値設定（1000分率）
void SetResignThreshold(const int resign_threshold)
{
	RESIGN_THRESHOLD = (float)resign_threshold / 1000.0f;
}

// 千日手の価値設定（1000分率）
void SetDrawValue(const int value_black, const int value_white)
{
	draw_value_black = (float)value_black / 1000.0f;
	draw_value_white = (float)value_white / 1000.0f;
}

// 1手にかける時間取得（ms）
int GetTimeLimit()
{
	return time_limit;
}

// 引き分けとする手数の設定
void SetDrawPly(const int ply)
{
	draw_ply = ply;
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

#ifdef THREAD_POOL
// スレッド終了
void
UCTSearcherGroup::Term()
{
	if (threads > 0) {
		// 探索用スレッド
		for (int i = 0; i < threads; i++) {
			searchers[i].Term();
		}
	}
}
#endif


//////////////////////////
//  ノード再利用の設定  //
//////////////////////////
void
SetReuseSubtree(bool flag)
{
	reuse_subtree = flag;
}

// PV表示間隔設定
void SetPvInterval(const int interval)
{
	pv_interval = interval;
}

/////////////////////////
//  UCT探索の初期設定  //
/////////////////////////
void
InitializeUctSearch(const unsigned int node_limit)
{
	uct_node_limit = node_limit;

	if (!tree) tree = std::make_unique<NodeTree>();
	search_groups = new UCTSearcherGroup[max_gpu];
}

//  UCT探索の終了処理
void TerminateUctSearch()
{
#ifdef THREAD_POOL
	if (search_groups) {
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Term();
	}
#endif
}

// position抜きで探索の条件を指定する
void SetLimits(const LimitsType& limits)
{
	begin_time = limits.startTime;
	time_limit = limits.moveTime;
	po_info.halt = limits.nodes;
	minimum_time = limits.moveTime;
}


// go cmd前に呼ばれ、探索の条件を指定する
void SetLimits(const Position* pos, const LimitsType& limits)
{
	begin_time = limits.startTime;
	if (const_playout > 0) {
		po_info.halt = const_playout;
		return;
	}
	const int color = pos->turn();
	const int divisor = 14 + std::max(0, 30 - pos->gamePly());
	remaining_time[color] = limits.time[color];
	time_limit = remaining_time[color] / divisor + limits.inc[color];
	minimum_time = limits.moveTime;
	if (time_limit < limits.moveTime) {
		time_limit = limits.moveTime;
	}
	if (limits.infinite)
		po_info.halt = INT_MAX;
	else
		po_info.halt = limits.nodes;
	extend_time = limits.moveTime == 0 && limits.nodes == 0;
}

// 1手のプレイアウト回数を固定したモード
// 0の場合は無効
void SetConstPlayout(const int playout)
{
	const_playout = playout;
}

////////////
//  終了  //
////////////
void
FinalizeUctSearch(void)
{
	delete[] search_groups;
}

void
StopUctSearch(void)
{
	uct_search_stop = true;
}


std::tuple<Move, float, Move> get_and_print_pv()
{
	const uct_node_t* current_root = tree->GetCurrentHead();
	const uct_node_t* uct_child = current_root->child.get();

	unsigned int select_index = 0;
	int max_count = 0;
	const int child_num = current_root->child_num;
	int child_win_count = 0;
	int child_lose_count = 0;

	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].evaled) {
			const uct_node_t* child_node = &uct_child[i];
			const float child_value_win = child_node->value_win;
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

	float best_wp = uct_child[select_index].win / uct_child[select_index].move_count;

	// 勝ちの場合
	if (child_lose_count > 0) {
		best_wp = 1.0f;
	}
	// すべて負けの場合
	else if (child_win_count == child_num) {
		best_wp = 0.0f;
	}

	const Move move = uct_child[select_index].move;
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
	Move ponderMove = Move::moveNone();
	int depth = 1;
	{
		unsigned int best_index = select_index;
		const uct_node_t* best_node = uct_child;

		while (best_node[best_index].evaled) {
			const uct_node_t* best_child_node = &best_node[best_index];

			best_node = best_child_node->child.get();
			max_count = 0;
			best_index = 0;
			for (int i = 0; i < best_child_node->child_num; i++) {
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

	// 探索にかかった時間を求める
	const int finish_time = std::max(1, begin_time.elapsed());

	cout << "info nps " << po_info.count * 1000LL / finish_time << " time " << finish_time << " nodes " << po_info.count << " hashfull " << current_root->move_count * 1000LL / uct_node_limit << " score cp " << cp << " depth " << depth << " pv " << pv << endl;

	return std::tuple<Move, float, Move>(move, best_wp, ponderMove);
}


/////////////////////////////////////
//  UCTアルゴリズムによる着手生成  //
/////////////////////////////////////
Move
UctSearchGenmove(Position *pos, const Key starting_pos_key, const std::vector<Move>& moves, Move &ponderMove, bool ponder)
{
	uct_search_stop = false;

	init_search_begin_time = false;
	interruption = false;

	// ゲーム木を現在の局面にリセット
	tree->ResetToPosition(starting_pos_key, moves);

	// ルート局面をグローバル変数に保存
	pos_root = pos;
	
	const uct_node_t* current_root = tree->GetCurrentHead();

	pondering = ponder;

	// 探索情報をクリア
	po_info.count = 0;

	// UCTの初期化
	ExpandRoot(pos);

	// 詰みのチェック
	const int child_num = current_root->child_num;
	if (child_num == 0) {
		return Move::moveNone();
	}
	if (current_root->value_win == VALUE_WIN) {
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
	// 詰みでも最後まで投了しないようにするためコメントアウト
	/*else if (uct_node[current_root].value_win == VALUE_LOSE) {
		// 自玉の詰み
		return Move::moveNone();
	}*/

	// 前回から持ち込んだ探索回数を記録
	int pre_simulated = current_root->move_count;

	// 探索時間とプレイアウト回数の予定値を出力
	PrintPlayoutLimits(time_limit, po_info.halt);

	// 探索スレッド開始
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Run();

	// 探索スレッド終了待機
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Join();

	if (pondering)
		return Move::moveNone();

	// 着手が21手以降で,
	// 時間延長を行う設定になっていて,
	// 探索時間延長をすべきときは
	// 探索回数を2倍に増やす
	if (!uct_search_stop &&
		pos->gamePly() > 20 &&
		extend_time &&
		remaining_time[pos->turn()] > time_limit * 2 &&
		ExtendTime()) {
		time_limit *= 2;
		init_search_begin_time = false;
		interruption = false;
		// 探索スレッド開始
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Run();
		cout << "ExtendTime" << endl;

		// 探索スレッド終了待機
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Join();
	}

	// PV取得と表示
	Move move;
	float best_wp;
	std::tie(move, best_wp, ponderMove) = get_and_print_pv();

	if (best_wp < RESIGN_THRESHOLD) {
		move = Move::moveNone();
	}

	// 探索にかかった時間を求める
	const int finish_time = begin_time.elapsed();
	remaining_time[pos->turn()] -= finish_time;

	// デバッグ用
	if (debug_message)
	{
		// 候補手の情報を出力
		for (int i = 0; i < child_num; i++) {
			const auto& child = current_root->child[i];
			cout << i << ":" << child.move.toUSI() << " move_count:" << child.move_count << " nnrate:" << child.nnrate
				<< " value_win:" << (child.evaled ? (float)child.value_win : 0)
				<< " win_rate:" << (child.evaled ? child.win / child.move_count : 0) << endl;
		}

		// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
		PrintPlayoutInformation(current_root, &po_info, finish_time, pre_simulated);
	}

	return move;
}


/////////////////////////
//  ルートノードの展開  //
/////////////////////////
static void
ExpandRoot(const Position *pos)
{
	uct_node_t* current_head = tree->GetCurrentHead();
	if (current_head->child_num == 0) {
		current_head->ExpandNode(pos);
	}
}



//////////////////////////////////////
//  ノードをキューに追加            //
//////////////////////////////////////
void
UCTSearcher::QueuingNode(const Position *pos, uct_node_t* node)
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
	policy_value_batch[current_policy_value_batch_index] = { node, pos->turn() };
#ifdef MAKE_BOOK
	policy_value_book_key[current_policy_value_batch_index] = Book::bookKey(*pos);
#endif
	current_policy_value_batch_index++;
}


//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
static bool
InterruptionCheck(void)
{
	// 消費時間が短い場合は打ち止めしない
	const auto spend_time = begin_time.elapsed();
	if (spend_time * 10 < time_limit || spend_time < minimum_time) {
		return false;
	}

	int max_searched = 0, second = 0;
	const uct_node_t* current_root = tree->GetCurrentHead();
	const int child_num = current_root->child_num;
	const uct_node_t* uct_child = current_root->child.get();

	// 探索回数が最も多い手と次に多い手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max_searched) {
			second = max_searched;
			max_searched = uct_child[i].move_count;
		}
		else if (uct_child[i].move_count > second) {
			second = uct_child[i].move_count;
		}
	}


	// 残りの探索を全て次善手に費やしても
	// 最善手を超えられない場合は探索を打ち切る
	const int rest_po = (int)((long long)po_info.count * ((long long)time_limit - (long long)spend_time) / spend_time);
	if (max_searched - second > rest_po) {
		//cout << "info string interrupt_no_movechange" << endl;
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
	float max_eval = 0, second_eval = 0;
	const uct_node_t* current_root = tree->GetCurrentHead();
	const int child_num = current_root->child_num;
	const uct_node_t* uct_child = current_root->child.get();

	// 探索回数が最も多い手と次に多い手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max) {
			second = max;
			max = uct_child[i].move_count;
			max_eval = uct_child[i].win / uct_child[i].move_count;
		}
		else if (uct_child[i].move_count > second) {
			second = uct_child[i].move_count;
			second_eval = uct_child[i].win / uct_child[i].move_count;
		}
	}

	// 最善手の探索回数がが次善手の探索回数の1.5倍未満
	// もしくは、勝率が逆なら探索延長
	if (max < second * 1.5 || max_eval < second_eval) {
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
	uct_node_t* current_root = tree->GetCurrentHead();
	// ルートノードを評価
	LOCK_EXPAND;
	if (!current_root->evaled) {
		current_policy_value_batch_index = 0;
		QueuingNode(pos_root, current_root);
		EvalNode();
	}
	UNLOCK_EXPAND;

	// いずれか一つのスレッドが時間を監視する
	bool monitoring_thread = false;
	if (!init_search_begin_time.exchange(true)) {
		search_begin_time.restart();
		last_pv_print = 0;
		monitoring_thread = true;
	}

	// 探索経路のバッチ
	vector<vector<uct_node_t*>> trajectories_batch;
	vector<vector<uct_node_t*>> trajectories_batch_discarded;
	vector<vector<uct_node_t*>> trajectories_batch_collided;
	vector<vector<uct_node_t*>> trajectories_batch_evaling;
	trajectories_batch.reserve(policy_value_batch_maxsize);
	trajectories_batch_discarded.reserve(policy_value_batch_maxsize);
	trajectories_batch_collided.reserve(policy_value_batch_maxsize);
	trajectories_batch_evaling.reserve(policy_value_batch_maxsize);

	// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
	do {
		trajectories_batch.clear();
		trajectories_batch_discarded.clear();
		trajectories_batch_collided.clear();
		current_policy_value_batch_index = 0;
		virtual_loss = VIRTUAL_LOSS;
		collision = false;

		// バッチサイズ分探索を繰り返す
		for (int i = 0; i < policy_value_batch_maxsize; i++) {
			// 盤面のコピー
			Position pos(*pos_root);
			
			// 1回プレイアウトする
			trajectories_batch.emplace_back();
			AddVirtualLoss(current_root);
			trajectories_batch.back().emplace_back(current_root);
			cache_hit = false;
			float result = UctSearch(&pos, current_root, 0, trajectories_batch.back());

			while (cache_hit) {
				// 仮展開したノードに到達した場合、再探索する
				atomic_fetch_add(&po_info.count, 1);

				if (result == QUEUING) {
					// 仮展開したノードが評価中だった場合、評価が完了後にバックアップする
					trajectories_batch_evaling.emplace_back(std::move(trajectories_batch.back()));

				}
				trajectories_batch.pop_back();

				// 再探索する
				Position pos(*pos_root);
				trajectories_batch.emplace_back();
				AddVirtualLoss(current_root);
				trajectories_batch.back().emplace_back(current_root);
				cache_hit = false;
				result = UctSearch(&pos, current_root, 0, trajectories_batch.back());
			}

			if (result != DISCARDED) {
				// 探索回数を1回増やす
				atomic_fetch_add(&po_info.count, 1);
			}
			else {
				// 破棄した探索経路を保存
				trajectories_batch_discarded.emplace_back(std::move(trajectories_batch.back()));
			}

			// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
			if (result == DISCARDED || result != QUEUING) {
				trajectories_batch.pop_back();
			}
		}

		// 衝突した数だけ再探索する
		if (trajectories_batch_discarded.size() > 0) {
			virtual_loss = 0;
			collision = true;

			for (int i = 0; i < trajectories_batch_discarded.size(); i++) {
				// 盤面のコピー
				Position pos(*pos_root);

				// バックアップしないため一時的なtrajectoriesを作成
				vector<uct_node_t*> trajectories_tmp;

				// 1回プレイアウトする
				trajectories_tmp.emplace_back(current_root);
				float result = UctSearch(&pos, current_root, 0, trajectories_tmp);

				// 探索回数は増やさない

				if (result == DISCARDED) {
					// 破棄した探索経路を保存
					trajectories_batch_collided.emplace_back(std::move(trajectories_tmp));
				}
			}
		}

		// 評価
		EvalNode();

		// 破棄した探索経路のVirtual Lossを戻す
		for (auto& trajectories : trajectories_batch_discarded) {
			for (int i = trajectories.size() - 1; i >= 0; i--) {
				trajectories[i]->collision_loss = 0;
				SubVirtualLoss(trajectories[i]);
			}
		}
		// collision_lossを戻す
		for (auto& trajectories : trajectories_batch_collided) {
			for (int i = trajectories.size() - 1; i >= 0; i--) {
				trajectories[i]->collision_loss = 0;
			}
		}

		// バックアップ
		float result = 0.0f;
		for (auto& trajectories : trajectories_batch) {
			for (int i = trajectories.size() - 1; i >= 0; i--) {
				uct_node_t* child_node = trajectories[i];
				if ((size_t)i == trajectories.size() - 1) {
					const float value_win = child_node->value_win;
					// 他スレッドの詰みの伝播によりvalue_winがVALUE_WINまたはVALUE_LOSEに上書きされる場合があるためチェックする
					if (value_win == VALUE_WIN)
						result = 0.0f;
					else if (value_win == VALUE_LOSE)
						result = 1.0f;
					else
						result = 1.0f - value_win;
				}
				UpdateResult(child_node, result);
				result = 1.0f - result;
			}
		}

		// 評価中だった仮展開したノードをバックアップ
		trajectories_batch_evaling.erase(
			std::remove_if(trajectories_batch_evaling.begin(), trajectories_batch_evaling.end(),
				[this](auto& trajectories) {
					if (trajectories.back()->evaled) {
						float result = 0.0f;
						for (int i = trajectories.size() - 1; i >= 0; i--) {
							uct_node_t* child_node = trajectories[i];
							if ((size_t)i == trajectories.size() - 1) {
								const float value_win = child_node->value_win;
								// 他スレッドの詰みの伝播によりvalue_winがVALUE_WINまたはVALUE_LOSEに上書きされる場合があるためチェックする
								if (value_win == VALUE_WIN)
									result = 0.0f;
								else if (value_win == VALUE_LOSE)
									result = 1.0f;
								else
									result = 1.0f - value_win;
							}
							UpdateResult(child_node, result);
							result = 1.0f - result;
						}
						return true;
					}
					return false;
				}), trajectories_batch_evaling.end());

		// PV表示
		if (monitoring_thread && pv_interval > 0) {
			const auto elapsed_time = search_begin_time.elapsed();
			// いずれかのスレッドが1回だけ表示する
			if (elapsed_time > last_pv_print + pv_interval) {
				const auto prev_last_pv_print = last_pv_print;
				last_pv_print = elapsed_time;
				if (elapsed_time > prev_last_pv_print + pv_interval) {
					// PV表示
					get_and_print_pv();
				}
			}
		}

		// stop
		if (uct_search_stop)
			break;
		// 探索の強制終了
		// 計算時間が予定の値を超えている
		if (!pondering && po_info.halt == 0 && begin_time.elapsed() > time_limit) {
			/*if (monitoring_thread)
				cout << "info string interrupt_time_limit" << endl;*/
			break;
		}
		// po_info.halt を超えたら打ち切る
		if (!pondering && po_info.halt > 0 && po_info.count > po_info.halt) {
			/*if (monitoring_thread)
				cout << "info string interrupt_node_limit" << endl;*/
			break;
		}
		// ハッシュフル
		if ((unsigned int)current_root->move_count >= uct_node_limit) {
			/*if (monitoring_thread)
				cout << "info string interrupt_no_hash" << endl;*/
			break;
		}
		// 探索を打ち切るか確認
		if (!pondering && po_info.halt == 0 && monitoring_thread)
			interruption = InterruptionCheck();
		// 探索打ち切り
		if (interruption) {
			break;
		}
	} while (true);

	return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position *pos, uct_node_t* current, const int depth, vector<uct_node_t*>& trajectories)
{
	// policy計算中のため破棄する(他のスレッドが同じノードを先に展開した場合)
	if (!current->evaled) {
		current->collision_loss += COLLISION_LOSS;
		return DISCARDED;
	}

	if (current != tree->GetCurrentHead()) {
		if (current->value_win == VALUE_WIN) {
			// 詰み、もしくはRepetitionWinかRepetitionSuperior
			return 0.0f;  // 反転して値を返すため0を返す
		}
		else if (current->value_win == VALUE_LOSE) {
			// 自玉の詰み、もしくはRepetitionLoseかRepetitionInferior
			return 1.0f; // 反転して値を返すため1を返す
		}

		// 千日手チェック
		if (current->value_win == VALUE_DRAW) {
			if (pos->turn() == Black) {
				// 白が選んだ手なので、白の引き分けの価値を返す
				return draw_value_white;
			}
			else {
				// 黒が選んだ手なので、黒の引き分けの価値を返す
				return draw_value_black;
			}
		}

		// 詰みのチェック
		if (current->child_num == 0) {
			return 1.0f; // 反転して値を返すため1を返す
		}
	}

	float result;

	// UCB値最大の手を求める
	assert(current->child_num > 0 && current->child_num <= UCT_CHILD_MAX);
	uct_node_t* child_node = SelectMaxUcbChild(pos, current, depth);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(child_node->move, st);

	// Virtual Lossを加算
	const int move_count = AddVirtualLoss(child_node);

	// 経路を記録
	trajectories.emplace_back(child_node);

	// 展開済みにする
	const bool expanded = child_node->expanded.exchange(true);

	// ノードの展開の確認（仮展開を含めて同時に展開できない）
	if (!expanded) {
		if (collision) {
			// 仮展開
			assert(child_node->child_num == 0 && !child_node->child);
			child_node->ExpandNode(pos);
			// ノードをキューに追加
			QueuingNode(pos, child_node);
			return QUEUING;
		}

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

		// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
		if (isDraw != 0) {
			if (isDraw == 1) {
				child_node->value_win = VALUE_WIN;
				result = 0.0f;
			}
			else if (isDraw == -1) {
				child_node->value_win = VALUE_LOSE;
				result = 1.0f;
			}
			else {
				child_node->value_win = VALUE_DRAW;
				if (pos->turn() == Black) {
					// 白が選んだ手なので、白の引き分けの価値を使う
					result = draw_value_white;
				}
				else {
					// 黒が選んだ手なので、黒の引き分けの価値を使う
					result = draw_value_black;
				}
			}
		}
		else {
			// 詰みチェック
			int isMate = 0;
			if (!pos->inCheck()) {
				if (mateMoveInOddPly<false>(*pos, MATE_SEARCH_DEPTH)) {
					isMate = 1;
				}
				// 入玉勝ちかどうかを判定
				else if (nyugyoku<false>(*pos)) {
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


			// 詰みの場合、ValueNetの値を上書き
			if (isMate == 1) {
				child_node->value_win = VALUE_WIN;
				result = 0.0f;
			}
			/*else if (isMate == -1) {
				uct_node[child_index].value_win = VALUE_LOSE;
				// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
				current->value_win = VALUE_WIN;
				result = 1.0f;
			}*/
			else {
				// 候補手を展開する（千日手や詰みの場合は候補手の展開が不要なため、タイミングを遅らせる）
				assert(child_node->child_num == 0 && !child_node->child);
				child_node->ExpandNode(pos);
				if (child_node->child_num == 0) {
					// 詰み
					child_node->value_win = VALUE_LOSE;
					result = 1.0f;
				}
				else
				{
					// ノードをキューに追加
					QueuingNode(pos, child_node);
					return QUEUING;
				}
			}
		}
		child_node->evaled = true;
	}
	else {

		// 仮展開済みノードの場合
		if (!collision && move_count == 0) {
			cache_hit = true;

			// 仮展開ノードは千日手と詰みチェックは未実施のためチェックする
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

			// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
			if (isDraw != 0) {
				if (isDraw == 1) {
					child_node->value_win = VALUE_WIN;
					result = 0.0f;
				}
				else if (isDraw == -1) {
					child_node->value_win = VALUE_LOSE;
					result = 1.0f;
				}
				else {
					child_node->value_win = VALUE_DRAW;
					if (pos->turn() == Black) {
						// 白が選んだ手なので、白の引き分けの価値を使う
						result = draw_value_white;
					}
					else {
						// 黒が選んだ手なので、黒の引き分けの価値を使う
						result = draw_value_black;
					}
				}
			}
			else {
				// 詰みチェック
				int isMate = 0;
				if (!pos->inCheck()) {
					if (mateMoveInOddPly<false>(*pos, MATE_SEARCH_DEPTH)) {
						isMate = 1;
					}
					// 入玉勝ちかどうかを判定
					else if (nyugyoku<false>(*pos)) {
						isMate = 1;
					}
				}
				else {
					if (mateMoveInOddPly<true>(*pos, MATE_SEARCH_DEPTH)) {
						isMate = 1;
					}
				}

				// 詰みの場合、ValueNetの値を上書き
				if (isMate == 1) {
					child_node->value_win = VALUE_WIN;
					result = 0.0f;
				}
				// 仮展開ノードが評価済みか
				else if (child_node->evaled) {
					result = child_node->value_win;
				}
				else {
					// 仮展開済みノードが評価中の場合
					return QUEUING;
				}
			}

			UpdateResult(child_node, result);
			return 1.0f - result;
		}

		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, child_node, depth + 1, trajectories);
	}

	if (result == QUEUING || result == DISCARDED2)
		return result;
	else if (result == DISCARDED) {
		current->collision_loss += COLLISION_LOSS;
		// Virtual Lossはバッチ完了までそのままにする
		return result;
	}

	// 探索結果の反映
	UpdateResult(child_node, result);

	return 1.0f - result;
}


/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
uct_node_t*
UCTSearcher::SelectMaxUcbChild(const Position *pos, uct_node_t* current, const int depth)
{
	uct_node_t* uct_child = current->child.get();
	const int child_num = current->child_num;
	uct_node_t* max_child = &uct_child[0];
	const int sum = current->move_count + current->collision_loss * collision;
	float q, u, max_value;
	float ucb_value;
	int child_win_count = 0;

	max_value = -FLT_MAX;

	const float fpu_reduction = (depth > 0 ? c_fpu_reduction : c_fpu_reduction_root) * sqrtf(current->visited_nnrate);

	// UCB値最大の手を求める
	for (int i = 0; i < child_num; i++) {
		uct_node_t* child_node = &uct_child[i];
		if (child_node->evaled) {
			const float child_value_win = child_node->value_win;
			if (child_value_win == VALUE_WIN) {
				child_win_count++;
				// 負けが確定しているノードは選択しない
				continue;
			}
			else if (child_value_win == VALUE_LOSE) {
				// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
				current->value_win = VALUE_WIN;
			}
		}
		const float win = child_node->win;
		const int move_count = child_node->move_count + child_node->collision_loss * collision;
		assert(child_node->move_count >= 0);

		if (move_count == 0) {
			// 未探索のノードの価値に、親ノードの価値を使用する
			if (current->win > 0)
				q = std::max(0.0f, current->win / current->move_count - fpu_reduction);
			else
				q = 0.0f;
			u = sum == 0 ? 1.0f : sqrtf(sum);
		}
		else {
			q = win / move_count;
			u = sqrtf(sum) / (1 + move_count);
		}

		const float rate = child_node->nnrate;

		const float c = depth > 0 ?
			FastLog((sum + c_base + 1.0f) / c_base) + c_init :
			FastLog((sum + c_base_root + 1.0f) / c_base_root) + c_init_root;
		ucb_value = q + c * u * rate;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = child_node;
		}
	}

	if (child_win_count == child_num) {
		// 子ノードがすべて勝ちのため、自ノードを負けにする
		current->value_win = VALUE_LOSE;
	}

	// for FPU reduction
	if (max_child->evaled) {
		atomic_fetch_add(&current->visited_nnrate, max_child->nnrate);
	}

	return max_child;
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
	grp->nn_forward(policy_value_batch_size, features1, features2, y1, y2);

	const DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
	const DType *value = y2;

	for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
		uct_node_t* node = policy_value_batch[i].node;
		Color color = policy_value_batch[i].color;

		const int child_num = node->child_num;
		uct_node_t* uct_child = node->child.get();

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
		node->value_win = __half2float(*value);
#else
		node->value_win = *value;
#endif

#ifdef MAKE_BOOK
		// 定跡作成時は、事前確率に定跡の遷移確率も使用する
		constexpr float alpha = 0.5f;
		const Key& key = policy_value_book_key[i];
		const auto itr = bookMap.find(key);
		if (itr != bookMap.end()) {
			const auto& entries = itr->second;
			// countから分布を作成
			std::map<u16, u16> count_map;
			int sum = 0;
			for (const auto& entry : entries) {
				count_map.insert(std::make_pair(entry.fromToPro, entry.count));
				sum += entry.count;
			}
			// policyと定跡から作成した分布の加重平均
			for (int j = 0; j < child_num; ++j) {
				const Move& move = uct_child[j].move;
				const auto itr2 = count_map.find((u16)move.proFromAndTo());
				const float bookrate = itr2 != count_map.end() ? (float)itr2->second / sum : 0.0f;
				uct_child[j].nnrate = (1.0f - alpha) * uct_child[j].nnrate + alpha * bookrate;
			}
		}
#endif
		node->evaled = true;
	}
}
