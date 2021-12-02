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
#ifdef PV_MATE_SEARCH
#include "PvMateSearch.h"
#endif

#ifdef ONNXRUNTIME
#include "nn_onnxruntime.h"
#else
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
constexpr uint64_t MUTEX_NUM = 65536; // must be 2^n
std::mutex mutexes[MUTEX_NUM];
inline std::mutex& GetPositionMutex(const Position* pos)
{
	return mutexes[pos->getKey() & (MUTEX_NUM - 1)];
}


////////////////
//  大域変数  //
////////////////

#ifdef MAKE_BOOK
#include "book.hpp"
extern std::map<Key, std::vector<BookEntry> > bookMap;
extern bool use_book_policy;
extern bool use_interruption;
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
int multi_pv = 1; // MultiPvの数
float eval_coef = 756; // 勝率から評価値に変換する際の係数

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
#ifndef MATE_SEARCH_DEPTH
constexpr int MATE_SEARCH_DEPTH = 5;
#endif

// 探索の結果を評価のキューに追加したか、破棄したか
constexpr float QUEUING = FLT_MAX;
constexpr float DISCARDED = -FLT_MAX;

// 千日手の価値
float draw_value_black = 0.5f;
float draw_value_white = 0.5f;

// 引き分けとする手数（この手数に達した場合引き分けとする）
int draw_ply = INT_MAX;

// ランダムムーブ設定
int random_ply = 0;
float random_reciprocal_temperature = 1.0f / 10.0f;
float random_cutoff = 0.020f;
std::unique_ptr<std::mt19937_64> random_mt_64;

#ifdef PV_MATE_SEARCH
// PVの詰み探索
std::vector<PvMateSearcher> pv_mate_searchers;
#endif

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

// Virtual Lossの加算
inline void
AddVirtualLoss(child_node_t* child, uct_node_t* current)
{
	current->move_count += VIRTUAL_LOSS;
	child->move_count += VIRTUAL_LOSS;
}

// Virtual Lossを減算
inline void
SubVirtualLoss(child_node_t* child, uct_node_t* current)
{
	current->move_count -= VIRTUAL_LOSS;
	child->move_count -= VIRTUAL_LOSS;
}

// 探索結果の更新
inline void
UpdateResult(child_node_t* child, float result, uct_node_t* current)
{
	atomic_fetch_add(&current->win, (WinType)result);
	if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
	atomic_fetch_add(&child->win, (WinType)result);
	if constexpr (VIRTUAL_LOSS != 1) child->move_count += 1 - VIRTUAL_LOSS;
}

typedef pair<uct_node_t*, unsigned int> trajectory_t;
typedef vector<trajectory_t> trajectories_t;

struct visitor_t {
	visitor_t() {
		trajectories.reserve(128);
	}
	trajectories_t trajectories;
	float value_win;
};

// バッチの要素
struct batch_element_t {
	uct_node_t* node;
	Color color;
	float* value_win;
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
			nn = (NN*)new NNTensorRT(model_path[gpu_id].c_str(), gpu_id, policy_value_batch_maxsize);
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
		checkCudaErrors(cudaHostAlloc((void**)&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc((void**)&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc((void**)&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc((void**)&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
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
	float UctSearch(Position* pos, child_node_t* parent, uct_node_t* current, visitor_t& visitor);
	// UCB値が最大の子ノードを返す
	int SelectMaxUcbChild(child_node_t* parent, uct_node_t* current);
	// ノードをキューに追加
	void QueuingNode(const Position* pos, uct_node_t* node, float* value_win);
	// ノードを評価
	void EvalNode();

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
#ifdef PV_MATE_SEARCH
	PvMateSearcher::Clear();
#endif
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
	draw_ply = ply > 0 ? ply : INT_MAX;
}

#ifdef PV_MATE_SEARCH
// PVの詰み探索の設定
void SetPvMateSearch(const int threads, const int depth, const int nodes)
{
	pv_mate_searchers.reserve(threads);
	for (int i = 0; i < threads; i++)
		pv_mate_searchers.emplace_back(depth, nodes);
}
#endif

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

// MultiPV設定
void SetMultiPV(const int multipv)
{
	multi_pv = multipv;
}

// 勝率から評価値に変換する際の係数設定
void SetEvalCoef(const int eval_coef)
{
	::eval_coef = (float)eval_coef;
}

// ランダムムーブの設定
void SetRandomMove(const int ply, const int random_temperature, const int cutoff)
{
	random_ply = ply;
	random_reciprocal_temperature = 1000.0f / random_temperature;
	random_cutoff = cutoff / 1000.0f;
	if (ply > 0 && !random_mt_64) {
		std::random_device seed_gen;
		random_mt_64.reset(new std::mt19937_64(seed_gen()));
	}
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
#ifdef PV_MATE_SEARCH
	for (auto& searcher : pv_mate_searchers)
		searcher.Term();
#endif

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
	po_info.halt = static_cast<int>(limits.nodes);
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
	int divisor = 14 + std::max(0, 30 - pos->gamePly());
	// 引き分けとする手数までに時間を使い切る
	if (draw_ply - pos->gamePly() >= 0)
		divisor = std::min(divisor, draw_ply - pos->gamePly() + 1);
	remaining_time[color] = limits.time[color];
	time_limit = remaining_time[color] / divisor + limits.inc[color];
	minimum_time = limits.moveTime;
	if (time_limit < limits.moveTime) {
		time_limit = limits.moveTime;
	}
	if (limits.infinite)
		po_info.halt = INT_MAX;
	else
		po_info.halt = static_cast<int>(limits.nodes);
	extend_time = time_limit > minimum_time && limits.nodes == 0;
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
#ifdef PV_MATE_SEARCH
	// PVの詰み探索スレッド停止
	for (auto& searcher : pv_mate_searchers)
		searcher.Stop();
#endif

	uct_search_stop = true;
}

bool IsUctSearchStoped()
{
	return uct_search_stop;
}

bool compare_child_node_ptr_descending(const child_node_t* lhs, const child_node_t* rhs)
{
	if (lhs->IsWin()) {
		// 負けが確定しているノードは選択しない
		if (rhs->IsWin()) {
			// すべて負けの場合は、探索回数が最大の手を選択する
			if (lhs->move_count == rhs->move_count)
				return lhs->nnrate > rhs->nnrate;
			return lhs->move_count > rhs->move_count;
		}
		return false;
	}
	else if (lhs->IsLose()) {
		// 子ノードに一つでも負けがあれば、勝ちなので選択する
		if (rhs->IsLose()) {
			// すべて勝ちの場合は、探索回数が最大の手を選択する
			if (lhs->move_count == rhs->move_count)
				return lhs->nnrate > rhs->nnrate;
			return lhs->move_count > rhs->move_count;
		}
		return true;
	}
	if (lhs->move_count == rhs->move_count)
		return lhs->nnrate > rhs->nnrate;
	return lhs->move_count > rhs->move_count;
}

inline std::tuple<std::string, int, int, Move, float, Move> get_pv(const uct_node_t* root_uct_node, const unsigned int best_root_child_index)
{
	const auto& best_root_uct_child = root_uct_node->child[best_root_child_index];
	float best_wp = best_root_uct_child.win / best_root_uct_child.move_count;

	// 勝ちの場合
	if (best_root_uct_child.IsLose()) {
		best_wp = 1.0f;
	}
	// すべて負けの場合
	else if (best_root_uct_child.IsWin()) {
		best_wp = 0.0f;
	}

	const Move move = best_root_uct_child.move;
	int cp;
	if (best_wp == 1.0f) {
		cp = 30000;
	}
	else if (best_wp == 0.0f) {
		cp = -30000;
	}
	else {
		cp = int(-logf(1.0f / best_wp - 1.0f) * eval_coef);
	}

	Move ponderMove = Move::moveNone();
	string pv = move.toUSI();
	int depth = 1;
	const uct_node_t* best_node = root_uct_node;
	unsigned int best_index = best_root_child_index;
	while (best_node->child_nodes && best_node->child_nodes[best_index]) {
		best_node = best_node->child_nodes[best_index].get();
		if (!best_node || best_node->child_num == 0)
			break;

		// 最大の子ノード
		best_index = select_max_child_node(best_node);
		const auto& best_uct_child = best_node->child[best_index];

		const auto best_move = best_uct_child.move;
		pv += " ";
		pv += best_move.toUSI();

		// ponderの着手
		if (pondering_mode && ponderMove == Move::moveNone())
			ponderMove = best_move;

		depth++;
	}

	return std::make_tuple(pv, cp, depth, move, best_wp, ponderMove);
}

// 訪問回数に応じてランダムに子ノードを選択
inline unsigned int select_random_child_node(const uct_node_t* uct_node)
{
	const child_node_t* uct_child = uct_node->child.get();
	const auto child_num = uct_node->child_num;

	// 訪問回数順にソート
	std::vector<const child_node_t*> sorted_uct_childs;
	sorted_uct_childs.reserve(child_num);
	for (int i = 0; i < child_num; i++)
		sorted_uct_childs.emplace_back(&uct_node->child[i]);
	std::stable_sort(sorted_uct_childs.begin(), sorted_uct_childs.end(), compare_child_node_ptr_descending);

	// 訪問数が最大のノードの価値の一定以下は除外
	const auto max_move_count_child = sorted_uct_childs[0];
	const auto cutoff_threshold = max_move_count_child->win / max_move_count_child->move_count - random_cutoff;
	vector<double> probabilities;
	probabilities.reserve(child_num);
	for (int i = 0; i < child_num; i++) {
		if (sorted_uct_childs[i]->move_count == 0) break;

		const auto win = sorted_uct_childs[i]->win / sorted_uct_childs[i]->move_count;
		if (win < cutoff_threshold) break;

		const auto probability = std::pow(sorted_uct_childs[i]->move_count, random_reciprocal_temperature);
		probabilities.emplace_back(probability);
		if (debug_message)
			std::cout << sorted_uct_childs[i]->move.toUSI() << " move_count:" << sorted_uct_childs[i]->move_count
			<< " nnrate:" << sorted_uct_childs[i]->nnrate << " win_rate:" << sorted_uct_childs[i]->win / (sorted_uct_childs[i]->move_count)
			<< " probability:" << probability << std::endl;
	}

	// 訪問回数に応じた確率で選択
	discrete_distribution<unsigned int> dist(probabilities.begin(), probabilities.end());
	const auto selected_index = dist(*random_mt_64);
	return static_cast<unsigned int>(sorted_uct_childs[selected_index] - uct_child);
}

std::tuple<Move, float, Move> get_and_print_pv(const bool use_random = false)
{
	const uct_node_t* current_root = tree->GetCurrentHead();

	// 探索にかかった時間
	const auto finish_time = std::max(1, begin_time.elapsed());
	// nps
	const auto nps = po_info.count * 1000LL / finish_time;
	// hashfull
	const auto hashfull = current_root->move_count * 1000LL / uct_node_limit;

	std::string pv;
	int cp;
	int depth;
	Move move;
	float best_wp;
	Move ponderMove;

	if (multi_pv == 1) {
		// 最大の子ノードを取得
		const unsigned int best_root_child_index =
			(use_random && pos_root->gamePly() <= random_ply)
			? select_random_child_node(current_root)
			: select_max_child_node(current_root);

		// PV表示
		std::tie(pv, cp, depth, move, best_wp, ponderMove) = get_pv(current_root, best_root_child_index);
		std::cout << "info nps " << nps << " time " << finish_time << " nodes " << po_info.count << " hashfull " << hashfull << " score cp " << cp << " depth " << depth << " pv " << pv << std::endl;
	}
	else {
		// 部分ソート
		const child_node_t* root_uct_child = current_root->child.get();
		const int child_num = current_root->child_num;
		const int multipv_num = std::min(multi_pv, child_num);
		std::vector<const child_node_t*> sorted_root_uct_childs;
		sorted_root_uct_childs.reserve(child_num);
		for (int i = 0; i < child_num; i++)
			sorted_root_uct_childs.emplace_back(&root_uct_child[i]);
		std::partial_sort(sorted_root_uct_childs.begin(), sorted_root_uct_childs.begin() + multipv_num, sorted_root_uct_childs.end(), compare_child_node_ptr_descending);

		// info文字列の共通部分
		std::stringstream info_ss;
		info_ss << " nps " << nps << " time " << finish_time << " nodes " << po_info.count << " hashfull " << hashfull;
		const std::string info_string = info_ss.str();

		Move move_tmp;
		float best_wp_tmp;
		Move ponderMove_tmp;

		// Multi PV表示
		for (int i = 0; i < multipv_num; i++) {
			const child_node_t* best_root_uct_child = sorted_root_uct_childs[i];
			const unsigned int best_root_child_index = static_cast<unsigned int>(best_root_uct_child - root_uct_child);

			std::tie(pv, cp, depth, move_tmp, best_wp_tmp, ponderMove_tmp) = get_pv(current_root, best_root_child_index);
			std::cout << "info multipv " << i + 1 << info_string;
			if (best_root_uct_child->move_count > 0)
				std::cout << " score cp " << cp;
			std::cout << " depth " << depth << " pv " << pv << "\n";

			if (i == 0) {
				move = move_tmp;
				best_wp = best_wp_tmp;
				ponderMove = ponderMove_tmp;
			}
		}
		std::cout << std::flush;

		// 訪問回数に応じた確率で選択する場合
		if (use_random && pos_root->gamePly() <= random_ply) {
			const unsigned int best_root_child_index = select_random_child_node(current_root);
			std::tie(pv, cp, depth, move, best_wp, ponderMove) = get_pv(current_root, best_root_child_index);
		}
	}

	return std::make_tuple(move, best_wp, ponderMove);
}


/////////////////////////////////////
//  UCTアルゴリズムによる着手生成  //
/////////////////////////////////////
Move
UctSearchGenmove(Position* pos, const Key starting_pos_key, const std::vector<Move>& moves, Move& ponderMove, bool ponder)
{
	uct_search_stop = false;
#ifdef PV_MATE_SEARCH
	for (auto& searcher : pv_mate_searchers)
		searcher.Stop(false);
#endif

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

	// 前回から持ち込んだ探索回数を記録
	const int pre_simulated = current_root->move_count;

	// 探索時間とプレイアウト回数の予定値を出力
	if (debug_message)
		PrintPlayoutLimits(time_limit, po_info.halt);

	// 探索スレッド開始
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Run();

#ifdef PV_MATE_SEARCH
	// PVの詰み探索スレッド開始
	for (auto& searcher : pv_mate_searchers)
		searcher.Run();
#endif

	// 探索スレッド終了待機
	for (int i = 0; i < max_gpu; i++)
		search_groups[i].Join();

	if (pondering) {
#ifdef PV_MATE_SEARCH
		// PVの詰み探索スレッド終了待機
		for (auto& searcher : pv_mate_searchers)
			searcher.Join();
#endif

		return Move::moveNone();
	}

#ifdef PV_MATE_SEARCH
	// PVの詰み探索スレッド停止
	for (auto& searcher : pv_mate_searchers)
		searcher.Stop();
	// PVの詰み探索スレッド終了待機
	for (auto& searcher : pv_mate_searchers)
		searcher.Join();
#endif

	// PV取得と表示
	Move move;
	float best_wp;
	std::tie(move, best_wp, ponderMove) = get_and_print_pv(random_ply > 0);

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
				<< " win_rate:" << (child.move_count > 0 ? child.win / child.move_count : 0)
				<< (child.IsLose() ? " win" : "") << (child.IsWin() ? " lose" : "") << endl;
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
UCTSearcher::QueuingNode(const Position *pos, uct_node_t* node, float* value_win)
{
	//cout << "QueuingNode:" << index << ":" << current_policy_value_queue_index << ":" << current_policy_value_batch_index << endl;
	//cout << pos->toSFEN() << endl;

	/* if (current_policy_value_batch_index >= policy_value_batch_maxsize) {
		std::cout << "error" << std::endl;
	}*/
	// set all zero
	std::fill_n((DType*)features1[current_policy_value_batch_index], sizeof(features1_t) / sizeof(DType), 0);
	std::fill_n((DType*)features2[current_policy_value_batch_index], sizeof(features2_t) / sizeof(DType), 0);

	make_input_features(*pos, &features1[current_policy_value_batch_index], &features2[current_policy_value_batch_index]);
	policy_value_batch[current_policy_value_batch_index] = { node, pos->turn(), value_win };
#ifdef MAKE_BOOK
	policy_value_book_key[current_policy_value_batch_index] = Book::bookKey(*pos);
#endif
	current_policy_value_batch_index++;
}

// 探索回数が最も多い手と次に多い手を求める
inline std::tuple<int, int, int, int> FindMaxAndSecondVisits(const uct_node_t* current_root, const child_node_t* uct_child)
{
	int max_searched = 0, second_searched = 0;
	int max_index = 0, second_index = 0;

	const int child_num = current_root->child_num;
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max_searched) {
			second_searched = max_searched;
			max_searched = uct_child[i].move_count;
			max_index = i;
		}
		else if (uct_child[i].move_count > second_searched) {
			second_searched = uct_child[i].move_count;
			second_index = i;
		}
	}

	return std::make_tuple(max_searched, second_searched, max_index, second_index);
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

	const uct_node_t* current_root = tree->GetCurrentHead();
	const child_node_t* uct_child = current_root->child.get();

	// 探索回数が最も多い手と次に多い手を求める
	int max_searched, second_searched;
	int max_index, second_index;
	std::tie(max_searched, second_searched, max_index, second_index) = FindMaxAndSecondVisits(current_root, uct_child);

	// 詰みが見つかった場合は探索を打ち切る
	if (uct_child[max_index].IsLose())
		return true;

	// 残りの探索で次善手が最善手を超える可能性がある場合は打ち切らない
	const int rest_po = (int)((long long)po_info.count * ((long long)time_limit - (long long)spend_time) / spend_time);
	if (max_searched - second_searched <= rest_po) {
		return false;
	}

	// 着手が21手以降で,
	// 時間延長を行う設定になっていて,
	// 探索時間延長をすべきときは
	// 探索回数を2倍に増やす
	if (extend_time &&
		pos_root->gamePly() > 20 &&
		remaining_time[pos_root->turn()] > time_limit * 2 &&
		// 最善手の探索回数が次善手の探索回数の1.5倍未満
		// もしくは、勝率が逆なら探索延長
		(max_searched < second_searched * 1.5 ||
		 uct_child[max_index].win / uct_child[max_index].move_count < uct_child[second_index].win / uct_child[second_index].move_count)) {
		time_limit *= 2;
		extend_time = false; // 探索延長は1回のみ
		cout << "ExtendTime" << endl;
		return false;
	}

	return true;
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
	if (!current_root->IsEvaled()) {
		current_policy_value_batch_index = 0;
		float value_win;
		QueuingNode(pos_root, current_root, &value_win);
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
	vector<visitor_t> visitor_pool(policy_value_batch_maxsize);
	vector<visitor_t*> visitor_batch;
	vector<trajectories_t*> trajectories_batch_discarded;
	visitor_batch.reserve(policy_value_batch_maxsize);
	trajectories_batch_discarded.reserve(policy_value_batch_maxsize);

	// 探索回数が閾値を超える, または探索が打ち切られたらループを抜ける
	do {
		visitor_batch.clear();
		trajectories_batch_discarded.clear();
		current_policy_value_batch_index = 0;

		// バッチサイズ分探索を繰り返す
		for (int i = 0; i < policy_value_batch_maxsize; i++) {
			// 盤面のコピー
			Position pos(*pos_root);
			
			// 1回プレイアウトする
			visitor_pool[i].trajectories.clear();
			const float result = UctSearch(&pos, nullptr, current_root, visitor_pool[i]);

			if (result != DISCARDED) {
				// 探索回数を1回増やす
				atomic_fetch_add(&po_info.count, 1);
			}
			else {
				// 破棄した探索経路を保存
				trajectories_batch_discarded.emplace_back(&visitor_pool[i].trajectories);
			}

			if (result == DISCARDED || result != QUEUING) {
				// 評価中の末端ノードに達した、もしくはバックアップ済みため破棄する
			}
			else {
				visitor_batch.emplace_back(&visitor_pool[i]);
			}
		}

		// 評価
		EvalNode();

		// 破棄した探索経路のVirtual Lossを戻す
		for (auto trajectories : trajectories_batch_discarded) {
			for (int i = static_cast<int>(trajectories->size() - 1); i >= 0; i--) {
				auto& current_next = trajectories->at(i);
				uct_node_t* current = current_next.first;
				child_node_t* uct_child = current->child.get();
				const unsigned int next_index = current_next.second;
				SubVirtualLoss(&uct_child[next_index], current);
			}
		}

		// バックアップ
		for (auto& visitor : visitor_batch) {
			auto& trajectories = visitor->trajectories;
			float result = 1.0f - visitor->value_win;
			for (int i = static_cast<int>(trajectories.size() - 1); i >= 0; i--) {
				auto& current_next = trajectories[i];
				uct_node_t* current = current_next.first;
				const unsigned int next_index = current_next.second;
				child_node_t* uct_child = current->child.get();
				UpdateResult(&uct_child[next_index], result, current);
				result = 1.0f - result;
			}
		}

		// stop
		if (uct_search_stop)
			break;
		// 探索の強制終了
		// ハッシュフル
		if ((unsigned int)current_root->move_count >= uct_node_limit) {
			/*if (monitoring_thread)
				cout << "info string interrupt_no_hash" << endl;*/
			break;
		}
		if (!pondering) {
			// po_info.halt を超えたら打ち切る
			if (po_info.halt > 0) {
				if (po_info.count > po_info.halt) {
					/*if (monitoring_thread)
						cout << "info string interrupt_node_limit" << endl;*/
					break;
				}
#ifdef MAKE_BOOK
				if (use_interruption && monitoring_thread) {
					const child_node_t* uct_child = current_root->child.get();

					// 探索回数が最も多い手と次に多い手を求める
					int max_searched, second_searched;
					int max_index, second_index;
					std::tie(max_searched, second_searched, max_index, second_index) = FindMaxAndSecondVisits(current_root, uct_child);


					// 残りの探索で次善手が最善手を超える可能性がない場合は打ち切る
					const int rest_po = po_info.halt - po_info.count;
					if (max_searched - second_searched > rest_po) {
						interruption = true;
					}
				}
#endif
			}
			else {
				// 探索を打ち切るか確認
				if (monitoring_thread)
					interruption = InterruptionCheck();
			}
			// 探索打ち切り
			if (interruption) {
				break;
			}
		}

		// PV表示
		if (monitoring_thread && pv_interval > 0) {
			const auto elapsed_time = search_begin_time.elapsed();
			// いずれかのスレッドが1回だけ表示する
			if (elapsed_time > last_pv_print + pv_interval) {
				last_pv_print = elapsed_time;
				// PV表示
				get_and_print_pv();
			}
		}
	} while (true);

	return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position *pos, child_node_t* parent, uct_node_t* current, visitor_t& visitor)
{
	float result;
	child_node_t *uct_child = current->child.get();
	auto& trajectories = visitor.trajectories;

	// 現在見ているノードをロック
	auto& mutex = GetPositionMutex(pos);
	mutex.lock();
	// 子ノードへのポインタ配列が初期化されていない場合、初期化する
	if (!current->child_nodes) current->InitChildNodes();
	// UCB値最大の手を求める
	const unsigned int next_index = SelectMaxUcbChild(parent, current);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	// Virtual Lossを加算
	AddVirtualLoss(&uct_child[next_index], current);
	// ノードの展開の確認
	if (!current->child_nodes[next_index]) {
		// ノードの作成
		uct_node_t* child_node = current->CreateChildNode(next_index);
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

		// 現在見ているノードのロックを解除
		mutex.unlock();

		// 経路を記録
		trajectories.emplace_back(current, next_index);

		int isDraw = 0;
		// 最大手数を超えていたら千日手扱いとする
		if (pos->gamePly() > draw_ply) {
			isDraw = 2; // Draw
		}
		else {
			// 千日手チェック
			switch (pos->isDraw(16)) {
			case NotRepetition: break;
			case RepetitionDraw: isDraw = 2; break; // Draw
			case RepetitionWin: isDraw = 1; break;
			case RepetitionLose: isDraw = -1; break;
			case RepetitionSuperior: isDraw = 1; break;
			case RepetitionInferior: isDraw = -1; break;
			default: UNREACHABLE;
			}
		}

		// 千日手の場合、ValueNetの値を使用しない（合流を処理しないため、value_winを上書きする）
		if (isDraw != 0) {
			if (isDraw == 1) {
				uct_child[next_index].SetWin();
				result = 0.0f;
			}
			else if (isDraw == -1) {
				uct_child[next_index].SetLose();
				result = 1.0f;
			}
			else {
				uct_child[next_index].SetDraw();
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
				if (mateMoveInOddPly<MATE_SEARCH_DEPTH, false>(*pos, draw_ply)) {
					isMate = 1;
				}
				// 入玉勝ちかどうかを判定
				else if (nyugyoku<false>(*pos)) {
					isMate = 1;
				}
			}
			else {
				if (mateMoveInOddPly<MATE_SEARCH_DEPTH, true>(*pos, draw_ply)) {
					isMate = 1;
				}
				// 偶数手詰めは親のノードの奇数手詰めでチェックされているためチェックしない
				/*else if (mateMoveInEvenPly(*pos, MATE_SEARCH_DEPTH - 1)) {
					isMate = -1;
				}*/
			}


			// 詰みの場合、ValueNetの値を上書き
			if (isMate == 1) {
				uct_child[next_index].SetWin();
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
				child_node->ExpandNode(pos);
				if (child_node->child_num == 0) {
					// 詰み
					uct_child[next_index].SetLose();
					result = 1.0f;
				}
				else
				{
					// ノードをキューに追加
					QueuingNode(pos, child_node, &visitor.value_win);
					return QUEUING;
				}
			}
		}
		child_node->SetEvaled();
	}
	else {
		// 現在見ているノードのロックを解除
		mutex.unlock();

		// 経路を記録
		trajectories.emplace_back(current, next_index);

		uct_node_t* next_node = current->child_nodes[next_index].get();

		// policy計算中のため破棄する(他のスレッドが同じノードを先に展開した場合)
		if (!next_node->IsEvaled())
			return DISCARDED;

		if (uct_child[next_index].IsWin()) {
			// 詰み、もしくはRepetitionWinかRepetitionSuperior
			result = 0.0f;  // 反転して値を返すため0を返す
		}
		else if (uct_child[next_index].IsLose()) {
			// 自玉の詰み、もしくはRepetitionLoseかRepetitionInferior
			result = 1.0f; // 反転して値を返すため1を返す
		}
		// 千日手チェック
		else if (uct_child[next_index].IsDraw()) {
			if (pos->turn() == Black) {
				// 白が選んだ手なので、白の引き分けの価値を返す
				result = draw_value_white;
			}
			else {
				// 黒が選んだ手なので、黒の引き分けの価値を返す
				result = draw_value_black;
			}
		}
		// 詰みのチェック
		else if (next_node->child_num == 0) {
			result = 1.0f; // 反転して値を返すため1を返す
		}
		else {
			// 手番を入れ替えて1手深く読む
			result = UctSearch(pos, &uct_child[next_index], next_node, visitor);
		}
	}

	if (result == QUEUING)
		return result;
	else if (result == DISCARDED) {
		// Virtual Lossはバッチ完了までそのままにする
		return result;
	}

	// 探索結果の反映
	UpdateResult(&uct_child[next_index], result, current);

	return 1.0f - result;
}


/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
int
UCTSearcher::SelectMaxUcbChild(child_node_t* parent, uct_node_t* current)
{
	const child_node_t *uct_child = current->child.get();
	const int child_num = current->child_num;
	int max_child = 0;
	const int sum = current->move_count;
	const WinType sum_win = current->win;
	float q, u, max_value;
	int child_win_count = 0;

	max_value = -FLT_MAX;

	const float sqrt_sum = sqrtf(static_cast<const float>(sum));
	const float c = parent == nullptr ?
		FastLog((sum + c_base_root + 1.0f) / c_base_root) + c_init_root :
		FastLog((sum + c_base + 1.0f) / c_base) + c_init;
	const float fpu_reduction = (parent == nullptr ? c_fpu_reduction_root : c_fpu_reduction) * sqrtf(current->visited_nnrate);
	const float parent_q = sum_win > 0 ? std::max(0.0f, (float)(sum_win / sum) - fpu_reduction) : 0.0f;
	const float init_u = sum == 0 ? 1.0f : sqrt_sum;

	// UCB値最大の手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].IsWin()) {
			child_win_count++;
			// 負けが確定しているノードは選択しない
			continue;
		}
		else if (uct_child[i].IsLose()) {
			// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
			if (parent != nullptr)
				parent->SetWin();
			// 勝ちが確定しているため、選択する
			return i;
		}

		const WinType win = uct_child[i].win;
		const int move_count = uct_child[i].move_count;

		if (move_count == 0) {
			// 未探索のノードの価値に、親ノードの価値を使用する
			q = parent_q;
			u = init_u;
		}
		else {
			q = (float)(win / move_count);
			u = sqrt_sum / (1 + move_count);
		}

		const float rate = uct_child[i].nnrate;

		const float ucb_value = q + c * u * rate;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	if (child_win_count == child_num) {
		// 子ノードがすべて勝ちのため、自ノードを負けにする
		if (parent != nullptr)
			parent->SetLose();
	}
	else {
		// for FPU reduction
		atomic_fetch_add(&current->visited_nnrate, uct_child[max_child].nnrate);
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
		const Color color = policy_value_batch[i].color;

		const int child_num = node->child_num;
		child_node_t *uct_child = node->child.get();

		// 合法手一覧
		for (int j = 0; j < child_num; j++) {
			const Move move = uct_child[j].move;
			const int move_label = make_move_label((u16)move.proFromAndTo(), color);
			const float logit = (*logits)[move_label];
			uct_child[j].nnrate = logit;
		}

		// Boltzmann distribution
		softmax_temperature_with_normalize(uct_child, child_num);

		*policy_value_batch[i].value_win = *value;

#ifdef MAKE_BOOK
		if (use_book_policy) {
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
		}
#endif
		node->SetEvaled();
	}
}
