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
#include "Utility.h"
#include "mate.h"
#include "nn_wideresnet10.h"
#include "nn_fused_wideresnet10.h"
#include "nn_wideresnet15.h"
#include "nn_senet10.h"
#include "nn_tensorrt.h"

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

// 持ち時間
double remaining_time[ColorNum];
double inc_time[ColorNum];
double po_per_sec = PLAYOUT_SPEED;

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

// 思考開始時刻
game_clock::time_point begin_time;
// 探索スレッドの思考開始時刻
game_clock::time_point search_begin_time;
atomic<bool> init_search_begin_time;

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

// 千日手の価値
float draw_value_black = 0.5f;
float draw_value_white = 0.5f;

// 引き分けとする手数（0以外の場合、この手数に達した場合引き分けとする）
int draw_ply = 0;

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
static void AddVirtualLoss(child_node_t *child, uct_node_t* current);
// Virtual Lossを減算
static void SubVirtualLoss(child_node_t *child, uct_node_t* current);

// 次のプレイアウト回数の設定
static void CalculatePlayoutPerSec(double finish_time);
static void CalculateNextPlayouts(const Position *pos);

// ルートの展開
static void ExpandRoot(const Position *pos);

// 思考時間を延長する処理
static bool ExtendTime(void);

// 候補手の初期化
static void InitializeCandidate(child_node_t *uct_child, Move move);

// 探索打ち切りの確認
static bool InterruptionCheck(void);

// 結果の更新
static void UpdateResult(child_node_t *child, float result, uct_node_t* current);

// 入玉宣言勝ち
bool nyugyoku(const Position& pos);


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
			if (model_path[gpu_id].find("onnx") != string::npos)
				nn = (NN*)new NNTensorRT(gpu_id, policy_value_batch_maxsize);
			else if (model_path[gpu_id].find("wideresnet15") != string::npos)
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
		checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
		policy_value_batch = new batch_element_t[policy_value_batch_maxsize];

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
		delete[] policy_value_batch;
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
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

					ready_th = false;
					cond_th.notify_all();

					// スレッドを停止しないで待機する
					std::unique_lock<std::mutex> lk(mtx_th);
					cond_th.wait(lk, [this] { return ready_th || term_th; });
				}
			});
		}
		else {
			// スレッドを再開する
			ready_th = true;
			cond_th.notify_all();
		}
#else
		handle = new thread([this]() {
			// スレッドにGPUIDを関連付けてから初期化する
			cudaSetDevice(grp->gpu_id);
			grp->InitGPU();

			this->ParallelUctSearch();
		});
#endif
	}
	// スレッド終了待機
	void Join() {
#ifdef THREAD_POOL
		std::unique_lock<std::mutex> lk(mtx_th);
		cond_th.wait(lk, [this] { return ready_th == false || term_th; });
#else
		handle->join();
		delete handle;
#endif
	}
#ifdef THREAD_POOL
	// スレッドを終了
	void Term() {
		term_th = true;
		ready_th = false;
		cond_th.notify_all();
		handle->join();
		delete handle;
	}
#endif

private:
	// UCT探索
	void ParallelUctSearch();
	//  UCT探索(1回の呼び出しにつき, 1回の探索)
	float UctSearch(Position* pos, uct_node_t* current, const int depth, vector<pair<uct_node_t*, unsigned int>>& trajectories);
	// ノードの展開
	unsigned int ExpandNode(Position* pos, const int depth);
	// UCB値が最大の子ノードを返す
	int SelectMaxUcbChild(const Position* pos, uct_node_t* current, const int depth);
	// ノードをキューに追加
	void QueuingNode(const Position* pos, uct_node_t* node);
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
	return (int)(time_limit * 1000);
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
	delete[] search_groups;
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
UctSearchGenmove(Position *pos, const Key starting_pos_key, const std::vector<Move>& moves, Move &ponderMove, bool ponder)
{
	Move move;
	double finish_time;

	// 探索開始時刻の記録
	begin_time = game_clock::now();
	init_search_begin_time = false;

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

	// 探索回数の閾値を設定
	CalculateNextPlayouts(pos);

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
		po_info.halt *= 2;
		time_limit *= 2;
		// 探索スレッド開始
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Run();
		cout << "ExtendTime" << endl;

		// 探索スレッド終了待機
		for (int i = 0; i < max_gpu; i++)
			search_groups[i].Join();
	}

	// 探索にかかった時間を求める
	finish_time = GetSpendTime(begin_time);

	const child_node_t* uct_child = current_root->child.get();

	int max_count = 0;
	unsigned int select_index = 0;
	int child_win_count = 0;
	int child_lose_count = 0;

	// 探索回数最大の手を見つける
	for (int i = 0; i < child_num; i++) {
		if (debug_message) {
			cout << i << ":" << uct_child[i].move.toUSI() << " move_count:" << uct_child[i].move_count << " nnrate:" << uct_child[i].nnrate << " value_win:";
			if (uct_child[i].node) cout << uct_child[i].node->value_win;
			cout << " win_rate:" << uct_child[i].win / (uct_child[i].move_count + 0.0001f) << endl;
		}

		if (uct_child[i].node) {
			const uct_node_t* child_node = uct_child[i].node.get();
			// 詰みの場合evaledは更新しないためevaledはチェックしない
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

			while (best_node[best_index].node) {
				const uct_node_t* best_child_node = best_node[best_index].node.get();

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

			cout << "info nps " << int(po_info.count / finish_time) << " time " << int(finish_time * 1000) << " nodes " << current_root->move_count << " hashfull " << current_root->move_count * 1000 / uct_node_limit << " score cp " << cp << " depth " << depth << " pv " << pv << endl;

			remaining_time[pos->turn()] -= finish_time;
		}
	}

	// 最善応手列を出力
	//PrintBestSequence(pos, uct_node, current_root);
	// 探索の情報を出力(探索回数, 勝敗, 思考時間, 勝率, 探索速度)
	if (debug_message) PrintPlayoutInformation(current_root, &po_info, finish_time, pre_simulated);

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
		MoveList<Legal> ml(*pos);
		current_head->CreateChildNode(ml);
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
	current_policy_value_batch_index++;
}


//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
static bool
InterruptionCheck(void)
{
	if (uct_search_stop)
		return true;

	if (pondering)
		return uct_search_stop;

	if (mode != CONST_PLAYOUT_MODE && po_info.halt < 0) {
		const auto spend_time = GetSpendTime(search_begin_time);
		// ハッシュの状況によってはすぐに返せる場合があるが、速度計測のため1/10は探索する
		if (spend_time * 10.0 < time_limit)
			return false;

		// プレイアウト速度を計算
		CalculatePlayoutPerSec(spend_time);
		po_info.num = (int)(po_per_sec * time_limit);
		po_info.halt = po_info.num;
	}

	int max = 0, second = 0;
	const uct_node_t* current_root = tree->GetCurrentHead();
	const int child_num = current_root->child_num;
	const int rest = po_info.halt - po_info.count;
	const child_node_t *uct_child = current_root->child.get();


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
	if (max - second >= rest) {
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
	const child_node_t *uct_child = current_root->child.get();

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

	if (!init_search_begin_time.exchange(true))
		search_begin_time = game_clock::now();

	// 探索経路のバッチ
	vector<vector<pair<uct_node_t*, unsigned int>>> trajectories_batch;

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
				pair<uct_node_t*, unsigned int>& current_next = trajectories[i];
				uct_node_t* current = current_next.first;
				const unsigned int next_index = current_next.second;
				child_node_t* uct_child = current->child.get();
				if ((size_t)i == trajectories.size() - 1) {
					const uct_node_t* child_node = uct_child[next_index].node.get();
					result = 1.0f - child_node->value_win;
				}
				UpdateResult(&uct_child[next_index], result, current);
				result = 1.0f - result;
			}
		}

		// 探索を打ち切るか確認
		if (!pondering && GetSpendTime(begin_time) > time_limit) break;
		// UCTノードに余裕があるか確認
		if ((unsigned int)current_root->move_count >= uct_node_limit) break;
	} while (!InterruptionCheck());

	return;
}


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position *pos, uct_node_t* current, const int depth, vector<pair<uct_node_t*, unsigned int>>& trajectories)
{
	if (current != tree->GetCurrentHead()) {
		// 詰みのチェック
		if (current->child_num == 0) {
			return 1.0f; // 反転して値を返すため1を返す
		}
		else if (current->value_win == VALUE_WIN) {
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
	}

	// policy計算中のため破棄する(他のスレッドが同じノードを先に展開した場合)
	if (!current->evaled)
		return DISCARDED;

	float result;
	unsigned int next_index;
	double score;
	child_node_t *uct_child = current->child.get();

	// 現在見ているノードをロック
	current->Lock();
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
	if (!uct_child[next_index].node) {
		// ノードの展開
		uct_node_t* child_node = uct_child[next_index].ExpandNode(pos);
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

		// 現在見ているノードのロックを解除
		current->UnLock();

		if (child_node->child_num == 0) {
			// 詰み
			child_node->value_win = VALUE_LOSE;
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
					// ノードをキューに追加
					QueuingNode(pos, child_node);
					return QUEUING;
				}
			}
		}
	}
	else {
		// 現在見ているノードのロックを解除
		current->UnLock();

		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, uct_child[next_index].node.get(), depth + 1, trajectories);
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
AddVirtualLoss(child_node_t *child, uct_node_t* current)
{
	atomic_fetch_add(&current->move_count, VIRTUAL_LOSS);
	atomic_fetch_add(&child->move_count, VIRTUAL_LOSS);
}

// Virtual Lossを減算
static void
SubVirtualLoss(child_node_t *child, uct_node_t* current)
{
	atomic_fetch_add(&current->move_count, -VIRTUAL_LOSS);
	atomic_fetch_add(&child->move_count, -VIRTUAL_LOSS);
}

//////////////////////
//  探索結果の更新  //
/////////////////////
static void
UpdateResult(child_node_t *child, float result, uct_node_t* current)
{
	atomic_fetch_add(&current->win, result);
	atomic_fetch_add(&current->move_count, 1 - VIRTUAL_LOSS);
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
UCTSearcher::SelectMaxUcbChild(const Position *pos, uct_node_t* current, const int depth)
{
	const child_node_t *uct_child = current->child.get();
	const int child_num = current->child_num;
	int max_child = 0;
	const int sum = current->move_count;
	float q, u, max_value;
	float ucb_value;
	int child_win_count = 0;

	max_value = -FLT_MAX;

	float fpu_reduction = (depth > 0 ? c_fpu_reduction : c_fpu_reduction_root) * sqrtf(current->visited_nnrate);

	// UCB値最大の手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].node) {
			const uct_node_t* child_node = uct_child[i].node.get();
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
		float win = uct_child[i].win;
		int move_count = uct_child[i].move_count;

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

		const float rate = uct_child[i].nnrate;

		const float c = depth > 0 ?
			FastLog((sum + c_base + 1.0f) / c_base) + c_init :
			FastLog((sum + c_base_root + 1.0f) / c_base_root) + c_init_root;
		ucb_value = q + c * u * rate;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	if (child_win_count == child_num) {
		// 子ノードがすべて勝ちのため、自ノードを負けにする
		current->value_win = VALUE_LOSE;
	}

	// for FPU reduction
	if (uct_child[max_child].node) {
		atomic_fetch_add(&current->visited_nnrate, uct_child[max_child].nnrate);
	}

	return max_child;
}


/////////////////////////////////
//  プレイアウト速度の計算     //
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
	if (pondering)
		return;

	// 探索の時の探索回数を求める
	if (mode == TIME_SETTING_MODE ||
		mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
		int color = pos->turn();
		int divisor = 14 + std::max(0, 30 - pos->gamePly());
		if (draw_ply > 0) {
			// 引き分けとする手数までに時間を使い切る
			divisor = std::min(divisor, draw_ply - pos->gamePly() + 1);
		}
		time_limit = remaining_time[color] / divisor + inc_time[color];
		// 秒読みの場合、秒読み時間未満にしない
		if (mode == TIME_SETTING_WITH_BYOYOMI_MODE &&
			time_limit < const_thinking_time) {
			time_limit = const_thinking_time;
		}
		po_info.halt = -1;
	}
	else {
		po_info.halt = po_info.num;
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
	grp->nn_forward(policy_value_batch_size, features1, features2, y1, y2);

	const DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
	const DType *value = y2;

	for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
		uct_node_t* node = policy_value_batch[i].node;
		Color color = policy_value_batch[i].color;

		node->Lock();

		const int child_num = node->child_num;
		child_node_t *uct_child = node->child.get();

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
		node->evaled = true;
		node->UnLock();
	}
}
