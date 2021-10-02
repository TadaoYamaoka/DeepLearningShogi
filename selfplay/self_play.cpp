#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"
#include "fastmath.h"

#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <signal.h>

#include "Node.h"
#include "LruCache.h"
#include "mate.h"
#include "nn_tensorrt.h"
#include "dfpn.h"
#include "USIEngine.h"

#include "cppshogi.h"

#include "cxxopts/cxxopts.hpp"

//#define SPDLOG_TRACE_ON
//#define SPDLOG_DEBUG_ON
#define SPDLOG_EOL "\n"
#include "spdlog/spdlog.h"
auto loggersink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
auto logger = std::make_shared<spdlog::async_logger>("selfplay", loggersink, 8192);

using namespace std;

// 候補手の最大数(盤上全体)
constexpr int UCT_CHILD_MAX = 593;

int threads = 2;

volatile sig_atomic_t stopflg = false;

void sigint_handler(int signum)
{
	stopflg = true;
}

// ランダムムーブの手数
int RANDOM_MOVE;
// 訪問数が最大のノードの価値の一定割合以下は除外
float RANDOM_CUTOFF = 0.0f;
// 訪問数に応じてランダムに選択する際の温度パラメータ
float RANDOM_TEMPERATURE = 1.0f;
// 訪問回数が最大の手が2番目の手のx倍以内の場合にランダムに選択する
float RANDOM2 = 0;
// 出力する最低手数
int MIN_MOVE;
// ルートの方策に加えるノイズの確率(千分率)
int ROOT_NOISE;

// 終局とする勝率の閾値
float WINRATE_THRESHOLD;

// 詰み探索の深さ
uint32_t ROOT_MATE_SEARCH_DEPTH;
// 詰み探索の最大ノード数
int64_t MATE_SEARCH_MAX_NODE;
constexpr int64_t MATE_SEARCH_MIN_NODE = 10000;

// モデルのパス
string model_path;

int playout_num = 1000;

// USIエンジンのパス
string usi_engine_path;
int usi_engine_num;
int usi_threads;
// USIエンジンオプション（name:value,...,name:value）
string usi_options;
int usi_byoyomi;
int usi_turn; // 0:先手、1:後手、それ以外:ランダム

std::mutex mutex_all_gpu;

int MAX_MOVE = 320; // 最大手数
bool OUT_MAX_MOVE = false; // 最大手数に達した対局の局面を出力するか
constexpr int EXTENSION_TIMES = 2; // 探索延長回数
bool REUSE_SUBTREE = false; // 探索済みノードを再利用するか

struct CachedNNRequest {
	CachedNNRequest(size_t size) : nnrate(size) {}
	float value_win;
	std::vector<float> nnrate;
};
typedef LruCache<uint64_t, CachedNNRequest> NNCache;
typedef LruCacheLock<uint64_t, CachedNNRequest> NNCacheLock;
unsigned int nn_cache_size = 8388608; // NNキャッシュサイズ

s64 teacherNodes; // 教師局面数
std::atomic<s64> idx(0);
std::atomic<s64> madeTeacherNodes(0);
std::atomic<s64> games(0);
std::atomic<s64> draws(0);
std::atomic<s64> nyugyokus(0);
// プレイアウト数
std::atomic<s64> sum_playouts(0);
std::atomic<s64> sum_nodes(0);
// USIエンジンとの対局結果
std::atomic<s64> usi_games(0);
std::atomic<s64> usi_wins(0);
std::atomic<s64> usi_draws(0);

ifstream ifs;
ofstream ofs;
bool OUT_MIN_HCP = false;
ofstream ofs_minhcp;
mutex imutex;
mutex omutex;
size_t entryNum;

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
#ifndef MATE_SEARCH_DEPTH
constexpr int MATE_SEARCH_DEPTH = 7;
#endif

// 探索の結果を評価のキューに追加したか
constexpr float QUEUING = FLT_MAX;

float c_init;
float c_base;
float c_fpu_reduction;
float c_init_root;
float c_base_root;
float temperature;


typedef pair<uct_node_t*, unsigned int> trajectory_t;
typedef vector<trajectory_t> trajectories_t;

struct visitor_t {
	trajectories_t trajectories;
	float value_win;
};

// バッチの要素
struct batch_element_t {
	uct_node_t* node;
	Color color;
	Key key;
	float* value_win;
};

// 探索結果の更新
inline void
UpdateResult(child_node_t* child, float result, uct_node_t* current)
{
	current->win += (WinType)result;
	current->move_count++;
	child->win += (WinType)result;
	child->move_count++;
}

// 詰み探索スロット
struct MateSearchEntry {
	Position *pos;
	enum State { RUNING, NOMATE, WIN, LOSE };
	atomic<State> status;
	Move move;
};

Searcher s;

class UCTSearcher;
class UCTSearcherGroupPair;
class UCTSearcherGroup {
public:
	UCTSearcherGroup(const int gpu_id, const int group_id, const int policy_value_batch_maxsize, UCTSearcherGroupPair* parent) :
		gpu_id(gpu_id), group_id(group_id), policy_value_batch_maxsize(policy_value_batch_maxsize), parent(parent),
		nn_cache(nn_cache_size),
		current_policy_value_batch_index(0), features1(nullptr), features2(nullptr), policy_value_batch(nullptr), y1(nullptr), y2(nullptr), running(false) {
		Initialize();
	}
	UCTSearcherGroup(UCTSearcherGroup&& o) {} // not use
	~UCTSearcherGroup() {
		delete[] policy_value_batch;
		delete[] mate_search_slot;
		checkCudaErrors(cudaFreeHost(features1));
		checkCudaErrors(cudaFreeHost(features2));
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
	}

	void QueuingNode(const Position *pos, uct_node_t* node, float* value_win);
	void EvalNode();
	void SelfPlay();
	void Run();
	void Join();
	void QueuingMateSearch(Position *pos, const int id) {
		lock_guard<mutex> lock(mate_search_mutex);
		mate_search_slot[id].pos = pos;
		mate_search_slot[id].status = MateSearchEntry::RUNING;
		mate_search_queue.push_back(id);
	}
	MateSearchEntry::State GetMateSearchStatus(const int id) {
		return mate_search_slot[id].status;
	}
	Move GetMateSearchMove(const int id) {
		return mate_search_slot[id].move;
	}
	void MateSearch();

	int group_id;
	int gpu_id;
	bool running;
	// USIEngine
	vector<USIEngine> usi_engines;

private:
	void Initialize();

	UCTSearcherGroupPair* parent;

	// NNキャッシュ
	NNCache nn_cache;

	// キュー
	int policy_value_batch_maxsize; // 最大バッチサイズ
	features1_t* features1;
	features2_t* features2;
	batch_element_t* policy_value_batch;
	int current_policy_value_batch_index;

	// UCTSearcher
	vector<UCTSearcher> searchers;
	thread* handle_selfplay;

	DType* y1;
	DType* y2;

	// 詰み探索
	DfPn dfpn;
	MateSearchEntry* mate_search_slot = nullptr;
	deque<int> mate_search_queue;
	mutex mate_search_mutex;
	thread* handle_mate_search = nullptr;
};

class UCTSearcher {
public:
	UCTSearcher(UCTSearcherGroup* grp, NNCache& nn_cache, const int id, const size_t entryNum) :
		grp(grp),
		nn_cache(nn_cache),
		id(id),
		mt_64(new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + id)),
		mt(new std::mt19937((unsigned int)std::chrono::system_clock::now().time_since_epoch().count() + id)),
		inputFileDist(0, entryNum - 1),
		max_playout_num(playout_num),
		playout(0),
		ply(0),
		states(MAX_MOVE + 1) {
		pos_root = new Position(DefaultStartPositionSFEN, s.thisptr);
		usi_engine_turn = (grp->usi_engines.size() > 0 && id < usi_engine_num) ? rnd(*mt) % 2 : -1;
		noise_count.reserve(UCT_CHILD_MAX);
	}
	UCTSearcher(UCTSearcher&& o) : nn_cache(o.nn_cache) {} // not use
	~UCTSearcher() {
		// USIエンジンが思考中の場合待機する
		if (usi_engine_turn >= 0) {
			grp->usi_engines[id % usi_threads].WaitThinking();
		}

		delete pos_root;
	}

	void Playout(visitor_t& visitor);
	void NextStep();

private:
	float UctSearch(Position* pos, child_node_t* parent, uct_node_t* current, visitor_t& visitor);
	int SelectMaxUcbChild(child_node_t* parent, uct_node_t* current);
	bool InterruptionCheck(const int playout_count, const int extension_times);
	void NextPly(const Move move);
	void NextGame();

	// キャッシュからnnrateをコピー
	void CopyNNRate(uct_node_t* node, const vector<float>& nnrate) {
		child_node_t* uct_child = node->child.get();
		for (int i = 0; i < node->child_num; i++) {
			uct_child[i].nnrate = nnrate[i];
		}
	}

	UCTSearcherGroup* grp;
	int id;
	unique_ptr<std::mt19937_64> mt_64;
	unique_ptr<std::mt19937> mt;

	// ルートノード
	std::unique_ptr<uct_node_t> root_node;

	// NNキャッシュ(UCTSearcherGroupで共有)
	NNCache& nn_cache;

	int max_playout_num;
	int playout;
	int ply;
	GameResult gameResult;
	u8 reason;

	std::vector<StateInfo> states;
	uniform_int_distribution<s64> inputFileDist;

	// 局面管理と探索スレッド
	Position* pos_root;

	// ノイズにより選んだ回数
	std::vector<int> noise_count;

	// 詰み探索のステータス
	MateSearchEntry::State mate_status;

	// 開始局面
	HuffmanCodedPos hcp;

	// USIエンジン
	int usi_engine_turn; // -1:未使用、0:偶数手、1:奇数手
	std::string usi_position;

	// 出力棋譜データ
	struct Record {
		Record() {}
		Record(const u16 selectedMove16, const s16 eval) : selectedMove16(selectedMove16), eval(eval) {}

		u16 selectedMove16; // 指し手
		s16 eval; // 評価値
		std::vector<MoveVisits> candidates;
	};
	std::vector<Record> records;

	// 局面追加
	// 訓練に使用しない手はtrainningをfalseにする
	void AddRecord(Move move, s16 eval, bool trainning) {
		Record& record = records.emplace_back(
			static_cast<u16>(move.value()),
			eval
		);
		if (trainning) {
			const auto child = root_node->child.get();
			record.candidates.reserve(root_node->child_num);
			for (size_t i = 0; i < root_node->child_num; ++i) {
				// ノイズにより選んだ回数を除く
				const auto move_count = child[i].move_count - noise_count[i];
				if (move_count > 0) {
					record.candidates.emplace_back(
						static_cast<u16>(child[i].move.value()),
						static_cast<u16>(move_count)
					);
				}
			}
			idx++;
		}
	}
};

class UCTSearcherGroupPair {
public:
	UCTSearcherGroupPair(const int gpu_id, const int policy_value_batch_maxsize) : nn(nullptr), gpu_id(gpu_id), policy_value_batch_maxsize(policy_value_batch_maxsize) {
		groups.reserve(threads);
		for (int i = 0; i < threads; i++)
			groups.emplace_back(gpu_id, i, policy_value_batch_maxsize, this);
	}
	UCTSearcherGroupPair(UCTSearcherGroupPair&& o) {} // not use
	~UCTSearcherGroupPair() {
		delete nn;
	}
	void InitGPU() {
		mutex_all_gpu.lock();
		if (nn == nullptr) {
			nn = (NN*)new NNTensorRT(model_path.c_str(), gpu_id, policy_value_batch_maxsize);
		}
		mutex_all_gpu.unlock();
	}
	void nn_forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2) {
		mutex_gpu.lock();
		nn->forward(batch_size, x1, x2, y1, y2);
		mutex_gpu.unlock();
	}
	int Running() {
		int running = 0;
		for (int i = 0; i < threads; i++)
			running += groups[i].running;
		return running;
	}
	void Run() {
		for (int i = 0; i < threads; i++)
			groups[i].Run();
	}
	void Join() {
		for (int i = 0; i < threads; i++)
			groups[i].Join();
	}

private:
	vector<UCTSearcherGroup> groups;
	int policy_value_batch_maxsize;
	int gpu_id;

	// neural network
	NN* nn;
	// mutex for gpu
	mutex mutex_gpu;
};

void
UCTSearcherGroup::Initialize()
{
	// USIエンジン起動
	if (usi_engine_path != "" && usi_engine_num != 0) {
		std::vector<std::pair<std::string, std::string>> options;
		std::istringstream ss(usi_options);
		std::string field;
		while (std::getline(ss, field, ',')) {
			const auto pos = field.find_first_of(":");
			options.emplace_back(field.substr(0, pos), field.substr(pos + 1));
		
		}
		usi_engines.reserve(usi_threads);
		for (int i = 0; i < usi_threads; ++i) {
			usi_engines.emplace_back(usi_engine_path, options, (usi_engine_num + usi_threads - 1) / usi_threads);
		}
	}

	// キューを動的に確保する
	checkCudaErrors(cudaHostAlloc((void**)&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	policy_value_batch = new batch_element_t[policy_value_batch_maxsize];

	// UCTSearcher
	searchers.clear();
	searchers.reserve(policy_value_batch_maxsize);
	for (int i = 0; i < policy_value_batch_maxsize; i++) {
		searchers.emplace_back(this, nn_cache, i, entryNum);
	}

	checkCudaErrors(cudaHostAlloc((void**)&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc((void**)&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));

	// 詰み探索
	if (ROOT_MATE_SEARCH_DEPTH > 0) {
		dfpn.init();
		dfpn.set_max_search_node(MATE_SEARCH_MAX_NODE);
		dfpn.set_maxdepth(ROOT_MATE_SEARCH_DEPTH);
		mate_search_slot = new MateSearchEntry[policy_value_batch_maxsize];
	}
}

// 連続自己対局
void UCTSearcherGroup::SelfPlay()
{
	// スレッドにGPUIDを関連付けてから初期化する
	cudaSetDevice(gpu_id);

	parent->InitGPU();

	// 探索経路のバッチ
	vector<visitor_t> visitor_batch(policy_value_batch_maxsize);

	// 全スレッドが生成した局面数が生成局面数以上になったら終了
	while (madeTeacherNodes < teacherNodes && !stopflg) {
		current_policy_value_batch_index = 0;

		// すべての対局についてシミュレーションを行う
		for (size_t i = 0; i < (size_t)policy_value_batch_maxsize; i++) {
			UCTSearcher& searcher = searchers[i];
			searcher.Playout(visitor_batch[i]);
		}

		// 評価
		EvalNode();

		// バックアップ
		for (auto& visitor : visitor_batch) {
			auto& trajectories = visitor.trajectories;
			float result = 1.0f - visitor.value_win;
			for (int i = (int)trajectories.size() - 1; i >= 0; i--) {
				auto& current_next = trajectories[i];
				uct_node_t* current = current_next.first;
				const unsigned int next_index = current_next.second;
				child_node_t* uct_child = current->child.get();
				UpdateResult(&uct_child[next_index], result, current);
				result = 1.0f - result;
			}
		}

		// 次のシミュレーションへ
		for (UCTSearcher& searcher : searchers) {
			searcher.NextStep();
		}
	}

	running = false;
}

// スレッド開始
void
UCTSearcherGroup::Run()
{
	// 自己対局用スレッド
	running = true;
	handle_selfplay = new thread([this]() { this->SelfPlay(); });

	// 詰み探索用スレッド
	if (ROOT_MATE_SEARCH_DEPTH > 0) {
		handle_mate_search = new thread([this]() { this->MateSearch(); });
	}
}

// スレッド終了待機
void
UCTSearcherGroup::Join()
{
	// 自己対局用スレッド
	handle_selfplay->join();
	delete handle_selfplay;
	// 詰み探索用スレッド
	if (handle_mate_search) {
		handle_mate_search->join();
		delete handle_mate_search;
	}
}

// 詰み探索スレッド
void UCTSearcherGroup::MateSearch()
{
	deque<int> queue;
	while (running) {
		// キューから取り出す
		mate_search_mutex.lock();
		if (mate_search_queue.size() > 0) {
			queue.swap(mate_search_queue);
			mate_search_mutex.unlock();
		}
		else {
			mate_search_mutex.unlock();
			this_thread::yield();
			continue;
		}

		for (int& id : queue) {
			// 盤面のコピー
			Position pos_copy(*mate_search_slot[id].pos);

			// 詰み探索
			if (!pos_copy.inCheck()) {
				bool mate = dfpn.dfpn(pos_copy);
				//SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} {} mate:{} nodes:{}", gpu_id, group_id, id, pos_copy.toSFEN(), mate, dfpn.searchedNode);
				if (mate)
					mate_search_slot[id].move = dfpn.dfpn_move(pos_copy);
				mate_search_slot[id].status = mate ? MateSearchEntry::WIN : MateSearchEntry::NOMATE;
			}
			else {
				// 自玉に王手がかかっている
				bool mate = dfpn.dfpn_andnode(pos_copy);
				//SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} {} mate_andnode:{} nodes:{}", gpu_id, group_id, id, pos_copy.toSFEN(), mate, dfpn.searchedNode);
				mate_search_slot[id].status = mate ? MateSearchEntry::LOSE : MateSearchEntry::NOMATE;
			}
		}
		queue.clear();
	}
}

//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position* pos, child_node_t* parent, uct_node_t* current, visitor_t& visitor)
{
	float result;
	child_node_t* uct_child = current->child.get();
	auto& trajectories = visitor.trajectories;

	// 子ノードへのポインタ配列が初期化されていない場合、初期化する
	if (!current->child_nodes) current->InitChildNodes();
	// UCB値最大の手を求める
	const unsigned int next_index = SelectMaxUcbChild(parent, current);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	// ノードの展開の確認
	if (!current->child_nodes[next_index]) {
		// ノードの作成
		uct_node_t* child_node = current->CreateChildNode(next_index);

		// 経路を記録
		trajectories.emplace_back(current, next_index);

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

		// 千日手の場合、ValueNetの値を使用しない
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
				result = 0.5f;
			}
		}
		else if (nn_cache.ContainsKey(pos->getKey())) {
			NNCacheLock cache_lock(&nn_cache, pos->getKey());
			// キャッシュヒット
			// 候補手を展開する
			child_node->ExpandNode(pos);
			assert(cache_lock->nnrate.size() == child_node->child_num);
			// キャッシュからnnrateをコピー
			CopyNNRate(child_node, cache_lock->nnrate);
			// 経路により詰み探索の結果が異なるためキャッシュヒットしても詰みの場合があるが、速度が落ちるため詰みチェックは行わない
			result = 1.0f - cache_lock->value_win;
		}
		else {
			// 詰みチェック
			int isMate = 0;
			if (!pos->inCheck()) {
				if (mateMoveInOddPly<MATE_SEARCH_DEPTH, false>(*pos)) {
					isMate = 1;
				}
				// 入玉勝ちかどうかを判定
				else if (nyugyoku<false>(*pos)) {
					isMate = 1;
				}
			}
			else {
				if (mateMoveInOddPly<MATE_SEARCH_DEPTH, true>(*pos)) {
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
					grp->QueuingNode(pos, child_node, &visitor.value_win);
					return QUEUING;
				}
			}
		}
		child_node->SetEvaled();
	}
	else {
		// 経路を記録
		trajectories.emplace_back(current, next_index);

		uct_node_t* next_node = current->child_nodes[next_index].get();

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
			result = 0.5f;
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
	const child_node_t* uct_child = current->child.get();
	const int child_num = current->child_num;
	int max_child = 0, max_child_nonoise = 0;
	const int sum = current->move_count;
	const WinType sum_win = current->win;
	float q, u, max_value, max_value_nonoise;
	int child_win_count = 0;

	max_value = max_value_nonoise = -FLT_MAX;

	const float sqrt_sum = sqrtf((float)sum);
	const float c = parent == nullptr ?
		FastLog((sum + c_base_root + 1.0f) / c_base_root) + c_init_root :
		FastLog((sum + c_base + 1.0f) / c_base) + c_init;
	const float fpu_reduction = (parent == nullptr ? 0.0f : c_fpu_reduction) * sqrtf(current->visited_nnrate);
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

		float rate = uct_child[i].nnrate;
		if (parent == nullptr) {
			const float ucb_value_nonoise = q + c * u * rate;
			// ノイズがない場合の選択
			if (ucb_value_nonoise > max_value_nonoise) {
				max_value_nonoise = ucb_value_nonoise;
				max_child_nonoise = i;
			}
			// ランダムに確率を上げる
			if (rnd(*mt) < ROOT_NOISE)
				rate = (rate + 1.0f) / 2.0f;
		}

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
		current->visited_nnrate += uct_child[max_child].nnrate;
	}

	// ノイズにより選んだ回数
	if (parent == nullptr && max_child != max_child_nonoise) {
		noise_count[max_child]++;
	}

	return max_child;
}

//////////////////////////////////////
//  ノードをキューに追加            //
//////////////////////////////////////
void
UCTSearcherGroup::QueuingNode(const Position *pos, uct_node_t* node, float* value_win)
{
	// set all zero
	std::fill_n((DType*)features1[current_policy_value_batch_index], sizeof(features1_t) / sizeof(DType), 0);
	std::fill_n((DType*)features2[current_policy_value_batch_index], sizeof(features2_t) / sizeof(DType), 0);

	make_input_features(*pos, &features1[current_policy_value_batch_index], &features2[current_policy_value_batch_index]);
	policy_value_batch[current_policy_value_batch_index] = { node, pos->turn(), pos->getKey(), value_win };
	current_policy_value_batch_index++;
}

//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
bool
UCTSearcher::InterruptionCheck(const int playout_count, const int extension_times)
{
	int max_index = 0;
	int max = 0, second = 0;
	const int child_num = root_node->child_num;
	const int rest = max_playout_num - playout_count;
	const child_node_t* uct_child = root_node->child.get();

	// 探索回数が最も多い手と次に多い手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].move_count > max) {
			second = max;
			max = uct_child[i].move_count;
			max_index = i;
		}
		else if (uct_child[i].move_count > second) {
			second = uct_child[i].move_count;
		}
	}

	// 詰みが見つかった場合は探索を打ち切る
	if (uct_child[max_index].IsLose())
		return true;

	// 残りの探索を全て次善手に費やしても
	// 最善手を超えられない場合は探索を打ち切る
	if (max - second > rest) {
		// 最善手の探索回数が次善手の探索回数の
		// 1.2倍未満なら探索延長
		if (max_playout_num < playout_num * extension_times && max < second * 1.2) {
			max_playout_num += playout_num / 2;
			return false;
		}

		return true;
	}
	else {
		return false;
	}
}

// 局面の評価
void UCTSearcherGroup::EvalNode() {
	if (current_policy_value_batch_index == 0)
		return;

	const int policy_value_batch_size = current_policy_value_batch_index;

	// predict
	parent->nn_forward(policy_value_batch_size, features1, features2, y1, y2);

	DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
	DType *value = y2;

	for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
		uct_node_t* node = policy_value_batch[i].node;
		const Color color = policy_value_batch[i].color;

		const int child_num = node->child_num;
		child_node_t* uct_child = node->child.get();

		// 合法手一覧
		for (int j = 0; j < child_num; j++) {
			Move move = uct_child[j].move;
			const int move_label = make_move_label((u16)move.proFromAndTo(), color);
			const float logit = (*logits)[move_label];
			uct_child[j].nnrate = logit;
		}

		// Boltzmann distribution
		softmax_temperature_with_normalize(uct_child, child_num);

		auto req = make_unique<CachedNNRequest>(child_num);
		for (int j = 0; j < child_num; j++) {
			req->nnrate[j] = uct_child[j].nnrate;
		}

		const float value_win = *value;

		req->value_win = value_win;
		nn_cache.Insert(policy_value_batch[i].key, std::move(req));

		if (policy_value_batch[i].value_win)
			*policy_value_batch[i].value_win = value_win;

		node->SetEvaled();
	}
}

// シミュレーションを1回行う
void UCTSearcher::Playout(visitor_t& visitor)
{
	while (true) {
		visitor.trajectories.clear();
		// 手番開始
		if (playout == 0) {
			// 新しいゲーム開始
			if (ply == 0) {
				ply = 1;

				// 開始局面を局面集からランダムに選ぶ
				{
					std::unique_lock<Mutex> lock(imutex);
					ifs.seekg(inputFileDist(*mt_64) * sizeof(HuffmanCodedPos), std::ios_base::beg);
					ifs.read(reinterpret_cast<char*>(&hcp), sizeof(hcp));
				}
				setPosition(*pos_root, hcp);
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());

				records.clear();
				reason = 0;
				root_node.release();

				// USIエンジン
				if (usi_engine_turn >= 0) {
					// 開始局面設定
					usi_position = "position " + pos_root->toSFEN() + " moves";

					// 先手後手指定
					if (usi_turn == Black)
						usi_engine_turn = pos_root->turn() == Black ? 1 : 0;
					else if (usi_turn == White)
						usi_engine_turn = pos_root->turn() == White ? 1 : 0;
					else
						usi_engine_turn = rnd(*mt) % 2;

					if (usi_engine_turn == 1 && RANDOM_MOVE == 0) {
						grp->usi_engines[id % usi_threads].ThinkAsync(id / usi_threads, *pos_root, usi_position, usi_byoyomi);
						return;
					}
				}
			}
			else if (ply % 2 == usi_engine_turn && ply > RANDOM_MOVE) {
				return;
			}

			if (!root_node || !REUSE_SUBTREE) {
				// ルートノード作成(以前のノードは再利用しないで破棄する)
				root_node = std::make_unique<uct_node_t>();

				// ルートノード展開
				root_node->ExpandNode(pos_root);
			}

			// ノイズ回数初期化
			noise_count.resize(root_node->child_num);
			std::fill(noise_count.begin(), noise_count.end(), 0);

			// 詰みのチェック
			if (root_node->child_num == 0) {
				gameResult = (pos_root->turn() == Black) ? GameResult::WhiteWin : GameResult::BlackWin;
				NextGame();
				continue;
			}
			else if (root_node->child_num == 1) {
				// 1手しかないときは、その手を指して次の手番へ
				const Move move = root_node->child[0].move;
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} skip:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), move.toUSI());
				AddRecord(move, 0, false);
				NextPly(move);
				continue;
			}
			else if (nyugyoku(*pos_root)) {
				// 入玉宣言勝ち
				gameResult = (pos_root->turn() == Black) ? GameResult::BlackWin : GameResult::WhiteWin;
				reason = GAMERESULT_NYUGYOKU;
				if (records.size() > 0)
					++nyugyokus;
				NextGame();
				continue;
			}

			// ルート局面を詰み探索キューに追加
			if (ROOT_MATE_SEARCH_DEPTH > 0) {
				mate_status = MateSearchEntry::RUNING;
				grp->QueuingMateSearch(pos_root, id);
			}

			// ルート局面をキューに追加
			if (!root_node->IsEvaled()) {
				NNCacheLock cache_lock(&nn_cache, pos_root->getKey());
				if (!cache_lock || cache_lock->nnrate.size() == 0) {
					grp->QueuingNode(pos_root, root_node.get(), nullptr);
					return;
				}
				else {
					assert(cache_lock->nnrate.size() == root_node->child_num);
					// キャッシュからnnrateをコピー
					CopyNNRate(root_node.get(), cache_lock->nnrate);
				}
			}
		}

		// 盤面のコピー
		Position pos_copy(*pos_root);
		// プレイアウト
		const float result = UctSearch(&pos_copy, nullptr, root_node.get(), visitor);
		if (result != QUEUING) {
			NextStep();
			continue;
		}

		return;
	}
}

// 次の手に進める
void UCTSearcher::NextStep()
{
	// USIエンジン
	if (ply % 2 == usi_engine_turn && ply > RANDOM_MOVE) {
		const Move move = grp->usi_engines[id % usi_threads].ThinkDone(id / usi_threads);
		if (move == Move::moveNone())
			return;

		if (move == moveResign()) {
			gameResult = (pos_root->turn() == Black) ? GameResult::WhiteWin : GameResult::BlackWin;
			NextGame();
			return;
		}
		else if (move == moveWin()) {
			gameResult = (pos_root->turn() == Black) ? GameResult::BlackWin : GameResult::WhiteWin;
			reason = GAMERESULT_NYUGYOKU;
			NextGame();
			return;
		}
		else if (move == moveAbort()) {
			if (stopflg)
				return;
			throw std::runtime_error("usi engine abort");
		}
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} usi_move:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), move.toUSI());

		AddRecord(move, 0, false);
		NextPly(move);
		return;
	}

	// 詰み探索の結果を調べる
	if (ROOT_MATE_SEARCH_DEPTH > 0 && mate_status == MateSearchEntry::RUNING) {
		mate_status = grp->GetMateSearchStatus(id);
		if (mate_status != MateSearchEntry::RUNING) {
			// 詰みの場合
			switch (mate_status) {
			case MateSearchEntry::WIN:
			{
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} mate win", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
				gameResult = (pos_root->turn() == Black) ? BlackWin : WhiteWin;

				// 局面追加（ランダム局面は除く）
				if (ply > RANDOM_MOVE)
					AddRecord(grp->GetMateSearchMove(id), 30000, false);

				NextGame();
				return;
			}
			case MateSearchEntry::LOSE:
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} mate lose", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
				gameResult = (pos_root->turn() == Black) ? WhiteWin : BlackWin;
				NextGame();
				return;
			}
		}
	}

	// プレイアウト回数加算
	playout++;

	// 探索終了判定
	if (InterruptionCheck(playout, (ply > RANDOM_MOVE) ? EXTENSION_TIMES : 0)) {
		// 平均プレイアウト数を計測
		sum_playouts += playout;
		++sum_nodes;

		// 詰み探索の結果を待つ
		if (ROOT_MATE_SEARCH_DEPTH > 0) {
			while (mate_status == MateSearchEntry::RUNING) {
				this_thread::yield();
				mate_status = grp->GetMateSearchStatus(id);
			}
			// 詰みの場合
			switch (mate_status) {
			case MateSearchEntry::WIN:
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} mate win", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
				gameResult = (pos_root->turn() == Black) ? BlackWin : WhiteWin;

				// 局面追加（初期局面は除く）
				if (ply > RANDOM_MOVE)
					AddRecord(grp->GetMateSearchMove(id), 30000, false);

				NextGame();
				return;
			case MateSearchEntry::LOSE:
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} mate lose", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
				gameResult = (pos_root->turn() == Black) ? WhiteWin : BlackWin;
				NextGame();
				return;
			}
		}

		const child_node_t* uct_child = root_node->child.get();
		unsigned int select_index = 0;
		Move best_move;
		if (ply <= RANDOM_MOVE) {
			// N手までは訪問数に応じた確率で選択する
			// 訪問数が最大のノードの価値の一定割合以下は除外
			const auto max_move_count_child = std::max_element(uct_child, uct_child + root_node->child_num, [](const child_node_t& l, const child_node_t& r) { return l.move_count < r.move_count; });
			const auto cutoff_threshold = max_move_count_child->win / max_move_count_child->move_count * RANDOM_CUTOFF;
			vector<int> indexes;
			vector<double> probabilities;
			indexes.reserve(root_node->child_num);
			probabilities.reserve(root_node->child_num);
			for (int i = 0; i < root_node->child_num; i++) {
				if (uct_child[i].move_count > 0) {
					const auto win = uct_child[i].win / uct_child[i].move_count;
					if (win >= cutoff_threshold) {
						indexes.emplace_back(i);
						probabilities.emplace_back(std::pow(uct_child[i].move_count, 1.0 / RANDOM_TEMPERATURE));
						SPDLOG_TRACE(logger, "gpu_id:{} group_id:{} id:{} {}:{} move_count:{} nnrate:{} win_rate:{}", grp->gpu_id, grp->group_id, id, i, uct_child[i].move.toUSI(), uct_child[i].move_count, uct_child[i].nnrate, uct_child[i].win / (uct_child[i].move_count));
					}
				}
			}

			discrete_distribution<unsigned int> dist(probabilities.begin(), probabilities.end());
			select_index = indexes[dist(*mt_64)];
			best_move = uct_child[select_index].move;
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} random_move:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), best_move.toUSI());
			AddRecord(best_move, 0, false);
		}
		else {
			// 探索回数最大の手を見つける
			int max_count = uct_child[0].move_count;
			int second_index = 0;
			int second_count = 0;
			int child_win_count = 0;
			int child_lose_count = 0;
			const int child_num = root_node->child_num;
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
					second_index = select_index;
					second_count = max_count;
					select_index = i;
					max_count = uct_child[i].move_count;
				}
			}

			if (RANDOM2 > 1) {
				// 訪問回数が最大の手が2番目の手のx倍以内の場合にランダムに選択する
				if (max_count < second_count * RANDOM2) {
					vector<int> probabilities{ second_count, max_count };
					discrete_distribution<unsigned int> dist(probabilities.begin(), probabilities.end());
					const auto i = dist(*mt_64);
					if (i == 0)
						select_index = second_index;
					SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} random2:{},{} selected:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), second_count, max_count, i);
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
			best_move = uct_child[select_index].move;
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} bestmove:{} winrate:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), best_move.toUSI(), best_wp);

			{
				// 勝率が閾値を超えた場合、ゲーム終了
				const float winrate = (best_wp - 0.5f) * 2.0f;
				if (WINRATE_THRESHOLD < abs(winrate)) {
					if (pos_root->turn() == Black)
						gameResult = (winrate < 0 ? WhiteWin : BlackWin);
					else
						gameResult = (winrate < 0 ? BlackWin : WhiteWin);

					NextGame();
					return;
				}
			}

			// 局面追加
			s16 eval;
			if (best_wp == 1.0f)
				eval = 30000;
			else if (best_wp == 0.0f)
				eval = -30000;
			else
				eval = s16(-logf(1.0f / best_wp - 1.0f) * 756.0864962951762f);
			AddRecord(best_move, eval, true);
		}

		NextPly(best_move);
	}
}

void UCTSearcher::NextPly(const Move move)
{
	// 一定の手数以上で引き分け
	if (ply >= MAX_MOVE) {
		gameResult = Draw;
		reason = GAMERESULT_MAXMOVE;
		// 最大手数に達した対局は出力しない
		if (!OUT_MAX_MOVE)
			records.clear();
		NextGame();
		return;
	}

	// 着手
	pos_root->doMove(move, states[ply]);
	ply++;

	// 千日手の場合
	switch (pos_root->isDraw(16)) {
	case RepetitionDraw:
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} RepetitionDraw", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
		gameResult = Draw;
		reason = GAMERESULT_SENNICHITE;
		NextGame();
		return;
	case RepetitionWin:
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} RepetitionWin", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
		gameResult = (pos_root->turn() == Black) ? BlackWin : WhiteWin;
		NextGame();
		return;
	case RepetitionLose:
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} RepetitionLose", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
		gameResult = (pos_root->turn() == Black) ? WhiteWin : BlackWin;
		NextGame();
		return;
	}

	// 次の手番
	max_playout_num = playout_num;
	playout = 0;

	if (usi_engine_turn >= 0) {
		usi_position += " " + move.toUSI();
		if (ply % 2 == usi_engine_turn)
			grp->usi_engines[id % usi_threads].ThinkAsync(id / usi_threads, *pos_root, usi_position, usi_byoyomi);
	}

	// ノード再利用
	if (root_node && REUSE_SUBTREE) {
		bool found = false;
		if (root_node->child_nodes) {
			for (int i = 0; i < root_node->child_num; i++) {
				if (root_node->child[i].move == move && root_node->child_nodes[i] && root_node->child_nodes[i]->child) {
					found = true;
					// 子ノードをルートノードにする
					auto root_node_tmp = std::move(root_node->child_nodes[i]);
					root_node = std::move(root_node_tmp);
					// ルートの訪問回数をクリア
					root_node->move_count = 0;
					root_node->win = 0;
					root_node->visited_nnrate = 0;
					for (int j = 0; j < root_node->child_num; j++) {
						root_node->child[j].move_count = 0;
						root_node->child[j].win = 0;
					}
					break;
				}
			}
		}
		// USIエンジンが選んだ手が見つからない可能性があるため、見つからなかったらルートノードを再作成する
		if (!found) {
			root_node.release();
		}
	}
}

void UCTSearcher::NextGame()
{
	SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} gameResult:{}", grp->gpu_id, grp->group_id, id, ply, gameResult);

	// 局面出力
	if (ply >= MIN_MOVE && records.size() > 0) {
		const Color start_turn = (ply % 2 == 1 && pos_root->turn() == Black || ply % 2 == 0 && pos_root->turn() == White) ? Black : White;
		const u8 opponent = usi_engine_turn < 0 ? 0 : (usi_engine_turn == 1 && start_turn == Black || usi_engine_turn == 0 && start_turn == White) ? 1 : 2;
		HuffmanCodedPosAndEval3 hcpe3{
			hcp,
			static_cast<u16>(records.size()),
			static_cast<u8>(gameResult | reason),
			opponent
		};
		{
			std::unique_lock<Mutex> lock(omutex);
			ofs.write(reinterpret_cast<char*>(&hcpe3), sizeof(HuffmanCodedPosAndEval3));
			for (auto& record : records) {
				MoveInfo moveInfo{ record.selectedMove16, record.eval, static_cast<u16>(record.candidates.size()) };
				ofs.write(reinterpret_cast<char*>(&moveInfo), sizeof(MoveInfo));
				if (record.candidates.size() > 0) {
					ofs.write(reinterpret_cast<char*>(record.candidates.data()), sizeof(MoveVisits) * record.candidates.size());
					madeTeacherNodes++;
				}
			}
		}
		++games;

		if (gameResult == Draw) {
			++draws;
		}
	}

	// USIエンジンとの対局結果
	if (ply >= MIN_MOVE && usi_engine_turn >= 0) {
		++usi_games;
		if (ply % 2 == 1 && (pos_root->turn() == Black && gameResult == (BlackWin + usi_engine_turn) || pos_root->turn() == White && gameResult == (WhiteWin - usi_engine_turn)) ||
			ply % 2 == 0 && (pos_root->turn() == Black && gameResult == (WhiteWin - usi_engine_turn) || pos_root->turn() == White && gameResult == (BlackWin + usi_engine_turn))) {
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} usi win", grp->gpu_id, grp->group_id, id, ply);
			++usi_wins;
		}
		else if (gameResult == Draw) {
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} usi draw", grp->gpu_id, grp->group_id, id, ply);
			++usi_draws;
		}
		else {
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} usi lose", grp->gpu_id, grp->group_id, id, ply);
		}
	}

	// すぐに終局した初期局面を出力する
	if (OUT_MIN_HCP && ply < MIN_MOVE) {
		std::unique_lock<Mutex> lock(omutex);
		ofs_minhcp.write(reinterpret_cast<char*>(&hcp), sizeof(HuffmanCodedPos));
	}

	// 新しいゲーム
	playout = 0;
	ply = 0;
}

// 教師局面生成
void make_teacher(const char* recordFileName, const char* outputFileName, const vector<int>& gpu_id, const vector<int>& batchsize)
{
	s.init();

	// 初期局面集
	ifs.open(recordFileName, ifstream::in | ifstream::binary | ios::ate);
	if (!ifs) {
		cerr << "Error: cannot open " << recordFileName << endl;
		exit(EXIT_FAILURE);
	}
	entryNum = ifs.tellg() / sizeof(HuffmanCodedPos);

	// 教師局面を保存するファイル
	ofs.open(outputFileName, ios::binary);
	if (!ofs) {
		cerr << "Error: cannot open " << outputFileName << endl;
		exit(EXIT_FAILURE);
	}
	// 削除候補の初期局面を出力するファイル
	if (OUT_MIN_HCP) ofs_minhcp.open(string(outputFileName) + "_min.hcp", ios::binary);

	vector<UCTSearcherGroupPair> group_pairs;
	group_pairs.reserve(gpu_id.size());
	for (size_t i = 0; i < gpu_id.size(); i++)
		group_pairs.emplace_back(gpu_id[i], batchsize[i]);

	// 探索スレッド開始
	for (size_t i = 0; i < group_pairs.size(); i++)
		group_pairs[i].Run();

	// 進捗状況表示
	auto progressFunc = [&gpu_id, &group_pairs](Timer& t) {
		ostringstream ss;
		for (size_t i = 0; i < gpu_id.size(); i++) {
			if (i > 0) ss << " ";
			ss << gpu_id[i];
		}
		while (!stopflg) {
			std::this_thread::sleep_for(std::chrono::seconds(10)); // 指定秒だけ待機し、進捗を表示する。
			const double progress = static_cast<double>(madeTeacherNodes) / teacherNodes;
			auto elapsed_msec = t.elapsed();
			if (progress > 0.0) // 0 除算を回避する。
				logger->info("Progress:{:.2f}%, nodes:{}, nodes/sec:{:.2f}, games:{}, draw:{}, nyugyoku:{}, ply/game:{:.2f}, playouts/node:{:.2f} gpu id:{}, usi_games:{}, usi_win:{}, usi_draw:{}, Elapsed:{}[s], Remaining:{}[s]",
					std::min(100.0, progress * 100.0),
					idx,
					static_cast<double>(idx) / elapsed_msec * 1000.0,
					games,
					draws,
					nyugyokus,
					static_cast<double>(madeTeacherNodes) / games,
					static_cast<double>(sum_playouts) / sum_nodes,
					ss.str(),
					usi_games,
					usi_wins,
					usi_draws,
					elapsed_msec / 1000,
					std::max<s64>(0, (s64)(elapsed_msec*(1.0 - progress) / (progress * 1000))));
			int running = 0;
			for (size_t i = 0; i < group_pairs.size(); i++)
				running += group_pairs[i].Running();
			if (running == 0)
				break;
		}
	};

	while (!stopflg) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		int running = 0;
		for (size_t i = 0; i < group_pairs.size(); i++)
			running += group_pairs[i].Running();
		if (running > 0)
			break;
	}
	Timer t = Timer::currentTime();
	std::thread progressThread([&progressFunc, &t] { progressFunc(t); });

	// 探索スレッド終了待機
	for (size_t i = 0; i < group_pairs.size(); i++)
		group_pairs[i].Join();

	progressThread.join();
	ifs.close();
	ofs.close();
	if (OUT_MIN_HCP) ofs_minhcp.close();

	logger->info("Made {} teacher nodes in {} seconds. games:{}, draws:{}, ply/game:{}, usi_games:{}, usi_win:{}, usi_draw:{}, usi_winrate:{:.2f}%",
		madeTeacherNodes, t.elapsed() / 1000,
		games,
		draws,
		static_cast<double>(madeTeacherNodes) / games,
		usi_games,
		usi_wins,
		usi_draws,
		static_cast<double>(usi_wins) / (usi_games - usi_draws) * 100);
}

int main(int argc, char* argv[]) {
	std::string recordFileName;
	std::string outputFileName;
	vector<int> gpu_id(1);
	vector<int> batchsize(1);

	cxxopts::Options options("selfplay");
	options.positional_help("modelfile hcp output nodes playout_num gpu_id batchsize [gpu_id batchsize]*");
	try {
		options.add_options()
			("modelfile", "model file path", cxxopts::value<std::string>(model_path))
			("hcp", "initial position file", cxxopts::value<std::string>(recordFileName))
			("output", "output file path", cxxopts::value<std::string>(outputFileName))
			("nodes", "nodes", cxxopts::value<s64>(teacherNodes))
			("playout_num", "playout number", cxxopts::value<int>(playout_num))
			("gpu_id", "gpu id", cxxopts::value<int>(gpu_id[0]))
			("batchsize", "batchsize", cxxopts::value<int>(batchsize[0]))
			("positional", "", cxxopts::value<std::vector<int>>())
			("threads", "thread number", cxxopts::value<int>(threads)->default_value("2"), "num")
			("random", "random move number", cxxopts::value<int>(RANDOM_MOVE)->default_value("4"), "num")
			("random_cutoff", "random cutoff ratio", cxxopts::value<float>(RANDOM_CUTOFF)->default_value("0.9"))
			("random_temperature", "random temperature", cxxopts::value<float>(RANDOM_TEMPERATURE)->default_value("1.0"))
			("random2", "random2", cxxopts::value<float>(RANDOM2)->default_value("0"))
			("min_move", "minimum move number", cxxopts::value<int>(MIN_MOVE)->default_value("10"), "num")
			("max_move", "maximum move number", cxxopts::value<int>(MAX_MOVE)->default_value("320"), "num")
			("out_max_move", "output the max move game", cxxopts::value<bool>(OUT_MAX_MOVE)->default_value("false"))
			("root_noise", "add noise to the policy prior at the root", cxxopts::value<int>(ROOT_NOISE)->default_value("3"), "per mille")
			("threshold", "winrate threshold", cxxopts::value<float>(WINRATE_THRESHOLD)->default_value("0.99"), "rate")
			("mate_depth", "mate search depth", cxxopts::value<uint32_t>(ROOT_MATE_SEARCH_DEPTH)->default_value("0"), "depth")
			("mate_nodes", "mate search max nodes", cxxopts::value<int64_t>(MATE_SEARCH_MAX_NODE)->default_value("100000"), "nodes")
			("c_init", "UCT parameter c_init", cxxopts::value<float>(c_init)->default_value("1.49"), "val")
			("c_base", "UCT parameter c_base", cxxopts::value<float>(c_base)->default_value("39470.0"), "val")
			("c_fpu_reduction", "UCT parameter c_fpu_reduction", cxxopts::value<float>(c_fpu_reduction)->default_value("20"), "val")
			("c_init_root", "UCT parameter c_init_root", cxxopts::value<float>(c_init_root)->default_value("1.49"), "val")
			("c_base_root", "UCT parameter c_base_root", cxxopts::value<float>(c_base_root)->default_value("39470.0"), "val")
			("temperature", "Softmax temperature", cxxopts::value<float>(temperature)->default_value("1.66"), "val")
			("reuse", "reuse sub tree", cxxopts::value<bool>(REUSE_SUBTREE)->default_value("false"))
			("out_min_hcp", "output minimum move hcp", cxxopts::value<bool>(OUT_MIN_HCP)->default_value("false"))
			("nn_cache_size", "nn cache size", cxxopts::value<unsigned int>(nn_cache_size)->default_value("8388608"))
			("usi_engine", "USIEngine exe path", cxxopts::value<std::string>(usi_engine_path))
			("usi_engine_num", "USIEngine number", cxxopts::value<int>(usi_engine_num)->default_value("0"), "num")
			("usi_threads", "USIEngine thread number", cxxopts::value<int>(usi_threads)->default_value("1"), "num")
			("usi_options", "USIEngine options", cxxopts::value<std::string>(usi_options))
			("usi_byoyomi", "USI byoyomi", cxxopts::value<int>(usi_byoyomi)->default_value("500"))
			("usi_turn", "USIEngine turn", cxxopts::value<int>(usi_turn)->default_value("-1"))
			("h,help", "Print help")
			;
		options.parse_positional({ "modelfile", "hcp", "output", "nodes", "playout_num", "gpu_id", "batchsize", "positional" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help({}) << std::endl;
			return 0;
		}

		const size_t positional_count = result.count("positional");
		if (positional_count > 0) {
			if (positional_count % 2 == 1) {
				throw cxxopts::option_required_exception("batchsize");
			}
			auto positional = result["positional"].as<std::vector<int>>();
			for (size_t i = 0; i < positional_count; i += 2) {
				gpu_id.push_back(positional[i]);
				batchsize.push_back(positional[i + 1]);
			}
		}
	}
	catch (cxxopts::OptionException &e) {
		std::cout << options.usage() << std::endl;
		std::cerr << e.what() << std::endl;
		return 0;
	}

	if (teacherNodes <= 0) {
		cerr << "too few teacherNodes" << endl;
		return 0;
	}
	if (playout_num <= 0) {
		cerr << "too few playout_num" << endl;
		return 0;
	}
	if (threads < 0) {
		cerr << "too few threads number" << endl;
		return 0;
	}
	if (RANDOM_MOVE < 0) {
		cerr << "too few random move number" << endl;
		return 0;
	}
	if (MIN_MOVE <= 0) {
		cerr << "too few min_move" << endl;
		return 0;
	}
	if (MAX_MOVE <= MIN_MOVE) {
		cerr << "too few max_move" << endl;
		return 0;
	}
	if (MAX_MOVE >= 1000) {
		cerr << "too large max_move" << endl;
		return 0;
	}
	if (ROOT_NOISE < 0) {
		cerr << "too few root_noise" << endl;
		return 0;
	}
	if (WINRATE_THRESHOLD <= 0) {
		cerr << "too few threshold" << endl;
		return 0;
	}
	if (MATE_SEARCH_MAX_NODE < MATE_SEARCH_MIN_NODE) {
		cerr << "too few mate nodes" << endl;
		return 0;
	}
	if (usi_engine_num < 0) {
		cerr << "too few usi_engine_num" << endl;
		return 0;
	}
	if (usi_threads < 0) {
		cerr << "too few usi_threads" << endl;
		return 0;
	}

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
	logger->set_level(spdlog::level::trace);
	logger->info("modelfile:{} roots.hcp:{} output:{} nodes:{} playout_num:{}", model_path, recordFileName, outputFileName, teacherNodes, playout_num);

	for (size_t i = 0; i < gpu_id.size(); i++) {
		logger->info("gpu_id:{} batchsize:{}", gpu_id[i], batchsize[i]);

		if (gpu_id[i] < 0) {
			cerr << "invalid gpu id" << endl;
			return 0;
		}
		if (batchsize[i] <= 0) {
			cerr << "too few batchsize" << endl;
			return 0;
		}
	}

	logger->info("threads:{}", threads);
	logger->info("random:{}", RANDOM_MOVE);
	logger->info("random_cutoff:{}", RANDOM_CUTOFF);
	logger->info("random_temperature:{}", RANDOM_TEMPERATURE);
	logger->info("random2:{}", RANDOM2);
	logger->info("min_move:{}", MIN_MOVE);
	logger->info("max_move:{}", MAX_MOVE);
	logger->info("out_max_move:{}", OUT_MAX_MOVE);
	logger->info("root_noise:{}", ROOT_NOISE);
	logger->info("threshold:{}", WINRATE_THRESHOLD);
	logger->info("mate depth:{}", ROOT_MATE_SEARCH_DEPTH);
	logger->info("mate nodes:{}", MATE_SEARCH_MAX_NODE);
	logger->info("c_init:{}", c_init);
	logger->info("c_base:{}", c_base);
	logger->info("c_fpu_reduction:{}", c_fpu_reduction);
	logger->info("c_init_root:{}", c_init_root);
	logger->info("c_base_root:{}", c_base_root);
	logger->info("temperature:{}", temperature);
	logger->info("reuse:{}", REUSE_SUBTREE);
	logger->info("nn_cache_size:{}", nn_cache_size);
	if (OUT_MIN_HCP) logger->info("out_min_hcp");
	logger->info("usi_engine:{}", usi_engine_path);
	logger->info("usi_engine_num:{}", usi_engine_num);
	logger->info("usi_threads:{}", usi_threads);
	logger->info("usi_options:{}", usi_options);
	logger->info("usi_byoyomi:{}", usi_byoyomi);
	logger->info("usi_turn:{}", usi_turn);

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

	set_softmax_temperature(temperature);

	signal(SIGINT, sigint_handler);

	logger->info("make_teacher");
	make_teacher(recordFileName.c_str(), outputFileName.c_str(), gpu_id, batchsize);

	spdlog::drop_all();
}
