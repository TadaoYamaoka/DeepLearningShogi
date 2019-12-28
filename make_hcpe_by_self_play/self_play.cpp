#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <signal.h>

#include "UctSearch.h"
#include "ZobristHash.h"
#include "LruCache.h"
#include "mate.h"
#include "nn_wideresnet10.h"
#include "nn_wideresnet15.h"
#include "nn_senet10.h"
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

volatile sig_atomic_t stopflg = false;

void sigint_handler(int signum)
{
	stopflg = true;
}

// ランダムムーブの手数
int RANDOM_MOVE;

// 終局とする勝率の閾値
float WINRATE_THRESHOLD;

// 詰み探索の深さ
uint32_t ROOT_MATE_SEARCH_DEPTH;
// 詰み探索の最大ノード数
int64_t MATE_SEARCH_MAX_NODE;
// 詰み探索の上限ノード数をランダムに決める
bool MATE_SEARCH_RAND_LIMIT_NODES = false;
constexpr int64_t MATE_SEARCH_MIN_NODE = 10000;

// モデルのパス
string model_path;

// 探索打ち止めの設定
constexpr int max_playout_num = 10000;
constexpr int interruption_interval = 100; // チェック間隔
constexpr double min_gain = 0.0001 * interruption_interval;
std::atomic<int> max_playout = 0;

// USIエンジンのパス
string usi_engine_path;
int usi_engine_id;
int usi_engine_num;
// USIエンジンオプション（name:value,...,name:value）
string usi_options;
int usi_byoyomi;

constexpr unsigned int uct_hash_size = 524288; // UCTハッシュサイズ
constexpr int MAX_PLY = 320; // 最大手数

struct CachedNNRequest {
	CachedNNRequest(size_t size) : nnrate(size) {}
	float value_win;
	std::vector<float> nnrate;
};
typedef LruCache<uint64_t, CachedNNRequest> NNCache;
typedef LruCacheLock<uint64_t, CachedNNRequest> NNCacheLock;
constexpr unsigned int nn_cache_size = 1048576; // NNキャッシュサイズ

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
//ofstream ofs_dup;
mutex imutex;
mutex omutex;
size_t entryNum;

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
constexpr int MATE_SEARCH_DEPTH = 7;

// 詰み探索で詰みの場合のvalue_winの定数
constexpr float VALUE_WIN = FLT_MAX;
constexpr float VALUE_LOSE = -FLT_MAX;

// 探索の結果を評価のキューに追加したか、破棄したか
constexpr float QUEUING = FLT_MAX;
constexpr float DISCARDED = -FLT_MAX;

float c_init = 1.49f;
float c_base = 39470.0f;
float temperature = 1.66f;


// 探索経路のノード
struct TrajectorEntry {
	TrajectorEntry(uct_node_t* uct_node, unsigned int current, unsigned int next_index) : uct_node(uct_node), current(current), next_index(next_index) {}
	uct_node_t* uct_node;
	unsigned int current;
	unsigned int next_index;
};

void UpdateResult(uct_node_t* uct_node, child_node_t *child, const float result, const unsigned int current);
bool nyugyoku(const Position& pos);

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
		uct_hash(uct_hash_size),
		uct_node(new uct_node_t[uct_hash_size]),
		nn_cache(nn_cache_size),
		current_policy_value_batch_index(0), features1(nullptr), features2(nullptr), policy_value_hash_index(nullptr), y1(nullptr), y2(nullptr), running(false) {
		Initialize();
	}
	UCTSearcherGroup(UCTSearcherGroup&& o) {} // not use
	~UCTSearcherGroup() {
		delete[] uct_node;
		delete[] mate_search_slot;
		delete[] mate_search_limit_nodes;
		checkCudaErrors(cudaFreeHost(features1));
		checkCudaErrors(cudaFreeHost(features2));
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
	}

	void QueuingNode(const Position *pos, unsigned int index);
	void EvalNode();
	void SelfPlay();
	void Run();
	void Join();
	void SetMateSearchLimitNodes(int limit_nodes, const int id) {
		mate_search_limit_nodes[id] = limit_nodes;
	}
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
	vector<USIEngine> engines;

private:
	void Initialize();

	UCTSearcherGroupPair* parent;

	// ハッシュ
	UctHash uct_hash;
	uct_node_t* uct_node;

	// NNキャッシュ
	NNCache nn_cache;

	// キュー
	int policy_value_batch_maxsize; // 最大バッチサイズ
	features1_t* features1;
	features2_t* features2;
	unsigned int* policy_value_hash_index;
	int current_policy_value_batch_index;

	// UCTSearcher
	vector<UCTSearcher> searchers;
	thread* handle_selfplay;

	DType* y1;
	DType* y2;

	// 詰み探索
	ns_dfpn::DfPn dfpn;
	MateSearchEntry* mate_search_slot = nullptr;
	int* mate_search_limit_nodes = nullptr;
	deque<int> mate_search_queue;
	mutex mate_search_mutex;
	thread* handle_mate_search;
};

class UCTSearcher {
public:
	UCTSearcher(UCTSearcherGroup* grp, UctHash& uct_hash, uct_node_t* uct_node, NNCache& nn_cache, const int id, const size_t entryNum) :
		grp(grp),
		uct_hash(uct_hash),
		uct_node(uct_node),
		nn_cache(nn_cache),
		id(id),
		mt_64(new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + id)),
		mt(new std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() + id)),
		inputFileDist(0, entryNum - 1),
		playout(0),
		ply(0) {
		pos_root = new Position(DefaultStartPositionSFEN, s.thisptr);
		need_usi_engine = grp->engines.size() > 0 && id < usi_engine_num;
	}
	UCTSearcher(UCTSearcher&& o) : uct_hash(o.uct_hash), nn_cache(o.nn_cache) {} // not use
	~UCTSearcher() {
		delete pos_root;
	}

	void Playout(vector<TrajectorEntry>& trajectories);
	void NextStep();

private:
	float UctSearch(Position* pos, unsigned int current, const int depth, vector<TrajectorEntry>& trajectories, bool& queued);
	int SelectMaxUcbChild(const Position* pos, unsigned int current, const int depth);
	unsigned int ExpandRoot(const Position* pos);
	unsigned int ExpandNode(Position* pos, const int depth);
	bool InterruptionCheck(const unsigned int current_root, const int playout_count);
	void NextPly(const Move move);
	void NextGame();

	// 局面追加
	void AddTeacher(s16 eval, Move move) {
		hcpevec.emplace_back(HuffmanCodedPosAndEval());
		HuffmanCodedPosAndEval& hcpe = hcpevec.back();
		hcpe.hcp = pos_root->toHuffmanCodedPos();
		const Color rootTurn = pos_root->turn();
		hcpe.eval = eval;
		hcpe.bestMove16 = static_cast<u16>(move.value());
		idx++;
	}

	UCTSearcherGroup* grp;
	int id;
	unique_ptr<std::mt19937_64> mt_64;
	unique_ptr<std::mt19937> mt;
	// スレッドのハンドル
	thread *handle;

	// ハッシュ(UCTSearcherGroupで共有)
	UctHash& uct_hash;
	uct_node_t* uct_node;

	// NNキャッシュ(UCTSearcherGroupで共有)
	NNCache& nn_cache;

	int playout;
	int ply;
	GameResult gameResult;
	unsigned int current_root;

	StateInfo states[MAX_PLY + 1];
	std::vector<HuffmanCodedPosAndEval> hcpevec;
	uniform_int_distribution<s64> inputFileDist;

	// 局面管理と探索スレッド
	Position* pos_root;

	// 詰み探索のステータス
	MateSearchEntry::State mate_status;

	HuffmanCodedPos hcp;

	// 探索打ち止め用の前回の訪問回数
	int prev_visits[UCT_CHILD_MAX];

	// USIエンジン
	bool need_usi_engine = false;
	std::string usi_position;
};

class UCTSearcherGroupPair {
public:
	static const int threads = 2;

	UCTSearcherGroupPair(const int gpu_id, const int policy_value_batch_maxsize) : nn(nullptr), policy_value_batch_maxsize(policy_value_batch_maxsize) {
		groups.reserve(threads);
		for (int i = 0; i < threads; i++)
			groups.emplace_back(gpu_id, i, policy_value_batch_maxsize, this);
	}
	UCTSearcherGroupPair(UCTSearcherGroupPair&& o) {} // not use
	~UCTSearcherGroupPair() {
		delete nn;
	}
	void InitGPU() {
		mutex_gpu.lock();
		if (nn == nullptr) {
			if (model_path.find("wideresnet15") != string::npos) {
				nn = (NN*)new NNWideResnet15(policy_value_batch_maxsize);
			}
			else if (model_path.find("senet10") != string::npos) {
				nn = (NN*)new NNSENet10(policy_value_batch_maxsize);
			}
			else {
				nn = (NN*)new NNWideResnet10(policy_value_batch_maxsize);
			}
			nn->load_model(model_path.c_str());
		}
		mutex_gpu.unlock();
	}
	void nn_forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2) {
		mutex_gpu.lock();
		nn->foward(batch_size, x1, x2, y1, y2);
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
	if (usi_engine_id == gpu_id && group_id == 0 && usi_engine_path != "" && usi_engine_num != 0) {
		std::vector<std::pair<std::string, std::string>> options;
		std::istringstream ss(usi_options);
		std::string field;
		while (std::getline(ss, field, ',')) {
			const auto pos = field.find_first_of(":");
			options.emplace_back(field.substr(0, pos), field.substr(pos + 1));
		}
		engines.reserve(usi_engine_num);
		for (int i = 0; i < usi_engine_num; ++i)
			engines.emplace_back(usi_engine_path, options);
	}

	// キューを動的に確保する
	checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	policy_value_hash_index = new unsigned int[policy_value_batch_maxsize];

	// UCTSearcher
	searchers.clear();
	searchers.reserve(policy_value_batch_maxsize);
	for (int i = 0; i < policy_value_batch_maxsize; i++) {
		searchers.emplace_back(this, uct_hash, uct_node, nn_cache, i, entryNum);
	}

	checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));

	// 詰み探索
	if (ROOT_MATE_SEARCH_DEPTH > 0) {
		dfpn.init();
		dfpn.set_max_search_node(MATE_SEARCH_MAX_NODE);
		mate_search_slot = new MateSearchEntry[policy_value_batch_maxsize];
		if (MATE_SEARCH_RAND_LIMIT_NODES) {
			mate_search_limit_nodes = new int[policy_value_batch_maxsize];
		}
	}
}

// 連続自己対局
void UCTSearcherGroup::SelfPlay()
{
	// スレッドにGPUIDを関連付けてから初期化する
	cudaSetDevice(gpu_id);

	parent->InitGPU();

	// 探索経路のバッチ
	vector<vector<TrajectorEntry>> trajectories_batch(policy_value_batch_maxsize);

	// 全スレッドが生成した局面数が生成局面数以上になったら終了
	while (madeTeacherNodes < teacherNodes && !stopflg) {
		current_policy_value_batch_index = 0;

		// すべての対局についてシミュレーションを行う
		for (size_t i = 0; i < (size_t)policy_value_batch_maxsize; i++) {
			UCTSearcher& searcher = searchers[i];
			searcher.Playout(trajectories_batch[i]);
		}

		// 評価
		EvalNode();

		// バックアップ
		float result = 0.0f;
		for (auto& trajectories : trajectories_batch) {
			for (int i = trajectories.size() - 1; i >= 0; i--) {
				TrajectorEntry& current_next = trajectories[i];
				uct_node_t* uct_node = current_next.uct_node;
				const unsigned int current = current_next.current;
				const unsigned int next_index = current_next.next_index;
				child_node_t* uct_child = uct_node[current].child;
				if ((size_t)i == trajectories.size() - 1) {
					const unsigned int child_index = uct_child[next_index].index;
					const float value_win = uct_node[child_index].value_win;
					result = 1.0f - value_win;
				}
				UpdateResult(uct_node, &uct_child[next_index], result, current);
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

			// 詰み探索の上限ノード数をランダムに決めた場合
			if (MATE_SEARCH_RAND_LIMIT_NODES) {
				dfpn.set_max_search_node(mate_search_limit_nodes[id]);
			}

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
UCTSearcher::UctSearch(Position *pos, unsigned int current, const int depth, vector<TrajectorEntry>& trajectories, bool& queued)
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

	float result;
	unsigned int next_index;
	child_node_t *uct_child = uct_node[current].child;

	// UCB値最大の手を求める
	next_index = SelectMaxUcbChild(pos, current, depth);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	// 経路を記録
	trajectories.emplace_back(uct_node, current, next_index);

	// ノードの展開の確認
	if (uct_child[next_index].index == NOT_EXPANDED) {
		// ノードの展開
		unsigned int child_index = ExpandNode(pos, depth + 1);
		uct_child[next_index].index = child_index;
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

		// 合流検知
		if (uct_node[child_index].evaled) {
			// 手番を入れ替えて1手深く読む
			result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories, queued);
		}
		else if (uct_node[child_index].child_num == 0) {
			// 詰み
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
				NNCacheLock cache_lock(&nn_cache, uct_node[child_index].key);
				if (!cache_lock) {
					grp->QueuingNode(pos, child_index);
					queued = true;
				}
			}
			else if (nn_cache.ContainsKey(uct_node[child_index].key)) {
				NNCacheLock cache_lock(&nn_cache, uct_node[child_index].key);
				const float value_win = cache_lock->value_win;
				// キャッシュヒット
				if (value_win == VALUE_WIN) {
					uct_node[child_index].value_win = VALUE_WIN;
					result = 0.0f;
				}
				else if (value_win == VALUE_LOSE) {
					uct_node[child_index].value_win = VALUE_LOSE;
					result = 1.0f;
				}
				else
					result = 1.0f - value_win;
			}
			else {
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

				// 入玉勝ちかどうかを判定
				if (nyugyoku(*pos)) {
					isMate = 1;
				}

				// 詰みの場合、ValueNetの値を上書き
				if (isMate == 1) {
					auto req = make_unique<CachedNNRequest>(0);
					req->value_win = VALUE_WIN;
					nn_cache.Insert(uct_node[child_index].key, std::move(req));
					uct_node[child_index].value_win = VALUE_WIN;
					result = 0.0f;
				}
				else if (isMate == -1) {
					auto req = make_unique<CachedNNRequest>(0);
					req->value_win = VALUE_LOSE;
					nn_cache.Insert(uct_node[child_index].key, std::move(req));
					uct_node[child_index].value_win = VALUE_LOSE;
					// 子ノードに一つでも負けがあれば、自ノードを勝ちにできる
					uct_node[current].value_win = VALUE_WIN;
					result = 1.0f;
				}
				else {
					// ノードをキューに追加
					grp->QueuingNode(pos, child_index);
					queued = true;
					return QUEUING;
				}
			}
		}
	}
	else {
		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories, queued);
	}

	if (result == QUEUING)
		return result;

	// 探索結果の反映
	UpdateResult(uct_node, &uct_child[next_index], result, current);

	return 1.0f - result;
}

//////////////////////
//  探索結果の更新  //
/////////////////////
void UpdateResult(uct_node_t* uct_node, child_node_t *child, const float result, const unsigned int current)
{
	uct_node[current].win += result;
	uct_node[current].move_count++;
	child->win += result;
	child->move_count++;
}

/////////////////////////////////////////////////////
//  UCBが最大となる子ノードのインデックスを返す関数  //
/////////////////////////////////////////////////////
int
UCTSearcher::SelectMaxUcbChild(const Position *pos, unsigned int current, const int depth)
{
	child_node_t *uct_child = uct_node[current].child;
	const int child_num = uct_node[current].child_num;
	int max_child = 0;
	const int sum = uct_node[current].move_count;
	float q, u, max_value;
	float ucb_value;
	int child_win_count = 0;

	max_value = -1;

	NNCacheLock cache_lock(&nn_cache, uct_node[current].key);

	// UCB値最大の手を求める
	for (int i = 0; i < child_num; i++) {
		if (uct_child[i].index != NOT_EXPANDED) {
			const float child_value_win = uct_node[uct_child[i].index].value_win;
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

		float win = uct_child[i].win;
		int move_count = uct_child[i].move_count;

		if (move_count == 0) {
			q = 0.5f;
			u = sum == 0 ? 1.0f : sqrtf(sum);
		}
		else {
			q = win / move_count;
			u = sqrtf(sum) / (1 + move_count);
		}

		float rate = cache_lock->nnrate[i];
		// ランダムに確率を上げる
		if (depth == 0 && rnd(*mt) <= 2) {
			rate = (rate + 1.0f) / 2.0f;
		}

		const float c = logf((sum + c_base + 1.0f) / c_base) + c_init;
		ucb_value = q + c * u * rate;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	if (child_win_count == child_num) {
		// 子ノードがすべて勝ちのため、自ノードを負けにする
		uct_node[current].value_win = VALUE_LOSE;
	}

	return max_child;
}

// 初期設定
void
InitializeUctSearch()
{
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
}

/////////////////////////
//  ルートノードの展開  //
/////////////////////////
unsigned int
UCTSearcher::ExpandRoot(const Position *pos)
{
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->gamePly(), id);
	child_node_t *uct_child;
	int child_num = 0;

	// 既に展開されていた時は, 探索結果を再利用する
	if (index != uct_hash_size) {
		return index;
	}
	else {
		// 空のインデックスを探す
		index = uct_hash.SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly(), id);

		assert(index != uct_hash_size);

		// ルートノードの初期化
		uct_node[index].move_count = 0;
		uct_node[index].win = 0;
		uct_node[index].child_num = 0;
		uct_node[index].evaled = false;
		uct_node[index].draw = false;
		uct_node[index].value_win = 0.0f;
		uct_node[index].key = pos->getKey();

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
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->gamePly() + depth, id);
	child_node_t *uct_child;

	// 合流先が検知できれば, それを返す
	if (index != uct_hash_size) {
		return index;
	}

	// 空のインデックスを探す
	index = uct_hash.SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth, id);

	assert(index != uct_hash_size);

	// 現在のノードの初期化
	uct_node[index].move_count = 0;
	uct_node[index].win = 0;
	uct_node[index].child_num = 0;
	uct_node[index].evaled = false;
	uct_node[index].draw = false;
	uct_node[index].value_win = 0.0f;
	uct_node[index].key = pos->getKey();
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
UCTSearcherGroup::QueuingNode(const Position *pos, unsigned int index)
{
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
bool
UCTSearcher::InterruptionCheck(const unsigned int current_root, const int playout_count)
{
	if (playout_count >= max_playout_num || playout_count % interruption_interval != 0) return false;

	const int child_num = uct_node[current_root].child_num;
	child_node_t *uct_child = uct_node[current_root].child;
	int new_visits[UCT_CHILD_MAX];

	// ルートの子ノードの訪問回数を取得
	for (int i = 0; i < child_num; i++) {
		new_visits[i] = uct_child[i].move_count;
	}

	if (playout_count > interruption_interval) {
		double kldgain = 0.0;
		const double prev_playout_count = (double)(playout_count - interruption_interval);
		const double new_playout_count = (double)playout_count;
		for (int i = 0; i < child_num; i++) {
			double o_p = prev_visits[i] / prev_playout_count;
			double n_p = new_visits[i] / new_playout_count;
			if (prev_visits[i] != 0) kldgain += o_p * log(o_p / n_p);
		}
		if (kldgain < min_gain) {
			if (playout_count > max_playout) max_playout = playout_count;
			return true;
		}
	}

	std::memcpy(prev_visits, new_visits, child_num * sizeof(int));
	return false;
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
		const unsigned int index = policy_value_hash_index[i];

		// 対局は1スレッドで行うためロックは不要
		const int child_num = uct_node[index].child_num;
		child_node_t *uct_child = uct_node[index].child;
		Color color = uct_hash[index].color;

		// 合法手一覧
		vector<float> legal_move_probabilities;
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

		auto req = make_unique<CachedNNRequest>(child_num);
		for (int j = 0; j < child_num; j++) {
			req->nnrate[j] = legal_move_probabilities[j];
		}

#ifdef FP16
		const float value_win = __half2float(*value);
#else
		const float value_win = *value;
#endif

		req->value_win = value_win;
		nn_cache.Insert(uct_node[index].key, std::move(req));

		uct_node[index].value_win = value_win;

		uct_node[index].evaled = true;
	}
}

// シミュレーションを1回行う
void UCTSearcher::Playout(vector<TrajectorEntry>& trajectories)
{
	while (true) {
		trajectories.clear();
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

				hcpevec.clear();

				// 詰み探索の上限ノード数をランダムに決める
				if (MATE_SEARCH_RAND_LIMIT_NODES) {
					std::uniform_int_distribution<> rand(MATE_SEARCH_MIN_NODE, MATE_SEARCH_MAX_NODE);
					grp->SetMateSearchLimitNodes(rand(*mt_64), id);
				}

				// USIエンジン
				if (need_usi_engine) {
					// 開始局面設定
					usi_position = "position " + pos_root->toSFEN() + " moves";
				}
			}
			else if (need_usi_engine && ply % 2 == 0 && ply > RANDOM_MOVE) {
				return;
			}

			// ハッシュクリア
			uct_hash.ClearUctHash(id);

			// ルートノード展開
			current_root = ExpandRoot(pos_root);

			// 詰みのチェック
			if (uct_node[current_root].child_num == 0) {
				gameResult = (pos_root->turn() == Black) ? GameResult::WhiteWin : GameResult::BlackWin;
				NextGame();
				continue;
			}
			else if (uct_node[current_root].child_num == 1) {
				// 1手しかないときは、その手を指して次の手番へ
				const Move move = uct_node[current_root].child[0].move;
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} skip:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), move.toUSI());
				NextPly(move);
				continue;
			}
			else if (nyugyoku(*pos_root)) {
				// 入玉宣言勝ち
				gameResult = (pos_root->turn() == Black) ? GameResult::BlackWin : GameResult::WhiteWin;
				if (hcpevec.size() > 0)
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
			NNCacheLock cache_lock(&nn_cache, uct_node[current_root].key);
			if (!cache_lock || cache_lock->nnrate.size() == 0) {
				grp->QueuingNode(pos_root, current_root);
				return;
			}
			else {
				NextStep();
				continue;
			}
		}

		// 盤面のコピー
		Position pos_copy(*pos_root);
		// プレイアウト
		bool queued = false;
		UctSearch(&pos_copy, current_root, 0, trajectories, queued);
		if (!queued) {
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
	if (need_usi_engine && ply % 2 == 0 && ply > RANDOM_MOVE) {
		const Move move = grp->engines[id].ThinkDone();
		if (move == Move::moveNone())
			return;

		if (move == moveResign()) {
			gameResult = (pos_root->turn() == Black) ? GameResult::WhiteWin : GameResult::BlackWin;
			NextGame();
			return;
		}
		else if (move == moveWin()) {
			gameResult = (pos_root->turn() == Black) ? GameResult::BlackWin : GameResult::WhiteWin;
			NextGame();
			return;
		}
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} usi_move:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), move.toUSI());

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
					AddTeacher(30000, grp->GetMateSearchMove(id));

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
	if (InterruptionCheck(current_root, playout)) {
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
					AddTeacher(30000, grp->GetMateSearchMove(id));

				NextGame();
				return;
			case MateSearchEntry::LOSE:
				SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} mate lose", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
				gameResult = (pos_root->turn() == Black) ? WhiteWin : BlackWin;
				NextGame();
				return;
			}
		}

		child_node_t* uct_child = uct_node[current_root].child;
		unsigned int select_index = 0;
		Move best_move;
		if (ply <= RANDOM_MOVE) {
			// N手までは訪問数に応じた確率で選択する
			vector<int> probabilities(uct_node[current_root].child_num);
			for (int i = 0; i < uct_node[current_root].child_num; i++) {
				probabilities[i] = uct_child[i].move_count;
				SPDLOG_TRACE(logger, "gpu_id:{} group_id:{} id:{} {}:{} move_count:{} win_rate:{}", grp->gpu_id, grp->group_id, id, i, uct_child[i].move.toUSI(), uct_child[i].move_count, uct_child[i].win / (uct_child[i].move_count + 0.0001f));
			}

			discrete_distribution<unsigned int> dist(probabilities.begin(), probabilities.end());
			select_index = dist(*mt_64);
			best_move = uct_child[select_index].move;
			SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} random_move:{}", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN(), best_move.toUSI());
		}
		else {
			// 探索回数最大の手を見つける
			int max_count = uct_child[0].move_count;
			int child_win_count = 0;
			int child_lose_count = 0;
			SPDLOG_TRACE(logger, "gpu_id:{} group_id:{} id:{} {}:{} move_count:{} win_rate:{}", grp->gpu_id, grp->group_id, id, 0, uct_child[0].move.toUSI(), uct_child[0].move_count, uct_child[0].win / (uct_child[0].move_count + 0.0001f));
			const int child_num = uct_node[current_root].child_num;
			for (int i = 1; i < child_num; i++) {
				if (uct_child[i].index != NOT_EXPANDED) {
					const float child_value_win = uct_node[uct_child[i].index].value_win;
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

				if (uct_child[i].move_count > max_count) {
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				SPDLOG_TRACE(logger, "gpu_id:{} group_id:{} id:{} {}:{} move_count:{} win_rate:{}", grp->gpu_id, grp->group_id, id, i, uct_child[i].move.toUSI(), uct_child[i].move_count, uct_child[i].win / (uct_child[i].move_count + 0.0001f));
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
			AddTeacher(eval, best_move);

			// 一定の手数以上で引き分け
			if (ply >= MAX_PLY) {
				gameResult = Draw;
				NextGame();
				return;
			}
		}

		NextPly(best_move);
	}
}

void UCTSearcher::NextPly(const Move move)
{
	// 着手
	pos_root->doMove(move, states[ply]);
	pos_root->setStartPosPly(ply + 1);

	// 千日手の場合
	switch (pos_root->isDraw(16)) {
	case RepetitionDraw:
		SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} {} RepetitionDraw", grp->gpu_id, grp->group_id, id, ply, pos_root->toSFEN());
		gameResult = Draw;
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
	playout = 0;
	ply++;

	if (need_usi_engine) {
		usi_position += " " + move.toUSI();
		if (ply % 2 == 0)
			grp->engines[id].ThinkAsync(*pos_root, usi_position, usi_byoyomi);
	}
}

void UCTSearcher::NextGame()
{
	SPDLOG_DEBUG(logger, "gpu_id:{} group_id:{} id:{} ply:{} gameResult:{}", grp->gpu_id, grp->group_id, id, ply, gameResult);
	// 勝敗を1局全てに付ける。
	for (auto& elem : hcpevec)
		elem.gameResult = gameResult;

	// 局面出力
	if (hcpevec.size() > 0) {
		std::unique_lock<Mutex> lock(omutex);
		ofs.write(reinterpret_cast<char*>(hcpevec.data()), sizeof(HuffmanCodedPosAndEval) * hcpevec.size());
		madeTeacherNodes += hcpevec.size();
		++games;

		if (gameResult == Draw) {
			++draws;
		}
	}

	// USIエンジンとの対局結果
	if (need_usi_engine) {
		++usi_games;
		if (ply % 2 == 1 && (pos_root->turn() == Black && gameResult == BlackWin || pos_root->turn() == White && gameResult == WhiteWin) ||
			ply % 2 == 0 && (pos_root->turn() == Black && gameResult == WhiteWin || pos_root->turn() == White && gameResult == BlackWin))
			++usi_wins;
		else if (gameResult == Draw)
			++usi_draws;
	}

	// すぐに終局した初期局面を削除候補とする
	/*if (ply < 10) {
		std::unique_lock<Mutex> lock(omutex);
		ofs_dup.write(reinterpret_cast<char*>(&hcp), sizeof(HuffmanCodedPos));
	}*/

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
	//ofs_dup.open(string(outputFileName) + "_dup", ios::binary);

	// 初期設定
	InitializeUctSearch();

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
				logger->info("Progress:{:.2f}%, nodes:{}, nodes/sec:{:.2f}, games:{}, draw:{}, nyugyoku:{}, ply/game:{:.2f}, playouts/node:{:.2f}, max_playout:{} gpu id:{}, usi_games:{}, usi_win:{}, usi_draw:{}, Elapsed:{}[s], Remaining:{}[s]",
					std::min(100.0, progress * 100.0),
					idx,
					static_cast<double>(idx) / elapsed_msec * 1000.0,
					games,
					draws,
					nyugyokus,
					static_cast<double>(madeTeacherNodes) / games,
					static_cast<double>(sum_playouts) / sum_nodes,
					max_playout,
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
	//ofs_dup.close();

	logger->info("Made {} teacher nodes in {} seconds. games:{}, draws:{}, ply/game:{}, usi_games:{}, usi_win:{}, usi_draw:{}",
		madeTeacherNodes, t.elapsed() / 1000,
		games,
		draws,
		static_cast<double>(madeTeacherNodes) / games,
		usi_games,
		usi_wins,
		usi_draws);
}

int main(int argc, char* argv[]) {
	std::string recordFileName;
	std::string outputFileName;
	vector<int> gpu_id(1);
	vector<int> batchsize(1);

	cxxopts::Options options("make_hcpe_by_self_play");
	options.positional_help("modelfile hcp output nodes playout_num gpu_id batchsize [gpu_id batchsize]*");
	try {
		std::string file;

		options.add_options()
			("modelfile", "model file path", cxxopts::value<std::string>(model_path))
			("hcp", "initial position file", cxxopts::value<std::string>(recordFileName))
			("output", "output file path", cxxopts::value<std::string>(outputFileName))
			("nodes", "nodes", cxxopts::value<s64>(teacherNodes))
			("gpu_id", "gpu id", cxxopts::value<int>(gpu_id[0]))
			("batchsize", "batchsize", cxxopts::value<int>(batchsize[0]))
			("positional", "", cxxopts::value<std::vector<int>>())
			("random", "random move number", cxxopts::value<int>(RANDOM_MOVE)->default_value("4"), "num")
			("threashold", "winrate threshold", cxxopts::value<float>(WINRATE_THRESHOLD)->default_value("0.99"), "rate")
			("mate_depth", "mate search depth", cxxopts::value<uint32_t>(ROOT_MATE_SEARCH_DEPTH)->default_value("0"), "depth")
			("mate_nodes", "mate search max nodes", cxxopts::value<int64_t>(MATE_SEARCH_MAX_NODE)->default_value("100000"), "nodes")
			("mate_rand_limit_nodes", "mate search randomize limit nodes", cxxopts::value<bool>(MATE_SEARCH_RAND_LIMIT_NODES))
			("c_init", "UCT parameter c_init", cxxopts::value<float>(c_init)->default_value("1.49"), "val")
			("c_base", "UCT parameter c_base", cxxopts::value<float>(c_base)->default_value("39470.0"), "val")
			("temperature", "Softmax temperature", cxxopts::value<float>(temperature)->default_value("1.66"), "val")
			("usi_engine", "USIEngine exe path", cxxopts::value<std::string>(usi_engine_path))
			("usi_engine_id", "USIEngine id corresponding to gpu_id", cxxopts::value<int>(usi_engine_id)->default_value("0"))
			("usi_engine_num", "USIEngine number", cxxopts::value<int>(usi_engine_num)->default_value("0"), "num")
			("usi_options", "USIEngine options", cxxopts::value<std::string>(usi_options))
			("usi_byoyomi", "USI byoyomi", cxxopts::value<int>(usi_byoyomi)->default_value("500"))
			("h,help", "Print help")
			;
		options.parse_positional({ "modelfile", "hcp", "output", "nodes", "gpu_id", "batchsize", "positional" });

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
	if (RANDOM_MOVE < 0) {
		cerr << "too few random move number" << endl;
		return 0;
	}
	if (WINRATE_THRESHOLD <= 0) {
		cerr << "too few threashold" << endl;
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

	ns_dfpn::DfPn::set_maxdepth(ROOT_MATE_SEARCH_DEPTH);

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
	logger->set_level(spdlog::level::trace);
	logger->info("modelfile:{} roots.hcp:{} output:{} nodes:{}", model_path, recordFileName, outputFileName, teacherNodes);

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

	logger->info("random:{}", RANDOM_MOVE);
	logger->info("threashold:{}", WINRATE_THRESHOLD);
	logger->info("mate depath:{}", ROOT_MATE_SEARCH_DEPTH);
	logger->info("mate nodes:{}", MATE_SEARCH_MAX_NODE);
	logger->info("mate rand limit nodes:{}", MATE_SEARCH_RAND_LIMIT_NODES);
	logger->info("c_init:{}", c_init);
	logger->info("c_base:{}", c_base);
	logger->info("temperature:{}", temperature);
	logger->info("usi_engine:{}", usi_engine_path);
	logger->info("usi_engine_id:{}", usi_engine_id);
	logger->info("usi_engine_num:{}", usi_engine_num);
	logger->info("usi_options:{}", usi_options);
	logger->info("usi_byoyomi:{}", usi_byoyomi);

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

	set_softmax_temperature(temperature);

	signal(SIGINT, sigint_handler);

	logger->info("make_teacher");
	make_teacher(recordFileName.c_str(), outputFileName.c_str(), gpu_id, batchsize);

	spdlog::drop_all();
}
