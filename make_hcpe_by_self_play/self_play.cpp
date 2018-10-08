#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "movePicker.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "book.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <memory>
#include <signal.h>

#include "ZobristHash.h"
#include "mate.h"
#include "nn.h"

#include "cppshogi.h"

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

// 終局とする勝率の閾値
const float WINRATE_THRESHOLD = 0.985f;

// モデルのパス
string model_path;

int playout_num = 1000;

const unsigned int uct_hash_size = 8192; // UCTハッシュサイズ

s64 teacherNodes; // 教師局面数
std::atomic<s64> idx(0);
std::atomic<s64> madeTeacherNodes(0);
std::atomic<s64> games(0);
std::atomic<s64> draws(0);

ifstream ifs;
ofstream ofs;
mutex imutex;
mutex omutex;
size_t entryNum;

// 候補手の最大数(盤上全体)
const int UCT_CHILD_MAX = 593;

struct child_node_t {
	Move move;          // 着手する座標
	int move_count;     // 探索回数
	float win;          // 勝った回数
	unsigned int index; // インデックス
	float nnrate;       // ニューラルネットワークでのレート
};

struct uct_node_t {
	int move_count;
	float win;
	int child_num;                      // 子ノードの数
	child_node_t child[UCT_CHILD_MAX];  // 子ノードの情報
	std::atomic<int> evaled; // 0:未評価 1:評価済 2:千日手の可能性あり
	float value_win;
};

struct policy_value_queue_node_t {
	uct_node_t* node;
	Color color;
};

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
const int MATE_SEARCH_DEPTH = 7;

// 詰み探索で詰みの場合のvalue_winの定数
const float VALUE_WIN = FLT_MAX;
const float VALUE_LOSE = -FLT_MAX;

// 探索の結果を評価のキューに追加したか、破棄したか
const float QUEUING = FLT_MAX;
const float DISCARDED = -FLT_MAX;

unsigned const int NOT_EXPANDED = -1; // 未展開のノードのインデックス

const float c_puct = 1.0f;

// 探索経路のノード
struct TrajectorEntry {
	TrajectorEntry(uct_node_t* uct_node, unsigned int current, unsigned int next_index) : uct_node(uct_node), current(current), next_index(next_index) {}
	uct_node_t* uct_node;
	unsigned int current;
	unsigned int next_index;
};

void UpdateResult(uct_node_t* uct_node, child_node_t *child, const float result, const unsigned int current);
bool nyugyoku(const Position& pos);

Searcher s;

class UCTSearcher;
class UCTSearcherGroupPair;
class UCTSearcherGroup {
public:
	UCTSearcherGroup(const int gpu_id, const int group_id, const int policy_value_batch_maxsize, UCTSearcherGroupPair* parent) :
		gpu_id(gpu_id), group_id(group_id), policy_value_batch_maxsize(policy_value_batch_maxsize), parent(parent),
		current_policy_value_batch_index(0), features1(nullptr), features2(nullptr), policy_value_queue_node(nullptr), y1(nullptr), y2(nullptr), running(false) {
		Initialize();
	}
	UCTSearcherGroup(UCTSearcherGroup&& o) {} // not use
	~UCTSearcherGroup() {
		checkCudaErrors(cudaFreeHost(features1));
		checkCudaErrors(cudaFreeHost(features2));
		checkCudaErrors(cudaFreeHost(y1));
		checkCudaErrors(cudaFreeHost(y2));
	}

	void QueuingNode(const Position *pos, unsigned int index, uct_node_t* uct_node);
	void EvalNode();
	void SelfPlay();
	void Run();
	void Join();

	int running;
private:
	void Initialize();

	UCTSearcherGroupPair* parent;
	int group_id;

	// GPUID
	int gpu_id;

	// キュー
	int policy_value_batch_maxsize; // 最大バッチサイズ
	features1_t* features1;
	features2_t* features2;
	policy_value_queue_node_t* policy_value_queue_node;
	int current_policy_value_batch_index;

	// UCTSearcher
	vector<UCTSearcher> searchers;
	thread* handle_selfplay;

	DType* y1;
	DType* y2;
};

class UCTSearcher {
public:
	UCTSearcher(UCTSearcherGroup* grp, const int id, const size_t entryNum) :
		grp(grp),
		id(id),
		mt_64(new std::mt19937_64(std::chrono::system_clock::now().time_since_epoch().count() + id)),
		mt(new std::mt19937(std::chrono::system_clock::now().time_since_epoch().count() + id)),
		uct_hash(new UctHash(uct_hash_size)),
		uct_node(new uct_node_t[uct_hash_size]),
		inputFileDist(0, entryNum - 1),
		playout(0),
		ply(0) {
		pos_root = new Position(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);
	}
	UCTSearcher(UCTSearcher&& o) {} // not use
	~UCTSearcher() {
		delete[] uct_node;
		delete pos_root;
	}

	float UctSearch(Position *pos, unsigned int current, const int depth, vector<TrajectorEntry>& trajectories);
	int SelectMaxUcbChild(const Position *pos, unsigned int current, const int depth);
	unsigned int ExpandRoot(const Position *pos);
	unsigned int ExpandNode(Position *pos, const int depth);
	bool InterruptionCheck(const unsigned int current_root, const int playout_count);
	void Playout(vector<TrajectorEntry>& trajectories);
	void NextStep();

private:
	void NextGame();

	UCTSearcherGroup* grp;
	int id;
	unique_ptr<UctHash> uct_hash;
	uct_node_t* uct_node;
	unique_ptr<std::mt19937_64> mt_64;
	unique_ptr<std::mt19937> mt;
	// スレッドのハンドル
	thread *handle;

	int playout;
	int ply;
	GameResult gameResult;
	unsigned int current_root;

	std::unordered_set<Key> keyHash;
	StateListPtr states = nullptr;
	std::vector<HuffmanCodedPosAndEval> hcpevec;
	uniform_int_distribution<s64> inputFileDist;

	// 局面管理と探索スレッド
	Position* pos_root;
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
			nn = new NN(policy_value_batch_maxsize);
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

void randomMove(Position& pos, std::mt19937& mt);

void
UCTSearcherGroup::Initialize()
{
	// キューを動的に確保する
	checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * policy_value_batch_maxsize, cudaHostAllocPortable));
	policy_value_queue_node = new policy_value_queue_node_t[policy_value_batch_maxsize];

	// UCTSearcher
	searchers.clear();
	searchers.reserve(policy_value_batch_maxsize);
	for (int i = 0; i < policy_value_batch_maxsize; i++) {
		searchers.emplace_back(this, gpu_id * 10000 + group_id * 1000 + i, entryNum);
	}

	checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (int)SquareNum * policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&y2, policy_value_batch_maxsize * sizeof(DType), cudaHostAllocPortable));
}

// 連続自己対局
void UCTSearcherGroup::SelfPlay()
{
	// スレッドにGPUIDを関連付けてから初期化する
	cudaSetDevice(gpu_id);

	parent->InitGPU();

	running = 1;

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
					result = 1.0f - uct_node[child_index].value_win;
				}
				UpdateResult(uct_node, &uct_child[next_index], result, current);
				result = 1.0f - result;
			}
		}

		// 次の手へ
		for (UCTSearcher& searcher : searchers) {
			searcher.NextStep();
		}
	}

	running = 0;
}

// スレッド開始
void
UCTSearcherGroup::Run()
{
	// 自己対局用スレッド
	handle_selfplay = new thread([this]() { this->SelfPlay(); });
}

// スレッド終了待機
void
UCTSearcherGroup::Join()
{
	// 自己対局用スレッド
	handle_selfplay->join();
	delete handle_selfplay;
}

//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UCTSearcher::UctSearch(Position *pos, unsigned int current, const int depth, vector<TrajectorEntry>& trajectories)
{
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
	if (uct_node[current].evaled == 2) {
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
		// ノード展開処理の中でvalueを計算する
		unsigned int child_index = ExpandNode(pos, depth + 1);
		uct_child[next_index].index = child_index;
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

		// 合流検知
		if (uct_node[child_index].evaled != 0) {
			// 手番を入れ替えて1手深く読む
			result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories);
		}
		else if (uct_node[child_index].child_num == 0) {
			// 詰み
			uct_node[child_index].value_win = VALUE_LOSE;
			uct_node[child_index].evaled = 1;
			result = 1.0f;
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
				uct_node[child_index].evaled = 2;
				if (isDraw == 1) {
					result = 0.0f;
				}
				else if (isDraw == -1) {
					result = 1.0f;
				}
				else {
					result = 0.5f;
				}

			}
			// 詰みの場合、ValueNetの値を上書き
			else if (isMate == 1) {
				uct_node[child_index].value_win = VALUE_WIN;
				result = 0.0f;
			}
			else if (isMate == -1) {
				uct_node[child_index].value_win = VALUE_LOSE;
				result = 1.0f;
			}
			else {
				// ノードをキューに追加
				grp->QueuingNode(pos, child_index, uct_node);
				return QUEUING;
			}
		}
	}
	else {
		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, uct_child[next_index].index, depth + 1, trajectories);
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
	//const bool debug = GetDebugMessageMode() && current == current_root && sum % 100 == 0;

	max_value = -1;

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
			q = 0.5f;
			u = 1.0f;
		}
		else {
			q = win / move_count;
			u = sqrtf(sum) / (1 + move_count);
		}

		float rate = max(uct_child[i].nnrate, 0.01f);
		// ランダムに確率を上げる
		if (depth == 0 && rnd(*mt) <= 2) {
			rate = (rate + 1.0f) / 2.0f;
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
	uct_child->nnrate = 0;
}

/////////////////////////
//  ルートノードの展開  //
/////////////////////////
unsigned int
UCTSearcher::ExpandRoot(const Position *pos)
{
	unsigned int index = uct_hash->FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly());
	child_node_t *uct_child;
	int child_num = 0;

	// 既に展開されていた時は, 探索結果を再利用する
	if (index != uct_hash_size) {
		return index;
	}
	else {
		// 空のインデックスを探す
		index = uct_hash->SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly());

		assert(index != uct_hash_size);

		// ルートノードの初期化
		uct_node[index].move_count = 0;
		uct_node[index].win = 0;
		uct_node[index].child_num = 0;
		uct_node[index].evaled = 0;
		uct_node[index].value_win = 0.0f;

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
	unsigned int index = uct_hash->FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth);
	child_node_t *uct_child;

	// 合流先が検知できれば, それを返す
	if (index != uct_hash_size) {
		return index;
	}

	// 空のインデックスを探す
	index = uct_hash->SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth);

	assert(index != uct_hash_size);

	// 現在のノードの初期化
	uct_node[index].move_count = 0;
	uct_node[index].win = 0;
	uct_node[index].child_num = 0;
	uct_node[index].evaled = 0;
	uct_node[index].value_win = 0.0f;
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
UCTSearcherGroup::QueuingNode(const Position *pos, unsigned int index, uct_node_t* uct_node)
{
	// set all zero
	std::fill_n((DType*)features1[current_policy_value_batch_index], sizeof(features1_t) / sizeof(DType), _zero);
	std::fill_n((DType*)features2[current_policy_value_batch_index], sizeof(features2_t) / sizeof(DType), _zero);

	make_input_features(*pos, &features1[current_policy_value_batch_index], &features2[current_policy_value_batch_index]);
	policy_value_queue_node[current_policy_value_batch_index].node = &uct_node[index];
	policy_value_queue_node[current_policy_value_batch_index].color = pos->turn();
	current_policy_value_batch_index++;
}

//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
bool
UCTSearcher::InterruptionCheck(const unsigned int current_root, const int playout_count)
{
	int max = 0, second = 0;
	const int child_num = uct_node[current_root].child_num;
	int rest = playout_num - playout_count;
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

	// 最善手の探索回数が次善手の探索回数の
	// 1.2倍未満なら探索延長
	if (max < second * 1.2) {
		rest += playout_num * 0.5;
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

// 局面の評価
void UCTSearcherGroup::EvalNode() {
	const int policy_value_batch_size = current_policy_value_batch_index;

	// predict
	parent->nn_forward(policy_value_batch_size, features1, features2, y1, y2);

	DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
	DType *value = y2;

	for (int i = 0; i < policy_value_batch_size; i++, logits++, value++) {
		policy_value_queue_node_t& queue_node = policy_value_queue_node[i];

		// 対局は1スレッドで行うためロックは不要
		const int child_num = queue_node.node->child_num;
		child_node_t *uct_child = queue_node.node->child;
		const Color color = queue_node.color;

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
		softmax_tempature_with_normalize(legal_move_probabilities);

		for (int j = 0; j < child_num; j++) {
			uct_child[j].nnrate = legal_move_probabilities[j];
		}

#ifdef FP16
		queue_node.node->value_win = __half2float(*value);
#else
		queue_node.node->value_win = *value;
#endif
		queue_node.node->evaled = 1;
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
				HuffmanCodedPos hcp;
				{
					std::unique_lock<Mutex> lock(imutex);
					ifs.seekg(inputFileDist(*mt_64) * sizeof(HuffmanCodedPos), std::ios_base::beg);
					ifs.read(reinterpret_cast<char*>(&hcp), sizeof(hcp));
				}
				setPosition(*pos_root, hcp);
				randomMove(*pos_root, *mt); // 教師局面を増やす為、取得した元局面からランダムに動かしておく。
				SPDLOG_DEBUG(logger, "id:{} ply:{} {}", id, ply, pos_root->toSFEN());

				keyHash.clear();
				states = StateListPtr(new std::deque<StateInfo>(1));
				hcpevec.clear();
			}

			// ハッシュクリア
			uct_hash->ClearUctHash();

			// ルートノード展開
			current_root = ExpandRoot(pos_root);

			// 詰みのチェック
			if (uct_node[current_root].child_num == 0) {
				gameResult = (pos_root->turn() == Black) ? WhiteWin : BlackWin;
				NextGame();
				continue;
			}
			else if (uct_node[current_root].child_num == 1) {
				// 1手しかないときは、その手を指して次の手番へ
				SPDLOG_DEBUG(logger, "id:{} ply:{} {} skip:{}", id, ply, pos_root->toSFEN(), uct_node[current_root].child[0].move.toUSI());
				states->push_back(StateInfo());
				pos_root->doMove(uct_node[current_root].child[0].move, states->back());
				playout = 0;
				ply++;
				continue;
			}
			else if (nyugyoku(*pos_root)) {
				// 入玉宣言勝ち
				gameResult = (pos_root->turn() == Black) ? BlackWin : WhiteWin;
				NextGame();
				continue;
			}

			// ルート局面をキューに追加
			grp->QueuingNode(pos_root, current_root, uct_node);
			return;
		}

		// 盤面のコピー
		Position pos_copy(*pos_root);
		// プレイアウト
		const float result = UctSearch(&pos_copy, current_root, 0, trajectories);
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
	// プレイアウト回数加算
	playout++;

	// 探索終了判定
	if (InterruptionCheck(current_root, playout)) {
		// 探索回数最大の手を見つける
		child_node_t* uct_child = uct_node[current_root].child;
		int max_count = uct_child[0].move_count;
		unsigned int select_index = 0;
		for (int i = 1; i < uct_node[current_root].child_num; i++) {
			if (uct_child[i].move_count > max_count) {
				select_index = i;
				max_count = uct_child[i].move_count;
			}
			SPDLOG_TRACE(logger, "id:{} {}:{} move_count:{} win_rate:{}", id, i, uct_child[i].move.toUSI(), uct_child[i].move_count, uct_child[i].win / (uct_child[i].move_count + 0.0001f));
		}

		// 選択した着手の勝率の算出
		float best_wp = uct_child[select_index].win / uct_child[select_index].move_count;
		Move best_move = uct_child[select_index].move;
		SPDLOG_DEBUG(logger, "id:{} ply:{} {} bestmove:{} winrate:{}", id, ply, pos_root->toSFEN(), best_move.toUSI(), best_wp);

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
		hcpevec.emplace_back(HuffmanCodedPosAndEval());
		HuffmanCodedPosAndEval& hcpe = hcpevec.back();
		hcpe.hcp = pos_root->toHuffmanCodedPos();
		const Color rootTurn = pos_root->turn();
		hcpe.eval = s16(-logf(1.0f / best_wp - 1.0f) * 756.0864962951762f);
		hcpe.bestMove16 = static_cast<u16>(uct_child[select_index].move.value());
		idx++;

		// 一定の手数以上で引き分け
		if (ply > 200) {
			gameResult = Draw;
			NextGame();
			return;
		}

		// 着手
		states->push_back(StateInfo());
		pos_root->doMove(best_move, states->back());

		// 次の手番
		playout = 0;
		ply++;
	}
}

void UCTSearcher::NextGame()
{
	SPDLOG_DEBUG(logger, "id:{} ply:{} gameResult:{}", id, ply, gameResult);
	// 引き分けは出力しない
	if (gameResult != Draw) {
		// 勝敗を1局全てに付ける。
		for (auto& elem : hcpevec)
			elem.gameResult = gameResult;

		// 局面出力
		if (hcpevec.size() > 0) {
			std::unique_lock<Mutex> lock(omutex);
			Position po;
			po.set(hcpevec[0].hcp, nullptr);
			ofs.write(reinterpret_cast<char*>(hcpevec.data()), sizeof(HuffmanCodedPosAndEval) * hcpevec.size());
			madeTeacherNodes += hcpevec.size();
			++games;
		}
	}
	else {
		++draws;
	}

	// 新しいゲーム
	playout = 0;
	ply = 0;
}

// 教師局面生成
void make_teacher(const char* recordFileName, const char* outputFileName, const vector<int>& gpu_id, const vector<int>& batchsize)
{
	s.init();
	const std::string options[] = {
		"name Threads value 1",
		"name MultiPV value 1",
		"name OwnBook value false",
		"name Max_Random_Score_Diff value 0" };
	for (auto& str : options) {
		std::istringstream is(str);
		s.setOption(is);
	}

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

	// 初期設定
	InitializeUctSearch();

	vector<UCTSearcherGroupPair> group_pairs;
	group_pairs.reserve(gpu_id.size());
	for (int i = 0; i < gpu_id.size(); i++)
		group_pairs.emplace_back(gpu_id[i], batchsize[i]);

	// 探索スレッド開始
	for (int i = 0; i < group_pairs.size(); i++)
		group_pairs[i].Run();

	// 進捗状況表示
	auto progressFunc = [&gpu_id, &group_pairs](Timer& t) {
		ostringstream ss;
		for (int i = 0; i < gpu_id.size(); i++) {
			if (i > 0) ss << " ";
			ss << gpu_id[i];
		}
		while (!stopflg) {
			std::this_thread::sleep_for(std::chrono::seconds(10)); // 指定秒だけ待機し、進捗を表示する。
			const double progress = static_cast<double>(madeTeacherNodes) / teacherNodes;
			auto elapsed_msec = t.elapsed();
			if (progress > 0.0) // 0 除算を回避する。
				logger->info("Progress:{:.2f}%, nodes:{}, nodes/sec:{:.2f}, games:{}, draws:{}, ply/game:{}, gpu id:{}, Elapsed:{}[s], Remaining:{}[s]",
					std::min(100.0, progress * 100.0),
					idx,
					static_cast<double>(idx) / elapsed_msec * 1000.0,
					games,
					draws,
					static_cast<double>(madeTeacherNodes) / games,
					ss.str(),
					elapsed_msec / 1000,
					std::max<s64>(0, (s64)(elapsed_msec*(1.0 - progress) / (progress * 1000))));
			int running = 0;
			for (int i = 0; i < group_pairs.size(); i++)
				running += group_pairs[i].Running();
			if (running == 0)
				break;
		}
	};

	while (!stopflg) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		int running = 0;
		for (int i = 0; i < group_pairs.size(); i++)
			running += group_pairs[i].Running();
		if (running > 0)
			break;
	}
	Timer t = Timer::currentTime();
	std::thread progressThread([&progressFunc, &t] { progressFunc(t); });

	// 探索スレッド終了待機
	for (int i = 0; i < group_pairs.size(); i++)
		group_pairs[i].Join();

	progressThread.join();
	ifs.close();
	ofs.close();

	logger->info("Made {} teacher nodes in {} seconds. games:{}, draws:{}, ply/game:{}",
		madeTeacherNodes, t.elapsed() / 1000,
		games,
		draws,
		static_cast<double>(madeTeacherNodes) / games);
}

int main(int argc, char* argv[]) {
	const int argnum = 8;
	if (argc < argnum) {
		cout << "make_hcpe_by_self_play <modelfile> <roots.hcp> <output> <nodes> <playout_num> <gpu_id> <batchsize> [<gpu_id> <batchsize>]*" << endl;
		return 0;
	}

	model_path = argv[1];
	char* recordFileName = argv[2];
	char* outputFileName = argv[3];
	teacherNodes = stoi(argv[4]);
	playout_num = stoi(argv[5]);

	if (teacherNodes <= 0) {
		cout << "too few teacherNodes" << endl;
		return 0;
	}
	if (playout_num <= 0)
		return 0;

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
	logger->set_level(spdlog::level::trace);
	logger->info("modelfile:{} roots.hcp:{} output:{} nodes:{} playout_num:{}", model_path, recordFileName, outputFileName, teacherNodes, playout_num);

	vector<int> gpu_id;
	vector<int> batchsize;
	for (int i = argnum - 2; i < argc; i += 2) {
		gpu_id.push_back(stoi(argv[i]));
		batchsize.push_back(stoi(argv[i + 1]));

		logger->info("gpu_id:{} batchsize:{}", gpu_id.back(), batchsize.back());

		if (gpu_id.back() < 0) {
			cout << "invalid gpu id" << endl;
			return 0;
		}
		if (batchsize.back() <= 0) {
			cout << "too few batchsize" << endl;
			return 0;
		}
	}


	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

	signal(SIGINT, sigint_handler);

	logger->info("make_teacher");
	make_teacher(recordFileName, outputFileName, gpu_id, batchsize);

	spdlog::drop_all();
}
