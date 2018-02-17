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
#include "ZobristHash.h"
#include "mate.h"

#include "cppshogi.h"
namespace py = boost::python;
namespace np = boost::python::numpy;

// ルートノードでの詰み探索を行う
//#define USE_MATE_ROOT_SEARCH

//#define SPDLOG_TRACE_ON
//#define SPDLOG_DEBUG_ON
#define SPDLOG_EOL "\n"
#include "spdlog/spdlog.h"
auto loggersink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
auto logger = std::make_shared<spdlog::async_logger>("selfplay", loggersink, 8192);

using namespace std;

// モデルのパス
string model_path;

int playout_num = 1000;

const unsigned int uct_hash_size = 16384; // UCTハッシュサイズ
int threads; // スレッド数
atomic<int> running_threads; // 実行中の探索スレッド数
thread **handle; // スレッドのハンドル

s64 teacherNodes; // 教師局面数
std::atomic<s64> idx(0);

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

// 2つのキューを交互に使用する
const int policy_value_batch_maxsize = 64; // スレッド数の以上確保する
static float features1[2][policy_value_batch_maxsize][ColorNum][MAX_FEATURES1_NUM][SquareNum];
static float features2[2][policy_value_batch_maxsize][MAX_FEATURES2_NUM][SquareNum];
struct policy_value_queue_node_t {
	uct_node_t* node;
	Color color;
};
static policy_value_queue_node_t policy_value_queue_node[2][policy_value_batch_maxsize];
static int current_policy_value_queue_index = 0;
static int current_policy_value_batch_index = 0;

mutex mutex_queue; // キューの排他制御
#define LOCK_QUEUE mutex_queue.lock();
#define UNLOCK_QUEUE mutex_queue.unlock();

// 予測関数
py::object dlshogi_predict;

// ランダム
uniform_int_distribution<int> rnd(0, 999);

// 末端ノードでの詰み探索の深さ(奇数であること)
const int MATE_SEARCH_DEPTH = 7;

// 詰み探索で詰みの場合のvalue_winの定数
const float VALUE_WIN = FLT_MAX;
const float VALUE_LOSE = -FLT_MAX;

unsigned const int NOT_EXPANDED = -1; // 未展開のノードのインデックス

const float c_puct = 1.0f;


class UctSercher {
public:
	UctSercher(const int threadID) : threadID(threadID), mt(std::chrono::system_clock::now().time_since_epoch().count() + threadID) {
		// UCTHash
		uct_hash.InitializeUctHash(uct_hash_size);
		// UCTのノード
		uct_node = new uct_node_t[uct_hash_size];
	}
	~UctSercher() {
		delete[] uct_node;
	}

	float UctSearch(Position *pos, unsigned int current, const int depth);
	int SelectMaxUcbChild(const Position *pos, unsigned int current, const int depth);
	unsigned int ExpandRoot(const Position *pos);
	unsigned int ExpandNode(Position *pos, unsigned int current, const int depth);
	bool InterruptionCheck(const unsigned int current_root, const int playout_count);
	void UpdateResult(child_node_t *child, float result, unsigned int current);
	void SelfPlay();


private:
	int threadID;
	UctHash uct_hash;
	uct_node_t* uct_node;
	std::mt19937 mt;
};

void randomMove(Position& pos, std::mt19937& mt);
static void QueuingNode(const Position *pos, unsigned int index, uct_node_t* uct_node, const Color color);


//////////////////////////////////////////////
//  UCT探索を行う関数                        //
//  1回の呼び出しにつき, 1プレイアウトする    //
//////////////////////////////////////////////
float
UctSercher::UctSearch(Position *pos, unsigned int current, const int depth)
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
			// case RepetitionSuperior: if (ss->ply != 2) { return ScoreMateInMaxPly; } break;
			// case RepetitionInferior: if (ss->ply != 2) { return ScoreMatedInMaxPly; } break;
		default: UNREACHABLE;
		}
	}

	float result;
	unsigned int next_index;
	double score;
	child_node_t *uct_child = uct_node[current].child;

	// UCB値最大の手を求める
	next_index = SelectMaxUcbChild(pos, current, depth);
	// 選んだ手を着手
	StateInfo st;
	pos->doMove(uct_child[next_index].move, st);

	// ノードの展開の確認
	if (uct_child[next_index].index == NOT_EXPANDED) {
		// ノードの展開
		// ノード展開処理の中でvalueを計算する
		unsigned int child_index = ExpandNode(pos, current, depth + 1);
		uct_child[next_index].index = child_index;
		//cerr << "value evaluated " << result << " " << v << " " << *value_result << endl;

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
			// case RepetitionSuperior: if (ss->ply != 2) { return ScoreMateInMaxPly; } break;
			// case RepetitionInferior: if (ss->ply != 2) { return ScoreMatedInMaxPly; } break;
		default: UNREACHABLE;
		}

		// valueが計算されるのを待つ
		//cout << "wait value:" << child_index << ":" << uct_node[child_index].evaled << endl;
		while (uct_node[child_index].evaled == 0)
			this_thread::sleep_for(chrono::milliseconds(0));

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
			// valueを勝敗として返す
			result = 1 - uct_node[child_index].value_win;
		}
	}
	else {
		// 手番を入れ替えて1手深く読む
		result = UctSearch(pos, uct_child[next_index].index, depth + 1);
	}

	// 探索結果の反映
	UpdateResult(&uct_child[next_index], result, current);

	pos->undoMove(uct_child[next_index].move);
	return 1 - result;
}

//////////////////////
//  探索結果の更新  //
/////////////////////
void
UctSercher::UpdateResult(child_node_t *child, float result, unsigned int current)
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
UctSercher::SelectMaxUcbChild(const Position *pos, unsigned int current, const int depth)
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
		if (depth == 0 && rnd(mt) <= 2) {
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

// モデル読み込み
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
UctSercher::ExpandRoot(const Position *pos)
{
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly());
	child_node_t *uct_child;
	int child_num = 0;

	// 既に展開されていた時は, 探索結果を再利用する
	if (index != uct_hash_size) {
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

		// 候補手のレーティング
		QueuingNode(pos, index, uct_node, pos->turn());

	}

	return index;
}

///////////////////
//  ノードの展開  //
///////////////////
unsigned int
UctSercher::ExpandNode(Position *pos, unsigned int current, const int depth)
{
	unsigned int index = uct_hash.FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth);
	child_node_t *uct_child;

	// 合流先が検知できれば, それを返す
	if (index != uct_hash_size) {
		return index;
	}

	// 空のインデックスを探す
	index = uct_hash.SearchEmptyIndex(pos->getKey(), pos->turn(), pos->gamePly() + depth);

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

	// 候補手のレーティング
	if (child_num > 0) {
		QueuingNode(pos, index, uct_node, pos->turn());
	}
	else {
		uct_node[index].value_win = 0.0f;
		uct_node[index].evaled = 1;
	}

	return index;
}

//////////////////////////////////////
//  ノードをキューに追加            //
//////////////////////////////////////
static void
QueuingNode(const Position *pos, unsigned int index, uct_node_t* uct_node, const Color color)
{
	LOCK_QUEUE;
	if (current_policy_value_batch_index >= policy_value_batch_maxsize) {
		std::cout << "error" << std::endl;
		exit(EXIT_FAILURE);
	}
	// set all zero
	std::fill_n((float*)features1[current_policy_value_queue_index][current_policy_value_batch_index], (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
	std::fill_n((float*)features2[current_policy_value_queue_index][current_policy_value_batch_index], MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

	make_input_features(*pos, &features1[current_policy_value_queue_index][current_policy_value_batch_index], &features2[current_policy_value_queue_index][current_policy_value_batch_index]);
	policy_value_queue_node[current_policy_value_queue_index][current_policy_value_batch_index].node = &uct_node[index];
	policy_value_queue_node[current_policy_value_queue_index][current_policy_value_batch_index].color = color;
	current_policy_value_batch_index++;
	UNLOCK_QUEUE;
}

//////////////////////////
//  探索打ち止めの確認  //
//////////////////////////
bool
UctSercher::InterruptionCheck(const unsigned int current_root, const int playout_count)
{
	int max = 0, second = 0;
	const int child_num = uct_node[current_root].child_num;
	const int rest = playout_num - playout_count;
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
void EvalNode() {
	bool enough_batch_size = false;
	while (true) {
		LOCK_QUEUE;
		if (running_threads == 0 && current_policy_value_batch_index == 0) {
			UNLOCK_QUEUE;
			break;
		}

		if (current_policy_value_batch_index == 0) {
			UNLOCK_QUEUE;
			this_thread::sleep_for(chrono::milliseconds(1));
			continue;
		}

		if (running_threads > 0 && !enough_batch_size && current_policy_value_batch_index < running_threads * 0.9) {
			UNLOCK_QUEUE;
			this_thread::sleep_for(chrono::milliseconds(1));
			enough_batch_size = true;
		}
		else {
			enough_batch_size = false;
			int policy_value_batch_size = current_policy_value_batch_index;
			int policy_value_queue_index = current_policy_value_queue_index;
			current_policy_value_batch_index = 0;
			current_policy_value_queue_index = current_policy_value_queue_index ^ 1;
			UNLOCK_QUEUE;
			SPDLOG_DEBUG(logger, "policy_value_batch_size:{}", policy_value_batch_size);

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
				policy_value_queue_node_t queue_node = policy_value_queue_node[policy_value_queue_index][i];

				/*if (index == current_root) {
				string str;
				for (int sq = 0; sq < SquareNum; sq++) {
				str += to_string((int)features1[policy_value_queue_index][i][0][0][sq]);
				str += " ";
				}
				cout << str << endl;
				}*/

				// 対局は1スレッドで行うためロックは不要
				const int child_num = queue_node.node->child_num;
				child_node_t *uct_child = queue_node.node->child;
				Color color = queue_node.color;

				// 合法手一覧
				vector<float> legal_move_probabilities;
				for (int j = 0; j < child_num; j++) {
					Move move = uct_child[j].move;
					const int move_label = make_move_label((u16)move.proFromAndTo(), color);
					legal_move_probabilities.emplace_back((*logits)[move_label]);
				}

				// Boltzmann distribution
				softmax_tempature_with_normalize(legal_move_probabilities);

				for (int j = 0; j < child_num; j++) {
					uct_child[j].nnrate = legal_move_probabilities[j];
				}

				queue_node.node->value_win = *value;
				queue_node.node->evaled = true;
			}
		}
	}
}

// 連続で自己対局する
void UctSercher::SelfPlay()
{
	int playout = 0;
	int ply = 0;
	GameResult gameResult;
	unsigned int current_root;

	std::unordered_set<Key> keyHash;
	StateListPtr states = nullptr;
	std::vector<HuffmanCodedPosAndEval> hcpevec;


	// 局面管理と探索スレッド
	Searcher s;
	s.init();
	const std::string options[] = {
		"name Threads value 1",
		"name MultiPV value 1",
		"name USI_Hash value 256",
		"name OwnBook value false",
		"name Max_Random_Score_Diff value 0" };
	for (auto& str : options) {
		std::istringstream is(str);
		s.setOption(is);
	}
	Position pos(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);

#ifdef USE_MATE_ROOT_SEARCH
	s.tt.clear();
	s.threads.main()->previousScore = ScoreInfinite;
	LimitsType limits;
	limits.depth = static_cast<Depth>(8);
#endif

	uniform_int_distribution<s64> inputFileDist(0, entryNum - 1);

	// プレイアウトを繰り返す
	while (true) {
		// 手番開始
		if (playout == 0) {
			// 新しいゲーム開始
			if (ply == 0) {
				// 全スレッドが生成した局面数が生成局面数以上になったら終了
				if (idx >= teacherNodes) {
					break;
				}

				ply = 1;

				// 開始局面を局面集からランダムに選ぶ
				HuffmanCodedPos hcp;
				{
					std::unique_lock<Mutex> lock(imutex);
					ifs.seekg(inputFileDist(mt) * sizeof(HuffmanCodedPos), std::ios_base::beg);
					ifs.read(reinterpret_cast<char*>(&hcp), sizeof(hcp));
				}
				setPosition(pos, hcp);
				randomMove(pos, mt); // 教師局面を増やす為、取得した元局面からランダムに動かしておく。
				SPDLOG_DEBUG(logger, "thread:{} ply:{} {}", threadID, ply, pos.toSFEN());

				keyHash.clear();
				states = StateListPtr(new std::deque<StateInfo>(1));
				hcpevec.clear();
			}

			// ハッシュクリア
			uct_hash.ClearUctHash();

			// ルートノード展開
			current_root = ExpandRoot(&pos);

			// 詰みのチェック
			if (uct_node[current_root].child_num == 0) {
				gameResult = (pos.turn() == Black) ? WhiteWin : BlackWin;
				goto L_END_GAME;
			}
			else if (uct_node[current_root].child_num == 1) {
				// 1手しかないときは、その手を指して次の手番へ
				states->push_back(StateInfo());
				pos.doMove(uct_node[current_root].child[0].move, states->back());
				playout = 0;
				continue;
			}
			else if (uct_node[current_root].value_win == VALUE_WIN) {
				// 詰み
				gameResult = (pos.turn() == Black) ? BlackWin : WhiteWin;
				goto L_END_GAME;
			}
			else if (uct_node[current_root].value_win == VALUE_LOSE) {
				// 自玉の詰み
				gameResult = (pos.turn() == Black) ? WhiteWin : BlackWin;
				goto L_END_GAME;
			}

#ifdef USE_MATE_ROOT_SEARCH
			// 詰み探索開始
			pos.searcher()->alpha = -ScoreMaxEvaluate;
			pos.searcher()->beta = ScoreMaxEvaluate;
			pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);
#endif
		}

		// プレイアウト
		UctSearch(&pos, current_root, 0);

		// プレイアウト回数加算
		playout++;

		// 探索終了判定
		if (InterruptionCheck(current_root, playout)) {
			// 探索回数最大の手を見つける
			child_node_t* uct_child = uct_node[current_root].child;
			int max_count = 0;
			unsigned int select_index;
			for (int i = 0; i < uct_node[current_root].child_num; i++) {
				if (uct_child[i].move_count > max_count) {
					select_index = i;
					max_count = uct_child[i].move_count;
				}
				SPDLOG_DEBUG(logger, "thread:{} {}:{} move_count:{} win_rate:{}", threadID, i, uct_child[i].move.toUSI(), uct_child[i].move_count, uct_child[i].win / (uct_child[i].move_count + 0.0001f));
			}

			// 選択した着手の勝率の算出
			float best_wp = uct_child[select_index].win / uct_child[select_index].move_count;
			Move best_move = uct_child[select_index].move;
			SPDLOG_DEBUG(logger, "thread:{} bestmove:{} winrate:{}", threadID, best_move.toUSI(), best_wp);

#ifdef USE_MATE_ROOT_SEARCH
			{
				// 詰み探索終了
				pos.searcher()->threads.main()->waitForSearchFinished();
				Score score = pos.searcher()->threads.main()->rootMoves[0].score;
				const Move bestMove = pos.searcher()->threads.main()->rootMoves[0].pv[0];

				// ゲーム終了判定
				// 条件：評価値が閾値を超えた場合
				const int ScoreThresh = 5000; // 自己対局を決着がついたとして止める閾値
				if (ScoreThresh < abs(score)) { // 差が付いたので投了した事にする。
					if (pos.turn() == Black)
						gameResult = (score < ScoreZero ? WhiteWin : BlackWin);
					else
						gameResult = (score < ScoreZero ? BlackWin : WhiteWin);

					goto L_END_GAME;
				}
				else if (!bestMove) { // 勝ち宣言
					gameResult = (pos.turn() == Black ? BlackWin : WhiteWin);
					goto L_END_GAME;
				}
			}
#else
			{
				// 勝率が閾値を超えた場合、ゲーム終了
				const float winrate = (best_wp - 0.5f) * 2.0f;
				const float winrate_threshold = 0.99f;
				if (winrate_threshold < abs(winrate)) {
					if (pos.turn() == Black)
						gameResult = (winrate < 0 ? WhiteWin : BlackWin);
					else
						gameResult = (winrate < 0 ? BlackWin : WhiteWin);

					goto L_END_GAME;
				}
			}
#endif

			// 局面追加
			hcpevec.emplace_back(HuffmanCodedPosAndEval());
			HuffmanCodedPosAndEval& hcpe = hcpevec.back();
			hcpe.hcp = pos.toHuffmanCodedPos();
			const Color rootTurn = pos.turn();
			hcpe.eval = s16(-logf(1.0f / best_wp - 1.0f) * 756.0864962951762f);
			hcpe.bestMove16 = static_cast<u16>(uct_child[select_index].move.value());
			idx++;

			// 一定の手数以上で引き分け
			if (ply > 200) {
				gameResult = Draw;
				goto L_END_GAME;
			}

			// 着手
			states->push_back(StateInfo());
			pos.doMove(best_move, states->back());

			// 次の手番
			playout = 0;
			ply++;
			SPDLOG_DEBUG(logger, "thread:{} ply:{} {}", threadID, ply, pos.toSFEN());
		}
		continue;

	L_END_GAME:
		SPDLOG_DEBUG(logger, "thread:{} ply:{} gameResult:{}", threadID, ply, gameResult);
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
			}
		}

		// 新しいゲーム
		playout = 0;
		ply = 0;
	}
}

void SelfPlay(const int threadID)
{
	logger->info("start selfplay thread:{}", threadID);
	UctSercher uct_sercher(threadID);
	uct_sercher.SelfPlay();
	logger->info("end selfplay thread:{}", threadID);
}

// 教師局面生成
void make_teacher(const char* recordFileName, const char* outputFileName)
{
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

	// モデル読み込み
	ReadWeights();

	// スレッド作成
	running_threads = threads;
	for (int i = 0; i < threads; i++) {
		handle[i] = new thread(SelfPlay, i);
	}

	// use_nn
	handle[threads] = new thread(EvalNode);

	// 進捗状況表示
	auto progressFunc = [](Timer& t) {
		while (true) {
			std::this_thread::sleep_for(std::chrono::seconds(5)); // 指定秒だけ待機し、進捗を表示する。
			const s64 madeTeacherNodes = idx;
			const double progress = static_cast<double>(madeTeacherNodes) / teacherNodes;
			auto elapsed_msec = t.elapsed();
			if (progress > 0.0) // 0 除算を回避する。
				std::cout << std::fixed << "Progress: " << std::setprecision(2) << std::min(100.0, progress * 100.0)
				<< "%, nodes:" << madeTeacherNodes
				<< ", nodes/sec:" << static_cast<double>(madeTeacherNodes) / elapsed_msec * 1000.0
				<< ", threads:" << running_threads
				<< ", Elapsed: " << elapsed_msec / 1000
				<< "[s], Remaining: " << std::max<s64>(0, elapsed_msec*(1.0 - progress) / (progress * 1000)) << "[s]" << std::endl;
			if (running_threads == 0)
				break;
		}
	};
	Timer t = Timer::currentTime();
	std::thread progressThread([&progressFunc, &t] { progressFunc(t); });

	// スレッドの終了を待機
	for (int i = 0; i < threads; i++) {
		handle[i]->join();
		running_threads--;
		delete handle[i];
		handle[i] = nullptr;
	}
	// use_nn
	handle[threads]->join();
	delete handle[threads];
	handle[threads] = nullptr;

	progressThread.join();
	ifs.close();
	ofs.close();

	std::cout << "Made " << idx << " teacher nodes in " << t.elapsed() / 1000 << " seconds." << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_MATE_ROOT_SEARCH
	const int argnum = 8;
#else
	const int argnum = 7;
#endif
	if (argc < argnum) {
		cout << "make_hcpe_by_self_play <modelfile> roots.hcp output.teacher <threads> <nodes> <playout_num>";
#ifdef USE_MATE_ROOT_SEARCH
		cout << " <eval_dir>";
#endif
		cout << endl;
		return 0;
	}

	model_path = argv[1];
	char* recordFileName = argv[2];
	char* outputFileName = argv[3];
	threads = stoi(argv[4]);
	teacherNodes = stoi(argv[5]);
	playout_num = stoi(argv[6]);
#ifdef USE_MATE_ROOT_SEARCH
	char* evalDir = argv[7];
#endif

	if (teacherNodes <= 0)
		return 0;
	if (threads <= 0 || threads > policy_value_batch_maxsize * 9 / 10)
		return 0;
	if (playout_num <= 0)
		return 0;

	logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
	logger->set_level(spdlog::level::trace);
	logger->info("{} {} {} {} {} {}", model_path, recordFileName, outputFileName, threads, teacherNodes, playout_num);

	handle = new thread*[threads + 1];

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

#ifdef USE_MATE_ROOT_SEARCH
	logger->info("init evaluator");
	// 一時オブジェクトを生成して Evaluator::init() を呼んだ直後にオブジェクトを破棄する。
	// 評価関数の次元下げをしたデータを格納する分のメモリが無駄な為、
	std::unique_ptr<Evaluator>(new Evaluator)->init(evalDir, true);
#endif

	logger->info("make_teacher");
	make_teacher(recordFileName, outputFileName);

	spdlog::drop_all();
}
