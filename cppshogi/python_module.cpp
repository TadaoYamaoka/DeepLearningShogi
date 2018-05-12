#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>

#include "cppshogi.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

// make result
inline int make_result(const GameResult gameResult, const Position& position) {
	if (position.turn() == Black && gameResult == BlackWin ||
		position.turn() == White && gameResult == WhiteWin) {
		return 1;
	}
	else {
		return 0;
	}
}

/*
HuffmanCodedPosAndEvalの配列からpolicy networkの入力ミニバッチに変換する。
ndhcpe : Python側で以下の構造体を定義して、その配列を入力する。
HuffmanCodedPosAndEval = np.dtype([
('hcp', np.uint8, 32),
('eval', np.int16),
('bestMove16', np.uint16),
('gameResult', np.uint8),
('dummy', np.uint8),
])
ndfeatures1, ndfeatures2, ndresult : 変換結果を受け取る。Python側でnp.emptyで事前に領域を確保する。
*/
void hcpe_decode_with_result(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndresult) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());

	// set all zero
	std::fill_n((float*)features1, (int)ColorNum * (PieceTypeNum - 1) * (int)SquareNum * len, 0.0f);
	std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, result++) {
		position.set(hcpe->hcp, nullptr);

		// input features
		make_input_features(position, features1, features2);

		// game result
		*result = make_result(hcpe->gameResult, position);
	}
}

void hcpe_decode_with_move(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++) {
		position.set(hcpe->hcp, nullptr);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position);
	}
}

void hcpe_decode_with_move_result(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++, result++) {
		position.set(hcpe->hcp, nullptr);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position);

		// game result
		*result = make_result(hcpe->gameResult, position);
	}
}

void hcpe_decode_with_value(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult, np::ndarray ndvalue) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());
	float *value = reinterpret_cast<float *>(ndvalue.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++) {
		position.set(hcpe->hcp, nullptr);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position);

		// game result
		*result = make_result(hcpe->gameResult, position);

		// eval
		*value = score_to_value((Score)hcpe->eval);
	}
}

class Engine {
private:
	Searcher *m_searcher;
	Position *m_position;
	std::deque<StateInfo> states;
	Ply ply;
public:
	static void setup_eval_dir(const char* eval_dir);

	Engine() : states(1) {
		m_searcher = new Searcher();
		m_position = new Position(m_searcher);

		m_searcher->init();
		const std::string options[] = {
			"name Threads value 1",
			"name MultiPV value 1",
			"name USI_Hash value 256",
			"name OwnBook value false",
			"name Max_Random_Score_Diff value 0" };
		for (auto& str : options) {
			std::istringstream is(str);
			m_searcher->setOption(is);
		}
	}

	void position(const char* cmd) {
		std::istringstream ssCmd(cmd);
		setPosition(*m_position, ssCmd);
		//g_position->print();
	}

	Move eval(Score* score) {
		m_position->searcher()->alpha = -ScoreMaxEvaluate;
		m_position->searcher()->beta = ScoreMaxEvaluate;

		// go
		LimitsType limits;
		limits.depth = static_cast<Depth>(6);
		m_position->searcher()->threads.startThinking(*m_position, limits, m_position->searcher()->states);
		m_position->searcher()->threads.main()->waitForSearchFinished();

		*score = m_position->searcher()->threads.main()->rootMoves[0].score;
		return m_position->searcher()->threads.main()->rootMoves[0].pv[0];
	}

	py::list go() {
		Score score;
		const Move bestMove = eval(&score);

		std::cout << "info score cp " << score << " pv " << bestMove.toUSI() << std::endl;

		py::list ret;
		ret.append(bestMove.toUSI());
		ret.append((int)score);
		return ret;
	}

	void make_input_features_inner(features1_t* features1, features2_t* features2) {
		// set all zero
		std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float), 0.0f);
		std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float), 0.0f);

		// input features
		::make_input_features(*m_position, features1, features2);
	}

	int make_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
		features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());

		// input features
		make_input_features_inner(features1, features2);

		return (int)m_position->turn();
	}

	const Move select_move_inner(float *logits) {
		// 合法手一覧
		std::vector<Move> legal_moves;
		std::vector<float> legal_move_probabilities;
		for (MoveList<Legal> ml(*m_position); !ml.end(); ++ml) {
			const Move move = ml.move();

			const int move_label = make_move_label((u16)move.proFromAndTo(), *m_position);

			legal_moves.emplace_back(move);
			legal_move_probabilities.emplace_back(logits[move_label]);
		}

		/*for (int i = 0; i < legal_move_probabilities.size(); i++) {
		std::cout << "info string i:" << i << " move:" << legal_moves[i].toUSI() << " p:" << legal_move_probabilities[i] << std::endl;
		}*/

		// Boltzmann distribution
		softmax_tempature(legal_move_probabilities);

		/*for (int i = 0; i < legal_move_probabilities.size(); i++) {
		std::cout << "info string i:" << i << " move:" << legal_moves[i].toUSI() << " temp_p:" << legal_move_probabilities[i] << std::endl;
		}*/

		// 確率に応じて手を選択
		std::discrete_distribution<int> distribution(legal_move_probabilities.begin(), legal_move_probabilities.end());
		int move_idx = distribution(g_randomTimeSeed);

		// greedy
		//int move_idx = std::distance(legal_move_probabilities.begin(), std::max_element(legal_move_probabilities.begin(), legal_move_probabilities.end()));

		return legal_moves[move_idx];
	}

	const std::string select_move(np::ndarray ndlogits) {
		float *logits = reinterpret_cast<float*>(ndlogits.get_data());

		Move move = select_move_inner(logits);

		// ラベルからusiに変換
		return move.toUSI();
	}

	void default_start_position() {
		m_position->set(DefaultStartPositionSFEN, m_searcher->threads.main());
		states.clear();
		ply = m_position->gamePly();
	}

	const Position& get_position() const { return *m_position; }

	void do_move(const Move& move) {
		states.push_back(StateInfo());
		m_position->doMove(move, states.back());
		ply++;
	}

	Ply get_ply() const { return ply; }

	void print() {
		std::cout << ply << std::endl;
		m_position->print();
	}
};
void Engine::setup_eval_dir(const char* eval_dir) {
	std::unique_ptr<Evaluator>(new Evaluator)->init(eval_dir);
}

class States {
private:
	std::vector<Engine> engines;
	std::vector<bool> unfinished;
	int unfinished_states_num;
	std::vector<int> wons;
	std::vector<Score> prev_score;

public:
	States(const int num) {
		for (int i = 0; i < num; i++) {
			engines.emplace_back();
			unfinished.emplace_back(true);
			wons.emplace_back();
		}
	}

	void default_start_position() {
		for (size_t i = 0; i < engines.size(); i++) {
			engines[i].default_start_position();
			unfinished[i] = true;
		}
		unfinished_states_num = unfinished.size();
		prev_score.clear();
	}

	void make_odd_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		const int len = engines.size() / 2;
		features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
		features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());

		// set all zero
		std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
		std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

		// input features
		for (int i = 0; i < len; i++, features1++, features2++) {
			::make_input_features(engines[i * 2 + 1].get_position(), features1, features2);
		}
	}

	void do_odd_moves(np::ndarray ndlogits) {
		const int len = engines.size() / 2;
		float(*logits)[MAX_MOVE_LABEL_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM][SquareNum]>(ndlogits.get_data());

		for (int i = 0; i < len; i++, logits++) {
			Move move = engines[i * 2 + 1].select_move_inner((float*)logits);
			engines[i * 2 + 1].do_move(move);
		}
	}

	py::list make_unfinished_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		const int len = engines.size();
		features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
		features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());

		py::list ret;

		// input features
		for (int i = 0; i < len; i++) {
			if (unfinished[i]) {
				engines[i].make_input_features_inner(features1, features2);
				ret.append(i);

				features1++;
				features2++;
			}
		}

		return ret;
	}

	int do_unfinished_moves_and_eval(const bool is_learner, np::ndarray ndlogits, np::ndarray ndlabels, np::ndarray ndvalues) {
		const int len = (int)ndlogits.shape(0);
		float(*logits)[MAX_MOVE_LABEL_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM][SquareNum]>(ndlogits.get_data());
		int *labels = reinterpret_cast<int*>(ndlabels.get_data());
		float *values = reinterpret_cast<float*>(ndvalues.get_data());

		// 初期局面の評価
		if (prev_score.size() == 0) {
			for (size_t i = 0; i < engines.size(); i++) {
				// eval
				Score score;
				engines[i].eval(&score);
				prev_score.emplace_back(score);
			}
		}

		for (size_t i = 0; i < engines.size(); i++) {
			if (unfinished[i]) {
				// select move
				Move move = engines[i].select_move_inner((float*)logits);
				*labels = make_move_label((u16)move.proFromAndTo(), engines[i].get_position());

				// 現在の局面の評価値は前のmoveの後で評価済み
				*values = score_to_value(is_learner ? -prev_score[i] : prev_score[i]);

				engines[i].do_move(move);

				// Move後の評価
				Score score;
				const Move bestMove = engines[i].eval(&score);

				// 終局判定
				if (abs(score) > 3000) {
					unfinished[i] = false;
					wons[i] = is_learner ? 1 : 0;
					unfinished_states_num--;
				}
				else if (engines[i].get_ply() > 255) {
					// 引き分け
					unfinished[i] = false;
					wons[i] = 0;
					unfinished_states_num--;
				}
				else {
					// 評価値を保存
					prev_score[i] = score;
				}

				logits++;
				labels++;
				values++;
			}
		}

		return unfinished_states_num;
	}

	void get_learner_wons(np::ndarray ndwons) {
		int *wons = reinterpret_cast<int*>(ndwons.get_data());
		for (size_t i = 0; i < this->wons.size(); i++, wons++) {
			*wons = this->wons[i];
		}
	}
};

BOOST_PYTHON_MODULE(cppshogi) {
	Py_Initialize();
	np::initialize();

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

	py::def("hcpe_decode_with_result", hcpe_decode_with_result);
	py::def("hcpe_decode_with_move", hcpe_decode_with_move);
	py::def("hcpe_decode_with_move_result", hcpe_decode_with_move_result);
	py::def("hcpe_decode_with_value", hcpe_decode_with_value);

	py::def("setup_eval_dir", Engine::setup_eval_dir);
	py::def("set_softmax_tempature", set_softmax_tempature);
	py::class_<Engine>("Engine")
		.def("position", &Engine::position)
		.def("go", &Engine::go)
		.def("make_input_features", &Engine::make_input_features)
		.def("select_move", &Engine::select_move)
		;
	py::class_<States>("States", py::init<const int>())
		.def("default_start_position", &States::default_start_position)
		.def("make_odd_input_features", &States::make_odd_input_features)
		.def("do_odd_moves", &States::do_odd_moves)
		.def("make_unfinished_input_features", &States::make_unfinished_input_features)
		.def("do_unfinished_moves_and_eval", &States::do_unfinished_moves_and_eval)
		.def("get_learner_wons", &States::get_learner_wons)
		;
}