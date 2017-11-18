#include <numeric>
#include <algorithm>

#include "cppshogi.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

// make input features
void make_input_features(const Position& position, float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum], float(*features2)[MAX_FEATURES2_NUM][SquareNum]) {
	float(*features2_hand)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_PIECES_IN_HAND_SUM][SquareNum]>(features2);

	const Bitboard occupied_bb = position.occupiedBB();

	// 駒の利き(駒種でマージ)
	Bitboard attacks[ColorNum][PieceTypeNum] = {
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
	};
	for (Square sq = SQ11; sq < SquareNum; sq++) {
		Piece p = position.piece(sq);
		if (p != Empty) {
			Color pc = pieceToColor(p);
			PieceType pt = pieceToPieceType(p);
			Bitboard bb = position.attacksFrom(pt, pc, sq, occupied_bb);
			attacks[pc][pt] |= bb;
		}
	}

	for (Color c = Black; c < ColorNum; ++c) {
		// 白の場合、色を反転
		Color c2 = c;
		if (position.turn() == White) {
			c2 = oppositeColor(c);
		}

		// 駒の配置
		Bitboard bb[PieceTypeNum];
		for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
			bb[pt] = position.bbOf(pt, c);
		}

		for (Square sq = SQ11; sq < SquareNum; ++sq) {
			// 白の場合、盤面を180度回転
			Square sq2 = sq;
			if (position.turn() == White) {
				sq2 = SQ99 - sq;
			}

			for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
				// 駒の配置
				if (bb[pt].isSet(sq)) {
					(*features1)[c2][pt - 1][sq2] = 1.0f;
				}

				// 駒の利き
				if (attacks[c][pt].isSet(sq)) {
					(*features1)[c2][PIECETYPE_NUM + pt - 1][sq2] = 1.0f;
				}
			}

			// 利き数
			const int num = std::min(MAX_ATTACK_NUM, position.attackersTo(c, sq, occupied_bb).popCount());
			for (int k = 0; k < num; k++) {
				(*features1)[c2][PIECETYPE_NUM + PIECETYPE_NUM + k][sq2] = 1.0f;
			}
		}

		// hand
		Hand hand = position.hand(c);
		int p = 0;
		for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
			u32 num = hand.numOf(hp);
			if (num >= MAX_PIECES_IN_HAND[hp]) {
				num = MAX_PIECES_IN_HAND[hp];
			}
			std::fill_n((*features2_hand)[c2][p], (int)SquareNum * num, 1.0f);
			p += MAX_PIECES_IN_HAND[hp];
		}
	}

	// is check
	if (position.inCheck()) {
		std::fill_n((*features2)[MAX_FEATURES2_HAND_NUM], SquareNum, 1.0f);
	}
}

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

// make move
int make_move_label(const u16 move16, const Position& position) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;

	// move direction
	int move_direction_label;
	if (from_sq < SquareNum) {
		// 駒の種類
		PieceType move_piece = pieceToPieceType(position.piece((Square)from_sq));

		// 白の場合、盤面を180度回転
		if (position.turn() == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;
		const int to_y = to_d.rem;
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
		const int dir_x = from_x - to_x;
		const int dir_y = to_y - from_y;

		MOVE_DIRECTION move_direction;
		if (dir_y < 0 && dir_x == 0) {
			move_direction = UP;
		}
		else if (dir_y == -2 && dir_x == -1) {
			move_direction = UP2_LEFT;
		}
		else if (dir_y == -2 && dir_x == 1) {
			move_direction = UP2_RIGHT;
		}
		else if (dir_y < 0 && dir_x < 0) {
			move_direction = UP_LEFT;
		}
		else if (dir_y < 0 && dir_x > 0) {
			move_direction = UP_RIGHT;
		}
		else if (dir_y == 0 && dir_x < 0) {
			move_direction = LEFT;
		}
		else if (dir_y == 0 && dir_x > 0) {
			move_direction = RIGHT;
		}
		else if (dir_y > 0 && dir_x == 0) {
			move_direction = DOWN;
		}
		else if (dir_y > 0 && dir_x < 0) {
			move_direction = DOWN_LEFT;
		}
		else if (dir_y > 0 && dir_x > 0) {
			move_direction = DOWN_RIGHT;
		}

		// promote
		if ((move16 & 0b100000000000000) > 0) {
			move_direction = MOVE_DIRECTION_PROMOTED[move_direction];
		}
		move_direction_label = move_direction;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (position.turn() == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
	}

	return 9 * 9 * move_direction_label + to_sq;
}

int make_move_label(const u16 move16, const PieceType move_piece, const Color color) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;

	// move direction
	int move_direction_label;
	if (from_sq < SquareNum) {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		const div_t to_d = div(to_sq, 9);
		const int to_x = to_d.quot;
		const int to_y = to_d.rem;
		const div_t from_d = div(from_sq, 9);
		const int from_x = from_d.quot;
		const int from_y = from_d.rem;
		const int dir_x = from_x - to_x;
		const int dir_y = to_y - from_y;

		MOVE_DIRECTION move_direction;
		if (dir_y < 0 && dir_x == 0) {
			move_direction = UP;
		}
		else if (dir_y == -2 && dir_x == -1) {
			move_direction = UP2_LEFT;
		}
		else if (dir_y == -2 && dir_x == 1) {
			move_direction = UP2_RIGHT;
		}
		else if (dir_y < 0 && dir_x < 0) {
			move_direction = UP_LEFT;
		}
		else if (dir_y < 0 && dir_x > 0) {
			move_direction = UP_RIGHT;
		}
		else if (dir_y == 0 && dir_x < 0) {
			move_direction = LEFT;
		}
		else if (dir_y == 0 && dir_x > 0) {
			move_direction = RIGHT;
		}
		else if (dir_y > 0 && dir_x == 0) {
			move_direction = DOWN;
		}
		else if (dir_y > 0 && dir_x < 0) {
			move_direction = DOWN_LEFT;
		}
		else if (dir_y > 0 && dir_x > 0) {
			move_direction = DOWN_RIGHT;
		}

		// promote
		if ((move16 & 0b100000000000000) > 0) {
			move_direction = MOVE_DIRECTION_PROMOTED[move_direction];
		}
		move_direction_label = move_direction;
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
	}

	return 9 * 9 * move_direction_label + to_sq;
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
	float (*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
	float (*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());

	// set all zero
	std::fill_n((float*)features1, (int)ColorNum * (PieceTypeNum-1) * (int)SquareNum * len, 0.0f);
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
	float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
	float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());

	// set all zero
	std::fill_n((float*)features1, (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum * len, 0.0f);
	std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

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
	float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
	float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());

	// set all zero
	std::fill_n((float*)features1, (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum * len, 0.0f);
	std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

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
	float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
	float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());
	float *value = reinterpret_cast<float *>(ndvalue.get_data());

	// set all zero
	std::fill_n((float*)features1, (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum * len, 0.0f);
	std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

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

void hcphe_decode_with_value(np::ndarray ndhcphe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult, np::ndarray ndvalue) {
	const int len = (int)ndhcphe.shape(0);
	HuffmanCodedPosWithHistoryAndEval *hcphe = reinterpret_cast<HuffmanCodedPosWithHistoryAndEval *>(ndhcphe.get_data());
	float(*features1)[HISTORY_NUM][ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[HISTORY_NUM][ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
	float(*features2)[HISTORY_NUM][MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[HISTORY_NUM][MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());
	int *move = reinterpret_cast<int *>(ndmove.get_data());
	int *result = reinterpret_cast<int *>(ndresult.get_data());
	float *value = reinterpret_cast<float *>(ndvalue.get_data());

	// set all zero
	std::fill_n((float*)features1, HISTORY_NUM * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum * len, 0.0f);
	std::fill_n((float*)features2, HISTORY_NUM * MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

	static Searcher* searcher = nullptr;
	if (searcher == nullptr) {
		searcher = new Searcher();
		searcher->init();
	}
	Position position(searcher);
	StateInfo state[7];
	for (int i = 0; i < len; i++, hcphe++, features1++, features2++, value++, move++, result++) {
		float(*features1_hist)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(features1);
		float(*features2_hist)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(features2);

		position.set(hcphe->hcp, searcher->threads.main());
		StateInfo* st = state;

		for (int j = 0; j < HISTORY_NUM - 1; j++, features1_hist++, features2_hist++) {
			// input features
			make_input_features(position, features1_hist, features2_hist);

			// 履歴を1手進める
			const Move mv = move16toMove(Move(hcphe->historyMove16[j]), position);
			position.doMove(mv, *st++);
		}
		// input features
		make_input_features(position, features1_hist, features2_hist);

		// move
		*move = make_move_label(hcphe->bestMove16, position);

		// game result
		*result = make_result(hcphe->gameResult, position);

		// eval
		*value = score_to_value((Score)hcphe->eval);
	}
}

// Boltzmann distribution
// see: Reinforcement Learning : An Introduction 2.3.SOFTMAX ACTION SELECTION
float beta = 1.0f / 0.67f;
void set_softmax_tempature(const float tempature) {
	beta = 1.0f / tempature;
}
void softmax_tempature(std::vector<float> &log_probabilities) {
	// apply beta exponent to probabilities(in log space)
	float max = 0.0f;
	for (float& x : log_probabilities) {
		x *= beta;
		if (x > max) {
			max = x;
		}
	}
	// オーバーフローを防止するため最大値で引く
	for (float& x : log_probabilities) {
		x = expf(x - max);
	}
}

void softmax_tempature_with_normalize(std::vector<float> &log_probabilities) {
	// apply beta exponent to probabilities(in log space)
	float max = 0.0f;
	for (float& x : log_probabilities) {
		x *= beta;
		if (x > max) {
			max = x;
		}
	}
	// オーバーフローを防止するため最大値で引く
	float sum = 0.0f;
	for (float& x : log_probabilities) {
		x = expf(x - max);
		sum += x;
	}
	// normalize
	for (float& x : log_probabilities) {
		x /= sum;
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

	void make_input_features_inner(float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum], float(*features2)[MAX_FEATURES2_NUM][SquareNum]) {
		// set all zero
		std::fill_n((float*)features1, (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
		std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

		// input features
		::make_input_features(*m_position, features1, features2);
	}

	int make_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
		float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());

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

	const int get_ply() const { return ply; }

	void print() {
		std::cout << ply << std::endl;
		m_position->print();
	}
};
void Engine::setup_eval_dir(const char* eval_dir) {
	std::unique_ptr<Evaluator>(new Evaluator)->init(eval_dir, true);
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
		for (int i = 0; i < engines.size(); i++) {
			engines[i].default_start_position();
			unfinished[i] = true;
		}
		unfinished_states_num = unfinished.size();
		prev_score.clear();
	}

	void make_odd_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		const int len = engines.size() / 2;
		float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
		float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());

		// set all zero
		std::fill_n((float*)features1, (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum * len, 0.0f);
		std::fill_n((float*)features2, MAX_FEATURES2_NUM * (int)SquareNum * len, 0.0f);

		// input features
		for (int i = 0; i < len; i++, features1++, features2++) {
			::make_input_features(engines[i * 2 + 1].get_position(), features1, features2);
		}
	}

	void do_odd_moves(np::ndarray ndlogits) {
		const int len = engines.size() / 2;
		float (*logits)[MAX_MOVE_LABEL_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM][SquareNum]>(ndlogits.get_data());

		for (int i = 0; i < len; i++, logits++) {
			Move move = engines[i * 2 + 1].select_move_inner((float*)logits);
			engines[i * 2 + 1].do_move(move);
		}
	}

	py::list make_unfinished_input_features(np::ndarray ndfeatures1, np::ndarray ndfeatures2) {
		const int len = engines.size();
		float(*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = reinterpret_cast<float(*)[ColorNum][MAX_FEATURES1_NUM][SquareNum]>(ndfeatures1.get_data());
		float(*features2)[MAX_FEATURES2_NUM][SquareNum] = reinterpret_cast<float(*)[MAX_FEATURES2_NUM][SquareNum]>(ndfeatures2.get_data());

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
			for (int i = 0; i < engines.size(); i++) {
				// eval
				Score score;
				engines[i].eval(&score);
				prev_score.emplace_back(score);
			}
		}

		for (int i = 0; i < engines.size(); i++) {
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
		for (int i = 0; i < this->wons.size(); i++, wons++) {
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
	py::def("hcphe_decode_with_value", hcphe_decode_with_value);

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