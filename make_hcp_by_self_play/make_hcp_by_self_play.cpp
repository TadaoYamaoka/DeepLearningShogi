#include <cstdlib>

#include "cppshogi.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

void randomMove(Position& pos, std::mt19937& mt);

const Move select_move(const Position pos, float *logits) {
	// 合法手一覧
	std::vector<Move> legal_moves;
	std::vector<float> legal_move_probabilities;
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		const Move move = ml.move();

		const int move_label = make_move_label((u16)move.proFromAndTo(), pos);

		legal_moves.emplace_back(move);
		legal_move_probabilities.emplace_back(logits[move_label]);
	}

	if (legal_moves.size() == 0) {
		return Move::moveNone();
	}

	// Boltzmann distribution
	softmax_tempature_with_normalize(legal_move_probabilities);

	// 確率に応じて手を選択
	std::discrete_distribution<int> distribution(legal_move_probabilities.begin(), legal_move_probabilities.end());
	int move_idx = distribution(g_randomTimeSeed);

	return legal_moves[move_idx];
}

int main(int argc, char** argv)
{
	if (argc < 5) {
		std::cout << "model_path outfile batch_size position_num" << std::endl;
		return 1;
	}

	const char* model_path = argv[1];
	const char* outfile = argv[2];
	int batch_size = std::atoi(argv[3]);
	int position_num = std::atoi(argv[4]);

	// Boost.PythonとBoost.Numpyの初期化
	Py_Initialize();
	np::initialize();

	// Pythonモジュール読み込み
	py::object dlshogi_ns = py::import("dlshogi.test").attr("__dict__");

	// modelロード
	py::object dlshogi_load_model = dlshogi_ns["load_model"];
	dlshogi_load_model(model_path);

	// 予測関数取得
	py::object dlshogi_predict = dlshogi_ns["predict"];

	initTable();
	Position::initZobrist();

	Searcher s;
	s.init();

	// ボルツマン温度設定
	set_softmax_tempature(1.25f);

	float (*features1)[ColorNum][MAX_FEATURES1_NUM][SquareNum] = new float[batch_size][ColorNum][MAX_FEATURES1_NUM][SquareNum];
	float (*features2)[MAX_FEATURES2_NUM][SquareNum] = new float[batch_size][MAX_FEATURES2_NUM][SquareNum];

	std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> dist(4, 250);
	std::uniform_int_distribution<int> doRandomDist(0, 30);

	std::vector<HuffmanCodedPos> hcpvec;

	std::ofstream ofs(outfile, std::ios::binary);
	if (!ofs) {
		std::cerr << "Error: cannot open " << outfile << std::endl;
		exit(EXIT_FAILURE);
	}

	// 進捗状況表示
	std::atomic<s64> index = 0;
	Timer t = Timer::currentTime();
	auto progressFunc = [&position_num](std::atomic<s64>& index, Timer& t) {
		while (true) {
			std::this_thread::sleep_for(std::chrono::seconds(5)); // 指定秒だけ待機し、進捗を表示する。
			const s64 madeTeacherNodes = index;
			const double progress = static_cast<double>(madeTeacherNodes) / position_num;
			auto elapsed_msec = t.elapsed();
			if (progress > 0.0) // 0 除算を回避する。
				std::cout << std::fixed << "Progress: " << std::setprecision(2) << std::min(100.0, progress * 100.0)
				<< "%, Elapsed: " << elapsed_msec / 1000
				<< "[s], Remaining: " << std::max<s64>(0, elapsed_msec*(1.0 - progress) / (progress * 1000)) << "[s]" << std::endl;
			if (index >= position_num)
				break;
		}
	};
	std::thread progressThread([&index, &progressFunc, &t] { progressFunc(index, t); });

	std::vector<Position> positions;
	std::vector<int> maxply;
	std::vector<int> tmpply;
	std::vector<int> tmpply2;
	std::vector<int> ply;
	std::vector<StateListPtr> stateLists;
	std::vector<HuffmanCodedPos> hcptmp(batch_size);
	std::vector<HuffmanCodedPos> hcptmp2(batch_size);

	// 局面初期化
	for (int i = 0; i < batch_size; i++) {
		positions.emplace_back(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);
		maxply.emplace_back(dist(mt));
		int maxply2 = std::uniform_int_distribution<int>(8, maxply[i])(mt);
		tmpply.emplace_back(maxply2);
		tmpply2.emplace_back(std::uniform_int_distribution<int>(8, maxply2)(mt));
		ply.emplace_back(1);
		stateLists.emplace_back(new std::deque<StateInfo>(1));
	}

	while (hcpvec.size() < position_num) {

		// set all zero
		std::fill_n((float*)features1, positions.size() * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
		std::fill_n((float*)features2, positions.size() * MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

		// make input_features
		for (int idx = 0; idx < positions.size(); idx++) {
			make_input_features(positions[idx], &features1[idx], &features2[idx]);
		}

		// predict
		np::ndarray ndfeatures1 = np::from_data(
			features1,
			np::dtype::get_builtin<float>(),
			py::make_tuple(positions.size(), (int)ColorNum * MAX_FEATURES1_NUM, 9, 9),
			py::make_tuple(sizeof(float)*(int)ColorNum*MAX_FEATURES1_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
			py::object());

		np::ndarray ndfeatures2 = np::from_data(
			features2,
			np::dtype::get_builtin<float>(),
			py::make_tuple(positions.size(), MAX_FEATURES2_NUM, 9, 9),
			py::make_tuple(sizeof(float)*MAX_FEATURES2_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
			py::object());

		auto ret = dlshogi_predict(ndfeatures1, ndfeatures2);
		np::ndarray y_data = py::extract<np::ndarray>(ret);
		float(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y_data.get_data());

		// do move
		for (int idx = 0; idx < positions.size(); idx++, logits++) {
			Move move = select_move(positions[idx], (float*)logits);

			if (move != Move::moveNone()) {

				stateLists[idx]->push_back(StateInfo());
				positions[idx].doMove(move, stateLists[idx]->back());

				ply[idx]++;

				// 出力判定
				if (ply[idx] == maxply[idx]) {
					hcpvec.emplace_back(positions[idx].toHuffmanCodedPos());
					index++;
				}
				else if (ply[idx] == tmpply[idx]) {
					hcptmp[idx] = positions[idx].toHuffmanCodedPos();
				}
				else if (ply[idx] == tmpply2[idx]) {
					hcptmp2[idx] = positions[idx].toHuffmanCodedPos();
				}
			}
			else {
				// 終局の場合、暫定で保存した局面を出力
				if (ply[idx] > tmpply[idx]) {
					hcpvec.emplace_back(hcptmp[idx]);
					index++;
				}
				else if (ply[idx] > tmpply2[idx]) {
					hcpvec.emplace_back(hcptmp2[idx]);
					index++;
				}
			}

			// 次のゲーム
			if (move == Move::moveNone() || ply[idx] >= maxply[idx]) {
				positions[idx].set(DefaultStartPositionSFEN, s.threads.main());
				maxply[idx] = dist(mt);
				int maxply2 = std::uniform_int_distribution<int>(8, maxply[idx])(mt);
				tmpply.emplace_back(maxply2);
				tmpply2.emplace_back(std::uniform_int_distribution<int>(8, maxply2)(mt));
				ply[idx] = 1;
				stateLists[idx]->clear();
			}
			else {
				// 低い確率でランダムムーブを入れる
				if (doRandomDist(mt) == 0 && !positions[idx].inCheck()) {
					randomMove(positions[idx], mt);
				}
			}
		}
	}

	// 出力
	ofs.write(reinterpret_cast<char*>(hcpvec.data()), sizeof(HuffmanCodedPos) * hcpvec.size());

	progressThread.join();
	Py_Finalize();

	return 0;
}