#include <cstdlib>

#include "nn_wideresnet10.h"
#include "nn_wideresnet15.h"
#include "cppshogi.h"
#include <string>

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
	softmax_temperature_with_normalize(legal_move_probabilities);

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
	int max_batch_size = std::atoi(argv[3]);
	int position_num = std::atoi(argv[4]);

	std::unique_ptr<NN> nn;
	if (std::string(model_path).find("wideresnet15") != std::string::npos)
		nn.reset((NN*)new NNWideResnet15(max_batch_size));
	else
		nn.reset((NN*)new NNWideResnet10(max_batch_size));
	nn->load_model(model_path);

	initTable();
	Position::initZobrist();

	Searcher s;
	s.init();

	// ボルツマン温度設定
	set_softmax_temperature(1.25f);

	features1_t *features1;
	features2_t *features2;
	checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * max_batch_size, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * max_batch_size, cudaHostAllocPortable));

	float* y1;
	float* y2;
	checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (int)SquareNum * max_batch_size * sizeof(float), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&y2, max_batch_size * sizeof(float), cudaHostAllocPortable));

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
	std::vector<HuffmanCodedPos> hcptmp(max_batch_size);
	std::vector<HuffmanCodedPos> hcptmp2(max_batch_size);

	// 局面初期化
	for (int i = 0; i < max_batch_size; i++) {
		positions.emplace_back(DefaultStartPositionSFEN, s.thisptr);
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
		nn->forward(max_batch_size, features1, features2, y1, y2);
		float(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);

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
				positions[idx].set(DefaultStartPositionSFEN);
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

	checkCudaErrors(cudaFreeHost(features1));
	checkCudaErrors(cudaFreeHost(features2));
	checkCudaErrors(cudaFreeHost(y1));
	checkCudaErrors(cudaFreeHost(y2));

	return 0;
}