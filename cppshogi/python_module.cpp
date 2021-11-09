#include <numeric>
#include "cppshogi.h"

void init() {
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
}

// make result
inline float make_result(const uint8_t result, const Color color) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if ((color == Black && gameResult == BlackWin) ||
		(color == White && gameResult == WhiteWin)) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}
template<typename T>
inline T is_sennichite(const uint8_t result) {
	return static_cast<T>(result & GAMERESULT_SENNICHITE ? 1 : 0);
}
template<typename T>
inline T is_nyugyoku(const uint8_t result) {
	return static_cast<T>(result & GAMERESULT_NYUGYOKU ? 1 : 0);
}

void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue) {
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	int64_t* move = reinterpret_cast<int64_t*>(ndmove);
	float *result = reinterpret_cast<float *>(ndresult);
	float *value = reinterpret_cast<float *>(ndvalue);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (size_t i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());

		// game result
		*result = make_result(hcpe->gameResult, position.turn());

		// eval
		*value = score_to_value((Score)hcpe->eval);
	}
}

void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux) {
	HuffmanCodedPosAndEval2 *hcpe = reinterpret_cast<HuffmanCodedPosAndEval2 *>(ndhcpe2);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	int64_t* move = reinterpret_cast<int64_t*>(ndmove);
	float* result = reinterpret_cast<float*>(ndresult);
	float* value = reinterpret_cast<float*>(ndvalue);
	auto aux = reinterpret_cast<float(*)[2]>(ndaux);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (size_t i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++, aux++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());

		// game result
		*result = make_result(hcpe->result, position.turn());

		// eval
		*value = score_to_value((Score)hcpe->eval);

		// sennichite
		(*aux)[0] = is_sennichite<float>(hcpe->result);

		// nyugyoku
		(*aux)[1] = is_nyugyoku<float>(hcpe->result);
	}
}

std::vector<TrainingData> trainingData;
// 重複チェック用 局面に対応するtrainingDataのインデックスを保持
std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;

// hcpe形式の指し手をone-hotの方策として読み込む
size_t load_hcpe(const std::string& filepath, std::ifstream& ifs, bool use_average, const double eval_scale, int& len) {
	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			break;
		}

		const int eval = (int)(hcpe.eval * eval_scale);
		if (use_average) {
			auto ret = duplicates.emplace(hcpe.hcp, trainingData.size());
			if (ret.second) {
				auto& data = trainingData.emplace_back(
					hcpe.hcp,
					eval,
					make_result(hcpe.gameResult, hcpe.hcp.color())
				);
				data.candidates[hcpe.bestMove16] = 1;
			}
			else {
				// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
				auto& data = trainingData[ret.first->second];
				data.eval += eval;
				data.result += make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates[hcpe.bestMove16] += 1;
				data.count++;
			}
		}
		else {
			auto& data = trainingData.emplace_back(
				hcpe.hcp,
				eval,
				make_result(hcpe.gameResult, hcpe.hcp.color())
			);
			data.candidates[hcpe.bestMove16] = 1;
		}
		++len;
	}

	return trainingData.size();
}

template <bool add>
inline void visits_to_proberbility(TrainingData& data, const std::vector<MoveVisits>& candidates, const double temperature) {
	if (candidates.size() == 1) {
		// one-hot
		const auto& moveVisits = candidates[0];
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 0) {
		// greedy
		const auto itr = std::max_element(candidates.begin(), candidates.end(), [](const MoveVisits& a, const MoveVisits& b) { return a.visitNum < b.visitNum; });
		const MoveVisits& moveVisits = *itr;
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 1) {
		const float sum_visitNum = (float)std::accumulate(candidates.begin(), candidates.end(), 0, [](int acc, const MoveVisits& move_visits) { return acc + move_visits.visitNum; });
		for (const auto& moveVisits : candidates) {
			const float proberbility = (float)moveVisits.visitNum / sum_visitNum;
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
	else {
		double exponentiated_visits[593];
		double sum = 0;
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const auto new_visits = std::pow(moveVisits.visitNum, 1.0 / temperature);
			exponentiated_visits[i] = new_visits;
			sum += new_visits;
		}
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const float proberbility = (float)(exponentiated_visits[i] / sum);
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
}

// フォーマット自動判別
bool is_hcpe(std::ifstream& ifs) {
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// 最後のデータがhcpeであるかで判別
		ifs.seekg(-sizeof(HuffmanCodedPosAndEval), std::ios_base::end);
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (hcpe.hcp.isOK() && hcpe.bestMove16 >= 1 && hcpe.bestMove16 <= 26703) {
			return true;
		}
	}
	return false;
}

// hcpe3形式のデータを読み込み、ランダムアクセス可能なように加工し、trainingDataに保存する
// 複数回呼ぶことで、複数ファイルの読み込みが可能
size_t __load_hcpe3(const std::string& filepath, bool use_average, bool use_opponent, double a, double a_opponent, double temperature, int& len) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return trainingData.size();

	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;
	const double eval_scale_opponent = a_opponent == 0 ? 1 : 756.0864962951762 / a_opponent;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return load_hcpe(filepath, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	std::vector<MoveVisits> candidates;

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}
		assert(hcpe3.moveNum <= 513);

		// 開始局面
		Position pos;
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}
		StateListPtr states{ new std::deque<StateInfo>(1) };

		int opponent_turn = -1;
		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);

			// candidateNum==0の手は読み飛ばす
			if (moveInfo.candidateNum > 0 || opponent_turn >= 0) {
				// opponentの場合、ランダムムーブを除外するため自分の教師局面現れた以降の局面対象とする
				if (use_opponent && opponent_turn < 0 && hcpe3.opponent > 0)
					opponent_turn = hcpe3.opponent - 1;

				int eval;
				if (moveInfo.candidateNum > 0) {
					candidates.resize(moveInfo.candidateNum);
					ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);
					eval = (int)(moveInfo.eval * eval_scale);
				}
				else if (pos.turn() == opponent_turn) {
					// opponentの場合、指し手のみを候補手にする
					candidates.resize(1);
					candidates[0].move16 = moveInfo.selectedMove16;
					candidates[0].visitNum = 1;
					eval = (int)(moveInfo.eval * eval_scale_opponent);
				}

				const auto hcp = pos.toHuffmanCodedPos();
				if (use_average) {
					auto ret = duplicates.emplace(hcp, trainingData.size());
					if (ret.second) {
						auto& data = trainingData.emplace_back(
							hcp,
							eval,
							make_result(hcpe3.result, pos.turn())
						);
						visits_to_proberbility<false>(data, candidates, temperature);
					}
					else {
						// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
						auto& data = trainingData[ret.first->second];
						data.eval += eval;
						data.result += make_result(hcpe3.result, pos.turn());
						visits_to_proberbility<true>(data, candidates, temperature);
						data.count++;

					}
				}
				else {
					auto& data = trainingData.emplace_back(
						hcp,
						eval,
						make_result(hcpe3.result, pos.turn())
					);
					visits_to_proberbility<false>(data, candidates, temperature);
				}
				++len;
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return trainingData.size();
}

// load_hcpe3で読み込み済みのtrainingDataから、インデックスを使用してサンプリングする
// 重複データは平均化する
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue) {
	unsigned int* index = reinterpret_cast<unsigned int*>(ndindex);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	auto probability = reinterpret_cast<float(*)[9 * 9 * MAX_MOVE_LABEL_NUM]>(ndprobability);
	float* result = reinterpret_cast<float*>(ndresult);
	float* value = reinterpret_cast<float*>(ndvalue);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)probability, 9 * 9 * MAX_MOVE_LABEL_NUM * len, 0.0f);

	Position position;
	for (size_t i = 0; i < len; i++, index++, features1++, features2++, value++, probability++, result++) {
		auto& hcpe3 = trainingData[*index];

		position.set(hcpe3.hcp);

		// input features
		make_input_features(position, features1, features2);

		// move probability
		for (const auto kv : hcpe3.candidates) {
			const auto label = make_move_label(kv.first, position.turn());
			assert(label < 9 * 9 * MAX_MOVE_LABEL_NUM);
			(*probability)[label] = kv.second / hcpe3.count;
		}

		// game result
		*result = hcpe3.result / hcpe3.count;

		// eval
		*value = score_to_value((Score)(hcpe3.eval / hcpe3.count));
	}
}

// evalの補正用データ準備
std::vector<int> eval;
std::vector<float> result;
std::vector<int> eval_opponent;
std::vector<float> result_opponent;
std::vector<size_t> __load_evalfix(const std::string& filepath) {
	eval.clear();
	result.clear();
	eval_opponent.clear();
	result_opponent.clear();

	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return { 0, 0 };

	// フォーマット自動判別
	bool hcpe3 = !is_hcpe(ifs);
	ifs.seekg(std::ios_base::beg);

	if (hcpe3) {
		for (int p = 0; ifs; ++p) {
			HuffmanCodedPosAndEval3 hcpe3;
			ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
			if (ifs.eof()) {
				break;
			}
			assert(hcpe3.moveNum <= 513);

			// 開始局面
			Position pos;
			if (!pos.set(hcpe3.hcp)) {
				std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
				ss << filepath << "(" << p << ")";
				throw std::runtime_error(ss.str());
			}
			StateListPtr states{ new std::deque<StateInfo>(1) };

			int opponent_turn = -1;
			for (int i = 0; i < hcpe3.moveNum; ++i) {
				MoveInfo moveInfo;
				ifs.read((char*)&moveInfo, sizeof(MoveInfo));
				assert(moveInfo.candidateNum <= 593);

				// candidateNum==0の手は読み飛ばす
				if (moveInfo.candidateNum > 0) {
					// opponentの場合、ランダムムーブを除外するため自分の教師局面現れた以降の局面対象とする
					if (opponent_turn < 0 && hcpe3.opponent > 0)
						opponent_turn = hcpe3.opponent - 1;

					ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
					// 詰みは除く
					if (std::abs(moveInfo.eval) < 30000) {
						eval.emplace_back(moveInfo.eval);
						result.emplace_back(make_result(hcpe3.result, pos.turn()));
					}
				}
				else if (pos.turn() == opponent_turn) {
					// 詰みは除く
					if (std::abs(moveInfo.eval) < 30000) {
						eval_opponent.emplace_back(moveInfo.eval);
						result_opponent.emplace_back(make_result(hcpe3.result, pos.turn()));
					}
				}

				assert(moveInfo.selectedMove16 <= 0x7fff);
				const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
				pos.doMove(move, states->emplace_back(StateInfo()));
			}
		}
	}
	else {
		// hcpeフォーマット
		for (int p = 0; ifs; ++p) {
			HuffmanCodedPosAndEval hcpe;
			ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
			if (ifs.eof()) {
				break;
			}

			// 詰みは除く
			if (std::abs(hcpe.eval) < 30000) {
				eval.emplace_back(hcpe.eval);
				result.emplace_back(make_result(hcpe.gameResult, hcpe.hcp.color()));
			}
		}
	}

	return { result.size(), result_opponent.size() };
}

void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult) {
	std::copy(eval.begin(), eval.end(), reinterpret_cast<int*>(ndeval));
	std::copy(result.begin(), result.end(), reinterpret_cast<float*>(ndresult));
}

void __hcpe3_prepare_evalfix_opponent(char* ndeval, char* ndresult) {
	std::copy(eval_opponent.begin(), eval_opponent.end(), reinterpret_cast<int*>(ndeval));
	std::copy(result_opponent.begin(), result_opponent.end(), reinterpret_cast<float*>(ndresult));
}
