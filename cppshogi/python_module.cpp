#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>

#include <numeric>
#include "cppshogi.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

class ReleaseGIL {
public:
	ReleaseGIL() {
		save_state = PyEval_SaveThread();
	}

	~ReleaseGIL() {
		PyEval_RestoreThread(save_state);
	}
private:
	PyThreadState* save_state;
};

// make result
inline float make_result(const uint8_t result, const Color color) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if (color == Black && gameResult == BlackWin ||
		color == White && gameResult == WhiteWin) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}
template<typename T>
inline T is_sennichite(const uint8_t result) {
	return result & GAMERESULT_SENNICHITE ? 1 : 0;
}
template<typename T>
inline T is_nyugyoku(const uint8_t result) {
	return result & GAMERESULT_NYUGYOKU ? 1 : 0;
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
	float *result = reinterpret_cast<float *>(ndresult.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, result++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, features1, features2);

		// game result
		*result = make_result(hcpe->gameResult, position.turn());
	}
}

void hcpe_decode_with_move(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int64_t* move = reinterpret_cast<int64_t*>(ndmove.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());
	}
}

void hcpe_decode_with_move_result(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int64_t* move = reinterpret_cast<int64_t*>(ndmove.get_data());
	float *result = reinterpret_cast<float *>(ndresult.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++, result++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, features1, features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());

		// game result
		*result = make_result(hcpe->gameResult, position.turn());
	}
}

void hcpe_decode_with_value(np::ndarray ndhcpe, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult, np::ndarray ndvalue) {
	const int len = (int)ndhcpe.shape(0);
	HuffmanCodedPosAndEval *hcpe = reinterpret_cast<HuffmanCodedPosAndEval *>(ndhcpe.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int64_t* move = reinterpret_cast<int64_t*>(ndmove.get_data());
	float *result = reinterpret_cast<float *>(ndresult.get_data());
	float *value = reinterpret_cast<float *>(ndvalue.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++) {
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

void hcpe2_decode_with_value(np::ndarray ndhcpe2, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult, np::ndarray ndvalue, np::ndarray ndaux) {
	const int len = (int)ndhcpe2.shape(0);
	HuffmanCodedPosAndEval2 *hcpe = reinterpret_cast<HuffmanCodedPosAndEval2 *>(ndhcpe2.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int64_t* move = reinterpret_cast<int64_t*>(ndmove.get_data());
	float* result = reinterpret_cast<float*>(ndresult.get_data());
	float* value = reinterpret_cast<float*>(ndvalue.get_data());
	auto aux = reinterpret_cast<float(*)[2]>(ndaux.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++, aux++) {
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
std::unordered_map<HuffmanCodedPos, int> duplicates;

// hcpe形式の指し手をone-hotの方策として読み込む
py::object load_hcpe(const std::string& filepath, std::ifstream& ifs, bool use_average, const double eval_scale) {
	int len = 0;

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

	return py::make_tuple((int)trainingData.size(), len);
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
			const auto new_visits = std::pow(moveVisits.visitNum, temperature);
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

// hcpe3形式のデータを読み込み、ランダムアクセス可能なように加工し、trainingDataに保存する
// 複数回呼ぶことで、複数ファイルの読み込みが可能
py::object load_hcpe3(std::string filepath, bool use_average, double a, double temperature) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return py::make_tuple((int)trainingData.size(), 0);

	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// 最後の1byteが0であるかで判別
		ifs.seekg(-1, std::ios_base::end);
		if (ifs.get() == 0) {
			ifs.seekg(std::ios_base::beg);
			return load_hcpe(filepath, ifs, use_average, eval_scale);
		}
	}
	ifs.seekg(std::ios_base::beg);

	int len = 0;
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

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);

			// candidateNum==0の手は読み飛ばす
			if (moveInfo.candidateNum > 0) {
				candidates.resize(moveInfo.candidateNum);
				ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

				const auto hcp = pos.toHuffmanCodedPos();
				const int eval = (int)(moveInfo.eval * eval_scale);
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

	return py::make_tuple((int)trainingData.size(), len);
}

// load_hcpe3で読み込み済みのtrainingDataから、インデックスを使用してサンプリングする
// 重複データは平均化する
void hcpe3_decode_with_value(np::ndarray ndindex, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndprobability, np::ndarray ndresult, np::ndarray ndvalue) {
	const size_t len = (size_t)ndindex.shape(0);
	int* index = reinterpret_cast<int*>(ndindex.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	auto probability = reinterpret_cast<float(*)[9 * 9 * MAX_MOVE_LABEL_NUM]>(ndprobability.get_data());
	float* result = reinterpret_cast<float*>(ndresult.get_data());
	float* value = reinterpret_cast<float*>(ndvalue.get_data());
	ReleaseGIL unlock = ReleaseGIL();

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)probability, 9 * 9 * MAX_MOVE_LABEL_NUM * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, index++, features1++, features2++, value++, probability++, result++) {
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
py::object hcpe3_prepare_evalfix(std::string filepath) {
	std::vector<int> eval;
	std::vector<float> result;

	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return py::object();

	// フォーマット自動判別
	bool hcpe3 = true;
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// 最後の1byteが0であるかで判別
		ifs.seekg(-1, std::ios_base::end);
		if (ifs.get() == 0) {
			hcpe3 = false;
		}
	}
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

			for (int i = 0; i < hcpe3.moveNum; ++i) {
				MoveInfo moveInfo;
				ifs.read((char*)&moveInfo, sizeof(MoveInfo));
				assert(moveInfo.candidateNum <= 593);

				// candidateNum==0の手は読み飛ばす
				if (moveInfo.candidateNum > 0) {
					ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
					// 詰みは除く
					if (std::abs(moveInfo.eval) < 30000) {
						eval.emplace_back(moveInfo.eval);
						result.emplace_back(make_result(hcpe3.result, pos.turn()));
					}
				}

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

	Py_intptr_t size[]{ eval.size() };
	auto ndeval = np::empty(1, size, np::dtype::get_builtin<int>());
	std::copy(eval.begin(), eval.end(), reinterpret_cast<int*>(ndeval.get_data()));
	auto ndresult = np::empty(1, size, np::dtype::get_builtin<float>());
	std::copy(result.begin(), result.end(), reinterpret_cast<float*>(ndresult.get_data()));
	return py::make_tuple(ndeval, ndresult);
}

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
	py::def("hcpe2_decode_with_value", hcpe2_decode_with_value);
	py::def("load_hcpe3", load_hcpe3);
	py::def("hcpe3_decode_with_value", hcpe3_decode_with_value);
	py::def("hcpe3_prepare_evalfix", hcpe3_prepare_evalfix);
}