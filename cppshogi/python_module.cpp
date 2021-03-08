#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

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
inline float make_result(const uint8_t result, const Position& position) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if (position.turn() == Black && gameResult == BlackWin ||
		position.turn() == White && gameResult == WhiteWin) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}
inline float is_sennichite(const uint8_t result) {
	return result & 0x4 ? 1.0f : 0.0f;
}
inline float is_nyugyoku(const uint8_t result) {
	return result & 0x8 ? 1.0f : 0.0f;
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
		*result = make_result(hcpe->gameResult, position);
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
		*result = make_result(hcpe->gameResult, position);
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
		*result = make_result(hcpe->gameResult, position);

		// eval
		*value = score_to_value((Score)hcpe->eval);
	}
}

void hcpe2_decode_with_value(np::ndarray ndhcpe2, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndmove, np::ndarray ndresult, np::ndarray ndaux, np::ndarray ndvalue) {
	const int len = (int)ndhcpe2.shape(0);
	HuffmanCodedPosAndEval2 *hcpe = reinterpret_cast<HuffmanCodedPosAndEval2 *>(ndhcpe2.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	int* move = reinterpret_cast<int*>(ndmove.get_data());
	float* result = reinterpret_cast<float*>(ndresult.get_data());
	auto aux = reinterpret_cast<float(*)[2]>(ndaux.get_data());
	float* value = reinterpret_cast<float*>(ndvalue.get_data());

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
		*result = make_result(hcpe->result, position);

		// sennichite
		(*aux)[0] = is_sennichite(hcpe->result);

		// nyugyoku
		(*aux)[1] = is_nyugyoku(hcpe->result);

		// eval
		*value = score_to_value((Score)hcpe->eval);
	}
}

struct MoveVisits {
	u16 move16;
	u16 visits;
};
static_assert(sizeof(MoveVisits) == 4, "");

struct HuffmanCodedPosAndEval3 {
	HuffmanCodedPos hcp;
	s16 eval;
	u16 bestMove16;
	u8 result; // xxxxxx11 : 勝敗、xxxxx1xx : 千日手、xxxx1xxx : 入玉宣言、xxx1xxxx : 持将棋、xx1xxxxx : 最大手数
	u8 seq;
	u16 candidateNum;
	std::vector<MoveVisits> candidates;
};
constexpr size_t hcpe3Size = sizeof(HuffmanCodedPosAndEval3) - sizeof(std::vector<MoveVisits>);
static_assert(hcpe3Size == 40, "");

std::vector<HuffmanCodedPosAndEval3> hcpe3Vec;

py::object load_hcpe3(std::string filepath) {
	std::ifstream ifs(filepath, std::ifstream::binary);
	if (!ifs) return py::make_tuple((int)hcpe3Vec.size(), 0, 0, 0);

	boost::iostreams::filtering_streambuf<boost::iostreams::input> filter;
	filter.push(boost::iostreams::gzip_decompressor());
	filter.push(ifs);

	int len = 0;
	int sum_sennichite = 0;
	int sum_nyugyoku = 0;

	std::istream is(&filter);
	while (is) {
		auto& hcpe3 = hcpe3Vec.emplace_back();
		is.read((char*)&hcpe3, hcpe3Size);
		if (is.eof()) {
			hcpe3Vec.pop_back();
			break;
		}
		assert(hcpe3.candidateNum <= 593);
		hcpe3.candidates.reserve(hcpe3.candidateNum);
		for (u16 i = 0; i < hcpe3.candidateNum; ++i) {
			auto& moveVisits = hcpe3.candidates.emplace_back();
			is.read((char*)&moveVisits, sizeof(MoveVisits));
		}
		sum_sennichite += (int)is_sennichite(hcpe3.result);
		sum_nyugyoku += (int)is_nyugyoku(hcpe3.result);
		++len;
	}

	return py::make_tuple((int)hcpe3Vec.size(), len, sum_sennichite, sum_nyugyoku);
}

void hcpe3_decode_with_value(np::ndarray ndindex, np::ndarray ndfeatures1, np::ndarray ndfeatures2, np::ndarray ndprobability, np::ndarray ndresult, np::ndarray ndaux, np::ndarray ndvalue) {
	const size_t len = (size_t)ndindex.shape(0);
	int* index = reinterpret_cast<int*>(ndindex.get_data());
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1.get_data());
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2.get_data());
	auto probability = reinterpret_cast<float(*)[9 * 9 * MAX_MOVE_LABEL_NUM]>(ndprobability.get_data());
	float* result = reinterpret_cast<float*>(ndresult.get_data());
	auto aux = reinterpret_cast<float(*)[2]>(ndaux.get_data());
	float* value = reinterpret_cast<float*>(ndvalue.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)probability, 9 * 9 * MAX_MOVE_LABEL_NUM * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, index++, features1++, features2++, value++, probability++, result++, aux++) {
		auto& hcpe3 = hcpe3Vec[*index];

		position.set(hcpe3.hcp);

		// input features
		make_input_features(position, features1, features2);

		// move probability
		const float sum_visits = (float)std::accumulate(hcpe3.candidates.begin(), hcpe3.candidates.end(), 0, [](int acc, MoveVisits& move_visits) { return acc + move_visits.visits; });
		for (int j = 0; j < (const int)hcpe3.candidateNum; j++) {
			auto label = make_move_label(hcpe3.candidates[j].move16, position.turn());
			assert(label < 9 * 9 * MAX_MOVE_LABEL_NUM);
			(*probability)[label] = (float)hcpe3.candidates[j].visits / sum_visits;
		}

		// game result
		*result = make_result(hcpe3.result, position);

		// sennichite
		(*aux)[0] = is_sennichite(hcpe3.result);

		// nyugyoku
		(*aux)[1] = is_nyugyoku(hcpe3.result);

		// eval
		*value = score_to_value((Score)hcpe3.eval);
	}
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
}