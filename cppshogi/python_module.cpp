#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>

#include "cppshogi.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

// make result
inline float make_result(const GameResult gameResult, const Position& position) {
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
	int *move = reinterpret_cast<int *>(ndmove.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++) {
		position.set(hcpe->hcp);

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
	float *result = reinterpret_cast<float *>(ndresult.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, move++, result++) {
		position.set(hcpe->hcp);

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
	float *result = reinterpret_cast<float *>(ndresult.get_data());
	float *value = reinterpret_cast<float *>(ndvalue.get_data());

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (int i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++) {
		position.set(hcpe->hcp);

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
}