#include "cppshogi.h"

#include <iostream>
#include <chrono>

namespace py = boost::python;
namespace np = boost::python::numpy;

int main() {
	// Boost.PythonとBoost.Numpyの初期化
	Py_Initialize();
	np::initialize();

	// 入力データ作成
	const int batchsize = 1;
	float features1[batchsize][ColorNum][PieceTypeNum - 1][SquareNum];
	float features2[batchsize][MAX_FEATURES2_NUM][SquareNum];

	np::ndarray ndfeatures1 = np::from_data(
		features1,
		np::dtype::get_builtin<float>(),
		py::make_tuple(batchsize, (int)ColorNum * ((int)PieceTypeNum - 1), 9, 9),
		py::make_tuple(sizeof(float)*(int)ColorNum*((int)PieceTypeNum - 1) * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
		py::object());

	np::ndarray ndfeatures2 = np::from_data(
		features2,
		np::dtype::get_builtin<float>(),
		py::make_tuple(batchsize, MAX_FEATURES2_NUM, 9, 9),
		py::make_tuple(sizeof(float)*MAX_FEATURES2_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
		py::object());

	Position position;
	position.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", nullptr);

	make_input_features(position, features1, features2);

	// Pythonモジュール読み込み
	py::object dlshogi_test_ns = py::import("dlshogi.test").attr("__dict__");

	py::object dlshogi_test_load_model = dlshogi_test_ns["load_model"];
	dlshogi_test_load_model("H:\\src\\DeepLearningShogi\\dlshogi\\model_sl_elmo1000-009");

	// Pythonの関数実行
	py::object dlshogi_test_predict = dlshogi_test_ns["predict"];
	const int max = 1000;
	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < max; i++) {
		auto ret = dlshogi_test_predict(ndfeatures1, ndfeatures2);
		np::ndarray y_data = py::extract<np::ndarray>(ret);
	}

	auto duration = std::chrono::system_clock::now() - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "predict call " << max << " times" << std::endl;
	std::cout << msec << "[msec]" << std::endl;
	std::cout << (double)msec / max << "[msec per each call]" << std::endl;
	std::cout << (double)max / msec * 1000 << "[nps]" << std::endl;


	// Python呼び出しのオーバーヘッド計測
	py::object dlshogi_test_dummy = dlshogi_test_ns["dummy"];
	const int max2 = 100000;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < max2; i++) {
		dlshogi_test_dummy();
	}

	duration = std::chrono::system_clock::now() - start;
	msec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "dummy call " << max2 << " times" << std::endl;
	std::cout << msec << "[msec]" << std::endl;
	std::cout << (double)msec / max2 << "[msec per each call]" << std::endl;


	// Python呼び出しのオーバーヘッド計測
	py::object dlshogi_test_dummy2 = dlshogi_test_ns["dummy2"];
	const int max3 = 1000;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < max3; i++) {
		auto ret = dlshogi_test_dummy2(ndfeatures1, ndfeatures2);
		np::ndarray y_data = py::extract<np::ndarray>(ret);
	}

	duration = std::chrono::system_clock::now() - start;
	msec = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "dummy2 call " << max3 << " times" << std::endl;
	std::cout << msec << "[msec]" << std::endl;
	std::cout << (double)msec / max3 << "[msec per each call]" << std::endl;

	return 0;
}