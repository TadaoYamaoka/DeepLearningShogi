#if 1
// Onnxベンチマーク
#include <iostream>
#include <chrono>
#include <random>

#ifdef _WIN32
#include <clocale>
#include <cstdlib>
#endif

#include "cppshogi.h"

#include <onnxruntime_cxx_api.h>
//#include <dnnl_provider_factory.h>
#include <dml_provider_factory.h>

#pragma comment(lib, "onnxruntime.lib")

using namespace std;

int main(int argc, char* argv[]) {
	if (argc < 5) {
		cout << "test <onnxfile> <hcpe> <num> <batchsize>" << endl;
		return 0;
	}

#ifdef _WIN32
	std::setlocale(LC_ALL, ".OCP");
	wchar_t tmpstr[2048];
	std::mbstowcs(tmpstr, argv[1], sizeof(tmpstr) / sizeof(wchar_t));
	std::wstring onnx_filename = tmpstr;
#else
	std::string onnx_filename(argv[1]);
#endif
	char* hcpe_path = argv[2];
	int num = stoi(argv[3]);
	size_t batchsize = stoi(argv[4]);

	cout << "onnx_filename = " << argv[1] << endl;
	cout << "num = " << num << endl;
	cout << "batchsize = " << batchsize << endl;

	initTable();
	HuffmanCodedPos::init();

	// 初期局面集
	ifstream ifs(hcpe_path, ifstream::in | ifstream::binary | ios::ate);
	if (!ifs) {
		cerr << "Error: cannot open " << hcpe_path << endl;
		exit(EXIT_FAILURE);
	}
	auto entry_num = ifs.tellg() / sizeof(HuffmanCodedPosAndEval);
	cout << "entry_num = " << entry_num << endl;

	std::mt19937_64 mt_64(0); // シード固定
	uniform_int_distribution<s64> inputFileDist(0, entry_num - 1);

	// Onnx初期化
	Ort::Env env;
	//Ort::SessionOptions session_options{ nullptr };
	Ort::SessionOptions session_options;
	//OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, 1);
	//session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	//session_options.EnableCpuMemArena();
	//session_options.SetInterOpNumThreads(8);
	//session_options.SetIntraOpNumThreads(8);
	//session_options.SetExecutionMode(ORT_PARALLEL);

	session_options.DisableMemPattern();
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
	OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

	Ort::Session session{ env, onnx_filename.c_str(), session_options };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	// input
	features1_t* features1 = new features1_t[batchsize];
	features2_t* features2 = new features2_t[batchsize];

	std::array<int64_t, 4> input_shape1{ batchsize, (size_t)ColorNum * MAX_FEATURES1_NUM, 9, 9 };
	std::array<int64_t, 4> input_shape2{ batchsize, MAX_FEATURES2_NUM, 9, 9 };

	std::array<Ort::Value, 2> input_values{
		Ort::Value::CreateTensor<float>(memory_info, (float*)features1, batchsize * (size_t)ColorNum * MAX_FEATURES1_NUM * (size_t)SquareNum, input_shape1.data(), input_shape1.size()),
		Ort::Value::CreateTensor<float>(memory_info, (float*)features2, batchsize * MAX_FEATURES2_NUM * (size_t)SquareNum, input_shape2.data(), input_shape2.size())
	};

	// output
	float* y1 = new float[batchsize * MAX_MOVE_LABEL_NUM * (size_t)SquareNum];
	float* y2 = new float[batchsize];

	std::array<int64_t, 2> output_shape1{ batchsize, MAX_MOVE_LABEL_NUM * (size_t)SquareNum };
	std::array<int64_t, 2> output_shape2{ batchsize, 1 };

	std::array<Ort::Value, 2> output_values{
		Ort::Value::CreateTensor<float>(memory_info, y1, batchsize * MAX_MOVE_LABEL_NUM * (size_t)SquareNum, output_shape1.data(), output_shape1.size()),
		Ort::Value::CreateTensor<float>(memory_info, y2, batchsize, output_shape2.data(), output_shape2.size())
	};

	// names
	const char* input_names[] = { "input1", "input2" };
	const char* output_names[] = { "output_policy", "output_value" };

	Color* color = new Color[batchsize];

	// 指し手の正解数
	int move_corrent = 0;

	// 勝敗の正解数
	int result_corrent = 0;

	// 評価値の2乗誤差
	float se_sum = 0;

	// 推論時間
	long long elapsed = 0;

	Position pos;
	HuffmanCodedPosAndEval* hcpe = new HuffmanCodedPosAndEval[batchsize];
	float* moves = new float[MAX_MOVE_LABEL_NUM * SquareNum];

	for (int n = 0; n < num / batchsize; n++) {
		// set all zero
		std::fill_n((DType*)features1, batchsize * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, _zero);
		std::fill_n((DType*)features2, batchsize * MAX_FEATURES2_NUM * (int)SquareNum, _zero);

		// hcpeをデコードして入力特徴作成
		for (int i = 0; i < batchsize; i++) {
			// hcpe読み込み
			ifs.seekg(inputFileDist(mt_64) * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
			ifs.read(reinterpret_cast<char*>(&hcpe[i]), sizeof(HuffmanCodedPosAndEval));

			pos.set(hcpe[i].hcp);
			color[i] = pos.turn();
			make_input_features(pos, features1 + i, features2 + i);
		}

		// 推論
		auto start = std::chrono::system_clock::now();

		session.Run(Ort::RunOptions{ nullptr }, input_names, input_values.data(), input_values.size(), output_names, output_values.data(), output_values.size());

		auto end = std::chrono::system_clock::now();

		// 時間集計
		elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		// 評価
		DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
		DType* value = reinterpret_cast<DType*>(y2);
		for (int i = 0; i < batchsize; i++, logits++, value++) {
			const DType* l = *logits;
			for (int j = 0; j < MAX_MOVE_LABEL_NUM * SquareNum; j++) {
				moves[j] = l[j];
			}
			const int move_label = (int)distance(moves, max_element(moves, moves + MAX_MOVE_LABEL_NUM * SquareNum));

			// 指し手の比較
			const int t_move_label = make_move_label(hcpe[i].bestMove16, color[i]);
			if (move_label == t_move_label) {
				++move_corrent;
			}

			// 勝敗の比較
			const float v = *value;

			if ((color[i] == Black && (hcpe[i].gameResult == BlackWin && v >= 0.5f || hcpe[i].gameResult == WhiteWin && v < 0.5f)) ||
				(color[i] == White && (hcpe[i].gameResult == BlackWin && v < 0.5f || hcpe[i].gameResult == WhiteWin && v >= 0.5f))) {
				++result_corrent;
			}

			// 評価値の誤差計算
			const float error = v - score_to_value((Score)hcpe[i].eval);
			se_sum += error * error;
		}
	}

	// 結果表示
	int num_actual = num / batchsize * batchsize;
	cout << "num_actual = " << num_actual << endl;
	cout << "elapsed = " << elapsed << " ns" << endl;
	cout << "move accuracy = " << (double)move_corrent / num_actual << endl;
	cout << "value accuracy = " << (double)result_corrent / num_actual << endl;
	cout << "value mse = " << (double)se_sum / num_actual << endl;

	return 0;
}
#endif