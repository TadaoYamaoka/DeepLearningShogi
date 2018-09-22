#include <iostream>
#include <chrono>
#include <random>

#include "cppshogi.h"

using namespace std;

#if 1
// GPU�x���`�}�[�N
#include "nn.h"

static void  showDevices(int i)
{
	struct cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, i));
	cout << "device " << i
		<< " sms " << prop.multiProcessorCount
		<< " Capabilities " << prop.major << "." << prop.minor
		<< ", SmClock " << (float)prop.clockRate*1e-3 << " Mhz"
		<< ", MemSize (Mb) " << (int)(prop.totalGlobalMem / (1024 * 1024))
		<< ", MemClock " << (float)prop.memoryClockRate*1e-3 << " Mhz"
		<< ", Ecc=" << prop.ECCEnabled
		<< ", boardGroupID=" << prop.multiGpuBoardGroupID << endl;
}

int main(int argc, char* argv[]) {
	if (argc < 5) {
		cout << "test <modelfile> <hcpe> <num> <gpu_id> <batchsize>" << endl;
		return 0;
	}

	char* model_path = argv[1];
	char* hcpe_path = argv[2];
	int num = stoi(argv[3]);
	int gpu_id = stoi(argv[4]);
	int batchsize = stoi(argv[5]);

	cout << "num = " << num << endl;
	cout << "gpu_id = " << gpu_id << endl;
	cout << "batchsize = " << batchsize << endl;

	initTable();
	HuffmanCodedPos::init();

	// �����ǖʏW
	ifstream ifs(hcpe_path, ifstream::in | ifstream::binary | ios::ate);
	if (!ifs) {
		cerr << "Error: cannot open " << hcpe_path << endl;
		exit(EXIT_FAILURE);
	}
	auto entry_num = ifs.tellg() / sizeof(HuffmanCodedPosAndEval);
	cout << "entry_num = " << entry_num << endl;

	std::mt19937_64 mt_64(0); // �V�[�h�Œ�
	uniform_int_distribution<s64> inputFileDist(0, entry_num - 1);

	showDevices(gpu_id);
	cudaSetDevice(gpu_id);
	NN nn(batchsize);
	nn.load_model(model_path);

	features1_t* features1;
	features2_t* features2;
	checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * batchsize, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * batchsize, cudaHostAllocPortable));

	float* y1;
	float* y2;
	checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (int)SquareNum * batchsize * sizeof(float), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&y2, batchsize * sizeof(float), cudaHostAllocPortable));

	Color* color = new Color[batchsize];

	// �w����̐���
	int move_corrent = 0;

	// ���s�̐���
	int result_corrent = 0;

	// �]���l��2��덷
	float se_sum = 0;

	// ���_����
	long long elapsed = 0;

	Position pos;
	HuffmanCodedPosAndEval* hcpe = new HuffmanCodedPosAndEval[batchsize];

	for (int n = 0; n < num / batchsize; n++) {
		// set all zero
		std::fill_n((float*)features1, batchsize * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
		std::fill_n((float*)features2, batchsize * MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

		// hcpe���f�R�[�h���ē��͓����쐬
		for (int i = 0; i < batchsize; i++) {
			// hcpe�ǂݍ���
			ifs.seekg(inputFileDist(mt_64) * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
			ifs.read(reinterpret_cast<char*>(&hcpe[i]), sizeof(HuffmanCodedPosAndEval));

			pos.set(hcpe[i].hcp, nullptr);
			color[i] = pos.turn();
			make_input_features(pos, features1 + i, features2 + i);
		}

		// ���_
		auto start = std::chrono::system_clock::now();
		nn.foward(batchsize, features1, features2, (float*)y1, y2);
		auto end = std::chrono::system_clock::now();

		// ���ԏW�v
		elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		// �]��
		float(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
		float *value = reinterpret_cast<float*>(y2);
		for (int i = 0; i < batchsize; i++, logits++, value++) {
			const float* l = *logits;
			const int move_label = (int)distance(l, max_element(l, l + MAX_MOVE_LABEL_NUM * SquareNum));

			// �w����̔�r
			const int t_move_label = make_move_label(hcpe[i].bestMove16, color[i]);
			if (move_label == t_move_label) {
				++move_corrent;
			}

			// ���s�̔�r
			if ((color[i] == Black && (hcpe[i].gameResult == BlackWin && *value > 0.5f || hcpe[i].gameResult == WhiteWin && *value < 0.5f)) ||
				(color[i] == White && (hcpe[i].gameResult == BlackWin && *value < 0.5f || hcpe[i].gameResult == WhiteWin && *value > 0.5f))) {
				++result_corrent;
			}

			// �]���l�̌덷�v�Z
			const float error = *value - score_to_value((Score)hcpe[i].eval);
			se_sum += error * error;
		}
	}

	// ���ʕ\��
	int num_actual = num / batchsize * batchsize;
	cout << "num_actual = " << num_actual << endl;
	cout << "elapsed = " << elapsed << " ns" << endl;
	cout << "move accuracy = " << (double)move_corrent / num_actual << endl;
	cout << "value accuracy = " << (double)result_corrent / num_actual << endl;
	cout << "value mse = " << (double)se_sum / num_actual << endl;

	return 0;
}
#endif