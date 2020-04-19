#if 1
// TensorRT�x���`�}�[�N
#include <iostream>
#include <chrono>
#include <random>

#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "cppshogi.h"
#include "cudnn_wrapper.h"

using namespace std;

class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg)
	{
		if (severity == Severity::kINTERNAL_ERROR) {
			switch (severity)
			{
			case Severity::kINTERNAL_ERROR: std::cout << "[F] "; break;
			case Severity::kERROR: std::cout << "[E] "; break;
			case Severity::kWARNING: std::cout << "[W] "; break;
			case Severity::kINFO: std::cout << "[I] "; break;
			case Severity::kVERBOSE: std::cout << "[V] "; break;
			default: assert(0);
			}
			std::cout << msg << std::endl;
		}
	}
} gLogger;

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

constexpr long long int operator"" _MiB(long long unsigned int val)
{
	return val * (1 << 20);
}

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

bool build(const string& onnx_filename, const int batchsize, const int fp16, InferUniquePtr<nvinfer1::ICudaEngine>& engine)
{
	auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		std::cerr << "Error: createBuilderConfig" << std::endl;
		return false;
	}

	auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
	if (!parser)
	{
		std::cerr << "Error: createParser" << std::endl;
		return false;
	}

	auto parsed = parser->parseFromFile(onnx_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
	if (!parsed)
	{
		std::cerr << "Error: parseFromFile" << std::endl;
		return false;
	}

	builder->setMaxBatchSize(batchsize);
	config->setMaxWorkspaceSize(64_MiB);

	if (fp16 == 1)
		config->setFlag(nvinfer1::BuilderFlag::kFP16);

	assert(network->getNbInputs() == 2);
	nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
	assert(inputDims[0].nbDims == 4);
	assert(inputDims[1].nbDims == 4);
	std::cout << "Input1 name : " << network->getInput(0)->getName() << std::endl;
	std::cout << "Input2 name : " << network->getInput(1)->getName() << std::endl;

	assert(network->getNbOutputs() == 2);
	nvinfer1::Dims outputDims[] = { network->getOutput(0)->getDimensions(), network->getOutput(1)->getDimensions() };
	assert(outputDims[0].nbDims == 2);
	assert(outputDims[1].nbDims == 2);
	std::cout << "Output1 name : " << network->getOutput(0)->getName() << std::endl;
	std::cout << "Output2 name : " << network->getOutput(1)->getName() << std::endl;

	// Optimization Profiles
	auto profile = builder->createOptimizationProfile();
	const auto dims1 = inputDims[0].d;
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(batchsize, dims1[1], dims1[2], dims1[3]));
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(batchsize, dims1[1], dims1[2], dims1[3]));
	const auto dims2 = inputDims[1].d;
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(batchsize, dims2[1], dims2[2], dims2[3]));
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(batchsize, dims2[1], dims2[2], dims2[3]));
	config->addOptimizationProfile(profile);

	engine = InferUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
	if (!engine)
	{
		std::cerr << "Error: buildEngineWithConfig" << std::endl;
		return false;
	}

	return true;
}

int main(int argc, char* argv[]) {
	if (argc < 6) {
		cout << "test <onnxfile> <hcpe> <num> <gpu_id> <batchsize> <fp16[0|1]>" << endl;
		return 0;
	}

	std::string onnx_filename(argv[1]);
	char* hcpe_path = argv[2];
	int num = stoi(argv[3]);
	int gpu_id = stoi(argv[4]);
	int batchsize = stoi(argv[5]);
	int fp16 = stoi(argv[6]);

	cout << "onnx_filename = " << onnx_filename << endl;
	cout << "num = " << num << endl;
	cout << "gpu_id = " << gpu_id << endl;
	cout << "batchsize = " << batchsize << endl;
	cout << "fp16 = " << fp16 << endl;

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

	InferUniquePtr<nvinfer1::ICudaEngine> engine;

	const string serialized_filename = onnx_filename + (fp16 == 1 ? ".fp16" : "") + ".serialized";
	std::ifstream seriarizedFile(serialized_filename, std::ios::binary);
	if (seriarizedFile.is_open())
	{
		// deserializing a model
		seriarizedFile.seekg(0, std::ios_base::end);
		const size_t modelSize = seriarizedFile.tellg();
		seriarizedFile.seekg(0, std::ios_base::beg);
		std::unique_ptr<char[]> blob(new char[modelSize]);
		seriarizedFile.read(blob.get(), modelSize);
		auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
		engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(blob.get(), modelSize, nullptr));
	}
	else
	{

		// build
		auto build_start = std::chrono::system_clock::now();
		if (!build(onnx_filename, batchsize, fp16, engine))
		{
			return 1;
		}

		// elapsed for building
		auto build_end = std::chrono::system_clock::now();
		std::cout << "elapsed for building = " << std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count() << " ms" << std::endl;

		// serializing a model
		auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
		if (!serializedEngine)
		{
			std::cerr << "Engine serialization failed" << std::endl;
			return 1;
		}
		std::ofstream engineFile(serialized_filename, std::ios::binary);
		if (!engineFile)
		{
			std::cerr << "Cannot open engine file: " << serialized_filename << std::endl;
			return 1;
		}
		engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
		if (engineFile.fail())
		{
			return 1;
		}
	}

	// infer
	auto context = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	if (!context)
	{
		return 1;
	}

	// Create host and device buffers
	features1_t* features1;
	features2_t* features2;
	checkCudaErrors(cudaHostAlloc(&features1, sizeof(features1_t) * batchsize, cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&features2, sizeof(features2_t) * batchsize, cudaHostAllocPortable));
	features1_t* x1_dev;
	features2_t* x2_dev;
	checkCudaErrors(cudaMalloc((void**)&x1_dev, sizeof(features1_t) * batchsize));
	checkCudaErrors(cudaMalloc((void**)&x2_dev, sizeof(features2_t) * batchsize));
	DType* y1;
	DType* y2;
	checkCudaErrors(cudaHostAlloc(&y1, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batchsize * sizeof(DType), cudaHostAllocPortable));
	checkCudaErrors(cudaHostAlloc(&y2, batchsize * sizeof(DType), cudaHostAllocPortable));
	DType* y1_dev;
	DType* y2_dev;
	checkCudaErrors(cudaMalloc((void**)&y1_dev, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batchsize * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&y2_dev, batchsize * sizeof(DType)));

	assert(engine->getNbBindings() == 4);
	assert(engine->getBindingDataType(0) == nvinfer1::DataType::kFLOAT);
	assert(engine->getBindingDataType(1) == nvinfer1::DataType::kFLOAT);
	assert(engine->getBindingDataType(2) == nvinfer1::DataType::kFLOAT);
	assert(engine->getBindingDataType(3) == nvinfer1::DataType::kFLOAT);
	assert(engine->getBindingDimensions(0).nbDims == 4);
	assert(engine->getBindingDimensions(1).nbDims == 4);
	assert(engine->getBindingDimensions(2).nbDims == 2);
	assert(engine->getBindingDimensions(3).nbDims == 2);

	void* inputBindings[] = { x1_dev, x2_dev, y1_dev, y2_dev };


	// Set the input size for the preprocessor
	nvinfer1::Dims inputDims[] = { engine->getBindingDimensions(0), engine->getBindingDimensions(1) };
	inputDims[0].d[0] = batchsize;
	inputDims[1].d[0] = batchsize;
	context->setBindingDimensions(0, inputDims[0]);
	context->setBindingDimensions(1, inputDims[1]);


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
	float *moves = new float[MAX_MOVE_LABEL_NUM * SquareNum];

	for (int n = 0; n < num / batchsize; n++) {
		// set all zero
		std::fill_n((DType*)features1, batchsize * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, _zero);
		std::fill_n((DType*)features2, batchsize * MAX_FEATURES2_NUM * (int)SquareNum, _zero);

		// hcpe���f�R�[�h���ē��͓����쐬
		for (int i = 0; i < batchsize; i++) {
			// hcpe�ǂݍ���
			ifs.seekg(inputFileDist(mt_64) * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
			ifs.read(reinterpret_cast<char*>(&hcpe[i]), sizeof(HuffmanCodedPosAndEval));

			pos.set(hcpe[i].hcp);
			color[i] = pos.turn();
			make_input_features(pos, features1 + i, features2 + i);
		}

		// ���_
		auto start = std::chrono::system_clock::now();

		checkCudaErrors(cudaMemcpy(x1_dev, features1, sizeof(features1_t) * batchsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(x2_dev, features2, sizeof(features2_t) * batchsize, cudaMemcpyHostToDevice));
		bool status = context->executeV2(inputBindings);
		if (!status)
		{
			return 1;
		}
		checkCudaErrors(cudaMemcpy(y1, y1_dev, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batchsize * sizeof(DType), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(y2, y2_dev, batchsize * sizeof(DType), cudaMemcpyDeviceToHost));

		auto end = std::chrono::system_clock::now();

		// ���ԏW�v
		elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

		// �]��
		DType(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<DType(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1);
		DType *value = reinterpret_cast<DType*>(y2);
		for (int i = 0; i < batchsize; i++, logits++, value++) {
			const DType* l = *logits;
			for (int j = 0; j < MAX_MOVE_LABEL_NUM * SquareNum; j++) {
				moves[j] = l[j];
			}
			const int move_label = (int)distance(moves, max_element(moves, moves + MAX_MOVE_LABEL_NUM * SquareNum));

			// �w����̔�r
			const int t_move_label = make_move_label(hcpe[i].bestMove16, color[i]);
			if (move_label == t_move_label) {
				++move_corrent;
			}

			// ���s�̔�r
			const float v = *value;

			if ((color[i] == Black && (hcpe[i].gameResult == BlackWin && v >= 0.5f || hcpe[i].gameResult == WhiteWin && v < 0.5f)) ||
				(color[i] == White && (hcpe[i].gameResult == BlackWin && v < 0.5f || hcpe[i].gameResult == WhiteWin && v >= 0.5f))) {
				++result_corrent;
			}

			// �]���l�̌덷�v�Z
			const float error = v - score_to_value((Score)hcpe[i].eval);
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

	checkCudaErrors(cudaFreeHost(features1));
	checkCudaErrors(cudaFreeHost(features2));
	checkCudaErrors(cudaFree(x1_dev));
	checkCudaErrors(cudaFree(x2_dev));
	checkCudaErrors(cudaFreeHost(y1));
	checkCudaErrors(cudaFreeHost(y2));
	checkCudaErrors(cudaFree(y1_dev));
	checkCudaErrors(cudaFree(y2_dev));

	return 0;
}
#endif