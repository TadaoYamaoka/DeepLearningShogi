#include "nn_tensorrt.h"

#include "cppshogi.h"

class Logger : public nvinfer1::ILogger
{
	const char* error_type(Severity severity)
	{
		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: return "[F] ";
		case Severity::kERROR: return "[E] ";
		case Severity::kWARNING: return "[W] ";
		case Severity::kINFO: return "[I] ";
		case Severity::kVERBOSE: return "[V] ";
		default: assert(0); return "";
		}
	}
	void log(Severity severity, const char* msg) noexcept
	{
		if (severity == Severity::kINTERNAL_ERROR) {
			std::cerr << error_type(severity) << msg << std::endl;
		}
	}
} gLogger;

constexpr long long int operator"" _MiB(long long unsigned int val)
{
	return val * (1 << 20);
}

NNTensorRT::NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size)
	: gpu_id(gpu_id), max_batch_size(max_batch_size), using_stream_index{}, semaphore_stream(MULTI_STREAM)
{
	// Create host and device buffers
	for (size_t i = 0; i < MULTI_STREAM; i++) {
		checkCudaErrors(cudaMalloc((void**)&x1_dev[i], sizeof(features1_t) * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&x2_dev[i], sizeof(features2_t) * max_batch_size));
		checkCudaErrors(cudaMalloc((void**)&y1_dev[i], MAX_MOVE_LABEL_NUM * (size_t)SquareNum * max_batch_size * sizeof(DType)));
		checkCudaErrors(cudaMalloc((void**)&y2_dev[i], max_batch_size * sizeof(DType)));

		inputBindings[i] = { x1_dev[i], x2_dev[i], y1_dev[i], y2_dev[i] };
	}

	load_model(filename);
}

NNTensorRT::~NNTensorRT()
{
	for (size_t i = 0; i < MULTI_STREAM; i++) {
		checkCudaErrors(cudaFree(x1_dev[i]));
		checkCudaErrors(cudaFree(x2_dev[i]));
		checkCudaErrors(cudaFree(y1_dev[i]));
		checkCudaErrors(cudaFree(y2_dev[i]));
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
}

void NNTensorRT::build(const std::string& onnx_filename)
{
	auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
	if (!builder)
	{
		throw std::runtime_error("createInferBuilder");
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		throw std::runtime_error("createNetworkV2");
	}

	auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		throw std::runtime_error("createBuilderConfig");
	}

	auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
	if (!parser)
	{
		throw std::runtime_error("createParser");
	}

	auto parsed = parser->parseFromFile(onnx_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
	if (!parsed)
	{
		throw std::runtime_error("parseFromFile");
	}

	builder->setMaxBatchSize(max_batch_size);
	config->setMaxWorkspaceSize(64_MiB);

	std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
	if (builder->platformHasFastInt8())
	{
		// キャリブレーションキャッシュがある場合のみINT8を使用
		std::string calibration_cache_filename = std::string(onnx_filename) + ".calibcache";
		std::ifstream calibcache(calibration_cache_filename);
		if (calibcache.is_open())
		{
			calibcache.close();

			config->setFlag(nvinfer1::BuilderFlag::kINT8);
			calibrator.reset(new Int8EntropyCalibrator2(onnx_filename.c_str(), 1));
			config->setInt8Calibrator(calibrator.get());
		}
		else if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
	}
	else if (builder->platformHasFastFp16())
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

#ifdef FP16
	network->getInput(0)->setType(nvinfer1::DataType::kHALF);
	network->getInput(1)->setType(nvinfer1::DataType::kHALF);
	network->getOutput(0)->setType(nvinfer1::DataType::kHALF);
	network->getOutput(1)->setType(nvinfer1::DataType::kHALF);
#endif

	assert(network->getNbInputs() == 2);
	nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
	assert(inputDims[0].nbDims == 4);
	assert(inputDims[1].nbDims == 4);

	assert(network->getNbOutputs() == 2);

	// Optimization Profiles
	auto profile = builder->createOptimizationProfile();
	const auto dims1 = inputDims[0].d;
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
	profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
	const auto dims2 = inputDims[1].d;
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
	profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
	config->addOptimizationProfile(profile);

	// TensorRT 8 より nvinfer1::IBuilder::buildSerializedNetwork() が追加され、 nvinfer1::IBuilder::buildEngineWithConfig() は非推奨となった。
	// nvinfer1::IBuilder::buildEngineWithConfig() は TensorRT 10.0 にて削除される見込み。
	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/deprecated.html
	// https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-800-ea/release-notes/tensorrt-8.html#rel_8-0-0-EA
#if NV_TENSORRT_MAJOR >= 8
	auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
	if (!serializedEngine)
	{
		throw std::runtime_error("buildSerializedNetwork");
	}
	auto runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
	engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
	if (!engine)
	{
		throw std::runtime_error("deserializeCudaEngine");
	}
	// 一旦シリアライズ化されたエンジンはデシリアライズを行った上で捨てているが、
	// この後またすぐにファイル書き出し用にシリアライズを行っているので、手順改善の余地あり。
	// // auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
#else
	engine.reset(builder->buildEngineWithConfig(*network, *config));
	if (!engine)
	{
		throw std::runtime_error("buildEngineWithConfig");
	}
#endif
}

void NNTensorRT::load_model(const char* filename)
{
	std::string serialized_filename = std::string(filename) + "." + std::to_string(gpu_id) + "." + std::to_string(max_batch_size) + ".serialized";
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
		engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(blob.get(), modelSize));
	}
	else
	{

		// build
		build(filename);

		// serializing a model
		auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
		if (!serializedEngine)
		{
			throw std::runtime_error("Engine serialization failed");
		}
		std::ofstream engineFile(serialized_filename, std::ios::binary);
		if (!engineFile)
		{
			throw std::runtime_error("Cannot open engine file");
		}
		engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
		if (engineFile.fail())
		{
			throw std::runtime_error("Cannot open engine file");
		}
	}

	for (size_t i = 0; i < MULTI_STREAM; i++) {
		context[i] = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
		if (!context[i])
		{
			throw std::runtime_error("createExecutionContext");
		}
		checkCudaErrors(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));

		inputDims1[i] = engine->getBindingDimensions(0);
		inputDims2[i] = engine->getBindingDimensions(1);
	}
}

void NNTensorRT::forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2)
{
	semaphore_stream.acquire();
	size_t i = 0;
	for (; i < MULTI_STREAM - 1; i++) {
		if (!using_stream_index[i].exchange(true))
			break;
	}

	inputDims1[i].d[0] = batch_size;
	inputDims2[i].d[0] = batch_size;
	context[i]->setBindingDimensions(0, inputDims1[i]);
	context[i]->setBindingDimensions(1, inputDims2[i]);

	checkCudaErrors(cudaMemcpyAsync(x1_dev[i], x1, sizeof(features1_t) * batch_size, cudaMemcpyHostToDevice, stream[i]));
	checkCudaErrors(cudaMemcpyAsync(x2_dev[i], x2, sizeof(features2_t) * batch_size, cudaMemcpyHostToDevice, stream[i]));
	const bool status = context[i]->enqueue(batch_size, inputBindings[i].data(), stream[i], nullptr);
	assert(status);
	checkCudaErrors(cudaMemcpyAsync(y1, y1_dev[i], sizeof(DType) * MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batch_size , cudaMemcpyDeviceToHost, stream[i]));
	checkCudaErrors(cudaMemcpyAsync(y2, y2_dev[i], sizeof(DType) * batch_size, cudaMemcpyDeviceToHost, stream[i]));
	checkCudaErrors(cudaStreamSynchronize(stream[i]));

	using_stream_index[i] = false;
	semaphore_stream.release();
}
