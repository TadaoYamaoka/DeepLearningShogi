#pragma once

#include "nn.h"

#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "int8_calibrator.h"

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

class NNTensorRT : NN {
public:
	NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size);
	~NNTensorRT();
	void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

private:
	const int gpu_id;
	const int max_batch_size;
	InferUniquePtr<nvinfer1::ICudaEngine> engine;
	features1_t* x1_dev;
	features2_t* x2_dev;
	DType* y1_dev;
	DType* y2_dev;
	std::vector<void*> inputBindings;
	InferUniquePtr<nvinfer1::IExecutionContext> context;
	nvinfer1::Dims inputDims1;
	nvinfer1::Dims inputDims2;

	void load_model(const char* filename);
	void build(const std::string& onnx_filename);
};
