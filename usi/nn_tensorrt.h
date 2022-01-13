#pragma once

#include "cppshogi.h"

#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "int8_calibrator.h"
#include "unpack.h"

struct InferDeleter
{
	// Deprecated And Removed Features
	// The following features are deprecated in TensorRT 8.0.0:
	// - Interface functions that provided a destroy function are deprecated in TensorRT 8.0. The destructors will be exposed publicly in order for the delete operator to work as expected on these classes.
	// - Destructors for classes with destroy() methods were previously protected. They are now public, enabling use of smart pointers for these classes. The destroy() methods are deprecated.
	// https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/deprecated.html
	// https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-800-ea/release-notes/tensorrt-8.html#rel_8-0-0-EA
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
#if NV_TENSORRT_MAJOR >= 8
			delete obj;
#else
			obj->destroy();
#endif
		}
	}
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class NNTensorRT {
public:
	NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size);
	~NNTensorRT();
	void forward(const int batch_size, packed_features1_t* x1, packed_features2_t* x2, DType* y1, DType* y2, packed_legal_moves_t* packed_legal_moves, const int sum_legal_moves_num);

private:
	const int gpu_id;
	const int max_batch_size;
	InferUniquePtr<nvinfer1::ICudaEngine> engine;
	packed_features1_t* p1_dev;
	packed_features2_t* p2_dev;
	features1_t* x1_dev;
	features2_t* x2_dev;
	packed_legal_moves_t* packed_legal_moves_dev;
	DType* t1_dev;
	DType* y1_dev;
	DType* y2_dev;
	std::vector<void*> inputBindings;
	InferUniquePtr<nvinfer1::IExecutionContext> context;
	nvinfer1::Dims inputDims1;
	nvinfer1::Dims inputDims2;

	void load_model(const char* filename);
	void build(const std::string& onnx_filename);
};

typedef NNTensorRT NN;
