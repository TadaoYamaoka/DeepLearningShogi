#pragma once

#include "cppshogi.h"

#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "int8_calibrator.h"

#include <queue>

#define MULTI_STREAM 2

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

class semaphore {
private:
	std::mutex mutex_;
	std::condition_variable condition_;
	size_t count_;

public:
	semaphore(const unsigned long count) : count_(count) {}

	void release() {
		std::lock_guard<decltype(mutex_)> lock(mutex_);
		++count_;
		condition_.notify_one();
	}

	size_t acquire() {
		std::unique_lock<decltype(mutex_)> lock(mutex_);
		while (!count_) // Handle spurious wake-ups.
			condition_.wait(lock);
		--count_;
		return count_;
	}
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class NNTensorRT {
public:
	NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size);
	~NNTensorRT();
	void forward(const int batch_size, packed_features1_t* x1, packed_features2_t* x2, DType* y1, DType* y2);

private:
	const int gpu_id;
	const int max_batch_size;
	InferUniquePtr<nvinfer1::ICudaEngine> engine;
	std::array<packed_features1_t*, MULTI_STREAM> p1_dev;
	std::array<packed_features2_t*, MULTI_STREAM> p2_dev;
	std::array<features1_t*, MULTI_STREAM> x1_dev;
	std::array<features2_t*, MULTI_STREAM> x2_dev;
	std::array<DType*, MULTI_STREAM> y1_dev;
	std::array<DType*, MULTI_STREAM> y2_dev;
	std::array<std::vector<void*>, MULTI_STREAM> inputBindings;
	std::array<InferUniquePtr<nvinfer1::IExecutionContext>, MULTI_STREAM> context;
	std::array<nvinfer1::Dims, MULTI_STREAM> inputDims1;
	std::array<nvinfer1::Dims, MULTI_STREAM> inputDims2;
	std::array<cudaStream_t, MULTI_STREAM> stream;
	std::array<std::atomic<bool>, MULTI_STREAM> using_stream_index;

	void load_model(const char* filename);
	void build(const std::string& onnx_filename);

	// semaphore for stream
	semaphore semaphore_stream;
};

typedef NNTensorRT NN;
