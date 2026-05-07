#pragma once

#include "cppshogi.h"

#include <cuda_runtime.h>
#include "NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "int8_calibrator.h"

#include <memory>
#include <mutex>
#include <vector>

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
	NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size, const int profile_count = 1);
	~NNTensorRT();
	int slot_capacity() const;
	void prepare_slots(const int slot_count);
	void forward(const int slot_id, const int batch_size, packed_features1_t* x1, packed_features2_t* x2, DType* y1, DType* y2);

private:
	struct InferenceSlot;

	const int gpu_id;
	const int max_batch_size;
	const int profile_count;
	InferUniquePtr<nvinfer1::ICudaEngine> engine;
	nvinfer1::Dims inputDims1;
	nvinfer1::Dims inputDims2;
	std::vector<std::unique_ptr<InferenceSlot>> slots;
	std::mutex slots_mutex;

	void load_model(const char* filename);
	void build(const std::string& onnx_filename);
	std::unique_ptr<InferenceSlot> create_slot(const int profile_index);
	InferenceSlot* get_slot(const int slot_id);
	void forward_impl(InferenceSlot* slot, const int batch_size, packed_features1_t* p1, packed_features2_t* p2, DType* y1, DType* y2);
};

typedef NNTensorRT NN;
