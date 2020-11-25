#pragma once

#include <onnxruntime_cxx_api.h>
#include "nn.h"

class NNOnnxRuntime : NN {
public:
	NNOnnxRuntime(const char* filename, const int gpu_id, const int max_batch_size);
	~NNOnnxRuntime() {};
	void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

private:
	Ort::Env env;
	std::unique_ptr<Ort::Session> session;
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
};
