#pragma once

#include <torch/torch.h>
#include "cppshogi.h"

class NNLibTorch {
public:
	NNLibTorch(const char* filename, const torch::DeviceIndex gpu_id, const int max_batch_size);
	~NNLibTorch() {};
	void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

private:
	torch::Device device;
	torch::jit::Module model;
};

typedef NNLibTorch NN;
