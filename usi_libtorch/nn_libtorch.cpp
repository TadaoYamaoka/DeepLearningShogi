#ifdef LIBTORCH
#include "nn_libtorch.h"
#include <torch/script.h>

NNLibTorch::NNLibTorch(const char* filename, const torch::DeviceIndex gpu_id, const int max_batch_size)
	: device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, torch::cuda::is_available() ? gpu_id : -1)
{
	c10::InferenceMode guard;
	model = torch::jit::load(filename);
	model.to(device);
	model.eval();
}

void NNLibTorch::forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2)
{
	c10::InferenceMode guard;

	std::vector<torch::jit::IValue> x = {
		torch::from_blob(x1, { batch_size, (size_t)ColorNum * MAX_FEATURES1_NUM, 9, 9 }, torch::dtype(torch::kFloat32)).to(device),
		torch::from_blob(x2, { batch_size, MAX_FEATURES2_NUM, 9, 9 }, torch::dtype(torch::kFloat32)).to(device)
	};

	const auto y = model.forward(x).toTuple();

	std::memcpy(y1, y->elements()[0].toTensor().cpu().data_ptr<float>(), sizeof(float) * batch_size * MAX_MOVE_LABEL_NUM * (size_t)SquareNum);
	std::memcpy(y2, y->elements()[1].toTensor().cpu().data_ptr<float>(), sizeof(float) * batch_size);
}

#endif
