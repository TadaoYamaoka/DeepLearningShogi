#pragma once

#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

inline void FatalError(const std::string& s) {
	std::cerr << s << "\nAborting...\n";
	cudaDeviceReset();
	exit(EXIT_FAILURE);
}

inline void checkCudaErrors(cudaError_t status) {
	if (status != 0) {
		std::stringstream _error;
		_error << "Cuda failure\nError: " << cudaGetErrorString(status);
		FatalError(_error.str());
	}
}
