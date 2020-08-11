#pragma once

#include <sstream>
#include <iostream>

inline void FatalError(const std::string& s) {
	std::cerr << s << "\nAborting...\n";
	cudaDeviceReset();
	exit(EXIT_FAILURE);
}

inline void checkCUDNN(cudnnStatus_t status) {
	if (status != CUDNN_STATUS_SUCCESS) {
		std::stringstream _error;
		_error << "CUDNN failure\nError: " << cudnnGetErrorString(status);
		FatalError(_error.str());
	}
}

inline void checkCudaErrors(cudaError_t status) {
	if (status != 0) {
		std::stringstream _error;
		_error << "Cuda failure\nError: " << cudaGetErrorString(status);
		FatalError(_error.str());
	}
}

inline void checkCublasErrors(cublasStatus_t status) {
	if (status != 0) {
		std::stringstream _error;
		_error << "Cublas failure\nError code " << status;
		FatalError(_error.str());
	}
}
