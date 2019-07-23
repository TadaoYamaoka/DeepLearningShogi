#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "error_util.h"
#include "cudnn_dtype.h"

class CudnnHandle
{
private:
	cudnnHandle_t cudnnHandle;

public:
	CudnnHandle() {
		checkCUDNN(cudnnCreate(&cudnnHandle));
	}
	~CudnnHandle() {
		checkCUDNN(cudnnDestroy(cudnnHandle));
	}

	cudnnHandle_t* operator &() {
		return &cudnnHandle;
	}

	operator cudnnHandle_t() {
		return cudnnHandle;
	}
};

class CudnnTensorDescriptor
{
private:
	cudnnTensorDescriptor_t cudnnTensorDescriptor;

public:
	CudnnTensorDescriptor() {
		checkCUDNN(cudnnCreateTensorDescriptor(&cudnnTensorDescriptor));
	}
	~CudnnTensorDescriptor() {
		checkCUDNN(cudnnDestroyTensorDescriptor(cudnnTensorDescriptor));
	}

	cudnnTensorDescriptor_t* operator &() {
		return &cudnnTensorDescriptor;
	}

	operator cudnnTensorDescriptor_t() {
		return cudnnTensorDescriptor;
	}
};

class CudnnFilterDescriptor
{
private:
	cudnnFilterDescriptor_t cudnnFilterDescriptor;

public:
	CudnnFilterDescriptor() {
		checkCUDNN(cudnnCreateFilterDescriptor(&cudnnFilterDescriptor));
	}
	~CudnnFilterDescriptor() {
		checkCUDNN(cudnnDestroyFilterDescriptor(cudnnFilterDescriptor));
	}

	cudnnFilterDescriptor_t* operator &() {
		return &cudnnFilterDescriptor;
	}

	operator cudnnFilterDescriptor_t() {
		return cudnnFilterDescriptor;
	}
};

class CudnnConvolutionDescriptor
{
private:
	cudnnConvolutionDescriptor_t cudnnConvolutionDescriptor;

public:
	CudnnConvolutionDescriptor() {
		checkCUDNN(cudnnCreateConvolutionDescriptor(&cudnnConvolutionDescriptor));
	}
	~CudnnConvolutionDescriptor() {
		checkCUDNN(cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor));
	}

	cudnnConvolutionDescriptor_t* operator &() {
		return &cudnnConvolutionDescriptor;
	}

	operator cudnnConvolutionDescriptor_t() {
		return cudnnConvolutionDescriptor;
	}
};

class CudnnActivationDescriptor
{
private:
	cudnnActivationDescriptor_t cudnnActivationDescriptor;

public:
	CudnnActivationDescriptor() {
		checkCUDNN(cudnnCreateActivationDescriptor(&cudnnActivationDescriptor));
	}
	~CudnnActivationDescriptor() {
		checkCUDNN(cudnnDestroyActivationDescriptor(cudnnActivationDescriptor));
	}

	cudnnActivationDescriptor_t* operator &() {
		return &cudnnActivationDescriptor;
	}

	operator cudnnActivationDescriptor_t() {
		return cudnnActivationDescriptor;
	}
};

class CudnnPoolingDescriptor
{
private:
	cudnnPoolingDescriptor_t cudnnPoolingDescriptor;

public:
	CudnnPoolingDescriptor() {
		checkCUDNN(cudnnCreatePoolingDescriptor(&cudnnPoolingDescriptor));
	}
	~CudnnPoolingDescriptor() {
		checkCUDNN(cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor));
	}

	cudnnPoolingDescriptor_t* operator &() {
		return &cudnnPoolingDescriptor;
	}

	operator cudnnPoolingDescriptor_t() {
		return cudnnPoolingDescriptor;
	}
};

class CudnnOpTensorDescriptor
{
private:
	cudnnOpTensorDescriptor_t cudnnOpTensorDescriptor;

public:
	CudnnOpTensorDescriptor() {
		checkCUDNN(cudnnCreateOpTensorDescriptor(&cudnnOpTensorDescriptor));
	}
	~CudnnOpTensorDescriptor() {
		checkCUDNN(cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor));
	}

	cudnnOpTensorDescriptor_t* operator &() {
		return &cudnnOpTensorDescriptor;
	}

	operator cudnnOpTensorDescriptor_t() {
		return cudnnOpTensorDescriptor;
	}
};

class CublasHandle
{
private:
	cublasHandle_t cublasHandle;

public:
	CublasHandle() {
		checkCublasErrors(cublasCreate(&cublasHandle));
#ifdef FP16
		checkCublasErrors(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
#endif
	}
	~CublasHandle() {
		checkCublasErrors(cublasDestroy(cublasHandle));
	}

	cublasHandle_t* operator &() {
		return &cublasHandle;
	}

	operator cublasHandle_t() {
		return cublasHandle;
	}
};
