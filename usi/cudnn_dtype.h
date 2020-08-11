#pragma once

#ifdef FP16
#include <cuda_fp16.h>
typedef __half DType;
#define CUDNN_DATA_TYPE CUDNN_DATA_HALF
#define CUDA_DATA_TYPE CUDA_R_16F
extern const DType _zero;
extern const DType _one;
inline float to_float(const DType x) {
	return __half2float(x);
}
#else
typedef float DType;
#define CUDNN_DATA_TYPE CUDNN_DATA_FLOAT
#define CUDA_DATA_TYPE CUDA_R_32F
constexpr const DType _zero = 0.0f;
constexpr const DType _one = 1.0f;
inline float to_float(DType x) {
	return x;
}
#endif
