#pragma once

#ifdef FP16
#include <cuda_fp16.h>
typedef __half DType;
extern const DType _zero;
extern const DType _one;
inline float dtype_to_float(const DType x) {
	return __half2float(x);
}
#else
typedef float DType;
constexpr DType _zero = 0.0f;
constexpr DType _one = 1.0f;
inline float dtype_to_float(const DType x) {
	return x;
}
#endif
