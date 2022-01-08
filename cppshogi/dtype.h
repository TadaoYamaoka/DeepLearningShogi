#pragma once

#ifdef FP16
#include <cuda_fp16.h>
typedef __half DType;
extern const DType _zero;
extern const DType _one;
#else
typedef float DType;
constexpr DType _zero = 0.0f;
constexpr DType _one = 1.0f;
#endif
