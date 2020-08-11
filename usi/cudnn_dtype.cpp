#include "cudnn_dtype.h"

#ifdef FP16
const DType _zero = __float2half(0.0f);
const DType _one = __float2half(1.0f);
#endif