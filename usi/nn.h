#pragma once

#include "cppshogi.h"
#include "layers.h"

class NN {
public:
	virtual ~NN() {};
	virtual void load_model(const char* filename) = 0;
	virtual void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2) = 0;
};