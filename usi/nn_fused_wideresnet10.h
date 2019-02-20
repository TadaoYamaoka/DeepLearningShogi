#pragma once

#include "nn.h"

class NNFusedWideResnet10 : NN {
public:
	NNFusedWideResnet10(const int max_batch_size);
	~NNFusedWideResnet10();

	void load_model(const char* filename);

	void foward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

private:
	void prepare_desc(const int batch_size);

	CudnnHandle cudnnHandle;
	CublasHandle cublasHandle;
	static constexpr int k = 192;
	static constexpr int fcl = 256;

	const int max_batch_size;

	// input layer
	ConvLayer<k, (int)ColorNum * MAX_FEATURES1_NUM, 3, 1> conv1_1_1;
	ConvLayer<k, (int)ColorNum * MAX_FEATURES1_NUM, 1, 0> conv1_1_2;
	ConvLayer<k, MAX_FEATURES2_NUM, 1, 0> conv1_2;
	// residual blocks
	BatchNormalization<k> bn1;
	ConvLayer<k, k, 3, 1> conv2;
	Bias<k, 1, 1> bias2;
	ConvLayer<k, k, 3, 1> conv3;
	BatchNormalization<k> bn3;
	ConvLayer<k, k, 3, 1> conv4;
	Bias<k, 1, 1> bias4;
	ConvLayer<k, k, 3, 1> conv5;
	BatchNormalization<k> bn5;
	ConvLayer<k, k, 3, 1> conv6;
	Bias<k, 1, 1> bias6;
	ConvLayer<k, k, 3, 1> conv7;
	BatchNormalization<k> bn7;
	ConvLayer<k, k, 3, 1> conv8;
	Bias<k, 1, 1> bias8;
	ConvLayer<k, k, 3, 1> conv9;
	BatchNormalization<k> bn9;
	ConvLayer<k, k, 3, 1> conv10;
	Bias<k, 1, 1> bias10;
	ConvLayer<k, k, 3, 1> conv11;
	BatchNormalization<k> bn11;
	ConvLayer<k, k, 3, 1> conv12;
	Bias<k, 1, 1> bias12;
	ConvLayer<k, k, 3, 1> conv13;
	BatchNormalization<k> bn13;
	ConvLayer<k, k, 3, 1> conv14;
	Bias<k, 1, 1> bias14;
	ConvLayer<k, k, 3, 1> conv15;
	BatchNormalization<k> bn15;
	ConvLayer<k, k, 3, 1> conv16;
	Bias<k, 1, 1> bias16;
	ConvLayer<k, k, 3, 1> conv17;
	BatchNormalization<k> bn17;
	ConvLayer<k, k, 3, 1> conv18;
	Bias<k, 1, 1> bias18;
	ConvLayer<k, k, 3, 1> conv19;
	BatchNormalization<k> bn19;
	ConvLayer<k, k, 3, 1> conv20;
	Bias<k, 1, 1> bias20;
	ConvLayer<k, k, 3, 1> conv21;
	BatchNormalization<k> bn21;
	// policy network
	ConvLayer<MAX_MOVE_LABEL_NUM, k, 1, 0> conv22;
	Bias<MAX_MOVE_LABEL_NUM, 9, 9> bias22;
	// value network
	ConvLayer<MAX_MOVE_LABEL_NUM, k, 1, 0> conv22v;
	Bias<MAX_MOVE_LABEL_NUM, 1, 1> bias22v;
	BatchNormalization<MAX_MOVE_LABEL_NUM> bn22v;
	Linear<9 * 9 * MAX_MOVE_LABEL_NUM, fcl> l23v;
	Bias<fcl, 1, 1> bias23v;
	Linear<fcl, 1> l24v;
	Bias<1, 1, 1> bias24v;

	ReLU relu;
	Add add;
	Sigmoid sigmoid;

	CudnnTensorDescriptor x1Desc;
	CudnnTensorDescriptor x2Desc;
	CudnnTensorDescriptor h1Desc;
	CudnnTensorDescriptor h22Desc;
	CudnnTensorDescriptor h22vDesc;
	CudnnTensorDescriptor h23vDesc;
	CudnnTensorDescriptor h24vDesc;
	CudnnTensorDescriptor y1Desc;
	CudnnTensorDescriptor y2Desc;

	// input layer
	DType* x1_dev;
	DType* x2_dev;
	DType* h1_1_1_dev;
	DType* h1_1_2_dev;
	DType* h1_2_dev;
	// residual block
	DType* h1_bn_dev;
	DType* h2_dev;
	DType* h3_dev;
	DType* h5_dev;
	DType* h7_dev;
	DType* h9_dev;
	DType* h11_dev;
	DType* h13_dev;
	DType* h15_dev;
	DType* h17_dev;
	DType* h19_dev;
	DType* h21_dev;
	// after residual blocks
	DType* h21_bn_dev;
	// policy network
	DType* y1_dev;
	// value network
	DType* h22v_dev;
	DType* h22v_bn_dev;
	DType* h23v_dev;
	DType* y2_dev;
};