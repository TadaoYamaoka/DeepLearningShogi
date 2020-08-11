#pragma once

#include "nn.h"

class NNWideResnet15 : NN {
public:
	NNWideResnet15(const int max_batch_size);
	~NNWideResnet15();

	void load_model(const char* filename);

	void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

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
	BatchNormalization<k> bn2;
	ConvLayer<k, k, 3, 1> conv3;
	BatchNormalization<k> bn3;
	ConvLayer<k, k, 3, 1> conv4;
	BatchNormalization<k> bn4;
	ConvLayer<k, k, 3, 1> conv5;
	BatchNormalization<k> bn5;
	ConvLayer<k, k, 3, 1> conv6;
	BatchNormalization<k> bn6;
	ConvLayer<k, k, 3, 1> conv7;
	BatchNormalization<k> bn7;
	ConvLayer<k, k, 3, 1> conv8;
	BatchNormalization<k> bn8;
	ConvLayer<k, k, 3, 1> conv9;
	BatchNormalization<k> bn9;
	ConvLayer<k, k, 3, 1> conv10;
	BatchNormalization<k> bn10;
	ConvLayer<k, k, 3, 1> conv11;
	BatchNormalization<k> bn11;
	ConvLayer<k, k, 3, 1> conv12;
	BatchNormalization<k> bn12;
	ConvLayer<k, k, 3, 1> conv13;
	BatchNormalization<k> bn13;
	ConvLayer<k, k, 3, 1> conv14;
	BatchNormalization<k> bn14;
	ConvLayer<k, k, 3, 1> conv15;
	BatchNormalization<k> bn15;
	ConvLayer<k, k, 3, 1> conv16;
	BatchNormalization<k> bn16;
	ConvLayer<k, k, 3, 1> conv17;
	BatchNormalization<k> bn17;
	ConvLayer<k, k, 3, 1> conv18;
	BatchNormalization<k> bn18;
	ConvLayer<k, k, 3, 1> conv19;
	BatchNormalization<k> bn19;
	ConvLayer<k, k, 3, 1> conv20;
	BatchNormalization<k> bn20;
	ConvLayer<k, k, 3, 1> conv21;
	BatchNormalization<k> bn21;
	ConvLayer<k, k, 3, 1> conv22;
	BatchNormalization<k> bn22;
	ConvLayer<k, k, 3, 1> conv23;
	BatchNormalization<k> bn23;
	ConvLayer<k, k, 3, 1> conv24;
	BatchNormalization<k> bn24;
	ConvLayer<k, k, 3, 1> conv25;
	BatchNormalization<k> bn25;
	ConvLayer<k, k, 3, 1> conv26;
	BatchNormalization<k> bn26;
	ConvLayer<k, k, 3, 1> conv27;
	BatchNormalization<k> bn27;
	ConvLayer<k, k, 3, 1> conv28;
	BatchNormalization<k> bn28;
	ConvLayer<k, k, 3, 1> conv29;
	BatchNormalization<k> bn29;
	ConvLayer<k, k, 3, 1> conv30;
	BatchNormalization<k> bn30;
	ConvLayer<k, k, 3, 1> conv31;
	BatchNormalization<k> bn31;
	// policy network
	ConvLayer<MAX_MOVE_LABEL_NUM, k, 1, 0> conv32;
	Bias<MAX_MOVE_LABEL_NUM, 9, 9> bias32;
	// value network
	ConvLayer<MAX_MOVE_LABEL_NUM, k, 1, 0> conv32v;
	Bias<MAX_MOVE_LABEL_NUM, 1, 1> bias32v;
	BatchNormalization<MAX_MOVE_LABEL_NUM> bn32v;
	Linear<9 * 9 * MAX_MOVE_LABEL_NUM, fcl> l33v;
	Bias<fcl, 1, 1> bias33v;
	Linear<fcl, 1> l34v;
	Bias<1, 1, 1> bias34v;

	ReLU relu;
	Add add;
	Sigmoid sigmoid;

	CudnnTensorDescriptor x1Desc;
	CudnnTensorDescriptor x2Desc;
	CudnnTensorDescriptor h1Desc;
	CudnnTensorDescriptor h32Desc;
	CudnnTensorDescriptor h32vDesc;
	CudnnTensorDescriptor h33vDesc;
	CudnnTensorDescriptor h34vDesc;
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
	DType* h2_bn_dev;
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
	DType* h23_dev;
	DType* h25_dev;
	DType* h27_dev;
	DType* h29_dev;
	DType* h31_dev;
	// after residual blocks
	DType* h31_bn_dev;
	// policy network
	DType* y1_dev;
	// value network
	DType* h32v_dev;
	DType* h32v_bn_dev;
	DType* h33v_dev;
	DType* y2_dev;
};