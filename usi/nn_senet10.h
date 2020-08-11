#pragma once

#include "nn.h"

class NNSENet10 : NN {
public:
	NNSENet10(const int max_batch_size);
	~NNSENet10();

	void load_model(const char* filename);

	void forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2);

private:
	static constexpr int k = 192;
	static constexpr int fcl = 256;
	static constexpr int reduction = 16;

	void prepare_desc(const int batch_size);
	void se(Linear<k, k / reduction>& se_l1, Linear<k / reduction, k>& se_l2, const int &batch_size, DType* x_dev);

	CudnnHandle cudnnHandle;
	CublasHandle cublasHandle;

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
	Linear<k, k / reduction> se3_l1;
	Linear<k / reduction, k> se3_l2;
	ConvLayer<k, k, 3, 1> conv4;
	BatchNormalization<k> bn4;
	ConvLayer<k, k, 3, 1> conv5;
	BatchNormalization<k> bn5;
	Linear<k, k / reduction> se5_l1;
	Linear<k / reduction, k> se5_l2;
	ConvLayer<k, k, 3, 1> conv6;
	BatchNormalization<k> bn6;
	ConvLayer<k, k, 3, 1> conv7;
	BatchNormalization<k> bn7;
	Linear<k, k / reduction> se7_l1;
	Linear<k / reduction, k> se7_l2;
	ConvLayer<k, k, 3, 1> conv8;
	BatchNormalization<k> bn8;
	ConvLayer<k, k, 3, 1> conv9;
	BatchNormalization<k> bn9;
	Linear<k, k / reduction> se9_l1;
	Linear<k / reduction, k> se9_l2;
	ConvLayer<k, k, 3, 1> conv10;
	BatchNormalization<k> bn10;
	ConvLayer<k, k, 3, 1> conv11;
	BatchNormalization<k> bn11;
	Linear<k, k / reduction> se11_l1;
	Linear<k / reduction, k> se11_l2;
	ConvLayer<k, k, 3, 1> conv12;
	BatchNormalization<k> bn12;
	ConvLayer<k, k, 3, 1> conv13;
	BatchNormalization<k> bn13;
	Linear<k, k / reduction> se13_l1;
	Linear<k / reduction, k> se13_l2;
	ConvLayer<k, k, 3, 1> conv14;
	BatchNormalization<k> bn14;
	ConvLayer<k, k, 3, 1> conv15;
	BatchNormalization<k> bn15;
	Linear<k, k / reduction> se15_l1;
	Linear<k / reduction, k> se15_l2;
	ConvLayer<k, k, 3, 1> conv16;
	BatchNormalization<k> bn16;
	ConvLayer<k, k, 3, 1> conv17;
	BatchNormalization<k> bn17;
	Linear<k, k / reduction> se17_l1;
	Linear<k / reduction, k> se17_l2;
	ConvLayer<k, k, 3, 1> conv18;
	BatchNormalization<k> bn18;
	ConvLayer<k, k, 3, 1> conv19;
	BatchNormalization<k> bn19;
	Linear<k, k / reduction> se19_l1;
	Linear<k / reduction, k> se19_l2;
	ConvLayer<k, k, 3, 1> conv20;
	BatchNormalization<k> bn20;
	ConvLayer<k, k, 3, 1> conv21;
	BatchNormalization<k> bn21;
	Linear<k, k / reduction> se21_l1;
	Linear<k / reduction, k> se21_l2;
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
	AveragePooling2D<9> averagePooling2D;

	CudnnTensorDescriptor x1Desc;
	CudnnTensorDescriptor x2Desc;
	CudnnTensorDescriptor h1Desc;
	CudnnTensorDescriptor se1Desc;
	CudnnTensorDescriptor se2Desc;
	CudnnTensorDescriptor se3Desc;
	CudnnTensorDescriptor h22Desc;
	CudnnTensorDescriptor h22vDesc;
	CudnnTensorDescriptor h23vDesc;
	CudnnTensorDescriptor h24vDesc;
	CudnnTensorDescriptor y1Desc;
	CudnnTensorDescriptor y2Desc;

	CudnnOpTensorDescriptor opMulDesc;

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
	// se layer
	DType* se1_dev;
	DType* se2_dev;
	DType* se3_dev;
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
