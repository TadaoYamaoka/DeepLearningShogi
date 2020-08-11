#include "nn_senet10.h"
#include "npz.h"

void debug_print_dev(DType* dev, const size_t size) {
	std::vector<DType> host(size);
	checkCudaErrors(cudaMemcpy(host.data(), dev, size * sizeof(DType), cudaMemcpyDeviceToHost));
	std::cout << to_float(host[0]);
	for (size_t i = 1; i < host.size(); i++) {
		std::cout << ", " << to_float(host[i]);
	}
	std::cout << std::endl;
}

NNSENet10::NNSENet10(const int max_batch_size) : max_batch_size(max_batch_size)
{
	prepare_desc(max_batch_size);

	// init conv layers
	conv1_1_1.init(cudnnHandle, x1Desc, h1Desc);
	conv1_1_2.init(cudnnHandle, x1Desc, h1Desc);
	conv1_2.init(cudnnHandle, x2Desc, h1Desc);
	conv2.init(cudnnHandle, h1Desc, h1Desc);
	conv3.init(cudnnHandle, h1Desc, h1Desc);
	conv4.init(cudnnHandle, h1Desc, h1Desc);
	conv5.init(cudnnHandle, h1Desc, h1Desc);
	conv6.init(cudnnHandle, h1Desc, h1Desc);
	conv7.init(cudnnHandle, h1Desc, h1Desc);
	conv8.init(cudnnHandle, h1Desc, h1Desc);
	conv9.init(cudnnHandle, h1Desc, h1Desc);
	conv10.init(cudnnHandle, h1Desc, h1Desc);
	conv11.init(cudnnHandle, h1Desc, h1Desc);
	conv12.init(cudnnHandle, h1Desc, h1Desc);
	conv13.init(cudnnHandle, h1Desc, h1Desc);
	conv14.init(cudnnHandle, h1Desc, h1Desc);
	conv15.init(cudnnHandle, h1Desc, h1Desc);
	conv16.init(cudnnHandle, h1Desc, h1Desc);
	conv17.init(cudnnHandle, h1Desc, h1Desc);
	conv18.init(cudnnHandle, h1Desc, h1Desc);
	conv19.init(cudnnHandle, h1Desc, h1Desc);
	conv20.init(cudnnHandle, h1Desc, h1Desc);
	conv21.init(cudnnHandle, h1Desc, h1Desc);
	conv22.init(cudnnHandle, h1Desc, y1Desc);
	conv22v.init(cudnnHandle, h1Desc, h22vDesc);

	// malloc
	checkCudaErrors(cudaMalloc((void**)&x1_dev, conv1_1_1.get_xsize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&x2_dev, conv1_2.get_xsize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h1_1_1_dev, conv1_1_1.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h1_1_2_dev, conv1_1_2.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h1_2_dev, conv1_2.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h1_bn_dev, conv1_1_1.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h2_dev, conv2.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h2_bn_dev, conv2.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h3_dev, conv3.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h5_dev, conv5.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h7_dev, conv7.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h9_dev, conv9.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h11_dev, conv11.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h13_dev, conv13.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h15_dev, conv15.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h17_dev, conv17.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h19_dev, conv19.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h21_dev, conv21.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&se1_dev, max_batch_size * k * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&se2_dev, max_batch_size * (k / reduction) * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&se3_dev, max_batch_size * k * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&h21_bn_dev, conv21.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&y1_dev, conv22.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h22v_dev, conv22v.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h22v_bn_dev, conv22v.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h23v_dev, max_batch_size * fcl * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&y2_dev, max_batch_size * sizeof(DType)));
}

NNSENet10::~NNSENet10() {
	checkCudaErrors(cudaFree(x1_dev));
	checkCudaErrors(cudaFree(x2_dev));
	checkCudaErrors(cudaFree(h1_1_1_dev));
	checkCudaErrors(cudaFree(h1_1_2_dev));
	checkCudaErrors(cudaFree(h1_2_dev));
	checkCudaErrors(cudaFree(h1_bn_dev));
	checkCudaErrors(cudaFree(h2_dev));
	checkCudaErrors(cudaFree(h2_bn_dev));
	checkCudaErrors(cudaFree(h3_dev));
	checkCudaErrors(cudaFree(h5_dev));
	checkCudaErrors(cudaFree(h7_dev));
	checkCudaErrors(cudaFree(h9_dev));
	checkCudaErrors(cudaFree(h11_dev));
	checkCudaErrors(cudaFree(h13_dev));
	checkCudaErrors(cudaFree(h15_dev));
	checkCudaErrors(cudaFree(h17_dev));
	checkCudaErrors(cudaFree(h19_dev));
	checkCudaErrors(cudaFree(h21_dev));
	checkCudaErrors(cudaFree(se1_dev));
	checkCudaErrors(cudaFree(se2_dev));
	checkCudaErrors(cudaFree(se3_dev));
	checkCudaErrors(cudaFree(h21_bn_dev));
	checkCudaErrors(cudaFree(y1_dev));
	checkCudaErrors(cudaFree(h22v_dev));
	checkCudaErrors(cudaFree(h22v_bn_dev));
	checkCudaErrors(cudaFree(h23v_dev));
	checkCudaErrors(cudaFree(y2_dev));
}

void NNSENet10::prepare_desc(const int batch_size)
{
	conv1_1_1.get_xdesc(x1Desc, batch_size, 9, 9);
	conv1_2.get_xdesc(x2Desc, batch_size, 9, 9);
	conv1_1_1.get_ydesc(h1Desc, batch_size, 9, 9);

	averagePooling2D.get_desc(se1Desc, batch_size, k, 1, 1);
	se3_l1.get_ydesc(se2Desc, batch_size);
	se3_l2.get_ydesc(se3Desc, batch_size);
	cudnnSetOpTensorDescriptor(opMulDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

	conv22.get_ydesc(y1Desc, batch_size, 9, 9);

	conv22v.get_ydesc(h22vDesc, batch_size, 9, 9);
	l23v.get_ydesc(h23vDesc, batch_size);
	l24v.get_ydesc(y2Desc, batch_size);
}

void NNSENet10::load_model(const char* filepath)
{
	// load nn params
	ParamMap params;
	load_npz(filepath, params);

	conv1_1_1.set_param(params["l1_1_1/W.npy"].data);
	conv1_1_2.set_param(params["l1_1_2/W.npy"].data);
	conv1_2.set_param(params["l1_2/W.npy"].data);
	bn1.set_param(params["norm1/gamma.npy"].data, params["norm1/beta.npy"].data, params["norm1/avg_mean.npy"].data, params["norm1/avg_var.npy"].data);
	conv2.set_param(params["l2/W.npy"].data);
	bn2.set_param(params["norm2/gamma.npy"].data, params["norm2/beta.npy"].data, params["norm2/avg_mean.npy"].data, params["norm2/avg_var.npy"].data);
	conv3.set_param(params["l3/W.npy"].data);
	bn3.set_param(params["norm3/gamma.npy"].data, params["norm3/beta.npy"].data, params["norm3/avg_mean.npy"].data, params["norm3/avg_var.npy"].data);
	se3_l1.set_param(params["se3/l1/W.npy"].data);
	se3_l2.set_param(params["se3/l2/W.npy"].data);
	conv4.set_param(params["l4/W.npy"].data);
	bn4.set_param(params["norm4/gamma.npy"].data, params["norm4/beta.npy"].data, params["norm4/avg_mean.npy"].data, params["norm4/avg_var.npy"].data);
	conv5.set_param(params["l5/W.npy"].data);
	bn5.set_param(params["norm5/gamma.npy"].data, params["norm5/beta.npy"].data, params["norm5/avg_mean.npy"].data, params["norm5/avg_var.npy"].data);
	se5_l1.set_param(params["se5/l1/W.npy"].data);
	se5_l2.set_param(params["se5/l2/W.npy"].data);
	conv6.set_param(params["l6/W.npy"].data);
	bn6.set_param(params["norm6/gamma.npy"].data, params["norm6/beta.npy"].data, params["norm6/avg_mean.npy"].data, params["norm6/avg_var.npy"].data);
	conv7.set_param(params["l7/W.npy"].data);
	bn7.set_param(params["norm7/gamma.npy"].data, params["norm7/beta.npy"].data, params["norm7/avg_mean.npy"].data, params["norm7/avg_var.npy"].data);
	se7_l1.set_param(params["se7/l1/W.npy"].data);
	se7_l2.set_param(params["se7/l2/W.npy"].data);
	conv8.set_param(params["l8/W.npy"].data);
	bn8.set_param(params["norm8/gamma.npy"].data, params["norm8/beta.npy"].data, params["norm8/avg_mean.npy"].data, params["norm8/avg_var.npy"].data);
	conv9.set_param(params["l9/W.npy"].data);
	bn9.set_param(params["norm9/gamma.npy"].data, params["norm9/beta.npy"].data, params["norm9/avg_mean.npy"].data, params["norm9/avg_var.npy"].data);
	se9_l1.set_param(params["se9/l1/W.npy"].data);
	se9_l2.set_param(params["se9/l2/W.npy"].data);
	conv10.set_param(params["l10/W.npy"].data);
	bn10.set_param(params["norm10/gamma.npy"].data, params["norm10/beta.npy"].data, params["norm10/avg_mean.npy"].data, params["norm10/avg_var.npy"].data);
	conv11.set_param(params["l11/W.npy"].data);
	bn11.set_param(params["norm11/gamma.npy"].data, params["norm11/beta.npy"].data, params["norm11/avg_mean.npy"].data, params["norm11/avg_var.npy"].data);
	se11_l1.set_param(params["se11/l1/W.npy"].data);
	se11_l2.set_param(params["se11/l2/W.npy"].data);
	conv12.set_param(params["l12/W.npy"].data);
	bn12.set_param(params["norm12/gamma.npy"].data, params["norm12/beta.npy"].data, params["norm12/avg_mean.npy"].data, params["norm12/avg_var.npy"].data);
	conv13.set_param(params["l13/W.npy"].data);
	bn13.set_param(params["norm13/gamma.npy"].data, params["norm13/beta.npy"].data, params["norm13/avg_mean.npy"].data, params["norm13/avg_var.npy"].data);
	se13_l1.set_param(params["se13/l1/W.npy"].data);
	se13_l2.set_param(params["se13/l2/W.npy"].data);
	conv14.set_param(params["l14/W.npy"].data);
	bn14.set_param(params["norm14/gamma.npy"].data, params["norm14/beta.npy"].data, params["norm14/avg_mean.npy"].data, params["norm14/avg_var.npy"].data);
	conv15.set_param(params["l15/W.npy"].data);
	bn15.set_param(params["norm15/gamma.npy"].data, params["norm15/beta.npy"].data, params["norm15/avg_mean.npy"].data, params["norm15/avg_var.npy"].data);
	se15_l1.set_param(params["se15/l1/W.npy"].data);
	se15_l2.set_param(params["se15/l2/W.npy"].data);
	conv16.set_param(params["l16/W.npy"].data);
	bn16.set_param(params["norm16/gamma.npy"].data, params["norm16/beta.npy"].data, params["norm16/avg_mean.npy"].data, params["norm16/avg_var.npy"].data);
	conv17.set_param(params["l17/W.npy"].data);
	bn17.set_param(params["norm17/gamma.npy"].data, params["norm17/beta.npy"].data, params["norm17/avg_mean.npy"].data, params["norm17/avg_var.npy"].data);
	se17_l1.set_param(params["se17/l1/W.npy"].data);
	se17_l2.set_param(params["se17/l2/W.npy"].data);
	conv18.set_param(params["l18/W.npy"].data);
	bn18.set_param(params["norm18/gamma.npy"].data, params["norm18/beta.npy"].data, params["norm18/avg_mean.npy"].data, params["norm18/avg_var.npy"].data);
	conv19.set_param(params["l19/W.npy"].data);
	bn19.set_param(params["norm19/gamma.npy"].data, params["norm19/beta.npy"].data, params["norm19/avg_mean.npy"].data, params["norm19/avg_var.npy"].data);
	se19_l1.set_param(params["se19/l1/W.npy"].data);
	se19_l2.set_param(params["se19/l2/W.npy"].data);
	conv20.set_param(params["l20/W.npy"].data);
	bn20.set_param(params["norm20/gamma.npy"].data, params["norm20/beta.npy"].data, params["norm20/avg_mean.npy"].data, params["norm20/avg_var.npy"].data);
	conv21.set_param(params["l21/W.npy"].data);
	bn21.set_param(params["norm21/gamma.npy"].data, params["norm21/beta.npy"].data, params["norm21/avg_mean.npy"].data, params["norm21/avg_var.npy"].data);
	se21_l1.set_param(params["se21/l1/W.npy"].data);
	se21_l2.set_param(params["se21/l2/W.npy"].data);
	conv22.set_param(params["l22/W.npy"].data);
	bias22.set_bias(params["l22_2/b.npy"].data);
	conv22v.set_param(params["l22_v/W.npy"].data);
	bias22v.set_bias(params["l22_v/b.npy"].data);
	bn22v.set_param(params["norm22_v/gamma.npy"].data, params["norm22_v/beta.npy"].data, params["norm22_v/avg_mean.npy"].data, params["norm22_v/avg_var.npy"].data);
	l23v.set_param(params["l23_v/W.npy"].data);
	bias23v.set_bias(params["l23_v/b.npy"].data);
	l24v.set_param(params["l24_v/W.npy"].data);
	bias24v.set_bias(params["l24_v/b.npy"].data);
}

void NNSENet10::forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2)
{
	prepare_desc(batch_size);

	// input
	checkCudaErrors(cudaMemcpy(x1_dev, x1, sizeof(features1_t) * batch_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(x2_dev, x2, sizeof(features2_t) * batch_size, cudaMemcpyHostToDevice));

	// layer1
	conv1_1_1(cudnnHandle, x1Desc, x1_dev, h1Desc, h1_1_1_dev);
	conv1_1_2(cudnnHandle, x1Desc, x1_dev, h1Desc, h1_1_2_dev);
	conv1_2(cudnnHandle, x2Desc, x2_dev, h1Desc, h1_2_dev);
	add(cudnnHandle, h1Desc, h1_1_2_dev, h1_1_1_dev);
	add(cudnnHandle, h1Desc, h1_2_dev, h1_1_1_dev);

	// residual block1
	bn1(cudnnHandle, h1Desc, h1_1_1_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv2(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn2(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv3(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h3_dev);
	se(se3_l1, se3_l2, batch_size, h3_dev);
	add(cudnnHandle, h1Desc, h1_1_1_dev, h3_dev);

	// residual block2
	bn3(cudnnHandle, h1Desc, h3_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv4(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn4(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv5(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h5_dev);
	se(se5_l1, se5_l2, batch_size, h5_dev);
	add(cudnnHandle, h1Desc, h3_dev, h5_dev);

	// residual block3
	bn5(cudnnHandle, h1Desc, h5_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv6(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn6(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv7(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h7_dev);
	se(se7_l1, se7_l2, batch_size, h7_dev);
	add(cudnnHandle, h1Desc, h5_dev, h7_dev);

	// residual block4
	bn7(cudnnHandle, h1Desc, h7_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv8(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn8(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv9(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h9_dev);
	se(se9_l1, se9_l2, batch_size, h9_dev);
	add(cudnnHandle, h1Desc, h7_dev, h9_dev);

	// residual block5
	bn9(cudnnHandle, h1Desc, h9_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv10(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn10(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv11(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h11_dev);
	se(se11_l1, se11_l2, batch_size, h11_dev);
	add(cudnnHandle, h1Desc, h9_dev, h11_dev);

	// residual block6
	bn11(cudnnHandle, h1Desc, h11_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv12(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn12(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv13(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h13_dev);
	se(se13_l1, se13_l2, batch_size, h13_dev);
	add(cudnnHandle, h1Desc, h11_dev, h13_dev);

	// residual block7
	bn13(cudnnHandle, h1Desc, h13_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv14(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn14(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv15(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h15_dev);
	se(se15_l1, se15_l2, batch_size, h15_dev);
	add(cudnnHandle, h1Desc, h13_dev, h15_dev);

	// residual block8
	bn15(cudnnHandle, h1Desc, h15_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv16(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn16(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv17(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h17_dev);
	se(se17_l1, se17_l2, batch_size, h17_dev);
	add(cudnnHandle, h1Desc, h15_dev, h17_dev);

	// residual block9
	bn17(cudnnHandle, h1Desc, h17_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv18(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn18(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv19(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h19_dev);
	se(se19_l1, se19_l2, batch_size, h19_dev);
	add(cudnnHandle, h1Desc, h17_dev, h19_dev);

	// residual block10
	bn19(cudnnHandle, h1Desc, h19_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv20(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn20(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv21(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h21_dev);
	se(se21_l1, se21_l2, batch_size, h21_dev);
	add(cudnnHandle, h1Desc, h19_dev, h21_dev);

	// after residual blocks
	bn21(cudnnHandle, h1Desc, h21_dev, h21_bn_dev);
	relu(cudnnHandle, h1Desc, h21_bn_dev);

	// policy network
	conv22(cudnnHandle, h1Desc, h21_bn_dev, y1Desc, y1_dev);
	bias22(cudnnHandle, y1Desc, y1_dev);

	// value network
	conv22v(cudnnHandle, h1Desc, h21_bn_dev, h22vDesc, h22v_dev);
	bias22v(cudnnHandle, h22vDesc, h22v_dev);
	bn22v(cudnnHandle, h22vDesc, h22v_dev, h22v_bn_dev);
	relu(cudnnHandle, h22vDesc, h22v_bn_dev);
	l23v(cublasHandle, batch_size, h22v_bn_dev, h23v_dev);
	bias23v(cudnnHandle, h23vDesc, h23v_dev);
	relu(cudnnHandle, h23vDesc, h23v_dev);
	l24v(cublasHandle, batch_size, h23v_dev, y2_dev);
	bias24v(cudnnHandle, y2Desc, y2_dev);
	sigmoid(cudnnHandle, y2Desc, y2_dev);

	// output
	checkCudaErrors(cudaMemcpy(y1, y1_dev, conv22.get_ysize(batch_size, 9, 9), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(y2, y2_dev, batch_size * sizeof(DType), cudaMemcpyDeviceToHost));
}

void NNSENet10::se(Linear<k, k / reduction>& se_l1, Linear<k / reduction, k>& se_l2, const int &batch_size, DType* x_dev)
{
	averagePooling2D(cudnnHandle, h1Desc, x_dev, se1Desc, se1_dev);
	se_l1(cublasHandle, batch_size, se1_dev, se2_dev);
	relu(cudnnHandle, se2Desc, se2_dev);
	se_l2(cublasHandle, batch_size, se2_dev, se3_dev);
	sigmoid(cudnnHandle, se3Desc, se3_dev);
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnOpTensor(cudnnHandle, opMulDesc, &alpha, h1Desc, x_dev, &alpha, se1Desc, se3_dev, &beta, h1Desc, x_dev);
}
