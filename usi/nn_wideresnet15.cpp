#include "nn_wideresnet15.h"
#include "npz.h"

NNWideResnet15::NNWideResnet15(const int max_batch_size) : max_batch_size(max_batch_size)
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
	conv22.init(cudnnHandle, h1Desc, h1Desc);
	conv23.init(cudnnHandle, h1Desc, h1Desc);
	conv24.init(cudnnHandle, h1Desc, h1Desc);
	conv25.init(cudnnHandle, h1Desc, h1Desc);
	conv26.init(cudnnHandle, h1Desc, h1Desc);
	conv27.init(cudnnHandle, h1Desc, h1Desc);
	conv28.init(cudnnHandle, h1Desc, h1Desc);
	conv29.init(cudnnHandle, h1Desc, h1Desc);
	conv30.init(cudnnHandle, h1Desc, h1Desc);
	conv31.init(cudnnHandle, h1Desc, h1Desc);
	conv32.init(cudnnHandle, h1Desc, y1Desc);
	conv32v.init(cudnnHandle, h1Desc, h32vDesc);

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
	checkCudaErrors(cudaMalloc((void**)&h23_dev, conv23.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h25_dev, conv25.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h27_dev, conv27.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h29_dev, conv29.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h31_dev, conv31.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h31_bn_dev, conv31.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&y1_dev, conv32.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h32v_dev, conv32v.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h32v_bn_dev, conv32v.get_ysize(max_batch_size, 9, 9)));
	checkCudaErrors(cudaMalloc((void**)&h33v_dev, max_batch_size * fcl * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&y2_dev, max_batch_size * sizeof(DType)));
}

NNWideResnet15::~NNWideResnet15() {
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
	checkCudaErrors(cudaFree(h23_dev));
	checkCudaErrors(cudaFree(h25_dev));
	checkCudaErrors(cudaFree(h27_dev));
	checkCudaErrors(cudaFree(h29_dev));
	checkCudaErrors(cudaFree(h31_dev));
	checkCudaErrors(cudaFree(h31_bn_dev));
	checkCudaErrors(cudaFree(y1_dev));
	checkCudaErrors(cudaFree(h32v_dev));
	checkCudaErrors(cudaFree(h32v_bn_dev));
	checkCudaErrors(cudaFree(h33v_dev));
	checkCudaErrors(cudaFree(y2_dev));
}

void NNWideResnet15::prepare_desc(const int batch_size)
{
	conv1_1_1.get_xdesc(x1Desc, batch_size, 9, 9);
	conv1_2.get_xdesc(x2Desc, batch_size, 9, 9);
	conv1_1_1.get_ydesc(h1Desc, batch_size, 9, 9);

	conv32.get_ydesc(y1Desc, batch_size, 9, 9);

	conv32v.get_ydesc(h32vDesc, batch_size, 9, 9);
	l33v.get_ydesc(h33vDesc, batch_size);
	l34v.get_ydesc(y2Desc, batch_size);
}

void NNWideResnet15::load_model(const char* filepath)
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
	conv4.set_param(params["l4/W.npy"].data);
	bn4.set_param(params["norm4/gamma.npy"].data, params["norm4/beta.npy"].data, params["norm4/avg_mean.npy"].data, params["norm4/avg_var.npy"].data);
	conv5.set_param(params["l5/W.npy"].data);
	bn5.set_param(params["norm5/gamma.npy"].data, params["norm5/beta.npy"].data, params["norm5/avg_mean.npy"].data, params["norm5/avg_var.npy"].data);
	conv6.set_param(params["l6/W.npy"].data);
	bn6.set_param(params["norm6/gamma.npy"].data, params["norm6/beta.npy"].data, params["norm6/avg_mean.npy"].data, params["norm6/avg_var.npy"].data);
	conv7.set_param(params["l7/W.npy"].data);
	bn7.set_param(params["norm7/gamma.npy"].data, params["norm7/beta.npy"].data, params["norm7/avg_mean.npy"].data, params["norm7/avg_var.npy"].data);
	conv8.set_param(params["l8/W.npy"].data);
	bn8.set_param(params["norm8/gamma.npy"].data, params["norm8/beta.npy"].data, params["norm8/avg_mean.npy"].data, params["norm8/avg_var.npy"].data);
	conv9.set_param(params["l9/W.npy"].data);
	bn9.set_param(params["norm9/gamma.npy"].data, params["norm9/beta.npy"].data, params["norm9/avg_mean.npy"].data, params["norm9/avg_var.npy"].data);
	conv10.set_param(params["l10/W.npy"].data);
	bn10.set_param(params["norm10/gamma.npy"].data, params["norm10/beta.npy"].data, params["norm10/avg_mean.npy"].data, params["norm10/avg_var.npy"].data);
	conv11.set_param(params["l11/W.npy"].data);
	bn11.set_param(params["norm11/gamma.npy"].data, params["norm11/beta.npy"].data, params["norm11/avg_mean.npy"].data, params["norm11/avg_var.npy"].data);
	conv12.set_param(params["l12/W.npy"].data);
	bn12.set_param(params["norm12/gamma.npy"].data, params["norm12/beta.npy"].data, params["norm12/avg_mean.npy"].data, params["norm12/avg_var.npy"].data);
	conv13.set_param(params["l13/W.npy"].data);
	bn13.set_param(params["norm13/gamma.npy"].data, params["norm13/beta.npy"].data, params["norm13/avg_mean.npy"].data, params["norm13/avg_var.npy"].data);
	conv14.set_param(params["l14/W.npy"].data);
	bn14.set_param(params["norm14/gamma.npy"].data, params["norm14/beta.npy"].data, params["norm14/avg_mean.npy"].data, params["norm14/avg_var.npy"].data);
	conv15.set_param(params["l15/W.npy"].data);
	bn15.set_param(params["norm15/gamma.npy"].data, params["norm15/beta.npy"].data, params["norm15/avg_mean.npy"].data, params["norm15/avg_var.npy"].data);
	conv16.set_param(params["l16/W.npy"].data);
	bn16.set_param(params["norm16/gamma.npy"].data, params["norm16/beta.npy"].data, params["norm16/avg_mean.npy"].data, params["norm16/avg_var.npy"].data);
	conv17.set_param(params["l17/W.npy"].data);
	bn17.set_param(params["norm17/gamma.npy"].data, params["norm17/beta.npy"].data, params["norm17/avg_mean.npy"].data, params["norm17/avg_var.npy"].data);
	conv18.set_param(params["l18/W.npy"].data);
	bn18.set_param(params["norm18/gamma.npy"].data, params["norm18/beta.npy"].data, params["norm18/avg_mean.npy"].data, params["norm18/avg_var.npy"].data);
	conv19.set_param(params["l19/W.npy"].data);
	bn19.set_param(params["norm19/gamma.npy"].data, params["norm19/beta.npy"].data, params["norm19/avg_mean.npy"].data, params["norm19/avg_var.npy"].data);
	conv20.set_param(params["l20/W.npy"].data);
	bn20.set_param(params["norm20/gamma.npy"].data, params["norm20/beta.npy"].data, params["norm20/avg_mean.npy"].data, params["norm20/avg_var.npy"].data);
	conv21.set_param(params["l21/W.npy"].data);
	bn21.set_param(params["norm21/gamma.npy"].data, params["norm21/beta.npy"].data, params["norm21/avg_mean.npy"].data, params["norm21/avg_var.npy"].data);
	conv22.set_param(params["l22/W.npy"].data);
	bn22.set_param(params["norm22/gamma.npy"].data, params["norm22/beta.npy"].data, params["norm22/avg_mean.npy"].data, params["norm22/avg_var.npy"].data);
	conv23.set_param(params["l23/W.npy"].data);
	bn23.set_param(params["norm23/gamma.npy"].data, params["norm23/beta.npy"].data, params["norm23/avg_mean.npy"].data, params["norm23/avg_var.npy"].data);
	conv24.set_param(params["l24/W.npy"].data);
	bn24.set_param(params["norm24/gamma.npy"].data, params["norm24/beta.npy"].data, params["norm24/avg_mean.npy"].data, params["norm24/avg_var.npy"].data);
	conv25.set_param(params["l25/W.npy"].data);
	bn25.set_param(params["norm25/gamma.npy"].data, params["norm25/beta.npy"].data, params["norm25/avg_mean.npy"].data, params["norm25/avg_var.npy"].data);
	conv26.set_param(params["l26/W.npy"].data);
	bn26.set_param(params["norm26/gamma.npy"].data, params["norm26/beta.npy"].data, params["norm26/avg_mean.npy"].data, params["norm26/avg_var.npy"].data);
	conv27.set_param(params["l27/W.npy"].data);
	bn27.set_param(params["norm27/gamma.npy"].data, params["norm27/beta.npy"].data, params["norm27/avg_mean.npy"].data, params["norm27/avg_var.npy"].data);
	conv28.set_param(params["l28/W.npy"].data);
	bn28.set_param(params["norm28/gamma.npy"].data, params["norm28/beta.npy"].data, params["norm28/avg_mean.npy"].data, params["norm28/avg_var.npy"].data);
	conv29.set_param(params["l29/W.npy"].data);
	bn29.set_param(params["norm29/gamma.npy"].data, params["norm29/beta.npy"].data, params["norm29/avg_mean.npy"].data, params["norm29/avg_var.npy"].data);
	conv30.set_param(params["l30/W.npy"].data);
	bn30.set_param(params["norm30/gamma.npy"].data, params["norm30/beta.npy"].data, params["norm30/avg_mean.npy"].data, params["norm30/avg_var.npy"].data);
	conv31.set_param(params["l31/W.npy"].data);
	bn31.set_param(params["norm31/gamma.npy"].data, params["norm31/beta.npy"].data, params["norm31/avg_mean.npy"].data, params["norm31/avg_var.npy"].data);
	conv32.set_param(params["l32/W.npy"].data);
	bias32.set_bias(params["l32_2/b.npy"].data);
	conv32v.set_param(params["l32_v/W.npy"].data);
	bn32v.set_param(params["norm32_v/gamma.npy"].data, params["norm32_v/beta.npy"].data, params["norm32_v/avg_mean.npy"].data, params["norm32_v/avg_var.npy"].data);
	l33v.set_param(params["l33_v/W.npy"].data);
	bias33v.set_bias(params["l33_v/b.npy"].data);
	l34v.set_param(params["l34_v/W.npy"].data);
	bias34v.set_bias(params["l34_v/b.npy"].data);
}

void NNWideResnet15::foward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2)
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
	add(cudnnHandle, h1Desc, h1_1_1_dev, h3_dev);

	// residual block2
	bn3(cudnnHandle, h1Desc, h3_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv4(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn4(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv5(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h5_dev);
	add(cudnnHandle, h1Desc, h3_dev, h5_dev);

	// residual block3
	bn5(cudnnHandle, h1Desc, h5_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv6(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn6(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv7(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h7_dev);
	add(cudnnHandle, h1Desc, h5_dev, h7_dev);

	// residual block4
	bn7(cudnnHandle, h1Desc, h7_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv8(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn8(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv9(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h9_dev);
	add(cudnnHandle, h1Desc, h7_dev, h9_dev);

	// residual block5
	bn9(cudnnHandle, h1Desc, h9_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv10(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn10(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv11(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h11_dev);
	add(cudnnHandle, h1Desc, h9_dev, h11_dev);

	// residual block6
	bn11(cudnnHandle, h1Desc, h11_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv12(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn12(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv13(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h13_dev);
	add(cudnnHandle, h1Desc, h11_dev, h13_dev);

	// residual block7
	bn13(cudnnHandle, h1Desc, h13_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv14(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn14(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv15(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h15_dev);
	add(cudnnHandle, h1Desc, h13_dev, h15_dev);

	// residual block8
	bn15(cudnnHandle, h1Desc, h15_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv16(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn16(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv17(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h17_dev);
	add(cudnnHandle, h1Desc, h15_dev, h17_dev);

	// residual block9
	bn17(cudnnHandle, h1Desc, h17_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv18(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn18(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv19(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h19_dev);
	add(cudnnHandle, h1Desc, h17_dev, h19_dev);

	// residual block10
	bn19(cudnnHandle, h1Desc, h19_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv20(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn20(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv21(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h21_dev);
	add(cudnnHandle, h1Desc, h19_dev, h21_dev);

	// residual block11
	bn21(cudnnHandle, h1Desc, h21_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv22(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn22(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv23(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h23_dev);
	add(cudnnHandle, h1Desc, h21_dev, h23_dev);

	// residual block12
	bn23(cudnnHandle, h1Desc, h23_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv24(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn24(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv25(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h25_dev);
	add(cudnnHandle, h1Desc, h23_dev, h25_dev);

	// residual block13
	bn25(cudnnHandle, h1Desc, h25_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv26(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn26(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv27(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h27_dev);
	add(cudnnHandle, h1Desc, h25_dev, h27_dev);

	// residual block14
	bn27(cudnnHandle, h1Desc, h27_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv28(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn28(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv29(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h29_dev);
	add(cudnnHandle, h1Desc, h27_dev, h29_dev);

	// residual block15
	bn29(cudnnHandle, h1Desc, h29_dev, h1_bn_dev);
	relu(cudnnHandle, h1Desc, h1_bn_dev);
	conv30(cudnnHandle, h1Desc, h1_bn_dev, h1Desc, h2_dev);
	bn30(cudnnHandle, h1Desc, h2_dev, h2_bn_dev);
	relu(cudnnHandle, h1Desc, h2_bn_dev);
	conv31(cudnnHandle, h1Desc, h2_bn_dev, h1Desc, h31_dev);
	add(cudnnHandle, h1Desc, h29_dev, h31_dev);

	// after residual blocks
	bn31(cudnnHandle, h1Desc, h31_dev, h31_bn_dev);
	relu(cudnnHandle, h1Desc, h31_bn_dev);

	// policy network
	conv32(cudnnHandle, h1Desc, h31_bn_dev, y1Desc, y1_dev);
	bias32(cudnnHandle, y1Desc, y1_dev);

	// value network
	conv32v(cudnnHandle, h1Desc, h31_bn_dev, h32vDesc, h32v_dev);
	bias32v(cudnnHandle, h32vDesc, h32v_dev);
	bn32v(cudnnHandle, h32vDesc, h32v_dev, h32v_bn_dev);
	relu(cudnnHandle, h32vDesc, h32v_bn_dev);

	l33v(cublasHandle, batch_size, h32v_bn_dev, h33v_dev);
	bias33v(cudnnHandle, h33vDesc, h33v_dev);
	relu(cudnnHandle, h33vDesc, h33v_dev);
	l34v(cublasHandle, batch_size, h33v_dev, y2_dev);
	bias34v(cudnnHandle, y2Desc, y2_dev);
	sigmoid(cudnnHandle, y2Desc, y2_dev);

	// output
	checkCudaErrors(cudaMemcpy(y1, y1_dev, conv32.get_ysize(batch_size, 9, 9), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(y2, y2_dev, batch_size * sizeof(DType), cudaMemcpyDeviceToHost));
}