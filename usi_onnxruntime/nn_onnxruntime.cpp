#ifdef ONNXRUNTIME
#include "nn_onnxruntime.h"
#ifndef ONNXRUNTIME_CPU
#include <dml_provider_factory.h>
#endif

#ifdef _WIN32
#include <clocale>
#include <cstdlib>
#endif

NNOnnxRuntime::NNOnnxRuntime(const char* filename, const int gpu_id, const int max_batch_size)
{
	Ort::SessionOptions session_options;
#ifndef ONNXRUNTIME_CPU
	session_options.DisableMemPattern();
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, gpu_id));
#endif

#ifdef _WIN32
	std::setlocale(LC_ALL, ".OCP");
	wchar_t tmpstr[2048];
	std::mbstowcs(tmpstr, filename, sizeof(tmpstr) / sizeof(wchar_t));
	std::wstring onnx_filename = tmpstr;
#else
	std::string onnx_filename(filename);
#endif

	session.reset(new Ort::Session(env, onnx_filename.c_str(), session_options));
	const auto shape1 = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	const auto shape2 = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	if (shape1.size() != 4 || shape2.size() != 4 || shape1[1] != (int)ColorNum * MAX_FEATURES1_NUM ||
		shape2[1] != MAX_FEATURES2_NUM || shape1[2] != 9 || shape1[3] != 9 || shape2[2] != 9 || shape2[3] != 9)
		throw std::runtime_error("Model input shape does not match dlshogi features (NYUGYOKU_FEATURES)");
	const auto policy_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	const auto value_shape = session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
	if (policy_shape.size() != 2 || policy_shape[1] != MAX_MOVE_LABEL_NUM * (int)SquareNum ||
		!((value_shape.size() == 2 && value_shape[1] == 1) || value_shape.size() == 1))
		throw std::runtime_error("Model must output dlshogi policy labels and one scalar value per position");
}

void NNOnnxRuntime::forward(const int batch_size, features1_t* x1, features2_t* x2, DType* y1, DType* y2)
{
	// input
	std::array<int64_t, 4> input_shape1{ batch_size, (size_t)ColorNum * MAX_FEATURES1_NUM, 9, 9 };
	std::array<int64_t, 4> input_shape2{ batch_size, MAX_FEATURES2_NUM, 9, 9 };

	std::array<Ort::Value, 2> input_values{
		Ort::Value::CreateTensor<float>(memory_info, (float*)x1, batch_size * sizeof(features1_t) / sizeof(float), input_shape1.data(), input_shape1.size()),
		Ort::Value::CreateTensor<float>(memory_info, (float*)x2, batch_size * sizeof(features2_t) / sizeof(float), input_shape2.data(), input_shape2.size())
	};

	// output
	std::array<int64_t, 2> output_shape1{ batch_size, MAX_MOVE_LABEL_NUM * (size_t)SquareNum };
	std::array<int64_t, 2> output_shape2{ batch_size, 1 };

	std::array<Ort::Value, 2> output_values{
		Ort::Value::CreateTensor<float>(memory_info, y1, batch_size * MAX_MOVE_LABEL_NUM * (size_t)SquareNum, output_shape1.data(), output_shape1.size()),
		Ort::Value::CreateTensor<float>(memory_info, y2, batch_size, output_shape2.data(), output_shape2.size())
	};

	// names
	const char* input_names[] = { "input1", "input2" };
	const char* output_names[] = { "output_policy", "output_value" };

	// run
	session->Run(Ort::RunOptions{ nullptr }, input_names, input_values.data(), input_values.size(), output_names, output_values.data(), output_values.size());
}

#endif
