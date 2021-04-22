#include <iostream>
#include <string>
#include "cxxopts/cxxopts.hpp"

#include "nn_tensorrt.h"

int main(int argc, char* argv[])
{
	std::string onnx_path;
	int batchsize;
	int gpu_id;
	std::string int8_calibration_hcpe;
	int int8_calibration_size;

	cxxopts::Options options("build_onnx");
	options.positional_help("build_onnx onnxfile");
	try {
		options.add_options()
			("onnxfile", "onnx file path", cxxopts::value<std::string>(onnx_path))
			("b,batchsize", "batch size", cxxopts::value<int>(batchsize)->default_value("128"))
			("g,gpu_id", "gpu id", cxxopts::value<int>(gpu_id)->default_value("0"))
			("calib_hcpe", "int8 calibration hcpe", cxxopts::value<std::string>(int8_calibration_hcpe)->default_value(""))
			("calib_size", "int calibration data size", cxxopts::value<int>(int8_calibration_size)->default_value("4096"))
			("h,help", "Print help")
			;
		options.parse_positional({ "onnxfile" });

		auto result = options.parse(argc, argv);

		if (result.count("help")) {
			std::cout << options.help({}) << std::endl;
			return 0;
		}
	}
	catch (cxxopts::OptionException& e) {
		std::cout << options.usage() << std::endl;
		std::cerr << e.what() << std::endl;
		return 1;
	}

	if (int8_calibration_hcpe != "") {
		initTable();
		HuffmanCodedPos::init();
	}

	try {
		cudaSetDevice(gpu_id);
		NNTensorRT nn(onnx_path.c_str(), gpu_id, batchsize, int8_calibration_hcpe == "" ? nullptr : int8_calibration_hcpe.c_str(), int8_calibration_size);
	}
	catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}