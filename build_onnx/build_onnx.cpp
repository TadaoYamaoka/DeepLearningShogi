#include <iostream>
#include <string>
#include "cxxopts/cxxopts.hpp"

#include "nn_tensorrt.h"

int main(int argc, char* argv[])
{
	std::string onnx_path;
	int batchsize;
	int gpu_id;

	cxxopts::Options options("build_onnx");
	options.positional_help("build_onnx onnxfile");
	try {
		options.add_options()
			("onnxfile", "onnx file path", cxxopts::value<std::string>(onnx_path))
			("b,batchsize", "batch size", cxxopts::value<int>(batchsize)->default_value("128"))
			("g,gpu_id", "gpu id", cxxopts::value<int>(gpu_id)->default_value("0"))
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

	try {
		NNTensorRT nn(onnx_path.c_str(), gpu_id, batchsize);
	}
	catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}