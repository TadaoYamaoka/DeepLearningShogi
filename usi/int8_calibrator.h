#pragma once

#include "cppshogi.h"
#include "error_util.h"
#include "NvInfer.h"

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
	Int8EntropyCalibrator2(const char* model_filename, const int batch_size, const char* hcpe_filename, const size_t int8_calibration_data_size) : batch_size(batch_size), int8_calibration_data_size(int8_calibration_data_size), features1(new features1_t[batch_size]), features2(new features2_t[batch_size]) {
		calibration_cache_filename = std::string(model_filename) + ".calibcache";
		checkCudaErrors(cudaMalloc(&input1, sizeof(features1_t) * batch_size));
		checkCudaErrors(cudaMalloc(&input2, sizeof(features2_t) * batch_size));

		// キャリブレーションキャッシュがあるかチェック
		std::ifstream calibcache(calibration_cache_filename);
		if (!calibcache.is_open())
		{
			if (hcpe_filename != nullptr)
			{
				// hcpeからランダムに選ぶ
				std::ifstream ifs;
				ifs.open(hcpe_filename, std::ifstream::in | std::ifstream::binary | std::ios::ate);
				const auto entryNum = ifs.tellg() / sizeof(HuffmanCodedPosAndEval);
				std::uniform_int_distribution<s64> inputFileDist(0, entryNum - 1);
				std::mt19937_64 mt_64(std::chrono::system_clock::now().time_since_epoch().count());
				HuffmanCodedPosAndEval hcpe;
				for (size_t i = 0; i < int8_calibration_data_size; ++i)
				{
					ifs.seekg(inputFileDist(mt_64) * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
					ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
					int8_calibration_data.emplace_back(hcpe.hcp);
				}
			}
			else
			{
				// キャリブレーションキャッシュがない場合エラー
				std::cerr << "missing calibration cache" << std::endl;
				throw std::runtime_error("missing calibration cache");
			}
		}
	}
	Int8EntropyCalibrator2(const char* model_filename, const int batch_size) : Int8EntropyCalibrator2(model_filename, batch_size, "", 0) {}

	~Int8EntropyCalibrator2() {
		checkCudaErrors(cudaFree(input1));
		checkCudaErrors(cudaFree(input2));
	}

	int getBatchSize() const override {
		return batch_size;
	}

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
		assert(nbBindings == 2);
		if (current_pos > int8_calibration_data_size - batch_size)
		{
			return false;
		}

		std::fill_n((float*)features1.get(), sizeof(features1_t) / sizeof(float) * batch_size, 0);
		std::fill_n((float*)features2.get(), sizeof(features2_t) / sizeof(float) * batch_size, 0);

		for (int i = 0; i < batch_size; ++i, ++current_pos)
		{
			pos.set(int8_calibration_data[current_pos]);
			make_input_features(pos, features1.get() + i, features2.get() + i);
		}

		checkCudaErrors(cudaMemcpy(input1, features1.get(), sizeof(features1_t) * batch_size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(input2, features2.get(), sizeof(features2_t) * batch_size, cudaMemcpyHostToDevice));
		bindings[0] = input1;
		bindings[1] = input2;

		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		calibration_cache.clear();
		std::ifstream input(calibration_cache_filename, std::ios::binary);
		input >> std::noskipws;
		if (input.good())
		{
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
				std::back_inserter(calibration_cache));
		}
		length = calibration_cache.size();
		return length ? calibration_cache.data() : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibration_cache_filename, std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
	std::vector<HuffmanCodedPos> int8_calibration_data;
	Position pos;
	const int batch_size;
	size_t current_pos = 0;
	std::unique_ptr<features1_t[]> features1;
	std::unique_ptr<features2_t[]> features2;
	void* input1;
	void* input2;
	std::string calibration_cache_filename;
	size_t int8_calibration_data_size;
	std::vector<char> calibration_cache;
};
