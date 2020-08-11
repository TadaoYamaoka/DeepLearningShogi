#pragma once

#include <string>
#include <map>

struct NPY
{
	//char magic_string[6]; // 6 bytes (0x93NUMPY)
	//unsigned char major_version; // 1 byte
	//unsigned char minor_version; // 1 byte
	//unsigned short header_len; // 2 bytes
	// ここまで10bytes
	unsigned char* uncompressed_data;
	float* data;

	NPY() : uncompressed_data(nullptr), data(nullptr) {}
	NPY(NPY&& o) : uncompressed_data(o.uncompressed_data), data(o.data) {
		o.uncompressed_data = nullptr;
		o.data = nullptr;
	}
	~NPY() {
		delete[] uncompressed_data;
	}
};

typedef std::map<const std::string, NPY> ParamMap;

void load_npz(const char* file, ParamMap& params);
