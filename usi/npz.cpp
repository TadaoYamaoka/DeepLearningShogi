#include "npz.h"

#include <zlib.h>
#include <fstream>

using namespace std;

// https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
struct LocalFileHeader
{
	uint32_t local_file_header_signature; // 4_bytes (0x04034b50)
	uint16_t version_needed_to_extract; // 2_bytes
	uint16_t general_purpose_bit_flag; // 2_bytes
	uint16_t compression_method; // 2_bytes
	uint16_t last_mod_file_time; // 2_bytes
	uint16_t last_mod_file_date; // 2_bytes
	uint32_t crc_32; // 4_bytes
	uint32_t compressed_size; // 4_bytes
	uint32_t uncompressed_size; // 4_bytes
	uint16_t file_name_length; // 2_bytes
	uint16_t extra_field_length; // 2_bytes
									   // ここまで30bytes

									   //char* file_name; // (variable_size)
									   //char* extra_field; // (variable_size)
};

ifstream& operator >> (ifstream& ifs, LocalFileHeader& lfh) {
	ifs.read((char*)&lfh.local_file_header_signature, sizeof(lfh.local_file_header_signature));
	ifs.read((char*)&lfh.version_needed_to_extract, sizeof(lfh.version_needed_to_extract));
	ifs.read((char*)&lfh.general_purpose_bit_flag, sizeof(lfh.general_purpose_bit_flag));
	ifs.read((char*)&lfh.compression_method, sizeof(lfh.compression_method));
	ifs.read((char*)&lfh.last_mod_file_time, sizeof(lfh.last_mod_file_time));
	ifs.read((char*)&lfh.last_mod_file_date, sizeof(lfh.last_mod_file_date));
	ifs.read((char*)&lfh.crc_32, sizeof(lfh.crc_32));
	ifs.read((char*)&lfh.compressed_size, sizeof(lfh.compressed_size));
	ifs.read((char*)&lfh.uncompressed_size, sizeof(lfh.uncompressed_size));
	ifs.read((char*)&lfh.file_name_length, sizeof(lfh.file_name_length));
	ifs.read((char*)&lfh.extra_field_length, sizeof(lfh.extra_field_length));
	return ifs;
}

void load_npz(const char* file, ParamMap& params)
{
	ifstream infile(file, ios_base::in | ios_base::binary);
	if (!infile)
		return;

	while (true)
	{
		// Local file header
		LocalFileHeader lfh;
		infile >> lfh;

		if (lfh.local_file_header_signature != 0x04034b50)
		{
			break;
		}

		char* file_name = new char[lfh.file_name_length + 1];

		infile.read(file_name, lfh.file_name_length);
		file_name[lfh.file_name_length] = '\0';

		infile.seekg(lfh.extra_field_length, ios_base::cur);

		// File data
		unsigned char* file_data = new unsigned char[lfh.compressed_size];
		infile.read((char*)file_data, lfh.compressed_size);

		NPY npy;
		npy.uncompressed_data = new unsigned char[lfh.uncompressed_size];

		z_stream strm = {};
		inflateInit2(&strm, -MAX_WBITS);

		strm.next_in = file_data;
		strm.avail_in = lfh.compressed_size;
		strm.next_out = npy.uncompressed_data;
		strm.avail_out = lfh.uncompressed_size;
		inflate(&strm, Z_NO_FLUSH);
		inflateEnd(&strm);

		// NPY
		const uint16_t header_len = *(uint16_t*)(npy.uncompressed_data + 8);
		npy.data = (float*)(npy.uncompressed_data + 10 + header_len);

		params.emplace(file_name, std::move(npy));

		delete[] file_data;
		delete[] file_name;
	}
}