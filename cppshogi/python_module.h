#pragma once

#include <string>

void init();
void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue);
void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux);
void __hcpe3_create_cache(const std::string& filepath);
size_t __hcpe3_load_cache(const std::string& filepath);
size_t __hcpe3_get_cache_num();
size_t __load_hcpe3(const std::string& filepath, bool use_average, double a, double temperature, size_t& len);
size_t __hcpe3_patch_with_hcpe(const std::string& filepath, size_t& add_len);
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue);
size_t __load_evalfix(const std::string& filepath);
void __hcpe3_get_hcpe(const size_t index, char* ndhcpe);
void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult);
void __hcpe3_merge_cache(const std::string& file1, const std::string& file2, const std::string& out);
std::pair<int, int> __hcpe3_to_hcpe(const std::string& file1, const std::string& file2);
