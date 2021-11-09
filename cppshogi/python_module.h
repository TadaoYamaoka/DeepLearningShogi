#pragma once

#include <string>
#include <vector>

void init();
void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue);
void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux);
size_t __load_hcpe3(const std::string& filepath, bool use_average, bool use_opponent, double a, double a_opponent, double temperature, int& len);
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue);
std::vector<size_t> __load_evalfix(const std::string& filepath);
void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult);
void __hcpe3_prepare_evalfix_opponent(char* ndeval, char* ndresult);
