﻿#pragma once

#include <string>

void init();
void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue);
void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux);
size_t __load_hcpe3(const std::string& filepath, bool use_average, double a, double temperature, int& len);
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue);
size_t __load_evalfix(const std::string& filepath);
void __hcpe3_get_hcpe(const size_t index, char* ndhcpe);
void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult);
