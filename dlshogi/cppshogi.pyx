from libcpp.string cimport string
from libcpp cimport bool

import numpy as np
cimport numpy as np

import locale

cdef extern from "python_module.h" nogil:
	void init()
	void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue)
	void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux)
	void __hcpe3_create_cache(const string& filepath)
	size_t __hcpe3_load_cache(const string& filepath)
	size_t __hcpe3_get_cache_num()
	size_t __load_hcpe3(const string& filepath, bool use_average, double a, double temperature, size_t& len)
	size_t __hcpe3_patch_with_hcpe(const string& filepath, size_t& add_len)
	void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue)
	size_t __load_evalfix(const string& filepath)
	void __hcpe3_get_hcpe(const size_t index, char* ndhcpe)
	void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult)

init()

def hcpe_decode_with_value(np.ndarray ndhcpe, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue):
	__hcpe_decode_with_value(len(ndhcpe), ndhcpe.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data)

def hcpe2_decode_with_value(np.ndarray ndhcpe2, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue, np.ndarray ndaux):
	__hcpe2_decode_with_value(len(ndhcpe2), ndhcpe2.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data, ndaux.data)

def hcpe3_create_cache(str filepath):
	__hcpe3_create_cache(filepath.encode(locale.getpreferredencoding()))

def hcpe3_load_cache(str filepath):
	return __hcpe3_load_cache(filepath.encode(locale.getpreferredencoding()))

def hcpe3_get_cache_num():
	return __hcpe3_get_cache_num()

def load_hcpe3(str filepath, bool use_average, double a, double temperature):
	cdef size_t len = 0
	cdef size_t size = __load_hcpe3(filepath.encode(locale.getpreferredencoding()), use_average, a, temperature, len)
	return size, len

def hcpe3_patch_with_hcpe(str filepath):
	cdef size_t add_len = 0
	cdef size_t sum_len = __hcpe3_patch_with_hcpe(filepath.encode(locale.getpreferredencoding()), add_len)
	return sum_len, add_len

def hcpe3_decode_with_value(np.ndarray ndindex, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndprobability, np.ndarray ndresult, np.ndarray ndvalue):
	__hcpe3_decode_with_value(len(ndindex), ndindex.data, ndfeatures1.data, ndfeatures2.data, ndprobability.data, ndresult.data, ndvalue.data)

def hcpe3_get_hcpe(size_t index, np.ndarray ndhcpe):
	__hcpe3_get_hcpe(index, ndhcpe.data)

def hcpe3_prepare_evalfix(str filepath):
	cdef size_t size = __load_evalfix(filepath.encode(locale.getpreferredencoding()))
	cdef np.ndarray ndeval = np.empty(size, np.int32)
	cdef np.ndarray ndresult = np.empty(size, np.float32)
	__hcpe3_prepare_evalfix(ndeval.data, ndresult.data)
	return ndeval, ndresult
