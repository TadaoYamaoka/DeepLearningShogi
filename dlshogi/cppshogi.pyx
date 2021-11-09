from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np

import locale

cdef extern from "python_module.h" nogil:
	void init()
	void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue);
	void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux);
	size_t __load_hcpe3(const string& filepath, bool use_average, bool use_opponent, double a, double a_opponent, double temperature, int& len);
	void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue);
	vector[size_t] __load_evalfix(const string& filepath);
	void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult);
	void __hcpe3_prepare_evalfix_opponent(char* ndeval, char* ndresult);

init()

def hcpe_decode_with_value(np.ndarray ndhcpe, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue):
	__hcpe_decode_with_value(len(ndhcpe), ndhcpe.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data)

def hcpe2_decode_with_value(np.ndarray ndhcpe2, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue, np.ndarray ndaux):
	__hcpe2_decode_with_value(len(ndhcpe2), ndhcpe2.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data, ndaux.data)

def load_hcpe3(str filepath, bool use_average, bool use_opponent, double a, double a_opponent, double temperature):
	cdef int len = 0
	cdef size_t size = __load_hcpe3(filepath.encode(locale.getpreferredencoding()), use_average, use_opponent, a, a_opponent, temperature, len)
	return size, len

def hcpe3_decode_with_value(np.ndarray ndindex, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndprobability, np.ndarray ndresult, np.ndarray ndvalue):
	__hcpe3_decode_with_value(len(ndindex), ndindex.data, ndfeatures1.data, ndfeatures2.data, ndprobability.data, ndresult.data, ndvalue.data)

def hcpe3_prepare_evalfix(str filepath, bool use_opponent=False):
	size, size_opponent = __load_evalfix(filepath.encode(locale.getpreferredencoding()))
	cdef np.ndarray ndeval = np.empty(size, np.int32)
	cdef np.ndarray ndresult = np.empty(size, np.float32)
	__hcpe3_prepare_evalfix(ndeval.data, ndresult.data)
	cdef np.ndarray ndeval_opponent;
	cdef np.ndarray ndresult_opponent;
	if use_opponent:
		ndeval_opponent = np.empty(size_opponent, np.int32)
		ndresult_opponent = np.empty(size_opponent, np.float32)
		__hcpe3_prepare_evalfix_opponent(ndeval_opponent.data, ndresult_opponent.data)
		return ndeval, ndresult, ndeval_opponent, ndresult_opponent
	else:
		return ndeval, ndresult
