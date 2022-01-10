#include "unpack.h"

constexpr int features1_size = sizeof(features1_t) / sizeof(DType) / SquareNum;
constexpr int features2_size = sizeof(features2_t) / sizeof(DType) / SquareNum;

__global__ void unpack_features1_kernel(char* p1, short* x1) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int p1_offset = sizeof(packed_features1_t) * 8 * blockIdx.x + threadIdx.x * 81;
	int x1_offset = tid * 81;
#pragma unroll
	for (int i = 0; i < 81; ++i) {
		int j = p1_offset + i;
		// p1[j / 8] >> (j % 8)�ŉ���1bit�ɐݒ肷��l�������Ă���
		// ����1bit�̃}�X�N���s���A�����𕉂ɂ��邱�Ƃ�1�̏ꍇ1byte�̑Sbit��1�ɂ���
		// 0x3c00�Ƙ_���ς���邱�Ƃ�float16��1.0�ɂ���
		x1[x1_offset + i] = (-(short)((p1[j >> 3] >> (j & 7)) & 1)) & 0x3c00;
	}
}

__global__ void unpack_features2_kernel(char* p2, short* x2) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int j = sizeof(packed_features2_t) * 8 * blockIdx.x + threadIdx.x;
	short v = (-(short)((p2[j >> 3] >> (j & 7)) & 1)) & 0x3c00;

	int x2_offset = tid * 81;
#pragma unroll
	for (int i = 0; i < 81; ++i) {
		x2[x2_offset + i] = v;
	}
}

void unpack_features1(const int batch_size, packed_features1_t* p1, features1_t* x1, cudaStream_t stream)
{
	unpack_features1_kernel<<<batch_size, features1_size, 0, stream>>>((char*)p1, (short*)x1);
}

void unpack_features2(const int batch_size, packed_features2_t* p2, features2_t* x2, cudaStream_t stream)
{
	unpack_features2_kernel<<<batch_size, features2_size, 0, stream>>> ((char*)p2, (short*)x2);
}
