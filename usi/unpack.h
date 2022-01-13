#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cppshogi.h"

constexpr int PACKED_LEGAL_MOVE_NUM = (MAX_MOVE_LABEL_NUM * SquareNum + 31) / 32;
typedef unsigned int packed_legal_moves_t[PACKED_LEGAL_MOVE_NUM];
typedef short move_label_index_t[MAX_MOVE_LABEL_NUM * SquareNum];

void unpack_features1(const int batch_size, packed_features1_t* p1, features1_t* x1, cudaStream_t stream);
void unpack_features2(const int batch_size, packed_features2_t* p2, features2_t* x2, cudaStream_t stream);
void gather_legal_moves(const int batch_size, packed_legal_moves_t* packed_legal_moves, DType* policies, DType* outputs, cudaStream_t stream);
