/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "sf_evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "sf_position.h"
#include "sf_types.h"
#include "sf_usi.h"
#include "sf_memory.h"

#include "bitboard.hpp"

#include "cppshogi.h"

namespace Stockfish {

namespace Eval {

// パラメータ情報を保持する構造体
struct Parameter {
    std::string name;
    std::vector<uint32_t> shape;  // 各次元のサイズ
    std::unique_ptr<float[], decltype(&std_aligned_free)> data;

    // コンストラクタ
    Parameter(const std::string& name, const std::vector<uint32_t>& shape, float* data)
        : name(name), shape(shape), data(data, std_aligned_free) { }
};

// パラメータ
std::vector<Parameter> parameters;

bool read_parameters(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    // 全パラメータ数 (unsigned int) を読み込む
    uint32_t num_params = 0;
    fin.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    std::cout << "Number of parameters: " << num_params << std::endl;

    parameters.clear();
    parameters.reserve(num_params);

    // 各パラメータについて読み込みを行う
    for (uint32_t i = 0; i < num_params; ++i) {
        // (a) パラメータ名のバイト長を読み込む
        uint32_t key_length = 0;
        fin.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        if (!fin) {
            std::cerr << "Error reading data" << std::endl;
            return false;
        }

        // (b) パラメータ名（UTF-8）の文字列を読み込む
        std::string name(key_length, ' ');
        fin.read(&name[0], key_length);
        if (!fin) {
            std::cerr << "Error reading data" << std::endl;
            return false;
        }

        // (c) テンソルの次元数を読み込む
        uint32_t ndim = 0;
        fin.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<uint32_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            fin.read(reinterpret_cast<char*>(&shape[d]), sizeof(uint32_t));
            if (!fin) {
                std::cerr << "Error reading data" << std::endl;
                return false;
            }
        }

        // (d) テンソルの要素数 (numel) を読み込む
        uint32_t numel = 0;
        fin.read(reinterpret_cast<char*>(&numel), sizeof(numel));
        if (!fin) {
            std::cerr << "Error reading data" << std::endl;
            return false;
        }

#ifndef NDEBUG
        {
            uint32_t prod = 1;
            for (uint32_t d : shape) {
                prod *= d;
            }
            assert(prod == numel && "Shape product does not match numel");
        }
#endif

        // (e) テンソルデータ（float32 の連続バイト列）を読み込む
        float* data = reinterpret_cast<float*>(std_aligned_alloc(32, numel * sizeof(float)));
        if (!data) {
            std::cerr << "Error: Memory allocation failed for parameter " << name << std::endl;
            return false;
        }
        fin.read(reinterpret_cast<char*>(data), numel * sizeof(float));
        if (!fin) {
            std::cerr << "Error reading data" << std::endl;
            return false;
        }

        // 32バイト整列を確認する assert
        assert(reinterpret_cast<std::uintptr_t>(data) % 32 == 0 && "Parameter data is not 32-byte aligned");

        // 読み込んだ内容を Parameter 構造体に格納して登録
        parameters.emplace_back(name, shape, data);

        // ログ出力：パラメータ名、shape、要素数
        std::cout << "Loaded parameter: " << name << " | shape: (";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j] << (j + 1 < shape.size() ? ", " : "");
        }
        std::cout << ") | numel: " << numel << std::endl;
    }

    fin.close();

    return true;
}

template <Color turn, int EmbeddingDim>
void embedding_layers(
    const Position& position,
    const float* __restrict embedding_table1,
    const float* __restrict embedding_table2,
    float* __restrict output   // 出力サイズ：EmbeddingDim * SquareNum
) {
    static_assert(EmbeddingDim % 8 == 0, "EmbeddingDim must be a multiple of 8");
    // 32バイト整列チェック
    assert(reinterpret_cast<uintptr_t>(embedding_table1) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(embedding_table2) % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(output) % 32 == 0);

    // 出力バッファを0で初期化（全チャネル・全マス）
    const __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < EmbeddingDim * SquareNum; i += 8) {
        _mm256_store_ps(output + i, zero);
    }

    // 持ち駒・王手フラグ用の一時バッファ (アライメント済み)
    alignas(32) float hand_embed[EmbeddingDim];
    for (int i = 0; i < EmbeddingDim; i += 8) {
        _mm256_store_ps(hand_embed + i, zero);
    }

    // --- 盤面上の駒の埋め込み (l1_1の処理) ---
    Bitboard occupied_bb = position.occupiedBB();

    // FOREACH_BB マクロ内で各盤面上の駒について処理
    FOREACH_BB(occupied_bb, Square sq, {
        const Piece pc = position.piece(sq);
        const PieceType pt = pieceToPieceType(pc);
        Color c = pieceToColor(pc);

        // 後手の場合、色を反転し、盤面を180度回転
        if (turn == White) {
            c = oppositeColor(c);
            sq = SQ99 - sq;
        }

        // 駒配置に対応するインデックス (例: idx = PIECETYPE_NUM * (int)c + pt - 1)
        const int idx = PIECETYPE_NUM * (int)c + pt - 1;

        // embedding_table1 の該当行から埋め込みベクトルを取得し、出力バッファに「転置して」格納する。
        // 具体的には、各チャネル j について出力のインデックスは [j * SquareNum + sq] となる。
        const float* embed_vec = &embedding_table1[idx * EmbeddingDim];
        float* out_ptr = output + sq;
        if constexpr (EmbeddingDim == 16) {
            // EmbeddingDim==16 の場合、アンローリングして各チャネルを個別に格納
            out_ptr[0 * SquareNum] = embed_vec[0];
            out_ptr[1 * SquareNum] = embed_vec[1];
            out_ptr[2 * SquareNum] = embed_vec[2];
            out_ptr[3 * SquareNum] = embed_vec[3];
            out_ptr[4 * SquareNum] = embed_vec[4];
            out_ptr[5 * SquareNum] = embed_vec[5];
            out_ptr[6 * SquareNum] = embed_vec[6];
            out_ptr[7 * SquareNum] = embed_vec[7];
            out_ptr[8 * SquareNum] = embed_vec[8];
            out_ptr[9 * SquareNum] = embed_vec[9];
            out_ptr[10 * SquareNum] = embed_vec[10];
            out_ptr[11 * SquareNum] = embed_vec[11];
            out_ptr[12 * SquareNum] = embed_vec[12];
            out_ptr[13 * SquareNum] = embed_vec[13];
            out_ptr[14 * SquareNum] = embed_vec[14];
            out_ptr[15 * SquareNum] = embed_vec[15];
        }
        else {
            for (int j = 0; j < EmbeddingDim; j++) {
                out_ptr[j * SquareNum] = embed_vec[j];
            }
        }
    });

    // --- 持ち駒の埋め込み (l1_2の処理) ---
    // 各色ごとに、保持している駒の埋め込みベクトルを hand_embed に加算する。
    for (Color c = Black; c < ColorNum; ++c) {
        // 後手の場合、色反転
        const Color c2 = (turn == Black) ? c : oppositeColor(c);

        // 持ち駒情報
        const Hand hand = position.hand(c);
        int p = 0;
        for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
            const u32 num = std::min(hand.numOf(hp), MAX_PIECES_IN_HAND[hp]);
            const int base_idx = MAX_PIECES_IN_HAND_SUM * static_cast<int>(c2) + p;

            for (u32 i = 0; i < num; ++i) {
                const int idx = base_idx + i;
                const float* embed_vec = &embedding_table2[idx * EmbeddingDim];

                if constexpr (EmbeddingDim == 16) {
                    // AVX2で8要素ずつ加算
                    __m256 curr1 = _mm256_load_ps(hand_embed);
                    __m256 curr2 = _mm256_load_ps(hand_embed + 8);
                    __m256 add1 = _mm256_load_ps(embed_vec);
                    __m256 add2 = _mm256_load_ps(embed_vec + 8);

                    curr1 = _mm256_add_ps(curr1, add1);
                    curr2 = _mm256_add_ps(curr2, add2);

                    _mm256_store_ps(hand_embed, curr1);
                    _mm256_store_ps(hand_embed + 8, curr2);
                }
                else {
                    for (int j = 0; j < EmbeddingDim; j += 8) {
                        __m256 curr = _mm256_load_ps(hand_embed + j);
                        __m256 add = _mm256_load_ps(embed_vec + j);
                        _mm256_store_ps(hand_embed + j, _mm256_add_ps(curr, add));
                    }
                }
            }
            p += MAX_PIECES_IN_HAND[hp];
        }
    }

    // 王手フラグの埋め込みを加算
    if (position.inCheck()) {
        const int idx = MAX_FEATURES2_HAND_NUM;
        const float* check_embed_vec = &embedding_table2[idx * EmbeddingDim];

        if constexpr (EmbeddingDim == 16) {
            __m256 curr1 = _mm256_load_ps(hand_embed);
            __m256 curr2 = _mm256_load_ps(hand_embed + 8);
            __m256 add1 = _mm256_load_ps(check_embed_vec);
            __m256 add2 = _mm256_load_ps(check_embed_vec + 8);
            _mm256_store_ps(hand_embed, _mm256_add_ps(curr1, add1));
            _mm256_store_ps(hand_embed + 8, _mm256_add_ps(curr2, add2));
        }
        else {
            for (int j = 0; j < EmbeddingDim; j += 8) {
                __m256 curr = _mm256_load_ps(hand_embed + j);
                __m256 add = _mm256_load_ps(check_embed_vec + j);
                _mm256_store_ps(hand_embed + j, _mm256_add_ps(curr, add));
            }
        }
    }

    // --- 持ち駒・王手フラグの埋め込みを盤面全体にブロードキャストして加算 ---
    // 出力バッファはチャネル優先のレイアウトになっているため、各チャネルごとにhand_embedの値を各マスに加算する。
    for (int c = 0; c < EmbeddingDim; c++) {
        float* out_channel = &output[c * SquareNum];
        __m256 hand_vec = _mm256_set1_ps(hand_embed[c]);

        // 32要素ずつ処理（4x8=32）
        for (int s = 0; s < 64; s += 32) {
            __m256 data1 = _mm256_loadu_ps(out_channel + s);
            __m256 data2 = _mm256_loadu_ps(out_channel + s + 8);
            __m256 data3 = _mm256_loadu_ps(out_channel + s + 16);
            __m256 data4 = _mm256_loadu_ps(out_channel + s + 24);

            data1 = _mm256_add_ps(data1, hand_vec);
            data2 = _mm256_add_ps(data2, hand_vec);
            data3 = _mm256_add_ps(data3, hand_vec);
            data4 = _mm256_add_ps(data4, hand_vec);

            _mm256_storeu_ps(out_channel + s, data1);
            _mm256_storeu_ps(out_channel + s + 8, data2);
            _mm256_storeu_ps(out_channel + s + 16, data3);
            _mm256_storeu_ps(out_channel + s + 24, data4);
        }

        // 残り17要素（64+16=80、残り1要素）
        __m256 data5 = _mm256_loadu_ps(out_channel + 64);
        __m256 data6 = _mm256_loadu_ps(out_channel + 72);
        data5 = _mm256_add_ps(data5, hand_vec);
        data6 = _mm256_add_ps(data6, hand_vec);
        _mm256_storeu_ps(out_channel + 64, data5);
        _mm256_storeu_ps(out_channel + 72, data6);

        // 最後の1要素
        out_channel[80] += hand_embed[c];
    }
}

template <int EmbeddingDim>
inline void embedding_layers(
    const Position& position,
    const float* __restrict embedding_table1,
    const float* __restrict embedding_table2,
    float* __restrict output
) {
    if (position.turn() == Black) {
        embedding_layers<Black, EmbeddingDim>(position, embedding_table1, embedding_table2, output);
    }
    else {
        embedding_layers<White, EmbeddingDim>(position, embedding_table1, embedding_table2, output);
    }
}

inline void embedding_layers(
    const Position& position,
    const float* __restrict embedding_table1,
    const float* __restrict embedding_table2,
    float* __restrict output
) {
    embedding_layers<EMBEDDING_DIM>(position, embedding_table1, embedding_table2, output);
}

// 9x9入力→8x8出力の2×2 depthwise畳み込み（バイアス付き、BN融合済み、ReLU付き）のAVX2実装
// 入力、重み、出力は32バイト整列済みであると仮定
// テンプレートパラメータ Channels は畳み込み対象のチャネル数
template <int Channels>
void conv2x2_depthwise_relu(
    const float* __restrict input,
    const float* __restrict weights,
    const float* __restrict bias,
    float* __restrict output)
{
    constexpr int INPUT_HEIGHT = 9;
    constexpr int INPUT_WIDTH = 9;
    constexpr int OUTPUT_HEIGHT = 8;
    constexpr int OUTPUT_WIDTH = 8;
    constexpr int INPUT_CHANNEL_SIZE = INPUT_HEIGHT * INPUT_WIDTH;
    constexpr int OUTPUT_CHANNEL_SIZE = OUTPUT_HEIGHT * OUTPUT_WIDTH;
    constexpr int CACHE_LINE_SIZE = 64; // バイト単位

    const __m256 zero = _mm256_setzero_ps();

    // Channels==16向けに8チャネルずつ処理（AVX2で8個まとめて処理）
    if constexpr (Channels == 16) {
        for (int c_group = 0; c_group < Channels; c_group += 8) {
            // 各チャネルごとの2×2の重みとバイアスをAVX2レジスタにロード
            __m256 w_avx[4][8];  // [重みのインデックス][チャネルグループ内インデックス]
            __m256 bias_avx[8];
            const float* in_ptr[8];
            float* out_ptr[8];

            // 重みとバイアスをレジスタに先行ロード（レジスタの再利用を改善）
            for (int i = 0; i < 8; ++i) {
                int c = c_group + i;
                in_ptr[i] = input + c * INPUT_CHANNEL_SIZE;
                out_ptr[i] = output + c * OUTPUT_CHANNEL_SIZE;

                // チャネルごとの重みをロード - メモリアクセスパターンを最適化
                const float* w_c = weights + c * 4;
                for (int w_idx = 0; w_idx < 4; ++w_idx) {
                    w_avx[w_idx][i] = _mm256_set1_ps(w_c[w_idx]);
                }
                bias_avx[i] = _mm256_set1_ps(bias[c]);
            }

            // すべての入力チャネルに対してプリフェッチ
            for (int i = 0; i < 8; ++i) {
                for (int r = 0; r < INPUT_HEIGHT; r += CACHE_LINE_SIZE / sizeof(float) / INPUT_WIDTH) {
                    _mm_prefetch(reinterpret_cast<const char*>(in_ptr[i] + r * INPUT_WIDTH), _MM_HINT_T0);
                }
            }

            // 出力は8x8なので、行を4行単位でループアンロール
            for (int r = 0; r < OUTPUT_HEIGHT; r += 4) {
                int rows = std::min(4, OUTPUT_HEIGHT - r);

                for (int ch = 0; ch < 8; ++ch) {
                    for (int rr = 0; rr < rows; ++rr) {
                        int curr_row = r + rr;

                        // 入力の該当行（上段と下段）のポインタ
                        const float* row_top = in_ptr[ch] + curr_row * INPUT_WIDTH;
                        const float* row_bottom = row_top + INPUT_WIDTH;
                        float* out_row = out_ptr[ch] + curr_row * OUTPUT_WIDTH;

                        // 入力データをロード（アライメントされている場合はload_psを使用）
                        __m256 top_left = _mm256_load_ps(row_top);
                        __m256 top_right = _mm256_load_ps(row_top + 1);
                        __m256 bottom_left = _mm256_load_ps(row_bottom);
                        __m256 bottom_right = _mm256_load_ps(row_bottom + 1);

                        // FMA命令で各位置の乗算・加算（重みのアクセスパターンを改善）
                        __m256 sum = _mm256_fmadd_ps(top_left, w_avx[0][ch], bias_avx[ch]);
                        sum = _mm256_fmadd_ps(top_right, w_avx[1][ch], sum);
                        sum = _mm256_fmadd_ps(bottom_left, w_avx[2][ch], sum);
                        sum = _mm256_fmadd_ps(bottom_right, w_avx[3][ch], sum);

                        // ReLU活性化関数を適用
                        sum = _mm256_max_ps(sum, zero);

                        // 結果を32バイト整列の出力バッファに保存
                        _mm256_store_ps(out_row, sum);
                    }
                }
            }
        }
    }
    // 一般のチャネル数向け（4チャネルずつ処理）
    else {
        for (int c = 0; c < Channels; c += 4) {
            int ch_block = std::min(4, Channels - c);
            __m256 weights_block[4][4];  // [重みのインデックス][チャネルブロック内インデックス]
            __m256 bias_block[4];
            const float* in_ptrs[4];
            float* out_ptrs[4];

            // 重みとバイアスをレジスタに先行ロード
            for (int i = 0; i < ch_block; ++i) {
                int channel = c + i;
                in_ptrs[i] = input + channel * INPUT_CHANNEL_SIZE;
                out_ptrs[i] = output + channel * OUTPUT_CHANNEL_SIZE;

                const float* w_c = weights + channel * 4;
                for (int w_idx = 0; w_idx < 4; ++w_idx) {
                    weights_block[w_idx][i] = _mm256_set1_ps(w_c[w_idx]);
                }
                bias_block[i] = _mm256_set1_ps(bias[channel]);
            }

            // プリフェッチの最適化 - 全行をプリフェッチ
            for (int i = 0; i < ch_block; ++i) {
                for (int r = 0; r < INPUT_HEIGHT; r += CACHE_LINE_SIZE / sizeof(float) / INPUT_WIDTH) {
                    _mm_prefetch(reinterpret_cast<const char*>(in_ptrs[i] + r * INPUT_WIDTH), _MM_HINT_T0);
                }
            }

            // 2行ずつ処理
            for (int r = 0; r < OUTPUT_HEIGHT; r += 2) {
                int rows = std::min(2, OUTPUT_HEIGHT - r);

                for (int ch_idx = 0; ch_idx < ch_block; ++ch_idx) {
                    for (int rr = 0; rr < rows; ++rr) {
                        int curr_row = r + rr;
                        const float* row_top = in_ptrs[ch_idx] + curr_row * INPUT_WIDTH;
                        const float* row_bottom = row_top + INPUT_WIDTH;
                        float* out_row = out_ptrs[ch_idx] + curr_row * OUTPUT_WIDTH;

                        // 32バイト整列を前提としたロード命令に変更
                        __m256 top_left = _mm256_load_ps(row_top);
                        __m256 top_right = _mm256_load_ps(row_top + 1);
                        __m256 bottom_left = _mm256_load_ps(row_bottom);
                        __m256 bottom_right = _mm256_load_ps(row_bottom + 1);

                        // インデックス順序を最適化
                        __m256 sum = _mm256_fmadd_ps(top_left, weights_block[0][ch_idx], bias_block[ch_idx]);
                        sum = _mm256_fmadd_ps(top_right, weights_block[1][ch_idx], sum);
                        sum = _mm256_fmadd_ps(bottom_left, weights_block[2][ch_idx], sum);
                        sum = _mm256_fmadd_ps(bottom_right, weights_block[3][ch_idx], sum);

                        // ReLU活性化関数を適用
                        sum = _mm256_max_ps(sum, zero);

                        _mm256_store_ps(out_row, sum);
                    }
                }
            }
        }
    }
}

// 水平方向の合計計算（AVX2）
inline float horizontal_sum_avx2(__m256 vec) {
    __m256 t1 = _mm256_hadd_ps(vec, vec);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 t3 = _mm256_extractf128_ps(t2, 1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
    return _mm_cvtss_f32(t4);
}

// 8x1の畳み込み処理
template <int InChannels, int OutChannels>
void conv8x1_avx2(const float* __restrict input,
    const float* __restrict weights,
    const float* __restrict bias,
    float* __restrict output) {
    constexpr int KERNEL_SIZE = 8;
    constexpr int inner_size = InChannels * KERNEL_SIZE;
    constexpr int padded_inner_size = ((inner_size + 7) / 8) * 8;

    for (int w = 0; w < 8; ++w) {
        alignas(32) float patch[inner_size];
        for (int ic = 0; ic < InChannels; ++ic) {
            const float* in_channel = input + ic * 64;  // 8x8 = 64
            int base = ic * KERNEL_SIZE;
            // 手動アンロールして8要素を展開
            patch[base + 0] = in_channel[0 * 8 + w];
            patch[base + 1] = in_channel[1 * 8 + w];
            patch[base + 2] = in_channel[2 * 8 + w];
            patch[base + 3] = in_channel[3 * 8 + w];
            patch[base + 4] = in_channel[4 * 8 + w];
            patch[base + 5] = in_channel[5 * 8 + w];
            patch[base + 6] = in_channel[6 * 8 + w];
            patch[base + 7] = in_channel[7 * 8 + w];
        }

        int oc = 0;
        constexpr int unroll_factor = 4;
        int main_loop_count = (OutChannels / unroll_factor) * unroll_factor;

        for (; oc < main_loop_count; oc += unroll_factor) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            const float* w_ptr0 = weights + (oc + 0) * padded_inner_size;
            const float* w_ptr1 = weights + (oc + 1) * padded_inner_size;
            const float* w_ptr2 = weights + (oc + 2) * padded_inner_size;
            const float* w_ptr3 = weights + (oc + 3) * padded_inner_size;
            int i = 0;
            for (; i <= inner_size - 8; i += 8) {
                __m256 patch_vec = _mm256_load_ps(patch + i);
                __m256 w0 = _mm256_load_ps(w_ptr0 + i);
                __m256 w1 = _mm256_load_ps(w_ptr1 + i);
                __m256 w2 = _mm256_load_ps(w_ptr2 + i);
                __m256 w3 = _mm256_load_ps(w_ptr3 + i);
                sum0 = _mm256_fmadd_ps(patch_vec, w0, sum0);
                sum1 = _mm256_fmadd_ps(patch_vec, w1, sum1);
                sum2 = _mm256_fmadd_ps(patch_vec, w2, sum2);
                sum3 = _mm256_fmadd_ps(patch_vec, w3, sum3);
            }
            float dot0 = horizontal_sum_avx2(sum0);
            float dot1 = horizontal_sum_avx2(sum1);
            float dot2 = horizontal_sum_avx2(sum2);
            float dot3 = horizontal_sum_avx2(sum3);
            for (; i < inner_size; ++i) {
                dot0 += patch[i] * w_ptr0[i];
                dot1 += patch[i] * w_ptr1[i];
                dot2 += patch[i] * w_ptr2[i];
                dot3 += patch[i] * w_ptr3[i];
            }
            // バイアスを加算して結果を書き込み
            output[(oc + 0) * 8 + w] = dot0 + bias[oc + 0];
            output[(oc + 1) * 8 + w] = dot1 + bias[oc + 1];
            output[(oc + 2) * 8 + w] = dot2 + bias[oc + 2];
            output[(oc + 3) * 8 + w] = dot3 + bias[oc + 3];
        }

        for (; oc < OutChannels; ++oc) {
            __m256 sum = _mm256_setzero_ps();
            const float* w_ptr = weights + oc * padded_inner_size;
            int i = 0;
            for (; i <= inner_size - 8; i += 8) {
                __m256 patch_vec = _mm256_load_ps(patch + i);
                __m256 w_vec = _mm256_load_ps(w_ptr + i);
                sum = _mm256_fmadd_ps(patch_vec, w_vec, sum);
            }
            float dot = horizontal_sum_avx2(sum);
            for (; i < inner_size; ++i) {
                dot += patch[i] * w_ptr[i];
            }
            output[oc * 8 + w] = dot + bias[oc];
        }
    }
}

// 1x8の畳み込み処理
template <int InChannels, int OutChannels>
void conv1x8_avx2(const float* __restrict input,
    const float* __restrict weights,
    const float* __restrict bias,
    float* __restrict output) {
    constexpr int KERNEL_SIZE = 8;
    constexpr int inner_size = InChannels * KERNEL_SIZE;
    constexpr int padded_inner_size = ((inner_size + 7) / 8) * 8;

    for (int h = 0; h < 8; ++h) {
        alignas(32) float patch[inner_size];
        for (int ic = 0; ic < InChannels; ++ic) {
            const float* in_channel = input + ic * 64;  // 8x8 = 64
            int base = ic * KERNEL_SIZE;
            patch[base + 0] = in_channel[h * 8 + 0];
            patch[base + 1] = in_channel[h * 8 + 1];
            patch[base + 2] = in_channel[h * 8 + 2];
            patch[base + 3] = in_channel[h * 8 + 3];
            patch[base + 4] = in_channel[h * 8 + 4];
            patch[base + 5] = in_channel[h * 8 + 5];
            patch[base + 6] = in_channel[h * 8 + 6];
            patch[base + 7] = in_channel[h * 8 + 7];
        }

        int oc = 0;
        constexpr int unroll_factor = 4;
        int main_loop_count = (OutChannels / unroll_factor) * unroll_factor;

        for (; oc < main_loop_count; oc += unroll_factor) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            const float* w_ptr0 = weights + (oc + 0) * padded_inner_size;
            const float* w_ptr1 = weights + (oc + 1) * padded_inner_size;
            const float* w_ptr2 = weights + (oc + 2) * padded_inner_size;
            const float* w_ptr3 = weights + (oc + 3) * padded_inner_size;
            int i = 0;
            for (; i <= inner_size - 8; i += 8) {
                __m256 patch_vec = _mm256_load_ps(patch + i);
                __m256 w0 = _mm256_load_ps(w_ptr0 + i);
                __m256 w1 = _mm256_load_ps(w_ptr1 + i);
                __m256 w2 = _mm256_load_ps(w_ptr2 + i);
                __m256 w3 = _mm256_load_ps(w_ptr3 + i);
                sum0 = _mm256_fmadd_ps(patch_vec, w0, sum0);
                sum1 = _mm256_fmadd_ps(patch_vec, w1, sum1);
                sum2 = _mm256_fmadd_ps(patch_vec, w2, sum2);
                sum3 = _mm256_fmadd_ps(patch_vec, w3, sum3);
            }
            float dot0 = horizontal_sum_avx2(sum0);
            float dot1 = horizontal_sum_avx2(sum1);
            float dot2 = horizontal_sum_avx2(sum2);
            float dot3 = horizontal_sum_avx2(sum3);
            for (; i < inner_size; ++i) {
                dot0 += patch[i] * w_ptr0[i];
                dot1 += patch[i] * w_ptr1[i];
                dot2 += patch[i] * w_ptr2[i];
                dot3 += patch[i] * w_ptr3[i];
            }
            // バイアスを加算して結果を書き込み
            output[(oc + 0) * 8 + h] = dot0 + bias[oc + 0];
            output[(oc + 1) * 8 + h] = dot1 + bias[oc + 1];
            output[(oc + 2) * 8 + h] = dot2 + bias[oc + 2];
            output[(oc + 3) * 8 + h] = dot3 + bias[oc + 3];
        }

        for (; oc < OutChannels; ++oc) {
            __m256 sum = _mm256_setzero_ps();
            const float* w_ptr = weights + oc * padded_inner_size;
            int i = 0;
            for (; i <= inner_size - 8; i += 8) {
                __m256 patch_vec = _mm256_load_ps(patch + i);
                __m256 w_vec = _mm256_load_ps(w_ptr + i);
                sum = _mm256_fmadd_ps(patch_vec, w_vec, sum);
            }
            float dot = horizontal_sum_avx2(sum);
            for (; i < inner_size; ++i) {
                dot += patch[i] * w_ptr[i];
            }
            output[oc * 8 + h] = dot + bias[oc];
        }
    }
}

// ReLU処理
template <int Output>
inline void relu(float* __restrict output) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    // 32要素ずつ処理
    for (; i <= Output - 32; i += 32) {
        __m256 vec1 = _mm256_load_ps(output + i);
        __m256 vec2 = _mm256_load_ps(output + i + 8);
        __m256 vec3 = _mm256_load_ps(output + i + 16);
        __m256 vec4 = _mm256_load_ps(output + i + 24);

        vec1 = _mm256_max_ps(vec1, zero);
        vec2 = _mm256_max_ps(vec2, zero);
        vec3 = _mm256_max_ps(vec3, zero);
        vec4 = _mm256_max_ps(vec4, zero);

        _mm256_store_ps(output + i, vec1);
        _mm256_store_ps(output + i + 8, vec2);
        _mm256_store_ps(output + i + 16, vec3);
        _mm256_store_ps(output + i + 24, vec4);
    }
    // 16要素ずつ処理
    for (; i <= Output - 16; i += 16) {
        __m256 vec1 = _mm256_load_ps(output + i);
        __m256 vec2 = _mm256_load_ps(output + i + 8);
        vec1 = _mm256_max_ps(vec1, zero);
        vec2 = _mm256_max_ps(vec2, zero);
        _mm256_store_ps(output + i, vec1);
        _mm256_store_ps(output + i + 8, vec2);
    }
    // 8要素ずつ処理
    for (; i <= Output - 8; i += 8) {
        __m256 vec = _mm256_load_ps(output + i);
        vec = _mm256_max_ps(vec, zero);
        _mm256_store_ps(output + i, vec);
    }
    // 余りのスカラー処理
    for (; i < Output; ++i) {
        output[i] = output[i] > 0.f ? output[i] : 0.f;
    }
}

// 畳み込み層+cat+ReLU
template <int InChannels, int OutChannels>
inline void conv_cat_relu(
    const float* __restrict input,
    const float* __restrict weights_conv8x1,
    const float* __restrict bias_conv8x1,
    const float* __restrict weights_conv1x8,
    const float* __restrict bias_conv1x8,
    float* __restrict output) {

    constexpr int output_offset = OutChannels * 8; // conv1x8 の出力開始オフセット

    // 1. 8x1の畳み込み処理
    conv8x1_avx2<InChannels, OutChannels>(input, weights_conv8x1, bias_conv8x1, output);

    // 2. 1x8の畳み込み処理（出力の後半部分に格納）
    conv1x8_avx2<InChannels, OutChannels>(input, weights_conv1x8, bias_conv1x8, output + output_offset);

    // 3. 連結後の出力全体に対してReLUを適用
    constexpr int total_output = OutChannels * 8 * 2;
    relu<total_output>(output);
}

// 全結合層
template <int InputSize, int OutputSize>
void fc_layer(const float* __restrict input,
    const float* __restrict weights,
    float* __restrict output,
    const float* __restrict bias) {
    // 入力サイズを8の倍数にパディングしたサイズ（重み行列の行サイズにも対応）
    constexpr int padded_input_size = (InputSize + 7) & ~7;
    // 入力データは8の倍数になるようにパディング（余分な領域はゼロ埋めされる）
    constexpr int input_aligned = (InputSize + 7) & ~7;
    // 出力チャネルのアンローリング因子。必要に応じて変更可能
    constexpr int unroll_factor = 4;
    // アンローリングで処理する出力チャネル数の上限
    constexpr int main_loop_count = (OutputSize / unroll_factor) * unroll_factor;

    // --- Main Loop: 複数の出力チャネルを同時に計算 ---
    for (int o = 0; o < main_loop_count; o += unroll_factor) {
        // 各出力チャネルごとに、前半(sum_a)と後半(sum_b)の集約器を初期化して並列計算を促進
        __m256 sum0_a = _mm256_setzero_ps();
        __m256 sum0_b = _mm256_setzero_ps();
        __m256 sum1_a = _mm256_setzero_ps();
        __m256 sum1_b = _mm256_setzero_ps();
        __m256 sum2_a = _mm256_setzero_ps();
        __m256 sum2_b = _mm256_setzero_ps();
        __m256 sum3_a = _mm256_setzero_ps();
        __m256 sum3_b = _mm256_setzero_ps();

        // 各出力チャネルに対応する重みの開始ポインタを計算
        const float* w_ptr0 = weights + (o + 0) * padded_input_size;
        const float* w_ptr1 = weights + (o + 1) * padded_input_size;
        const float* w_ptr2 = weights + (o + 2) * padded_input_size;
        const float* w_ptr3 = weights + (o + 3) * padded_input_size;

        // 入力サイズが32の倍数の場合、32要素ずつループアンローリングで処理
        if constexpr (InputSize % 32 == 0) {
            for (int i = 0; i < input_aligned; i += 32) {
                // プリフェッチ命令により、次の64要素先のデータをキャッシュにロード（範囲内であれば）
                if (i + 64 < input_aligned) {
                    _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w_ptr0 + i + 64), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w_ptr1 + i + 64), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w_ptr2 + i + 64), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w_ptr3 + i + 64), _MM_HINT_T0);
                }
                // 現在のブロックの入力データの先頭アドレスを取得
                const float* in_ptr = input + i;
                // 32要素を4つの256ビットレジスタにロード（各レジスタに8要素ずつ）
                __m256 x0 = _mm256_load_ps(in_ptr);       // 要素 i ～ i+7
                __m256 x1 = _mm256_load_ps(in_ptr + 8);     // 要素 i+8 ～ i+15
                __m256 x2 = _mm256_load_ps(in_ptr + 16);    // 要素 i+16 ～ i+23
                __m256 x3 = _mm256_load_ps(in_ptr + 24);    // 要素 i+24 ～ i+31

                // --- 前半16要素の処理 ---
                {
                    // 各出力チャネルごとに、前半部分の重みをロードし乗算累積演算
                    __m256 w0_0 = _mm256_load_ps(w_ptr0 + i);
                    __m256 w0_1 = _mm256_load_ps(w_ptr0 + i + 8);
                    sum0_a = _mm256_fmadd_ps(x0, w0_0, sum0_a);
                    sum0_a = _mm256_fmadd_ps(x1, w0_1, sum0_a);

                    __m256 w1_0 = _mm256_load_ps(w_ptr1 + i);
                    __m256 w1_1 = _mm256_load_ps(w_ptr1 + i + 8);
                    sum1_a = _mm256_fmadd_ps(x0, w1_0, sum1_a);
                    sum1_a = _mm256_fmadd_ps(x1, w1_1, sum1_a);

                    __m256 w2_0 = _mm256_load_ps(w_ptr2 + i);
                    __m256 w2_1 = _mm256_load_ps(w_ptr2 + i + 8);
                    sum2_a = _mm256_fmadd_ps(x0, w2_0, sum2_a);
                    sum2_a = _mm256_fmadd_ps(x1, w2_1, sum2_a);

                    __m256 w3_0 = _mm256_load_ps(w_ptr3 + i);
                    __m256 w3_1 = _mm256_load_ps(w_ptr3 + i + 8);
                    sum3_a = _mm256_fmadd_ps(x0, w3_0, sum3_a);
                    sum3_a = _mm256_fmadd_ps(x1, w3_1, sum3_a);
                }

                // --- 後半16要素の処理 ---
                {
                    // 各出力チャネルごとに、後半部分の重みをロードし乗算累積演算
                    __m256 w0_2 = _mm256_load_ps(w_ptr0 + i + 16);
                    __m256 w0_3 = _mm256_load_ps(w_ptr0 + i + 24);
                    sum0_b = _mm256_fmadd_ps(x2, w0_2, sum0_b);
                    sum0_b = _mm256_fmadd_ps(x3, w0_3, sum0_b);

                    __m256 w1_2 = _mm256_load_ps(w_ptr1 + i + 16);
                    __m256 w1_3 = _mm256_load_ps(w_ptr1 + i + 24);
                    sum1_b = _mm256_fmadd_ps(x2, w1_2, sum1_b);
                    sum1_b = _mm256_fmadd_ps(x3, w1_3, sum1_b);

                    __m256 w2_2 = _mm256_load_ps(w_ptr2 + i + 16);
                    __m256 w2_3 = _mm256_load_ps(w_ptr2 + i + 24);
                    sum2_b = _mm256_fmadd_ps(x2, w2_2, sum2_b);
                    sum2_b = _mm256_fmadd_ps(x3, w2_3, sum2_b);

                    __m256 w3_2 = _mm256_load_ps(w_ptr3 + i + 16);
                    __m256 w3_3 = _mm256_load_ps(w_ptr3 + i + 24);
                    sum3_b = _mm256_fmadd_ps(x2, w3_2, sum3_b);
                    sum3_b = _mm256_fmadd_ps(x3, w3_3, sum3_b);
                }
            }
        }
        else {
            // InputSize が32の倍数でない場合は、8要素ずつループして安全に処理
            for (int i = 0; i < input_aligned; i += 8) {
                const float* in_ptr = input + i;
                __m256 x = _mm256_load_ps(in_ptr);
                sum0_a = _mm256_fmadd_ps(x, _mm256_load_ps(w_ptr0 + i), sum0_a);
                sum1_a = _mm256_fmadd_ps(x, _mm256_load_ps(w_ptr1 + i), sum1_a);
                sum2_a = _mm256_fmadd_ps(x, _mm256_load_ps(w_ptr2 + i), sum2_a);
                sum3_a = _mm256_fmadd_ps(x, _mm256_load_ps(w_ptr3 + i), sum3_a);
            }
        }
        // 前半部分(sum_a)と後半部分(sum_b)の和をとって、全体の加算結果を得る
        __m256 sum0 = _mm256_add_ps(sum0_a, sum0_b);
        __m256 sum1 = _mm256_add_ps(sum1_a, sum1_b);
        __m256 sum2 = _mm256_add_ps(sum2_a, sum2_b);
        __m256 sum3 = _mm256_add_ps(sum3_a, sum3_b);

        // 各出力チャネルについて、水平加算関数で最終的な出力値を計算し、バイアスを加算
        output[o + 0] = horizontal_sum_avx2(sum0) + bias[o + 0];
        output[o + 1] = horizontal_sum_avx2(sum1) + bias[o + 1];
        output[o + 2] = horizontal_sum_avx2(sum2) + bias[o + 2];
        output[o + 3] = horizontal_sum_avx2(sum3) + bias[o + 3];
    }

    // --- Tail 部分: アンローリングで処理できなかった残りの出力チャネルの計算 ---
    for (int o = main_loop_count; o < OutputSize; ++o) {
        __m256 sum_a = _mm256_setzero_ps();
        __m256 sum_b = _mm256_setzero_ps();
        const float* w_ptr = weights + o * padded_input_size;
        if constexpr (InputSize % 32 == 0) {
            for (int i = 0; i < input_aligned; i += 32) {
                if (i + 64 < input_aligned) {
                    _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(w_ptr + i + 64), _MM_HINT_T0);
                }
                const float* in_ptr = input + i;
                __m256 x0 = _mm256_load_ps(in_ptr);
                __m256 x1 = _mm256_load_ps(in_ptr + 8);
                __m256 x2 = _mm256_load_ps(in_ptr + 16);
                __m256 x3 = _mm256_load_ps(in_ptr + 24);

                __m256 w0 = _mm256_load_ps(w_ptr + i);
                __m256 w1 = _mm256_load_ps(w_ptr + i + 8);
                sum_a = _mm256_fmadd_ps(x0, w0, sum_a);
                sum_a = _mm256_fmadd_ps(x1, w1, sum_a);

                __m256 w2 = _mm256_load_ps(w_ptr + i + 16);
                __m256 w3 = _mm256_load_ps(w_ptr + i + 24);
                sum_b = _mm256_fmadd_ps(x2, w2, sum_b);
                sum_b = _mm256_fmadd_ps(x3, w3, sum_b);
            }
        }
        else {
            // InputSize が32の倍数でない場合は、8要素ずつ処理
            for (int i = 0; i < input_aligned; i += 8) {
                const float* in_ptr = input + i;
                __m256 x = _mm256_load_ps(in_ptr);
                sum_a = _mm256_fmadd_ps(x, _mm256_load_ps(w_ptr + i), sum_a);
            }
        }
        __m256 sum = _mm256_add_ps(sum_a, sum_b);
        output[o] = horizontal_sum_avx2(sum) + bias[o];
    }
}

}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Position& pos) {
    constexpr int EMBEDDING_DIM = 16;
    constexpr int CONV_OUT_CHANNELS = 4;
    constexpr int FC_DIM = 32;

    // パラメータが12個あること、かつ名前が想定通りであることをチェック
    assert(parameters.size() == 12 && "Insufficient number of parameters loaded");
    assert(parameters[0].name == "l1_1.weight" && "Parameter 0 should be l1_1.weight");
    assert(parameters[1].name == "l1_2.weight" && "Parameter 1 should be l1_2.weight");
    assert(parameters[2].name == "l2.weight" && "Parameter 2 should be l2.weight");
    assert(parameters[3].name == "l2.bias" && "Parameter 3 should be l2.bias");
    assert(parameters[4].name == "l3_1.weight" && "Parameter 4 should be l3_1.weight");
    assert(parameters[5].name == "l3_1.bias" && "Parameter 5 should be l3_1.bias");
    assert(parameters[6].name == "l3_2.weight" && "Parameter 6 should be l3_2.weight");
    assert(parameters[7].name == "l3_2.bias" && "Parameter 7 should be l3_2.bias");
    assert(parameters[8].name == "l4.weight" && "Parameter 8 should be l4.weight");
    assert(parameters[9].name == "l4.bias" && "Parameter 9 should be l4.bias");
    assert(parameters[10].name == "l5.weight" && "Parameter 10 should be l5.weight");
    assert(parameters[11].name == "l5.bias" && "Parameter 11 should be l5.bias");

    alignas(32) float h1[EMBEDDING_DIM * (int)SquareNum];
    alignas(32) float h2[EMBEDDING_DIM * 8 * 8];
    alignas(32) float h3[CONV_OUT_CHANNELS * 8 * 2];
    alignas(32) float h4[FC_DIM];
    alignas(32) float h5;

    // 1. 埋め込み層の処理
    const float* l1_1_weight = parameters[0].data.get();
    const float* l1_2_weight = parameters[1].data.get();
    embedding_layers(pos, l1_1_weight, l1_2_weight, h1);

    // 2. 畳み込み層の処理
    const float* l2_weight = parameters[2].data.get();
    const float* l2_bias = parameters[3].data.get();
    conv2x2_depthwise_relu<EMBEDDING_DIM>(
        h1, l2_weight, l2_bias, h2);

    // 3. 畳み込み層の処理
    const float* l3_1_weight = parameters[4].data.get();
    const float* l3_1_bias = parameters[5].data.get();
    const float* l3_2_weight = parameters[6].data.get();
    const float* l3_2_bias = parameters[7].data.get();
    conv_cat_relu<EMBEDDING_DIM, CONV_OUT_CHANNELS>(
        h2, l3_1_weight, l3_1_bias, l3_2_weight, l3_2_bias, h3);

    // 4. 1つ目の全結合層
    const float* l4_weight = parameters[8].data.get();
    const float* l4_bias = parameters[9].data.get();
    fc_layer<CONV_OUT_CHANNELS * 8 * 2, FC_DIM>(
        h3, l4_weight, h4, l4_bias);

    // 5. ReLU活性化関数の適用
    relu<FC_DIM>(h4);

    // 6. 2つ目の全結合層（出力層）
    const float* l5_weight = parameters[10].data.get();
    const float* l5_bias = parameters[11].data.get();
    fc_layer<FC_DIM, 1>(
        h4, l5_weight, &h5, l5_bias);

    // 7. 評価値への変換
    // 出力を将棋エンジンの評価値スケールに変換
    constexpr float EVAL_SCALE = 600.0f;
    const Value v = static_cast<Value>(h5 * EVAL_SCALE);

    return v;
}

}  // namespace Stockfish
