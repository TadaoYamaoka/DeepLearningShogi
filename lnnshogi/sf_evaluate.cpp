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

#include <torch/torch.h>
#include <torch/script.h>

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

#include "bitboard.hpp"

#include "cppshogi.h"

namespace Stockfish {

struct TorchInitializer {
    TorchInitializer() {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
    }
};
TorchInitializer torch_initializer;
c10::InferenceMode guard;
auto model = torch::jit::load(R"(D:\src\DeepLearningShogiNN\dlshogi\model_lite-008.pt)");

    // Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Position& pos) {

    Value v = 0;

    features1_lite_t features1;
    features2_lite_t features2;

    // set all padding index
    std::fill_n((int64_t*)features1, sizeof(features1_lite_t) / sizeof(int64_t), PIECETYPE_NUM * 2);
    std::fill_n((int64_t*)features2, sizeof(features2_lite_t) / sizeof(int64_t), MAX_FEATURES2_NUM);

    make_input_features_lite(pos, features1, features2);

    std::vector<torch::jit::IValue> x = {
        torch::from_blob(features1, { 1, (size_t)SquareNum }, torch::dtype(torch::kInt64)),
        torch::from_blob(features2, { 1, MAX_FEATURES2_NUM }, torch::dtype(torch::kInt64))
    };

    const auto y = model.forward(x);
    const float value = *y.toTensor().data_ptr<float>();

    v = int(value * 600);

    return v;
}

}  // namespace Stockfish
