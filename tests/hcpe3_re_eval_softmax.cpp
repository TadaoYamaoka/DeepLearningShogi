#include "cppshogi.h"
#include "python_module.h"
#include "usi.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>

extern std::vector<TrainingData> trainingData;
void close_hcpe3_cache();

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

TrainingData re_eval(float base, float best, float temperature, u16 selected, float alpha = 1.0f) {
    trainingData.clear();
    std::vector<float> logits(9 * 9 * MAX_MOVE_LABEL_NUM, base);
    logits[make_move_label(selected, Black)] = best;
    size_t index = 0;
    float value = 0.5f;
    __hcpe3_cache_re_eval(1, reinterpret_cast<char*>(&index),
        reinterpret_cast<char*>(logits.data()), reinterpret_cast<char*>(&value),
        alpha, 0, 0, 1, MaxLegalMoves, temperature);
    const auto& data = trainingData.front();
    check(!data.candidates.empty(), "softmax discarded every candidate");
    float sum = 0;
    for (const auto& kv : data.candidates) {
        check(std::isfinite(kv.second) && kv.second >= 0, "invalid probability");
        sum += kv.second;
    }
    check(std::abs(sum - 1.0f) < 1e-5f, "probabilities do not sum to one");
    return data;
}
}

int main() {
    const char* path = "hcpe3_re_eval_softmax.cache";
    try {
        init();
        Position pos;
        pos.set(DefaultStartPositionSFEN);
        const auto selected = static_cast<u16>(usiToMove(pos, "7g7f").proFromAndTo());
        const auto alternative = static_cast<u16>(usiToMove(pos, "2g2f").proFromAndTo());
        const size_t legalCount = MoveList<Legal>(pos).size();
        auto& seed = trainingData.emplace_back(pos.toHuffmanCodedPos(), 0.75f, 1.0f);
        seed.candidates[selected] = 0.75f;
        seed.candidates[alternative] = 0.25f;
        __hcpe3_create_cache(path);
        check(__hcpe3_load_cache(path) == 1, "failed to load fixture");

        // Negative offsets must preserve the softmax distribution, including
        // cases where exp(logit) would underflow for every legal move.
        for (const float temperature : {0.5f, 1.0f, 2.0f}) {
            const auto reference = re_eval(0, 1, temperature, selected);
            const float weight = std::exp(1.0f / temperature);
            check(std::abs(reference.candidates.at(selected) - weight / (legalCount - 1 + weight)) < 1e-6f,
                "softmax differs from the expected distribution");
            for (const float offset : {-200.0f, -1000.0f, 200.0f}) {
                const auto shifted = re_eval(offset, offset + 1, temperature, selected);
                check(shifted.candidates.size() == legalCount, "legal moves were lost");
                for (const auto& kv : reference.candidates) {
                    check(std::abs(shifted.candidates.at(kv.first) - kv.second) < 1e-6f,
                        "adding a constant changed softmax");
                }
            }
        }

        const float max = std::numeric_limits<float>::max();
        const float tiny = std::numeric_limits<float>::min();
        for (const float logit : {-max, max}) {
            const auto uniform = re_eval(logit, logit, tiny, selected);
            check(uniform.candidates.size() == legalCount, "uniform softmax lost moves");
            for (const auto& kv : uniform.candidates) {
                check(std::abs(kv.second - 1.0f / legalCount) < 1e-6f, "uniform softmax changed");
            }
        }
        const auto extreme = re_eval(-max, max, 0.5f, selected);
        check(extreme.candidates.size() == 1 && extreme.candidates.at(selected) == 1,
            "extreme finite logits were not normalized");

        // The float difference -FLT_MAX - FLT_MAX overflows, but dividing
        // the double difference by FLT_MAX gives -2 and must retain all moves.
        const auto highTemperature = re_eval(-max, max, max, selected);
        const double otherWeight = std::exp(-2.0);
        const double denominator = 1.0 + (legalCount - 1) * otherWeight;
        check(highTemperature.candidates.size() == legalCount,
            "extreme logits and temperature lost legal moves");
        for (const auto& kv : highTemperature.candidates) {
            const double expected = (kv.first == selected ? 1.0 : otherWeight) / denominator;
            check(std::abs(kv.second - expected) < 1e-6,
                "float subtraction overflow changed the softmax distribution");
        }

        const auto reference = re_eval(-200, -199, 1, selected);
        const auto blended = re_eval(-200, -199, 1, selected, 0.5f);
        check(std::abs(blended.candidates.at(selected)
            - (0.375f + 0.5f * reference.candidates.at(selected))) < 1e-6f,
            "policy blending changed");

        for (const float temperature : {0.0f, -1.0f,
            std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            bool rejected = false;
            try { re_eval(-200, -199, temperature, selected); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected && trainingData.empty(), "invalid temperature was not rejected before mutation");
        }

        close_hcpe3_cache();
        std::remove(path);
        std::cout << "All hcpe3 re-evaluation softmax tests passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        close_hcpe3_cache();
        std::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
