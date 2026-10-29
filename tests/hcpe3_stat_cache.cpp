#include "cppshogi.h"
#include "python_module.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>

extern std::vector<TrainingData> trainingData;
void close_hcpe3_cache();

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string capture_stats() {
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    try {
        __hcpe3_stat_cache();
    }
    catch (...) {
        std::cout.rdbuf(previous);
        throw;
    }
    std::cout.rdbuf(previous);
    return output.str();
}

void check_stat(std::istream& input, const char* name, size_t count,
    const std::array<double, 7>& expected) {
    std::string key;
    size_t actualCount;
    check(static_cast<bool>(input >> key) && key == name, "statistic heading changed");
    check(static_cast<bool>(input >> key >> actualCount) && key == "count"
        && actualCount == count, "statistic count changed");
    if (count == 0) return;
    const char* keys[]{"mean", "std", "min", "25%", "50%", "75%", "max"};
    for (size_t i = 0; i < expected.size(); ++i) {
        double actual;
        check(static_cast<bool>(input >> key >> actual) && key == keys[i], "statistic field missing");
        check(std::isfinite(actual) && std::abs(actual - expected[i]) < 0.000051,
            "incorrect statistic value");
    }
}

void check_empty(const std::string& output) {
    std::istringstream input(output);
    for (const char* name : {"counts:", "candidates:", "values:", "results:"}) {
        check_stat(input, name, 0, {});
    }
    std::string extra;
    check(!(input >> extra), "empty cache printed undefined statistics");
}
}

int main() {
    const char* path = "hcpe3_stat_cache.cache";
    try {
        close_hcpe3_cache();
        check_empty(capture_stats());
        __hcpe3_create_cache(path);
        check(__hcpe3_load_cache(path) == 0, "empty cache was not loaded");
        check_empty(capture_stats());
        close_hcpe3_cache();

        for (size_t count : {size_t(1), size_t(2), size_t(3)}) {
            for (size_t i = 0; i < count; ++i) {
                const float value = count == 1 ? 0.6f : 0.2f + 0.8f * i / (count - 1);
                auto& data = trainingData.emplace_back(HuffmanCodedPos{}, value, 0.5f);
                data.candidates[1] = 0.75f;
                data.candidates[2] = 0.25f;
            }
            __hcpe3_create_cache(path);
            check(__hcpe3_load_cache(path) == count, "cache count changed");
            std::istringstream input(capture_stats());
            check_stat(input, "counts:", count, {1, 0, 1, 1, 1, 1, 1});
            check_stat(input, "candidates:", count, {2, 0, 2, 2, 2, 2, 2});
            const std::array<double, 7> expectedValues = count == 1
                ? std::array<double, 7>{0.6, 0, 0.6, 0.6, 0.6, 0.6, 0.6}
                : std::array<double, 7>{0.6, count == 2 ? 0.4 : std::sqrt(0.32 / 3),
                    0.2, 0.4, 0.6, 0.8, 1};
            check_stat(input, "values:", count, expectedValues);
            check_stat(input, "results:", count, {0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5});
            close_hcpe3_cache();
        }
        std::remove(path);
        std::cout << "All hcpe3 cache statistics tests passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        close_hcpe3_cache();
        std::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
