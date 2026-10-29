#include "cppshogi.h"
#include "python_module.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

extern std::vector<TrainingData> trainingData;
extern std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

HuffmanCodedPosAndEval patch(unsigned char key, s16 eval, u16 move, GameResult result) {
    HuffmanCodedPosAndEval hcpe{};
    hcpe.hcp.data[0] = key;
    hcpe.eval = eval;
    hcpe.bestMove16 = move;
    hcpe.gameResult = result;
    return hcpe;
}

void check_data(size_t index, const HuffmanCodedPosAndEval& expected) {
    const auto& actual = trainingData[index];
    const float result = expected.gameResult == Draw ? 0.5f
        : ((expected.gameResult == BlackWin) == (expected.hcp.color() == Black) ? 1.0f : 0.0f);
    check(actual.hcp == expected.hcp && actual.count == 1
        && actual.value == score_to_value((Score)expected.eval) && actual.result == result
        && actual.candidates.size() == 1 && actual.candidates.at(expected.bestMove16) == 1,
        "patched position differs from expected data");
}
}

int main() {
    const char* path = "hcpe3_patch_with_hcpe.hcpe";
    try {
        const auto a = patch(0, 100, 1, BlackWin);
        const auto b = patch(1, -200, 2, WhiteWin);
        const auto c = patch(2, 300, 3, Draw);
        const auto cLast = patch(2, -400, 4, BlackWin);
        const auto aLast = patch(0, 500, 5, WhiteWin);
        const auto d = patch(3, 600, 6, BlackWin);
        for (const auto& hcpe : {a, b, a}) {
            auto& data = trainingData.emplace_back(hcpe.hcp, 0.25f, 0.75f);
            data.count = 3;
            data.candidates[7] = 2;
            data.candidates[8] = 1;
        }
        const auto untouched = trainingData[1];
        duplicates.emplace(a.hcp, 0);
        const auto originalDuplicates = duplicates;
        {
            std::ofstream ofs(path, std::ios::binary);
            for (const auto& hcpe : {a, c, cLast, aLast, d}) {
                ofs.write(reinterpret_cast<const char*>(&hcpe), sizeof(hcpe));
            }
            check(static_cast<bool>(ofs), "failed to write patch fixture");
        }
        size_t added = 7;
        check(__hcpe3_patch_with_hcpe(path, added) == 5 && added == 9,
            "patch counters changed");
        check(trainingData.size() == 5, "repeated patch appended duplicate positions");
        check_data(0, aLast);
        check_data(2, aLast);
        check_data(3, cLast);
        check_data(4, d);
        const auto& actual = trainingData[1];
        check(actual.hcp == untouched.hcp && actual.count == untouched.count
            && actual.value == untouched.value && actual.result == untouched.result
            && actual.candidates == untouched.candidates, "unmatched position changed");
        check(duplicates == originalDuplicates, "global duplicate index changed");
        check(__hcpe3_patch_with_hcpe(path, added) == 5 && added == 9
            && trainingData.size() == 5, "repeated call changed append behavior");
        {
            std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        }
        check(__hcpe3_patch_with_hcpe(path, added) == 0 && added == 9,
            "empty patch changed counters");
        std::remove(path);
        check(__hcpe3_patch_with_hcpe(path, added) == 0 && added == 9,
            "missing patch changed counters");
        std::cout << "hcpe3 patch tests passed\n";
    }
    catch (const std::exception& error) {
        std::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
