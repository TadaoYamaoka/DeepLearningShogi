#include "cppshogi.h"
#include "python_module.h"
#include "usi.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

extern std::vector<TrainingData> trainingData;
extern std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;

namespace {
const char* path = "hcpe3_truncated_input.hcpe3";

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void append(std::vector<char>& bytes, const T& value) {
    const auto data = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void write_file(const std::vector<char>& bytes, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), size);
    check(static_cast<bool>(out), "failed to write test fixture");
}

void write_input(const std::vector<char>& bytes, size_t size) {
    trainingData.clear();
    duplicates.clear();
    write_file(bytes, size);
}

void expect_rejected(const std::vector<char>& bytes, size_t size, bool average, size_t expectedLen = 0) {
    write_input(bytes, size);
    size_t len = 0;
    bool rejected = false;
    try {
        __load_hcpe3(path, average, 0, 1, len);
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "incomplete or invalid hcpe3 was accepted");
    check(trainingData.size() == expectedLen && len == expectedLen, "incomplete move was added to training data");
}
}

int main() {
    try {
        init();
        Position pos;
        pos.set(DefaultStartPositionSFEN);
        const auto selected = static_cast<u16>(usiToMove(pos, "7g7f").proFromAndTo());
        const auto alternative = static_cast<u16>(usiToMove(pos, "2g2f").proFromAndTo());
        // Draw also prevents a 40-byte prefix from being mistaken for hcpe.
        HuffmanCodedPosAndEval3 header{pos.toHuffmanCodedPos(), 1, Draw, 0};
        MoveInfo info{selected, 100, 2};
        std::vector<char> bytes;
        append(bytes, header);
        append(bytes, info);
        append(bytes, MoveVisits(selected, 3));
        append(bytes, MoveVisits(alternative, 1));

        for (bool average : {false, true}) {
            // Every possible nonempty truncated prefix, including missing entire records.
            for (size_t size = 1; size < bytes.size(); ++size) {
                expect_rejected(bytes, size, average);
            }
            write_input(bytes, bytes.size());
            size_t len = 0;
            check(__load_hcpe3(path, average, 0, 1, len) == 1 && len == 1,
                "valid hcpe3 was rejected");
            const auto& data = trainingData.front();
            check(data.candidates.at(selected) == 0.75f && data.candidates.at(alternative) == 0.25f,
                "valid probabilities changed");
            check(data.count == 1 && data.result == 0.5f, "valid targets changed");

            write_input(bytes, 0);
            len = 0;
            check(__load_hcpe3(path, average, 0, 1, len) == 0 && len == 0,
                "empty input was rejected");

            // A complete game followed by each possible incomplete next game.
            for (size_t tail = 1; tail < bytes.size(); ++tail) {
                auto broken = bytes;
                broken.insert(broken.end(), bytes.begin(), bytes.begin() + tail);
                expect_rejected(broken, broken.size(), average, 1);
            }
        }

        auto invalid = bytes;
        header.moveNum = 514;
        std::memcpy(invalid.data(), &header, sizeof(header));
        expect_rejected(invalid, invalid.size(), false);
        invalid = bytes;
        info.candidateNum = 594;
        std::memcpy(invalid.data() + sizeof(header), &info, sizeof(info));
        expect_rejected(invalid, invalid.size(), false);

        std::remove(path);
        std::cout << "All hcpe3 truncated input tests passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
