#include "cppshogi.h"
#include "python_module.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

TrainingData get_cache(size_t i);
TrainingData get_cache_with_lock(size_t i);
void close_hcpe3_cache();

namespace {
const char* path = "hcpe3_cache_validation.cache";

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_cache(size_t num, const std::vector<size_t>& offsets, size_t bodySize) {
    close_hcpe3_cache();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&num), sizeof(num));
    out.write(reinterpret_cast<const char*>(offsets.data()), offsets.size() * sizeof(size_t));
    const std::vector<char> body(bodySize, 0);
    out.write(body.data(), body.size());
    check(static_cast<bool>(out), "failed to write test fixture");
}

void expect_rejected() {
    bool rejected = false;
    try {
        __hcpe3_load_cache(path);
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "malformed cache was accepted");
    check(__hcpe3_get_cache_num() == 0, "rejected cache remained loaded");
}
}

int main() {
    try {
        const size_t dataBegin = 2 * sizeof(size_t);
        for (const size_t candidates : {size_t(0), size_t(1), size_t(MaxLegalMoves)}) {
            write_cache(1, {dataBegin}, sizeof(Hcpe3CacheBody) + candidates * sizeof(Hcpe3CacheCandidate));
            check(__hcpe3_load_cache(path) == 1, "valid cache was rejected");
            // Both readers must accept the largest record that fits their buffers.
            get_cache(0);
            get_cache_with_lock(0);
        }

        write_cache(0, {}, 0);
        check(__hcpe3_load_cache(path) == 0, "empty cache was rejected");

        write_cache(1, {dataBegin}, sizeof(Hcpe3CacheBody) + (MaxLegalMoves + 1) * sizeof(Hcpe3CacheCandidate));
        expect_rejected();
        write_cache(1, {dataBegin}, sizeof(Hcpe3CacheBody) - 1);
        expect_rejected();
        write_cache(1, {dataBegin}, sizeof(Hcpe3CacheBody) + 1);
        expect_rejected();
        write_cache(1, {0}, sizeof(Hcpe3CacheBody));
        expect_rejected();
        write_cache(1, {dataBegin + sizeof(Hcpe3CacheBody) + 1}, sizeof(Hcpe3CacheBody));
        expect_rejected();
        write_cache(2, {3 * sizeof(size_t) + 1, 3 * sizeof(size_t)}, 2 * sizeof(Hcpe3CacheBody));
        expect_rejected();
        write_cache(std::numeric_limits<size_t>::max(), {}, 0);
        expect_rejected();
        write_cache(1, {}, 0);
        expect_rejected();
        close_hcpe3_cache();
        std::remove(path);
        std::cout << "All hcpe3 cache validation tests passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        close_hcpe3_cache();
        std::remove(path);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
