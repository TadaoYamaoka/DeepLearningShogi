#include "cppshogi.h"
#include "python_module.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

extern std::vector<TrainingData> trainingData;
extern std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;
TrainingData get_cache(size_t i);
void close_hcpe3_cache();

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void expect_preserved(const std::string& path) {
    const auto originalData = trainingData;
    const auto originalDuplicates = duplicates;
    bool rejected = false;
    try {
        __hcpe3_create_cache(path);
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "failed save did not throw");
    check(trainingData.size() == originalData.size(), "failed save released training data");
    for (size_t i = 0; i < originalData.size(); ++i) {
        const auto& actual = trainingData[i];
        const auto& original = originalData[i];
        check(actual.hcp == original.hcp && actual.value == original.value
            && actual.result == original.result && actual.count == original.count
            && actual.candidates == original.candidates, "failed save changed training data");
    }
    check(duplicates == originalDuplicates, "failed save changed duplicate index");
}
}

int main() {
    const char* output = "hcpe3_cache_save_failure.cache";
    try {
        HuffmanCodedPos hcp{};
        auto& data = trainingData.emplace_back(hcp, 0.75f, 1.0f);
        for (u16 move = 1; move <= MaxLegalMoves; ++move) data.candidates[move] = 1.0f / MaxLegalMoves;
        duplicates.emplace(hcp, 0);

        // A directory cannot be opened as an output file.
        expect_preserved(std::filesystem::current_path().string());

#ifdef _WIN32
        // Force an actual write failure without filling a disk: the pipe reader
        // disconnects after receiving one byte of a record larger than its buffer.
        const std::string pipePath = "\\\\.\\pipe\\dlshogi-cache-save-" + std::to_string(GetCurrentProcessId());
        const HANDLE pipe = CreateNamedPipeA(pipePath.c_str(), PIPE_ACCESS_INBOUND,
            PIPE_TYPE_BYTE | PIPE_WAIT, 1, 0, 1, 0, nullptr);
        check(pipe != INVALID_HANDLE_VALUE, "failed to create test pipe");
        bool readByte = false;
        std::thread reader([&] {
            if (ConnectNamedPipe(pipe, nullptr) || GetLastError() == ERROR_PIPE_CONNECTED) {
                char byte;
                DWORD read = 0;
                readByte = ReadFile(pipe, &byte, 1, &read, nullptr) && read == 1;
            }
            DisconnectNamedPipe(pipe);
            CloseHandle(pipe);
        });
        std::exception_ptr error;
        try { expect_preserved(pipePath); }
        catch (...) { error = std::current_exception(); }
        // Unblock the reader even if the C++ stream failed to open the pipe.
        const HANDLE fallback = CreateFileA(pipePath.c_str(), GENERIC_WRITE, 0,
            nullptr, OPEN_EXISTING, 0, nullptr);
        if (fallback != INVALID_HANDLE_VALUE) CloseHandle(fallback);
        reader.join();
        if (error) std::rethrow_exception(error);
        check(readByte, "test did not reach the write phase");
#else
        expect_preserved("/dev/full");
#endif

        // Retrying to a working destination must preserve the original format.
        const auto original = trainingData.front();
        __hcpe3_create_cache(output);
        check(trainingData.empty() && duplicates.empty(), "successful save did not release data");
        check(__hcpe3_load_cache(output) == 1, "saved cache count changed");
        const auto saved = get_cache(0);
        check(saved.hcp == original.hcp && saved.value == original.value
            && saved.result == original.result && saved.count == original.count
            && saved.candidates == original.candidates, "saved cache contents changed");
        close_hcpe3_cache();
        __hcpe3_create_cache(output);
        check(__hcpe3_load_cache(output) == 0, "empty cache save changed");
        close_hcpe3_cache();
        std::remove(output);
        std::cout << "All hcpe3 cache save failure tests passed\n";
        return 0;
    }
    catch (const std::exception& e) {
        close_hcpe3_cache();
        std::remove(output);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
