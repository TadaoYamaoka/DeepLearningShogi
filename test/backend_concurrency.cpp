// Exercise the actual dlshogi backend, including feature/policy correspondence.
#include "backend.h"
#include "book.hpp"
#include "init.hpp"
#include <algorithm>
#include <barrier>
#include <cmath>
#include <iostream>
#include <thread>

// The engine normally defines these; this test keeps book blending disabled.
std::map<Key, std::vector<BookEntry>> bookMap;
bool use_book_policy = false;

static void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  try {
    Require(argc == 2 || argc == 3, "usage: backend_concurrency model [GPU-count]");
    initTable();
    ::Position::initZobrist();
    const int gpus = argc == 3 ? std::stoi(argv[2]) : 1;
    Require(gpus >= 1 && gpus <= 8, "GPU count must be 1..8");
    std::array<lczero::shogi::DeviceConfig, 8> configs{};
    for (int gpu = 0; gpu < gpus; ++gpu)
      configs[gpu] = {argv[1], 2, 128};
    if (gpus < 8)
      configs[gpus] = {"disabled-device-model-does-not-exist", 0, 1};
    lczero::shogi::DlshogiBackend backend(configs, 0, 1.74f);
    const auto attributes = backend.GetAttributes();
    Require(attributes.suggested_num_search_threads == 2 * gpus &&
                attributes.recommended_batch_size == 128,
            "backend recommends slots, not global search worker count");
    constexpr int count = 64, writers = 8;
    std::vector<lczero::PositionHistory> histories;
    std::vector<lczero::MoveList> legal;
    lczero::PositionHistory history;
    history.Reset(lczero::Position(DefaultStartPositionSFEN));
    for (int i = 0; i < count; ++i) {
      histories.push_back(history);
      legal.push_back(history.Last().GetBoard().GenerateLegalMoves());
      Require(!legal.back().empty(), "nonterminal test position");
      history.Append(legal.back()[(i * 17) % legal.back().size()]);
    }
    auto input = [&](int i) {
      return lczero::EvalPosition{histories[i].GetPositions(), legal[i]};
    };
    std::vector<lczero::EvalResult> reference(count);
    {
      auto serial = backend.CreateComputation();
      for (int i = 0; i < count; ++i) {
        reference[i].p.resize(legal[i].size());
        serial->AddInput(input(i), reference[i].AsPtr());
      }
      serial->ComputeBlocking();
    }
    for (int round = 0; round < 8; ++round) {
      auto computation = backend.CreateComputation();
      std::vector<lczero::EvalResult> results(count);
      for (int i = 0; i < count; ++i)
        results[i].p.resize(legal[i].size());
      std::barrier start(writers);
      std::vector<std::thread> tasks;
      for (int tid = 0; tid < writers; ++tid)
        tasks.emplace_back([&, tid] {
          start.arrive_and_wait();
          for (int i = tid; i < count; i += writers) {
            computation->AddInput(input(i), results[i].AsPtr());
            if (i % 2 == 0)
              computation->AddInput(input(i), {}); // Concurrent prefetch too.
          }
        });
      for (auto &task : tasks)
        task.join();
      Require(computation->UsedBatchSize() == 96, "all requests preserved");
      computation->ComputeBlocking();
      for (int i = 0; i < count; ++i) {
        Require(std::abs(results[i].q - reference[i].q) <= 1e-3f,
                "parallel request value/position mismatch");
        Require(results[i].d == 0 && results[i].m == 0, "scalar NN output");
        for (size_t j = 0; j < legal[i].size(); ++j)
          Require(std::abs(results[i].p[j] - reference[i].p[j]) <= 1e-3f,
                  "parallel request policy/move mismatch");
      }
    }
    {
      auto computation = backend.CreateComputation();
      for (int i = 0; i < 160; ++i)
        computation->AddInput(input(0), {});
      bool overflow = false;
      try {
        computation->AddInput(input(0), {});
      } catch (const std::length_error &) {
        overflow = true;
      }
      Require(overflow && computation->UsedBatchSize() == 160,
              "real AddInput capacity overflow must throw without corruption");
      // Discarding an uncomputed computation must also return its storage.
    }
    {
      // Explicit classic minibatches can exceed a GPU's inference maximum.
      backend.SetCollectionBatchSize(384);
      const auto before = backend.GetInferenceStats();
      auto computation = backend.CreateComputation();
      for (int i = 0; i < 384; ++i)
        computation->AddInput(input(i % count), {});
      computation->ComputeBlocking();
      const auto stats = backend.GetInferenceStats();
      Require(stats.evaluations == before.evaluations + 384 &&
                  stats.batches == before.batches + 3,
              "large classic minibatch is safely split at GPU batch maximum");
    }
    backend.SetCollectionBatchSize(0);
    {
      auto computation = backend.CreateComputation();
      for (int i = 0; i < 160; ++i)
        computation->AddInput(input(0), {});
      bool overflow = false;
      try {
        computation->AddInput(input(0), {});
      } catch (const std::length_error &) {
        overflow = true;
      }
      Require(overflow, "shrinking the request pool restores the capacity guard");
    }
    std::cout << "backend concurrent AddInput passed; writers=8 rounds=8 "
                 "queued=96 capacity=160 gpus=" << gpus << std::endl;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
