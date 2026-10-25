// Concurrent classic AddInput storage. GPL-3.0-or-later.
#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "utils/atomic_vector.h"

namespace lczero::shogi {
template <class T> class RequestPool {
public:
  class Batch {
  public:
    explicit Batch(size_t capacity) : entries_(capacity) {}
    void Append(T &&value) {
      // AtomicVector's own bound check is an assert. Reserve a separate ticket
      // BEFORE touching it, so concurrent overflow is also safe in Release.
      static_assert(std::is_nothrow_move_constructible_v<T>);
      size_t reserved = reserved_.load(std::memory_order_relaxed);
      do {
        if (reserved >= entries_.capacity())
          throw std::length_error("dlshogi NN request capacity exceeded");
      } while (!reserved_.compare_exchange_weak(
          reserved, reserved + 1, std::memory_order_relaxed));
      entries_.emplace_back(std::move(value));
    }
    size_t size() const { return entries_.size(); }
    T &operator[](size_t i) { return entries_[i]; }
    // Only after all AddInput tasks have joined and inference has finished.
    void Clear() {
      entries_.clear();
      reserved_.store(0, std::memory_order_relaxed);
    }

  private:
    AtomicVector<T> entries_;
    std::atomic<size_t> reserved_{0};
  };

  explicit RequestPool(size_t capacity) : capacity_(capacity) {}
  std::unique_ptr<Batch> Borrow() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!available_.empty()) {
        auto result = std::move(available_.back());
        available_.pop_back();
        return result;
      }
    }
    // ONNX features are large: retain this allocation across computations.
    return std::make_unique<Batch>(capacity_);
  }
  void Return(std::unique_ptr<Batch> batch) noexcept {
    batch->Clear();
    std::lock_guard<std::mutex> lock(mutex_);
    // Destruction must not throw if the pool's small pointer array cannot grow.
    try {
      available_.push_back(std::move(batch));
    } catch (const std::bad_alloc &) {
    }
  }

private:
  size_t capacity_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<Batch>> available_;
};
} // namespace lczero::shogi
