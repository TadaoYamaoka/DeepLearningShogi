// dlshogi inference adapter for lc0 classic. GPL-3.0-or-later.
#pragma once
#include "PolicyValueCache.h"
#include "neural/backend.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
namespace lczero::shogi {
template <class T> class RequestPool;
struct DeviceConfig {
  std::string model;
  int slots = 0;
  int batch = 128;
};
class DlshogiBackend : public Backend {
public:
  DlshogiBackend(const std::array<DeviceConfig, 8> &devices, size_t cache_size,
                 float temperature, int collection_batch = 0);
  ~DlshogiBackend();
  BackendAttributes GetAttributes() const override;
  std::unique_ptr<BackendComputation> CreateComputation() override;
  // Call only after every computation has been destroyed (isready joins workers).
  void SetCollectionBatchSize(int collection_batch);
  void ResetCache(size_t capacity);
  struct InferenceStats {
    uint64_t evaluations = 0, batches = 0;
  };
  // Query after workers have drained for a consistent pair of counters.
  InferenceStats GetInferenceStats() const;

private:
  struct Device;
  struct Request;
  class Computation;
  std::vector<std::unique_ptr<Device>> devices_;
  std::vector<std::pair<int, int>> slots_;
  std::deque<size_t> available_slots_;
  std::unique_ptr<RequestPool<Request>> request_pool_;
  int request_capacity_ = 0;
  std::mutex slots_mutex_;
  std::condition_variable slots_available_;
  std::atomic<uint64_t> evaluations_{0}, batches_{0};
  PolicyValueCache cache_;
  float temperature_;
  int max_batch_ = 1;
  int recommended_batch_ = 0;
};
} // namespace lczero::shogi
