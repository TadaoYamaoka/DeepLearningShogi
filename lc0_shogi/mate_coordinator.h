#pragma once
#include "dfpn.h"
#include "search/classic/search.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
namespace lczero::shogi {
// A pool independent of GPU workers. Proof identity is a path, never Node*.
class PvMatePool {
public:
  PvMatePool(int threads, int depth, int nodes);
  ~PvMatePool();
  void Start(classic::Search *search, const PositionHistory &root);
  void Stop();
  void Clear();

private:
  struct Worker {
    std::unique_ptr<::DfPn> dfpn;
    std::thread thread;
  };
  struct Proof {
    std::vector<Move> path;
    int plies;
    std::string identity;
  };
  void Run(size_t index);
  std::mutex mutex_;
  std::condition_variable changed_;
  bool terminate_ = false, running_ = false;
  size_t epoch_ = 0, active_ = 0;
  std::atomic<bool> stop_{true};
  classic::Search *search_ = nullptr;
  PositionHistory root_;
  std::string root_identity_;
  std::int64_t pending_retry_visits_ = -1;
  std::set<std::string> searched_;
  std::vector<Proof> pending_;
  std::vector<std::unique_ptr<Worker>> workers_;
};
} // namespace lczero::shogi
