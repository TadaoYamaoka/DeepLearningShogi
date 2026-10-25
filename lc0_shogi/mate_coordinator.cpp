#include "mate_coordinator.h"
#include <chrono>
namespace lczero::shogi {
PvMatePool::PvMatePool(int threads, int depth, int nodes) {
  for (int i = 0; i < threads; ++i) {
    auto worker = std::make_unique<Worker>();
    worker->dfpn = std::make_unique<::DfPn>();
    worker->dfpn->init();
    worker->dfpn->set_maxdepth(depth);
    worker->dfpn->set_max_search_node(nodes);
    workers_.push_back(std::move(worker));
  }
  for (size_t i = 0; i < workers_.size(); ++i)
    workers_[i]->thread = std::thread([this, i] { Run(i); });
}
PvMatePool::~PvMatePool() {
  Stop();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    terminate_ = true;
    changed_.notify_all();
  }
  for (auto &worker : workers_)
    worker->thread.join();
}
void PvMatePool::Start(classic::Search *search, const PositionHistory &root) {
  Stop();
  std::lock_guard<std::mutex> lock(mutex_);
  search_ = search;
  root_ = root;
  root_identity_ = root_.HistoryIdentity();
  // Retain only descendants of the new root, including the delimiter so a
  // numeric move prefix cannot match a different move. Never retain Node*.
  const auto prefix = root_identity_ + ':';
  for (auto entry = searched_.begin(); entry != searched_.end();) {
    if (!entry->starts_with(prefix))
      entry = searched_.erase(entry);
    else
      ++entry;
  }
  pending_retry_visits_ = -1;
  pending_.clear();
  stop_ = false;
  for (auto &worker : workers_)
    worker->dfpn->dfpn_stop(false);
  active_ = workers_.size();
  running_ = true;
  ++epoch_;
  changed_.notify_all();
}
void PvMatePool::Stop() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!running_)
    return;
  stop_ = true;
  for (auto &worker : workers_)
    worker->dfpn->dfpn_stop(true);
  changed_.wait(lock, [this] { return active_ == 0; });
  // Search workers have already drained when called by SearchSession.
  for (const auto &proof : pending_)
    if (!search_->ApplyProof(proof.path, GameResult::BLACK_WON, proof.plies))
      searched_.erase(proof.identity);
  pending_.clear();
  running_ = false;
}
void PvMatePool::Clear() {
  Stop();
  std::lock_guard<std::mutex> lock(mutex_);
  searched_.clear();
  pending_.clear();
}
void PvMatePool::Run(size_t index) {
  size_t seen_epoch = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    changed_.wait(lock, [&] { return terminate_ || epoch_ != seen_epoch; });
    if (terminate_)
      return;
    seen_epoch = epoch_;
    PositionHistory cursor(root_);
    auto *search = search_;
    const auto root_identity = root_identity_;
    lock.unlock();
    while (!stop_) {
      // N==0 targets cannot become publishable without a completed backup.
      // One pool-wide attempt per observed progress avoids exclusive node-lock
      // acquisition on every idle PV iteration or once per PV worker.
      const auto visits = search->GetTotalPlayouts();
      lock.lock();
      if (!pending_.empty() && visits != pending_retry_visits_) {
        pending_retry_visits_ = visits;
        for (auto proof = pending_.begin(); proof != pending_.end();) {
          if (search->ApplyProof(proof->path, GameResult::BLACK_WON,
                                 proof->plies))
            proof = pending_.erase(proof);
          else
            ++proof;
        }
      }
      lock.unlock();
      const auto pv = search->GetPrincipalVariation();
      cursor = root_;
      std::vector<Move> path;
      auto identity = root_identity;
      bool worked = false;
      for (auto move : pv) {
        cursor.Append(move);
        path.push_back(move);
        identity += ':';
        identity += std::to_string(move.raw_data());
        {
          std::lock_guard<std::mutex> guard(mutex_);
          if (!searched_.insert(identity).second)
            continue;
        }
        worked = true;
        auto &position =
            const_cast<::Position &>(cursor.Last().GetBoard().Shogi());
        auto &dfpn = *workers_[index]->dfpn;
        if (dfpn.dfpn(position)) {
          auto [text, plies, mate_move] = dfpn.get_pv(position);
          if (!search->ApplyProof(path, GameResult::BLACK_WON, plies)) {
            std::lock_guard<std::mutex> guard(mutex_);
            pending_.push_back({path, plies, identity});
          }
        } else if (stop_) {
          std::lock_guard<std::mutex> guard(mutex_);
          searched_.erase(identity);
        }
        break; // Always re-read the latest PV after doing one dfpn search.
      }
      if (!worked)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    lock.lock();
    --active_;
    changed_.notify_all();
  }
}
} // namespace lczero::shogi
