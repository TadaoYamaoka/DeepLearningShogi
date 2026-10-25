#include "search_session.h"
#include "backend.h"
#include "classic_options.h"
#include "chess/gamestate.h"
#include "mate_coordinator.h"
#include "neural/shared_params.h"
#include "rules.h"
#include "search/classic/search.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <type_traits>
#ifdef MAKE_BOOK
extern bool use_interruption;
#endif
namespace {
using Classic = lczero::classic::Search;
using Params = lczero::classic::BaseSearchParams;
std::array<lczero::shogi::DeviceConfig, 8> devices;
std::unique_ptr<lczero::shogi::DlshogiBackend> backend;
std::unique_ptr<lczero::classic::NodeTree> tree;
std::unique_ptr<lczero::shogi::PvMatePool> mate_pool;
std::mutex active_mutex;
Classic *active_search = nullptr;
std::atomic<bool> stopped{false}, pondering{false};
std::condition_variable state_changed;
struct ResponseGate {
  std::mutex mutex;
  std::condition_variable changed;
  bool released = true, cancelled = false;
};
std::shared_ptr<ResponseGate> response_gate = std::make_shared<ResponseGate>();
LimitsType limits;
int time_limit = 0, minimum_time = 0, constant_playouts = 0, draw_ply = 0,
    multi_pv = 1, pv_interval = 500, eval_coef = 756;
int root_game_ply = 1, root_remaining_time = 0;
int task_workers = -1, concurrent_searchers = 1;
int search_threads = 0, minibatch_size = 0;
std::unique_ptr<lczero::shogi::ClassicOptions> classic_options;
bool may_extend = false;
unsigned int node_limit = 10000000;
uint32_t last_root_visits_before_search = 0;
float resign_threshold = .01f, draw_black = .5f, draw_white = .5f,
      temperature = 1.74f;
bool reuse = true, backend_dirty = true, cache_dirty = false, debug = false;
bool pondering_mode = false;
size_t cache_capacity = 100000;
std::vector<SearchCandidate> last_candidates;
#ifdef MULTI_PONDER
std::atomic<bool> multi_preparing{false};
#endif
struct RandomSettings {
  int ply = 0;
  float temperature = 10, drop = 0, cutoff = .015f, cutoff_drop = 0,
        probability = 1, limit = 1;
};
RandomSettings random_settings, random2_settings;
std::mt19937_64 random_generator(std::random_device{}());
void EnsureBackend() {
  if (!backend || backend_dirty) {
    if(mate_pool)mate_pool->Clear();
    backend = std::make_unique<lczero::shogi::DlshogiBackend>(
        devices, cache_capacity, temperature, minibatch_size);
    backend_dirty = false;
    tree = std::make_unique<lczero::classic::NodeTree>();
  } else {
    backend->SetCollectionBatchSize(minibatch_size);
    if (cache_dirty) {
      backend->ResetCache(cache_capacity);
      NewGame();
    }
  }
  cache_dirty = false;
}
int ToCp(float q) {
  const float p = (q + 1) / 2;
  return p <= 0   ? -30000
         : p >= 1 ? 30000
                  : static_cast<int>(std::log(p / (1 - p)) * eval_coef);
}
class Responder : public lczero::UciResponder {
public:
  explicit Responder(const lczero::PositionHistory &root) : root_(root) {}
  void OutputBestMove(lczero::BestMoveInfo *) override {
  } // main.cpp owns bestmove.
  void OutputThinkingInfo(std::vector<lczero::ThinkingInfo> *infos) override {
    pending_ = *infos;
    Emit(*infos, false);
  }
  void Final(const std::vector<Classic::RootCandidate> &candidates) {
    std::vector<lczero::ThinkingInfo> infos;
    for (size_t i = 0;
         i < candidates.size() && i < static_cast<size_t>(multi_pv); ++i) {
      auto info = pending_.empty() ? lczero::ThinkingInfo{} : pending_.front();
      info.scalar_q = candidates[i].q;
      info.pv = candidates[i].pv;
      info.multipv = static_cast<int>(i) + 1;
      info.mate = candidates[i].decisive && std::abs(candidates[i].q) > .9999f
                      ? std::optional<int>(candidates[i].q < 0 ? -1 : 1)
                      : std::nullopt;
      infos.push_back(std::move(info));
    }
    Emit(infos, true);
  }

private:
  void Emit(const std::vector<lczero::ThinkingInfo> &infos, bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && (pv_interval <= 0 ||
                   (last_.time_since_epoch().count() != 0 &&
                    now - last_ < std::chrono::milliseconds(pv_interval))))
      return;
    last_ = now;
    for (const auto &info : infos) {
      auto pv = info.pv;
      bool checkmate = false;
      if (info.mate) {
        lczero::PositionHistory cursor(root_);
        const auto ended_by_rule = [](const ::Position &position) {
          return (draw_ply > 0 && position.gamePly() > draw_ply) ||
                 position.isDraw(16) != NotRepetition;
        };
        bool legal_line = true;
        for (auto move : pv) {
          const auto legal = cursor.Last().GetBoard().GenerateLegalMoves();
          if (std::find(legal.begin(), legal.end(), move) == legal.end()) {
            legal_line = false;
            break;
          }
          cursor.Append(move);
          if (ended_by_rule(cursor.Last().GetBoard().Shogi())) {
            legal_line = false;
            break;
          }
        }
        if (legal_line) {
          auto &tail =
              const_cast<::Position &>(cursor.Last().GetBoard().Shogi());
          checkmate = tail.inCheck() && ::MoveList<Legal>(tail).size() == 0;
          if (!checkmate && !tail.inCheck()) {
            const auto mate = tail.mateMoveIn1Ply();
            if (mate) {
              cursor.Append(lczero::Move(mate));
              const auto &end = cursor.Last().GetBoard().Shogi();
              if (!ended_by_rule(end) && end.inCheck() &&
                  ::MoveList<Legal>(end).size() == 0) {
                pv.emplace_back(mate);
                checkmate = true;
              }
            }
          }
        }
      }
      std::ostringstream line;
      line << "info depth " << info.depth << " nodes " << info.nodes;
      if (info.nps >= 0)
        line << " nps " << info.nps;
      line << " time " << info.time;
      if (multi_pv > 1)
        line << " multipv " << info.multipv;
      if (checkmate)
        line << " score mate "
             << (*info.mate < 0 ? -1 : 1) * static_cast<int>(pv.size());
      else if (info.scalar_q)
        line << " score cp " << ToCp(*info.scalar_q);
      line << " pv";
      for (auto move : pv)
        line << ' ' << move.ToString();
      std::cout << line.str() << std::endl;
    }
  }
  std::vector<lczero::ThinkingInfo> pending_;
  lczero::PositionHistory root_;
  std::chrono::steady_clock::time_point last_{};
};
class Stopper : public lczero::classic::SearchStopper {
public:
  bool ShouldStop(const lczero::classic::IterationStats &stats,
                  lczero::classic::StoppersHints *hints) override {
    if (stopped.load())
      return true;
    if (stats.total_nodes >= node_limit)
      return true;
    if (pondering.load() || limits.infinite)
      return false;
    const int64_t count =
        constant_playouts > 0 ? constant_playouts : limits.nodes;
    if (count > 0) {
      if (stats.nodes_since_movestart >= count)
        return true;
#ifdef MAKE_BOOK
      if (use_interruption && stats.edge_n.size() > 1) {
        uint32_t best = 0, second = 0;
        for (auto visits : stats.edge_n) {
          if (visits > best) {
            second = best;
            best = visits;
          } else if (visits > second) {
            second = visits;
          }
        }
        return static_cast<int64_t>(best) - second >
               count - stats.nodes_since_movestart;
      }
#endif
      return false;
    }
    hints->UpdateEstimatedRemainingTimeMs(
        std::max<int64_t>(0, time_limit - stats.time_since_movestart));
    const auto elapsed = stats.time_since_movestart;
    if (elapsed * 10 < time_limit || elapsed < minimum_time)
      return false;
    if (stats.win_found)
      return true;
    size_t best = 0, second = 0;
    for (size_t i = 0; i < stats.edge_n.size(); ++i) {
      if (stats.edge_n[i] > stats.edge_n[best]) {
        second = best;
        best = i;
      } else if (i != best &&
                 (second == best || stats.edge_n[i] > stats.edge_n[second]))
        second = i;
    }
    if (stats.edge_n.empty())
      return elapsed >= time_limit;
    const auto remaining = stats.nodes_since_movestart *
                           (time_limit - elapsed) /
                           std::max<int64_t>(1, elapsed);
    if (static_cast<int64_t>(stats.edge_n[best]) - stats.edge_n[second] <=
        remaining)
      return false;
    if (may_extend && root_game_ply > 20 &&
        root_remaining_time > time_limit * 2 &&
        (stats.edge_n[best] < stats.edge_n[second] * 1.5 ||
         stats.edge_q[best] < stats.edge_q[second])) {
      time_limit *= 2;
      may_extend = false;
      std::cout << "info string ExtendTime" << std::endl;
      return false;
    }
    return true;
  }
};
void Configure(lczero::OptionsDict *options) {
  options->Set<std::string>(Params::kContemptModeId, "disable");
  options->Set<std::string>(Params::kContemptId, "0");
  options->Set<float>(Params::kWDLDrawRateTargetId, .58f);
  options->Set<float>(Params::kWDLDrawRateReferenceId, .58f);
  options->Set<float>(Params::kWDLCalibrationEloId, 0);
  options->Set<float>(Params::kMovesLeftMaxEffectId, 0);
  options->Set<float>(Params::kTemperatureId, 0);
  options->Set<float>(Params::kNoiseEpsilonId, 0);
  options->Set<bool>(Params::kTwoFoldDrawsId, false);
  options->Set<int>(Params::kTaskWorkersPerSearchWorkerId, task_workers);
  options->Set<int>(Params::kMaxConcurrentSearchersId, concurrent_searchers);
  options->Set<int>(Params::kMiniBatchSizeId, minibatch_size);
  options->Set<int>(Params::kMultiPvId, multi_pv);
  options->Set<std::string>(Params::kScoreTypeId, "Q");

}
} // namespace
void InitializeSearch(unsigned int limit) {
  node_limit = limit;
  tree = std::make_unique<lczero::classic::NodeTree>();
}
void CancelSearchResponse() {
  std::lock_guard<std::mutex> lock(response_gate->mutex);
  response_gate->cancelled = true;
  response_gate->changed.notify_all();
}
void SetSearchResponseBlocked(bool blocked) {
  CancelSearchResponse();
  response_gate = std::make_shared<ResponseGate>();
  response_gate->released = !blocked;
}
void ReleaseSearchResponse() {
  std::lock_guard<std::mutex> lock(response_gate->mutex);
  response_gate->released = true;
  response_gate->changed.notify_all();
}
std::function<bool()> CaptureSearchResponseWaiter() {
  auto gate = response_gate;
  return [gate] {
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->changed.wait(lock, [&] { return gate->released || gate->cancelled; });
    return !gate->cancelled;
  };
}
void InitSearchStop() { stopped = false; }
void StopSearch() {
  std::lock_guard<std::mutex> lock(active_mutex);
  stopped = true;
  state_changed.notify_all();
  if (active_search)
    active_search->Stop();
}
bool IsSearchStopped() { return stopped; }
void TerminateSearch() { StopSearch(); }
void FinalizeSearch() {
  mate_pool.reset();
  backend.reset();
  tree.reset();
}
void SetPondering(bool value) {
  std::lock_guard<std::mutex> lock(active_mutex);
  pondering = value;
  state_changed.notify_all();
}
void SetPonderingMode(bool value) { pondering_mode = value; }
void SetInferenceSlots(const int slots[8], const int batches[8]) {
  for (int i = 0; i < 8; ++i) {
    const int batch = batches[i] > 0 ? batches[i] : batches[0];
    backend_dirty |= devices[i].slots != slots[i] || devices[i].batch != batch;
    devices[i].slots = slots[i];
    devices[i].batch = batch;
  }
}
void SetSearchSettings(int threads, int minibatch) {
  search_threads = threads;
  minibatch_size = minibatch;
}
void SetSearchConcurrency(int helpers, int searchers) {
  task_workers = helpers;
  concurrent_searchers = searchers;
}
void RegisterClassicSearchOptions(OptionsMap& options) {
  classic_options = std::make_unique<lczero::shogi::ClassicOptions>();
  classic_options->Register(options);
}
void ApplyClassicSearchOptions(const OptionsMap& options) {
  classic_options->Snapshot(options);
  set_softmax_temperature(classic_options->PolicyTemperature());
}
void SetModelPath(const std::string paths[8]) {
  for (int i = 0; i < 8; ++i) {
    const auto &model = paths[i].empty() ? paths[0] : paths[i];
    backend_dirty |= devices[i].model != model;
    devices[i].model = model;
  }
}
void SetDnnCacheSize(size_t capacity) {
  cache_dirty |= cache_capacity != capacity;
  cache_capacity = capacity;
}
void InvalidateNeuralCache() { cache_dirty = true; }
void set_softmax_temperature(float value) {
  backend_dirty |= temperature != value;
  temperature = value;
}
void SetResignThreshold(int value) { resign_threshold = value / 1000.0f; }
void SetDrawValue(int black, int white) {
  draw_black = black / 1000.0f;
  draw_white = white / 1000.0f;
}
void SetDrawPly(int ply) { draw_ply = ply; }
void SetMultiPV(int count) { multi_pv = count; }
void SetPvInterval(int value) { pv_interval = value; }
void SetEvalCoef(int value) { eval_coef = value; }
void SetReuseSubtree(bool value) { reuse = value; }
void SetDebugMessageMode(bool value) { debug = value; }
void SetConstPlayout(int count) { constant_playouts = count; }
void SetLimits(const LimitsType &value) {
  limits = value;
  time_limit = value.moveTime;
  minimum_time = value.moveTime;
  pondering = value.ponder;
  may_extend = false;
}
void SetLimits(const ::Position *pos, const LimitsType &value) {
  SetLimits(value);
  int divisor = 14 + std::max(0, 30 - pos->gamePly());
  if (draw_ply > 0 && draw_ply >= pos->gamePly())
    divisor = std::min(divisor, draw_ply - pos->gamePly() + 1);
  time_limit = std::max(value.moveTime, value.time[pos->turn()] / divisor +
                                            value.inc[pos->turn()]);
  root_game_ply = pos->gamePly();
  root_remaining_time = value.time[pos->turn()];
  may_extend = time_limit > minimum_time && value.nodes == 0;
}
int GetTimeLimit() { return time_limit; }
void NewGame() {
  if(mate_pool)mate_pool->Clear();
  tree = std::make_unique<lczero::classic::NodeTree>();
}
void GameOver() {}
void SetRandomMove(int ply, int temperature, int drop, int cutoff,
                   int cutoff_drop) {
  random_settings = {ply,
                     temperature / 1000.0f,
                     drop / 1000.0f,
                     cutoff / 1000.0f,
                     cutoff_drop / 1000.0f,
                     1,
                     1};
}
void SetRandomMove2(int ply, int probability, int temperature, int cutoff,
                    int limit) {
  random2_settings = {ply,
                      temperature / 1000.0f,
                      0,
                      cutoff / 1000.0f,
                      0,
                      probability / 1000.0f,
                      limit / 1000.0f};
}
void SetPvMateSearch(int threads, int depth, int nodes) {
  mate_pool =
      threads > 0
          ? std::make_unique<lczero::shogi::PvMatePool>(threads, depth, nodes)
          : nullptr;
}
std::vector<SearchCandidate> GetSearchCandidates() { return last_candidates; }
void DumpSearchSettings() {
  lczero::OptionsParser parser;
  lczero::SharedBackendParams::Populate(&parser);
  lczero::classic::SearchParams::Populate(&parser);
  classic_options->ApplyTo(&parser);
  auto* values = parser.GetMutableOptions();
  Configure(values);
  const Params effective(*values);
  std::ostringstream out;
  out << std::setprecision(9) << std::boolalpha << "info string searchsettings {";
  bool first = true;
  const auto add = [&](const std::string& key, const auto& value) {
    if (!first) out << ',';
    first = false;
    out << std::quoted(key) << ':';
    if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>)
      out << std::quoted(value);
    else
      out << value;
  };
  add("CPuct", values->Get<float>(Params::kCpuctId));
  add("CPuctBase", values->Get<float>(Params::kCpuctBaseId));
  add("CPuctFactor", values->Get<float>(Params::kCpuctFactorId));
  add("CPuctAtRoot", values->Get<float>(Params::kCpuctAtRootId));
  add("CPuctBaseAtRoot", values->Get<float>(Params::kCpuctBaseAtRootId));
  add("CPuctFactorAtRoot", values->Get<float>(Params::kCpuctFactorAtRootId));
  add("FpuValue", values->Get<float>(Params::kFpuValueId));
  add("FpuValueAtRoot", values->Get<float>(Params::kFpuValueAtRootId));
  add("RootHasOwnCpuctParams", values->Get<bool>(Params::kRootHasOwnCpuctParamsId));
  add("FpuStrategy", values->Get<std::string>(Params::kFpuStrategyId));
  add("FpuStrategyAtRoot", values->Get<std::string>(Params::kFpuStrategyAtRootId));
  add("PolicyTemperature", temperature);
  add("ClassicPolicyTemperature", effective.GetPolicySoftmaxTemp());
  add("EffectiveCPuctAtRoot", effective.GetCpuct(true));
  add("EffectiveFpuValueAtRoot", effective.GetFpuValue(true));
  add("ReuseSubtree", reuse);
  add("Threads", search_threads);
  add("TaskWorkers", task_workers);
  add("MinibatchSize", minibatch_size);
  add("MaxConcurrentSearchers", concurrent_searchers);
  add("DNN_Cache_Size", cache_capacity);
  add("Resign_Threshold", resign_threshold);
  add("Draw_Ply", draw_ply);
  add("Draw_Value_Black", draw_black);
  add("Draw_Value_White", draw_white);
  add("MultiPV", multi_pv);
  add("PV_Interval", pv_interval);
  add("Eval_Coef", eval_coef);
  add("DebugMessage", debug);
  add("Const_Playout", constant_playouts);
  add("PonderingMode", pondering_mode);
  add("LastRootVisitsBeforeSearch", last_root_visits_before_search);
  add("Random_Ply", random_settings.ply);
  add("Random_Temperature", random_settings.temperature);
  add("Random_Temperature_Drop", random_settings.drop);
  add("Random_Cutoff", random_settings.cutoff);
  add("Random_Cutoff_Drop", random_settings.cutoff_drop);
  add("Random2_Ply", random2_settings.ply);
  add("Random2_Probability", random2_settings.probability);
  add("Random2_Temperature", random2_settings.temperature);
  add("Random2_Cutoff", random2_settings.cutoff);
  add("Random2_Value_Limit", random2_settings.limit);
  for (int i = 0; i < max_gpu; ++i) {
    const std::string suffix = i ? std::to_string(i + 1) : "";
    add("DNN_Slots" + suffix, devices[i].slots);
    add("DNN_Batch_Size" + suffix, devices[i].batch);
    add("DNN_Model" + suffix, devices[i].model);
  }
  std::cout << out.str() << '}' << std::endl;
}
void DumpNeuralStats() {
  const auto stats = backend ? backend->GetInferenceStats()
                             : lczero::shogi::DlshogiBackend::InferenceStats{};
  std::cout << "info string nnstats evaluations " << stats.evaluations
            << " batches " << stats.batches << std::endl;
}
void DumpNeuralEvaluation(const ::Position &position) {
  EnsureBackend();
  const lczero::Position current(position.toSFEN());
  const auto legal = current.GetBoard().GenerateLegalMoves();
  const lczero::EvalPosition request{{&current, 1}, legal};
  const auto result = backend->EvaluateBatch({&request, 1}).front();
  std::ostringstream line;
  line << std::setprecision(9) << "info string nnprobe value "
       << (result.q + 1) / 2;
  for (size_t i = 0; i < legal.size(); ++i)
    line << ' ' << legal[i].ToString() << ':' << result.p[i];
  std::cout << line.str() << std::endl;
}
::Move SearchGenmove(::Position *position, Key,
                     const std::vector<::Move> &moves, ::Move &ponder) {
  EnsureBackend();
  if (!tree || !reuse) {
    if(mate_pool)mate_pool->Clear();
    tree = std::make_unique<lczero::classic::NodeTree>();
  }
  ::Position initial(*position);
  for (auto move = moves.rbegin(); move != moves.rend(); ++move)
    initial.undoMove(*move);
  lczero::GameState game{lczero::Position(initial.toSFEN()), {}};
  for (auto move : moves)
    game.moves.emplace_back(move);
  // A rewind or a different branch can discard the previously searched head.
  // Its path identities must become eligible for dfpn in the replacement tree.
  if (!tree->ResetToPosition(game) && mate_pool)
    mate_pool->Clear();
  last_root_visits_before_search = tree->GetCurrentHead()->GetN();
  lczero::shogi::ConfigureRules(draw_ply, draw_black, draw_white);
  lczero::OptionsParser parser;
  lczero::SharedBackendParams::Populate(&parser);
  lczero::classic::SearchParams::Populate(&parser);
  auto *options = parser.GetMutableOptions();
  classic_options->ApplyTo(&parser);
  Configure(options);
  const auto start = std::chrono::steady_clock::now() -
                     std::chrono::milliseconds(limits.startTime.elapsed());
  auto responder = std::make_unique<Responder>(tree->GetPositionHistory());
  auto *output = responder.get();
  Classic search(*tree, backend.get(), std::move(responder), {}, start,
                 std::make_unique<Stopper>(), false, false, *options, nullptr);
  {
    std::lock_guard<std::mutex> lock(active_mutex);
    active_search = &search;
    if (stopped)
      search.Stop();
  }
  search.StartThreads(search_threads); // 0: upstream recommended workers + 1.
#ifdef MULTI_PONDER
  if (pondering)
    PrepareMultiPonder();
#endif
  if (mate_pool)
    mate_pool->Start(&search, tree->GetPositionHistory());
  search.Wait();
  if (mate_pool)
    mate_pool->Stop();
  // Even an internally completed ponder/infinite search must hold its result.
  {
    std::unique_lock<std::mutex> lock(active_mutex);
    state_changed.wait(
        lock, [] { return stopped || (!pondering && !limits.infinite); });
    active_search = nullptr;
  }
  last_candidates.clear();
  const auto snapshot = search.GetRootSnapshot();
  output->Final(snapshot);
  for (const auto &edge : snapshot) {
    auto move = ::move16toMove(::Move(edge.move.raw_data()), *position);
    last_candidates.push_back({move, static_cast<int>(edge.visits),
                               (edge.q + 1) * .5 * edge.visits, edge.prior});
  }
  ponder = ::Move::moveNone();
  if (snapshot.empty())
    return ::Move::moveNone();
  size_t selected = 0;
  const RandomSettings *selection = nullptr;
  const float best_win = (snapshot.front().q + 1) / 2;
  if (position->gamePly() <= random2_settings.ply &&
      best_win < random2_settings.limit &&
      std::uniform_real_distribution<float>(0, 1)(random_generator) <
          random2_settings.probability)
    selection = &random2_settings;
  else if (position->gamePly() <= random_settings.ply)
    selection = &random_settings;
  if (selection) {
    const int step = (position->gamePly() - 1) / 2;
    const float cutoff =
        std::max(0.0f, selection->cutoff - selection->cutoff_drop * step);
    const float temp =
        std::max(.1f, selection->temperature - selection->drop * step);
    std::vector<double> weights;
    for (const auto &candidate : snapshot) {
      if (candidate.visits == 0 || (candidate.q + 1) / 2 < best_win - cutoff)
        break;
      weights.push_back(std::pow(candidate.visits, 1.0 / temp));
    }
    if (!weights.empty())
      selected = std::discrete_distribution<size_t>(
          weights.begin(), weights.end())(random_generator);
  }
  const auto &best = snapshot[selected];
  auto result = ::move16toMove(::Move(best.move.raw_data()), *position);
  if (pondering_mode && best.pv.size() > 1) {
    ::Position next(*position);
    ::StateInfo state;
    next.doMove(result, state);
    ponder = ::move16toMove(::Move(best.pv[1].raw_data()), next);
  }
  if(debug)std::cout<<"info string classic visits "<<search.GetTotalPlayouts()<<" candidates "<<snapshot.size()<<" node_bytes "<<sizeof(lczero::classic::Node)<<" edge_bytes "<<sizeof(lczero::classic::Edge)<<std::endl;
  if ((best.q + 1) / 2 < resign_threshold)
    return ::Move::moveNone();
  return result;
}
#ifdef MULTI_PONDER
void ResetMultiPonder() { multi_preparing = false; }
void PrepareMultiPonder() {
  multi_preparing = true;
  state_changed.notify_all();
}
void WaitPrepareMultiPonder() {
  std::unique_lock<std::mutex> lock(active_mutex);
  while (!stopped) {
    if (multi_preparing && pondering && active_search &&
        active_search->GetTotalPlayouts() > 0)
      return;
    state_changed.wait_for(lock, std::chrono::milliseconds(1));
  }
}
bool GetMultiPonderMove(std::vector<::Move> &moves, int count) {
  std::lock_guard<std::mutex> lock(active_mutex);
  if (!active_search || !pondering || stopped)
    return false;
  const auto candidates = active_search->GetRootSnapshot();
  if (candidates.size() < 2)
    return false;
  const auto &position = tree->HeadPosition().GetBoard().Shogi();
  for (const auto &candidate : candidates) {
    if (static_cast<int>(moves.size()) >= count)
      break;
    moves.push_back(
        ::move16toMove(::Move(candidate.move.raw_data()), position));
  }
  return !moves.empty();
}
#endif
