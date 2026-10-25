#include "chess/gamestate.h"
#include "init.hpp"
#include "usi.hpp"
#include "classic_options.h"
#include <sstream>
#include "neural/backend.h"
#include "neural/shared_params.h"
#include "rules.h"
#include "request_pool.h"
#include "search/classic/search.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>
extern std::ostream& operator << (std::ostream&, const OptionsMap&);
static void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
class ConstantBackend : public lczero::Backend {
public:
  std::function<void()> before_compute;
  size_t cache_queries = 0;
  std::optional<lczero::EvalResult>
  GetCachedEvaluation(const lczero::EvalPosition &) override {
    ++cache_queries;
    return std::nullopt;
  }
  lczero::BackendAttributes GetAttributes() const override {
    return {false, false, true, 1, 1, 1};
  }
  class Computation : public lczero::BackendComputation {
  public:
    explicit Computation(ConstantBackend *owner) : owner_(owner) {}
    size_t UsedBatchSize() const override { return queued_; }
    AddInputResult AddInput(const lczero::EvalPosition &position,
                            lczero::EvalResultPtr result) override {
      if (result.q)
        *result.q = 0.6f;
      if (result.d)
        *result.d = 0;
      if (result.m)
        *result.m = 0;
      for (auto &p : result.p)
        p = 1.0f / position.legal_moves.size();
      if (owner_->before_compute) {
        ++queued_;
        return ENQUEUED_FOR_EVAL;
      }
      return FETCHED_IMMEDIATELY;
    }
    void ComputeBlocking() override {
      if (owner_->before_compute)
        owner_->before_compute();
    }

  private:
    ConstantBackend *owner_;
    size_t queued_ = 0;
  };
  std::unique_ptr<lczero::BackendComputation> CreateComputation() override {
    return std::make_unique<Computation>(this);
  }
};
class OneVisit : public lczero::classic::SearchStopper {
public:
  bool ShouldStop(const lczero::classic::IterationStats &stats,
                  lczero::classic::StoppersHints *) override {
    return stats.nodes_since_movestart >= 1;
  }
};
class Sink : public lczero::UciResponder {
public:
  void OutputBestMove(lczero::BestMoveInfo *) override {}
  void OutputThinkingInfo(std::vector<lczero::ThinkingInfo> *) override {}
};
int main() {
  try {
    // Integer USI options decode to lc0 units and reject fractional values;
    // combos and model-ini logging must also work in assertion-enabled builds.
    {
      OptionsMap advertised;
      advertised.init(nullptr);
      lczero::shogi::ClassicOptions bridge;
      bridge.Register(advertised);
      auto& value = advertised.at("FpuValue");
      value = std::string("-1000");
      for (const auto* invalid : {"nan", "inf", "-inf", "100001", "-100001", "540junk", "0.54", "", "1e3", "1e999", "9223372036854775808"}) {
        value = std::string(invalid);
        Require(std::string(value) == "-1000", "FPU rejects invalid/out-of-range values");
      }
      for (const auto* valid : {"-100000", "100000", "0", "540", "10"}) {
        value = std::string(valid);
        Require(std::string(value) == valid, "FPU retains valid integer presentation");
      }
      auto& strategy = advertised.at("FpuStrategy");
      strategy = std::string("absolute");
      strategy = std::string("same");
      Require(std::string(strategy) == "absolute", "same is root-only");
      std::ostringstream printed;
      printed << strategy << ' ' << value;
      Require(printed.str() == "absolute 10", "model-ini logs string/combo values");
      for (const auto* retired : {"C_init", "C_base", "C_init_root", "C_base_root", "Softmax_Temperature", "C_fpu_reduction", "C_fpu_reduction_root"})
        Require(!advertised.count(retired), "legacy search names retired");
      std::ostringstream listing;
      listing << advertised;
      Require(listing.str().find("option name FpuStrategyAtRoot type combo default reduction var reduction var absolute var same") != std::string::npos,
              "root FPU combo includes same");
      // Validate upstream limits, and snapshot only at isready. Main-thread
      // setoption must not alter the parameters of an already running search.
      advertised.at("CPuct") = std::string("3250");
      advertised.at("CPuctBase") = std::string("0");
      Require(std::stof(std::string(advertised.at("CPuctBase"))) == 28288,
              "upstream CPuctBase minimum is enforced");
      advertised.at("CPuctFactor") = std::string("4500");
      advertised.at("CPuctAtRoot") = std::string("7000");
      advertised.at("RootHasOwnCpuctParams") = std::string("false");
      advertised.at("PolicyTemperature") = std::string("1600");
      bridge.Snapshot(advertised);
      advertised.at("CPuct") = std::string("99000");
      lczero::OptionsParser parser;
      lczero::SharedBackendParams::Populate(&parser);
      lczero::classic::SearchParams::Populate(&parser);
      bridge.ApplyTo(&parser);
      const lczero::classic::SearchParams params(parser.GetOptionsDict());
      Require(params.GetCpuct(false) == 3.25f && params.GetCpuct(true) == 3.25f &&
                  params.GetCpuctFactor(false) == 4.5f,
              "1000-based units, root inheritance and immutable ready snapshot");
      Require(bridge.PolicyTemperature() == 1.6f && params.GetPolicySoftmaxTemp() == 1,
              "policy temperature applied once by dlshogi backend");
      advertised.at("RootHasOwnCpuctParams") = std::string("true");
      advertised.at("CPuctFactorAtRoot") = std::string("5000");
      advertised.at("CPuctBaseAtRoot") = std::string("12345");
      bridge.Snapshot(advertised);
      bridge.ApplyTo(&parser);
      const lczero::classic::SearchParams own(parser.GetOptionsDict());
      Require(own.GetCpuct(true) == 7 && own.GetCpuctFactor(true) == 5 &&
                  own.GetCpuctBase(true) == 12345 && own.GetCpuct(false) == 99,
              "independent root PUCT values decode exactly once");
    }
    {
      using P = lczero::classic::BaseSearchParams;
      lczero::OptionsParser parser;
      lczero::SharedBackendParams::Populate(&parser);
      lczero::classic::SearchParams::Populate(&parser);
      auto* values = parser.GetMutableOptions();
      values->Set<std::string>(P::kFpuStrategyId, "absolute");
      values->Set<float>(P::kFpuValueId, -1);
      values->Set<std::string>(P::kFpuStrategyAtRootId, "same");
      values->Set<float>(P::kFpuValueAtRootId, 99);
      const lczero::classic::SearchParams same(*values);
      Require(same.GetFpuAbsolute(false) && same.GetFpuAbsolute(true) &&
                  same.GetFpuValue(false) == -1 && same.GetFpuValue(true) == -1,
              "classic root same inherits both FPU strategy and scalar value");
      values->Set<std::string>(P::kFpuStrategyId, "reduction");
      values->Set<float>(P::kFpuValueId, .54f);
      values->Set<std::string>(P::kFpuStrategyAtRootId, "absolute");
      values->Set<float>(P::kFpuValueAtRootId, -1);
      const lczero::classic::SearchParams independent(*values);
      Require(!independent.GetFpuAbsolute(false) && independent.GetFpuAbsolute(true) &&
                  independent.GetFpuValue(false) == .54f && independent.GetFpuValue(true) == -1,
              "classic root FPU settings are independent and use unscaled values");
    }
    // The real backend shares this buffer among classic task helpers. Exercise
    // contended appends and Release overflow, then reuse the same allocation.
    {
      lczero::shogi::RequestPool<std::unique_ptr<int>> pool(256);
      auto batch = pool.Borrow();
      const auto *allocation = batch.get();
      auto other = pool.Borrow();
      Require(other.get() != allocation, "live computations own separate batches");
      for (int round = 0; round < 8; ++round) {
        std::atomic<int> overflow{0};
        std::vector<std::thread> helpers;
        for (int tid = 0; tid < 8; ++tid)
          helpers.emplace_back([&, tid] {
            for (int i = 0; i < 64; ++i) {
              try {
                batch->Append(std::make_unique<int>(tid * 64 + i));
              } catch (const std::length_error &) {
                ++overflow;
              }
            }
          });
        for (auto &helper : helpers)
          helper.join();
        Require(batch->size() == 256 && overflow == 256,
                "concurrent request overflow is bounded in Release too");
        std::array<bool, 512> seen{};
        for (size_t i = 0; i < batch->size(); ++i) {
          const int id = *(*batch)[i];
          Require(id >= 0 && id < 512 && !seen[id],
                  "concurrent appends preserve distinct complete requests");
          seen[id] = true;
        }
        pool.Return(std::move(batch));
        batch = pool.Borrow();
        Require(batch.get() == allocation && batch->size() == 0,
                "request storage is cleared and reused across computations");
      }
      pool.Return(std::move(batch));
      pool.Return(std::move(other));
    }
    initTable();
    ::Position::initZobrist();
    lczero::PositionHistory history;
    lczero::Position start;
    history.Reset(start);
    const auto startkey = history.Last().Hash();
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    Require(legal.size() == 30, "startpos legal moves");
    history.Append(history.Last().GetBoard().ParseMove("7g7f"));
    lczero::PositionHistory copy(history);
    const auto key = copy.Last().Hash();
    history.Pop();
    Require(history.Last().Hash() == startkey && copy.Last().Hash() == key,
            "independent owned history");
    copy = history;
    Require(copy.Last().Hash() == startkey, "history assignment reset");
    {
      lczero::PositionHistory first, second;
      first.Reset(start);
      second.Reset(start);
      for (auto text : {"7g7f", "3c3d", "2g2f", "8c8d"})
        first.Append(first.Last().GetBoard().ParseMove(text));
      for (auto text : {"2g2f", "8c8d", "7g7f", "3c3d"})
        second.Append(second.Last().GetBoard().ParseMove(text));
      Require(first.Last().Hash() == second.Last().Hash(),
              "transposed positions share a board key");
      Require(first.Last().GetBoard().GenerateLegalMoves() ==
                  second.Last().GetBoard().GenerateLegalMoves(),
              "canonical legal order is independent of move history");
    }
    history.Reset(start);
    for (const auto *text : {"5i5h", "5a5b", "5h5i", "5b5a"})
      history.Append(history.Last().GetBoard().ParseMove(text));
    Require(history.Last().GetBoard().Shogi().isDraw(16) == RepetitionDraw,
            "owned repetition chain");
    lczero::PositionHistory repetition_copy(history);
    history.Reset(start);
    Require(repetition_copy.Last().GetBoard().Shogi().isDraw(16) ==
                RepetitionDraw,
            "copied repetition chain survives reset");
    {
      lczero::GameState replay{start, {}};
      lczero::PositionHistory cursor;
      cursor.Reset(start);
      for (const auto *text : {"5i5h", "5a5b", "5h5i", "5b5a"}) {
        const auto move = cursor.Last().GetBoard().ParseMove(text);
        replay.moves.push_back(move);
        cursor.Append(move);
      }
      lczero::classic::NodeTree reuse_tree;
      reuse_tree.ResetToPosition(replay);
      Require(reuse_tree.GetPositionHistory().HistoryIdentity() ==
                  cursor.HistoryIdentity(),
              "tree owns complete path identity");
      const auto next = cursor.Last().GetBoard().ParseMove("5i5h");
      auto *head = reuse_tree.GetCurrentHead();
      head->CreateEdges({next});
      auto *child = head->Edges().begin().GetOrSpawnNode(head);
      child->MakeTerminal(lczero::GameResult::DRAW);
      child->IncrementNInFlight(1);
      child->FinalizeScoreUpdate(0, 1, 0, 1);
      replay.moves.push_back(next);
      cursor.Append(next);
      reuse_tree.ResetToPosition(replay);
      Require(!reuse_tree.GetCurrentHead()->IsTerminal(),
              "reused root is reopened");
      const auto &reused_history = reuse_tree.GetPositionHistory();
      Require(reused_history.Last().GetBoard().Shogi().isDraw(16) ==
                      RepetitionDraw &&
                  reused_history.HistoryIdentity() == cursor.HistoryIdentity(),
              "repetition path survives root advance");
      auto check_history = reused_history;
      lczero::shogi::ConfigureRules(5, .5f, .5f);
      auto rule = lczero::shogi::EvaluateRules(
          check_history, false,
          check_history.Last().GetBoard().GenerateLegalMoves().empty());
      Require(rule && rule->reason == lczero::shogi::RuleResult::MoveLimit,
              "reused tree keeps absolute Draw_Ply");
      lczero::shogi::ConfigureRules(6, .5f, .5f);
      rule = lczero::shogi::EvaluateRules(
          check_history, false,
          check_history.Last().GetBoard().GenerateLegalMoves().empty());
      Require(rule && rule->reason == lczero::shogi::RuleResult::Repetition,
              "Draw_Ply boundary and repetition precedence");
    }
    history.Reset(lczero::Position("8k/9/9/8R/9/9/9/9/4K4 w - 1"));
    for (const auto *text : {"1a2a", "1d2d", "2a1a", "2d1d"})
      history.Append(history.Last().GetBoard().ParseMove(text));
    Require(history.Last().GetBoard().Shogi().isDraw(16) == RepetitionWin,
            "continuous checker loses: checked side");
    history.Append(history.Last().GetBoard().ParseMove("1a2a"));
    Require(history.Last().GetBoard().Shogi().isDraw(16) == RepetitionLose,
            "continuous checker loses: checking side");
    history.Reset(lczero::Position(
        "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b PLNSGBR17p3n3g 1"));
    auto many = history.Last().GetBoard().GenerateLegalMoves();
    Require(many.size() > 255 && many.size() < 594,
            "high branching shogi fixture");
    lczero::classic::Node node(nullptr, 0);
    node.CreateEdges(many);
    Require(node.GetNumEdges() == many.size(), "wide edge count");
    Require(sizeof(lczero::classic::Edge) == 4, "compact edge representation");
    lczero::shogi::ConfigureRules(1, 0.4f, 0.4f);
    Require(std::abs(lczero::shogi::DrawScore(false) + 0.2f) < 1e-6f,
            "black draw value");
    Require(std::abs(lczero::shogi::DrawScore(true) + 0.2f) < 1e-6f,
            "white draw value");
    history.Reset(start);
    history.Append(legal.front());
    auto draw = lczero::shogi::EvaluateRules(
        history, false, history.Last().GetBoard().GenerateLegalMoves().empty());
    Require(draw && draw->result == lczero::GameResult::DRAW,
            "absolute move limit");
    lczero::shogi::ConfigureRules(0, 0.5f, 0.5f);
    {
      lczero::PositionHistory mated;
      mated.Reset(lczero::Position("8k/6G2/7G1/9/9/9/9/9/4K4 b R 1"));
      mated.Append(mated.Last().GetBoard().ParseMove("R*1b"));
      Require(mated.Last().GetBoard().GenerateLegalMoves().empty(),
              "shared legal generation detects checkmate");
      auto result = lczero::shogi::EvaluateRules(mated, false, true);
      Require(result && result->reason == lczero::shogi::RuleResult::NoMoves,
              "empty legal list preserves no-move adjudication");
      lczero::shogi::ConfigureRules(1, 0.5f, 0.5f);
      result = lczero::shogi::EvaluateRules(mated, false, true);
      Require(result && result->reason == lczero::shogi::RuleResult::MoveLimit,
              "move-limit adjudication retains priority over no-moves");
      lczero::shogi::ConfigureRules(0, 0.5f, 0.5f);
    }
    lczero::OptionsParser options;
    lczero::SharedBackendParams::Populate(&options);
    lczero::classic::SearchParams::Populate(&options);
    auto *settings = options.GetMutableOptions();
    settings->Set<int>(
        lczero::classic::BaseSearchParams::kTaskWorkersPerSearchWorkerId, 0);
    settings->Set<int>(lczero::classic::BaseSearchParams::kMiniBatchSizeId, 1);
    settings->Set<bool>(lczero::classic::BaseSearchParams::kOutOfOrderEvalId,
                        false);
    settings->Set<std::string>(
        lczero::classic::BaseSearchParams::kContemptModeId, "disable");
    lczero::classic::NodeTree tree;
    tree.ResetToPosition(lczero::GameState{start, {}});
    ConstantBackend backend;
    {
      lczero::classic::NodeTree wide_tree;
      wide_tree.ResetToPosition(lczero::GameState{
          lczero::Position(
              "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b PLNSGBR17p3n3g 1"),
          {}});
      for (int iteration = 0; iteration < 2; ++iteration) {
        lczero::classic::Search wide_search(
            wide_tree, &backend, std::make_unique<Sink>(), {},
            std::chrono::steady_clock::now(), std::make_unique<OneVisit>(),
            false, false, *settings, nullptr);
        lczero::classic::SearchWorker worker(&wide_search,
                                             wide_search.GetParams());
        worker.ExecuteOneIteration();
      }
      Require(wide_tree.GetCurrentHead()->GetNumEdges() == 562 &&
                  wide_tree.GetCurrentHead()->GetN() >= 2,
              "classic selection workspaces support more than 256 moves");
    }
    lczero::classic::Search search(tree, &backend, std::make_unique<Sink>(), {},
                                   std::chrono::steady_clock::now(),
                                   std::make_unique<OneVisit>(), false, false,
                                   *settings, nullptr);
    {
      lczero::classic::SearchWorker worker(&search, search.GetParams());
      worker.ExecuteOneIteration();
    }
    Require(tree.GetCurrentHead()->GetN() >= 1, "root backed up");
    Require(std::abs(tree.GetCurrentHead()->GetWL() + 0.6f) < 1e-5f,
            "single root NN sign");
    Require(tree.GetCurrentHead()->GetD() == 0, "no synthetic NN draw");
    Require(std::abs(search.GetRootSnapshot().front().q - .6f) < 1e-5f,
            "unvisited root move inherits scalar NN evaluation");
    lczero::classic::Search proof_search(
        tree, &backend, std::make_unique<Sink>(), {},
        std::chrono::steady_clock::now(), std::make_unique<OneVisit>(), false,
        false, *settings, nullptr);
    {
      lczero::classic::SearchWorker worker(&proof_search,
                                           proof_search.GetParams());
      worker.ExecuteOneIteration();
    }
    auto *root = tree.GetCurrentHead();
    lczero::classic::Node *target = nullptr;
    lczero::Move target_move;
    for (auto edge : root->Edges())
      if (edge.GetN() > 0) {
        target = edge.node();
        target_move = edge.GetMove();
        break;
      }
    Require(target != nullptr, "visited PV target");
    const auto old_root_n = root->GetN(), old_target_n = target->GetN();
    root->IncrementNInFlight(1);
    target->IncrementNInFlight(1);
    Require(proof_search.ApplyProof({target_move},
                                    lczero::GameResult::BLACK_WON, 3),
            "proof applied while root and target in flight");
    Require(target->IsTerminal() && target->GetWL() == -1,
            "proof terminal sign");
    Require(target->GetBounds().first == target->GetBounds().second,
            "proof exact bounds");
    Require(root->GetN() == old_root_n && target->GetN() == old_target_n,
            "proof does not invent visits");
    Require(root->GetNInFlight() == 1 && target->GetNInFlight() == 1,
            "proof retains pending visits");
    target->FinalizeScoreUpdate(target->GetWL(), target->GetD(), target->GetM(),
                                1);
    root->FinalizeScoreUpdate(1, 0, 4, 1);
    Require(target->GetWL() == -1 && target->GetNInFlight() == 0 &&
                root->GetNInFlight() == 0,
            "pending terminal visit drains");
    auto ranked = proof_search.GetRootSnapshot();
    Require(!ranked.empty(), "post-proof root snapshot");
    {
      lczero::classic::NodeTree delayed_tree;
      delayed_tree.ResetToPosition(lczero::GameState{start, {}});
      lczero::PositionHistory cursor;
      cursor.Reset(start);
      const auto first = cursor.Last().GetBoard().ParseMove("7g7f");
      cursor.Append(first);
      const auto second = cursor.Last().GetBoard().ParseMove("3c3d");
      const auto other = cursor.Last().GetBoard().ParseMove("8c8d");
      auto *delayed_root = delayed_tree.GetCurrentHead();
      delayed_root->CreateEdges({first});
      auto *ancestor =
          delayed_root->Edges().begin().GetOrSpawnNode(delayed_root);
      ancestor->CreateEdges({second, other});
      auto *descendant = ancestor->Edges().begin().GetOrSpawnNode(ancestor);
      for (auto *seed : {delayed_root, ancestor, descendant}) {
        seed->IncrementNInFlight(1);
        seed->FinalizeScoreUpdate(0, 0, 0, 1);
      }
      lczero::classic::Search delayed_search(
          delayed_tree, &backend, std::make_unique<Sink>(), {},
          std::chrono::steady_clock::now(), std::make_unique<OneVisit>(), false,
          false, *settings, nullptr);
      Require(
          delayed_search.ApplyProof({first}, lczero::GameResult::WHITE_WON, 9),
          "ancestor proof applied first");
      Require(delayed_search.ApplyProof({first, second},
                                        lczero::GameResult::BLACK_WON, 1),
              "delayed descendant proof applied");
      Require(ancestor->GetBounds().first == lczero::GameResult::WHITE_WON &&
                  ancestor->GetBounds().second ==
                      lczero::GameResult::WHITE_WON &&
                  ancestor->GetWL() == 1 && ancestor->GetM() == 9,
              "late descendant cannot weaken or rewrite an existing ancestor "
              "proof");
    }
    {
      lczero::classic::NodeTree publishing_tree;
      publishing_tree.ResetToPosition(lczero::GameState{start, {}});
      auto *publishing_root = publishing_tree.GetCurrentHead();
      const auto first = start.GetBoard().ParseMove("7g7f");
      const auto second = start.GetBoard().ParseMove("2g2f");
      publishing_root->CreateEdges({first, second});
      auto edge = publishing_root->Edges().begin();
      auto *published = edge.GetOrSpawnNode(publishing_root);
      ++edge;
      auto *unpublished = edge.GetOrSpawnNode(publishing_root);
      published->IncrementNInFlight(1);
      published->FinalizeScoreUpdate(0, 0, 0, 1);
      unpublished->IncrementNInFlight(1);
      unpublished->MakeTerminal(lczero::GameResult::BLACK_WON);
      publishing_root->IncrementNInFlight(3);
      publishing_root->FinalizeScoreUpdate(0, 0, 0, 2);
      lczero::classic::Search publishing_search(
          publishing_tree, &backend, std::make_unique<Sink>(), {},
          std::chrono::steady_clock::now(), std::make_unique<OneVisit>(), false,
          false, *settings, nullptr);
      Require(publishing_search.ApplyProof({first},
                                           lczero::GameResult::BLACK_WON, 3),
              "proof with unpublished sibling");
      Require(publishing_root->GetBounds().first ==
                      lczero::GameResult::BLACK_WON &&
                  publishing_root->GetBounds().second ==
                      lczero::GameResult::WHITE_WON,
              "unvisited sibling bounds remain unknown until first backup");
      unpublished->CancelScoreUpdate(1);
      publishing_root->CancelScoreUpdate(1);
    }
    // Apply a proof while a real worker is waiting for its descendant NN batch.
    // This exercises classic's terminal override in DoBackupUpdateSingleNode,
    // rather than manually substituting the expected value in a Node update.
    {
      lczero::classic::NodeTree pending_tree;
      pending_tree.ResetToPosition(lczero::GameState{start, {}});
      lczero::PositionHistory path;
      path.Reset(start);
      const auto first = path.Last().GetBoard().ParseMove("7g7f");
      path.Append(first);
      const auto second = path.Last().GetBoard().ParseMove("3c3d");
      auto *pending_root = pending_tree.GetCurrentHead();
      pending_root->CreateEdges({first});
      auto *pending_target =
          pending_root->Edges().begin().GetOrSpawnNode(pending_root);
      pending_target->CreateEdges({second});
      pending_root->IncrementNInFlight(2);
      pending_root->FinalizeScoreUpdate(0, 0, 0, 2);
      pending_target->IncrementNInFlight(1);
      pending_target->FinalizeScoreUpdate(0, 0, 0, 1);
      ConstantBackend pending_backend;
      lczero::classic::Search pending_search(
          pending_tree, &pending_backend, std::make_unique<Sink>(), {},
          std::chrono::steady_clock::now(), std::make_unique<OneVisit>(), false,
          false, *settings, nullptr);
      bool applied = false;
      pending_backend.before_compute = [&] {
        Require(pending_root->GetNInFlight() > 0 &&
                    pending_target->GetNInFlight() > 0,
                "real pending batch crosses proof target");
        applied = pending_search.ApplyProof({first},
                                            lczero::GameResult::BLACK_WON, 3);
      };
      lczero::classic::SearchWorker worker(&pending_search,
                                           pending_search.GetParams());
      worker.ExecuteOneIteration();
      Require(applied && pending_target->GetWL() == -1,
              "pending NN result cannot overwrite terminal proof");
      Require(pending_target->GetN() == 2 && pending_root->GetN() == 3 &&
                  pending_target->GetNInFlight() == 0 &&
                  pending_root->GetNInFlight() == 0,
              "real backup drains all pending visits exactly once");
      Require(
          std::abs(pending_root->GetWL() - 1.0f / 3) < 1e-5f,
          "real backup uses terminal ancestor value without past correction");
      Require(
          pending_root->GetBounds().first == lczero::GameResult::WHITE_WON &&
              pending_root->GetBounds().second == lczero::GameResult::WHITE_WON,
          "external proof propagates exact root bounds immediately");
    }
    for (const auto values :
         {std::pair<float, float>{.5f, .5f}, {.4f, .6f}, {.4f, .4f}}) {
      for (bool white : {false, true}) {
        std::string sfen = DefaultStartPositionSFEN;
        if (white) {
          const auto side = sfen.find(" b ");
          Require(side != std::string::npos, "fixture side");
          sfen.replace(side, 3, " w ");
        }
        lczero::GameState game{lczero::Position(sfen), {}};
        lczero::classic::NodeTree draw_tree;
        draw_tree.ResetToPosition(game);
        auto *draw_root = draw_tree.GetCurrentHead();
        draw_root->CreateEdges(game.startpos.GetBoard().GenerateLegalMoves());
        Require(draw_root->TryStartScoreUpdate(), "reserve seeded root");
        draw_root->FinalizeScoreUpdate(-.6f, 0, 0, 1);
        lczero::shogi::ConfigureRules(1, values.first, values.second);
        lczero::classic::Search draw_search(
            draw_tree, &backend, std::make_unique<Sink>(), {},
            std::chrono::steady_clock::now(), std::make_unique<OneVisit>(),
            false, false, *settings, nullptr);
        {
          lczero::classic::SearchWorker worker(&draw_search,
                                               draw_search.GetParams());
          worker.ExecuteOneIteration();
        }
        const auto snapshot = draw_search.GetRootSnapshot();
        Require(!snapshot.empty() && snapshot.front().visits > 0,
                "draw terminal visited");
        const float expected = 2 * (white ? values.second : values.first) - 1;
        Require(std::abs(snapshot.front().q - expected) < 1e-5f,
                "draw color and depth mapping");

        lczero::classic::NodeTree deep_tree;
        deep_tree.ResetToPosition(game);
        lczero::PositionHistory path;
        path.Reset(game.startpos);
        const auto first =
            path.Last().GetBoard().ParseMove(white ? "3c3d" : "7g7f");
        path.Append(first);
        const auto second =
            path.Last().GetBoard().ParseMove(white ? "7g7f" : "3c3d");
        auto *deep_root = deep_tree.GetCurrentHead();
        deep_root->CreateEdges({first});
        lczero::classic::Node *middle = nullptr;
        for (auto edge : deep_root->Edges()) {
          middle = edge.GetOrSpawnNode(deep_root);
          break;
        }
        Require(middle != nullptr, "forced two-ply path");
        middle->CreateEdges({second});
        deep_root->IncrementNInFlight(2);
        deep_root->FinalizeScoreUpdate(0, 0, 0, 2);
        middle->IncrementNInFlight(1);
        middle->FinalizeScoreUpdate(0, 0, 0, 1);
        lczero::shogi::ConfigureRules(2, values.first, values.second);
        lczero::classic::Search deep_search(
            deep_tree, &backend, std::make_unique<Sink>(), {},
            std::chrono::steady_clock::now(), std::make_unique<OneVisit>(),
            false, false, *settings, nullptr);
        {
          lczero::classic::SearchWorker worker(&deep_search,
                                               deep_search.GetParams());
          worker.ExecuteOneIteration();
        }
        Require(middle->IsTerminal(),
                "forced draw bounds propagate to ancestor");
        Require(std::abs(deep_search.GetRootSnapshot().front().q - expected) <
                    1e-5f,
                "two-ply forced draw uses root color");
        for (auto edge : middle->Edges()) {
          Require(edge.node() && edge.node()->GetD() == 1,
                  "two-ply rule draw flag");
          const float other = 2 * (white ? values.first : values.second) - 1;
          Require(std::abs(edge.node()->GetQ(lczero::shogi::DrawScore(!white)) -
                           other) < 1e-5f,
                  "two-ply opposite mover draw score");
        }
      }
    }
    lczero::shogi::ConfigureRules(0, .5f, .5f);
    for (bool verbose : {false, true}) {
      lczero::OptionsParser stats_options;
      lczero::SharedBackendParams::Populate(&stats_options);
      lczero::classic::SearchParams::Populate(&stats_options);
      auto *stats_settings = stats_options.GetMutableOptions();
      stats_settings->Set<int>(
          lczero::classic::BaseSearchParams::kTaskWorkersPerSearchWorkerId, 0);
      stats_settings->Set<int>(
          lczero::classic::BaseSearchParams::kMiniBatchSizeId, 1);
      stats_settings->Set<int>(
          lczero::classic::SearchParams::kMaxPrefetchBatchId, 0);
      stats_settings->Set<bool>(
          lczero::classic::BaseSearchParams::kVerboseStatsId, verbose);
      ConstantBackend stats_backend;
      lczero::classic::NodeTree stats_tree;
      stats_tree.ResetToPosition(lczero::GameState{start, {}});
      lczero::classic::Search stats_search(
          stats_tree, &stats_backend, std::make_unique<Sink>(), {},
          std::chrono::steady_clock::now(), std::make_unique<OneVisit>(), false,
          false, *stats_settings, nullptr);
      lczero::classic::SearchWorker stats_worker(&stats_search,
                                                 stats_search.GetParams());
      stats_worker.ExecuteOneIteration();
      Require((stats_backend.cache_queries != 0) == verbose,
              "final history/cache statistics are constructed only in verbose "
              "mode");
    }
    std::cout << "classic adapter tests passed; max fixture=" << many.size()
              << " node=" << sizeof(node)
              << " edge=" << sizeof(lczero::classic::Edge) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
