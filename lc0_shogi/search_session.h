// USI-facing facade. No legacy MCTS node types cross this boundary.
#pragma once
#include "position.hpp"
#include "search.hpp"
#include <atomic>
#include <functional>
#include <vector>
inline constexpr int max_gpu = 8;
struct OptionsMap;
void RegisterClassicSearchOptions(OptionsMap& options);
void ApplyClassicSearchOptions(const OptionsMap& options);
void SetLimits(const LimitsType &limits);
void SetLimits(const Position *position, const LimitsType &limits);
void SetConstPlayout(int count);
void SetPondering(bool value);
void SetPonderingMode(bool value);
void InitSearchStop();
void StopSearch();
bool IsSearchStopped();
void SetInferenceSlots(const int slots[max_gpu], const int batches[max_gpu]);
void SetSearchSettings(int threads, int minibatch);
void SetSearchConcurrency(int task_workers, int concurrent_searchers);
void SetDnnCacheSize(size_t capacity);
// Policy blending changed; discard cached values/tree at the next quiescent use.
void InvalidateNeuralCache();
void SetResignThreshold(int threshold);
void SetDrawValue(int black, int white);
void InitializeSearch(unsigned int limit);
void TerminateSearch();
void FinalizeSearch();
Move SearchGenmove(Position *position, Key starting_key,
                   const std::vector<Move> &moves, Move &ponder);
void SetReuseSubtree(bool value);
void SetPvInterval(int interval);
void SetMultiPV(int count);
void SetEvalCoef(int coefficient);
void SetRandomMove(int ply, int temperature, int drop, int cutoff,
                   int cutoff_drop);
void SetRandomMove2(int ply, int probability, int temperature, int cutoff,
                    int limit);
void SetModelPath(const std::string paths[max_gpu]);
void NewGame();
void GameOver();
int GetTimeLimit();
void SetDrawPly(int ply);
void SetPvMateSearch(int threads, int depth, int nodes);
void set_softmax_temperature(float temperature);
void SetDebugMessageMode(bool value);
struct SearchCandidate {
  Move move;
  int move_count;
  double win;
  float prior;
};
std::vector<SearchCandidate> GetSearchCandidates();
void DumpNeuralEvaluation(const Position &position);
void DumpNeuralStats();
// Quiescent diagnostic: actual session/search values, not the USI option map.
void DumpSearchSettings();
void SetSearchResponseBlocked(bool blocked);
void ReleaseSearchResponse();
void CancelSearchResponse();
std::function<bool()> CaptureSearchResponseWaiter();
#ifdef MULTI_PONDER
void ResetMultiPonder();
void PrepareMultiPonder();
void WaitPrepareMultiPonder();
bool GetMultiPonderMove(std::vector<Move> &moves, int count);
#endif
