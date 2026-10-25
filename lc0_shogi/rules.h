#pragma once
#include "chess/position.h"
#include <optional>
namespace lczero::shogi {
struct RuleResult {
  GameResult result;
  enum Reason { Repetition, MoveLimit, Mate, Declaration, NoMoves } reason;
  int plies = 0; // Proven distance or the short-mate search's upper bound.
};
std::optional<RuleResult> EvaluateRules(PositionHistory &history, bool root,
                                        bool no_moves);
void ConfigureRules(int draw_ply, float black_draw, float white_draw);
float DrawScore(bool white_to_move);
} // namespace lczero::shogi
