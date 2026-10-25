#include "rules.h"
#include "cppshogi.h"
#include "mate.h"
#include "search.hpp"
#include <climits>
namespace lczero::shogi {
namespace {
int draw_ply = INT_MAX;
float draw_black = 0, draw_white = 0;
} // namespace
void ConfigureRules(int ply, float black, float white) {
  draw_ply = ply > 0 ? ply : INT_MAX;
  draw_black = 2 * black - 1;
  draw_white = 2 * white - 1;
}
float DrawScore(bool white) { return white ? draw_white : draw_black; }
std::optional<RuleResult> EvaluateRules(PositionHistory &history, bool root,
                                        bool no_moves) {
  // This cursor belongs exclusively to the worker performing leaf expansion.
  auto &pos = const_cast<::Position &>(history.Last().GetBoard().Shogi());
  if (!root) {
    if (pos.gamePly() > draw_ply)
      return RuleResult{GameResult::DRAW, RuleResult::MoveLimit};
    switch (pos.isDraw(16)) {
    case NotRepetition:
      break;
    case RepetitionDraw:
      return RuleResult{GameResult::DRAW, RuleResult::Repetition};
    // classic stores a node's value from its incoming edge's perspective.
    case RepetitionWin:
    case RepetitionSuperior:
      return RuleResult{GameResult::BLACK_WON, RuleResult::Repetition};
    case RepetitionLose:
    case RepetitionInferior:
      return RuleResult{GameResult::WHITE_WON, RuleResult::Repetition};
    }
#ifndef MATE_SEARCH_DEPTH
#define MATE_SEARCH_DEPTH 5
#endif
    const bool check = pos.inCheck();
    if (check ? ::mateMoveInOddPly<MATE_SEARCH_DEPTH, true>(pos, draw_ply)
              : ::mateMoveInOddPly<MATE_SEARCH_DEPTH, false>(pos, draw_ply)) {
      int plies = MATE_SEARCH_DEPTH;
      if (!check && pos.mateMoveIn1Ply())
        plies = 1;
      else if constexpr (MATE_SEARCH_DEPTH > 3) {
        if (check ? ::mateMoveInOddPly<3, true>(pos, draw_ply)
                  : ::mateMoveInOddPly<3, false>(pos, draw_ply))
          plies = 3;
      }
      return RuleResult{GameResult::BLACK_WON, RuleResult::Mate, plies};
    }
    if (!check && ::nyugyoku<false>(pos))
      return RuleResult{GameResult::BLACK_WON, RuleResult::Declaration};
  }
  if (no_moves)
    return RuleResult{GameResult::WHITE_WON, RuleResult::NoMoves};
  return std::nullopt;
}
} // namespace lczero::shogi
