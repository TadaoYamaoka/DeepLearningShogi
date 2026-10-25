// Shogi compatibility boundary for lc0 classic. GPL-3.0-or-later.
#pragma once
#include "chess/types.h"
#include "generateMoves.hpp"
#include "position.hpp"
#include "usi.hpp"
#include <memory>
namespace lczero {
class ChessBoard {
public:
  ChessBoard();
  explicit ChessBoard(const std::string &sfen);
  const ::Position &Shogi() const { return *position_; }
  ::Position &MutableShogi() { return *position_; }
  MoveList GenerateLegalMoves() const;
  Move ParseMove(const std::string &text) const;
  bool IsUnderCheck() const { return Shogi().inCheck(); }
  void Mirror() {} // cppshogi keeps absolute coordinates and side to move.
  bool flipped() const { return Shogi().turn() == White; }

private:
  std::shared_ptr<::Position> position_;
};
} // namespace lczero
