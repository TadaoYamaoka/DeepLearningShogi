// Shogi compatibility boundary for lc0 classic. GPL-3.0-or-later.
#pragma once
#include "chess/board.h"
#include <deque>
#include <span>
namespace lczero {
enum class GameResult : uint8_t { UNDECIDED, BLACK_WON, DRAW, WHITE_WON };
inline GameResult operator-(GameResult result) {
  if (result == GameResult::WHITE_WON)
    return GameResult::BLACK_WON;
  if (result == GameResult::BLACK_WON)
    return GameResult::WHITE_WON;
  return result;
}
class Position {
public:
  Position() = default;
  explicit Position(const std::string &sfen) : board_(sfen) {}
  static Position FromFen(std::string_view sfen) {
    return Position(std::string(sfen));
  }
  const ChessBoard &GetBoard() const { return board_; }
  bool IsBlackToMove() const { return board_.flipped(); }
  int GetGamePly() const { return board_.Shogi().gamePly(); }
  uint64_t Hash() const { return board_.Shogi().getKey(); }
  bool operator==(const Position &other) const {
    return board_.Shogi().toSFEN() == other.board_.Shogi().toSFEN();
  }
  std::string DebugString() const { return board_.Shogi().toSFEN(); }

private:
  ChessBoard board_;
  friend class PositionHistory;
};
class PositionHistory {
public:
  PositionHistory() = default;
  PositionHistory(const PositionHistory &other) { *this = other; }
  PositionHistory(PositionHistory &&) noexcept = default;
  PositionHistory &operator=(PositionHistory &&) noexcept = default;
  PositionHistory &operator=(const PositionHistory &other);
  const Position &Starting() const { return starting_; }
  const Position &Last() const { return current_; }
  void Reserve(int) {
  } // deque entries never move after doMove takes their address.
  int GetLength() const { return static_cast<int>(moves_.size()) + 1; }
  void Reset(const Position &position);
  void Append(Move move);
  void Pop();
  void Trim(int size);
  bool IsBlackToMove() const { return Last().IsBlackToMove(); }
  // dlshogi models need the current position only, not chess history planes.
  std::span<const Position> GetPositions() const { return {&current_, 1}; }
  std::string HistoryIdentity() const;

private:
  Position starting_;
  Position current_;
  std::string initial_sfen_;
  std::deque<::StateInfo> states_;
  std::vector<::Move> moves_;
};
} // namespace lczero
