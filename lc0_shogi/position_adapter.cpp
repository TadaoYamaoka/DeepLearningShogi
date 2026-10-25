// GPL-3.0-or-later. cppshogi owns all shogi rules and move interpretation.
#include "chess/position.h"
#include <stdexcept>
namespace lczero {
std::string PositionHistory::HistoryIdentity() const {
  std::string identity = initial_sfen_;
  for (auto move : moves_) {
    identity += ':';
    identity += std::to_string(move.proFromAndTo());
  }
  return identity;
}
namespace {
const char *const kStart =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
}
ChessBoard::ChessBoard() : ChessBoard(kStart) {}
ChessBoard::ChessBoard(const std::string &sfen)
    : position_(std::make_shared<::Position>()) {
  position_->setSearcher(nullptr);
  position_->set(sfen.starts_with("sfen ") ? sfen.substr(5) : sfen);
}
MoveList ChessBoard::GenerateLegalMoves() const {
  ::MoveList<Legal> moves(Shogi());
  MoveList result;
  result.reserve(moves.size());
  for (; !moves.end(); ++moves)
    result.emplace_back(moves.move());
  return result;
}
Move ChessBoard::ParseMove(const std::string &text) const {
  for (auto move : GenerateLegalMoves())
    if (move.ToString() == text)
      return move;
  throw std::invalid_argument("Illegal shogi move: " + text);
}
void PositionHistory::Reset(const Position &position) {
  const auto sfen = position.GetBoard().Shogi().toSFEN();
  if (sfen == initial_sfen_) {
    Trim(1);
    return;
  }
  moves_.clear();
  states_.clear();
  initial_sfen_ = sfen;
  starting_ = Position(sfen);
  current_ = Position(sfen);
}
PositionHistory &PositionHistory::operator=(const PositionHistory &other) {
  if (this == &other)
    return *this;
  if (initial_sfen_ != other.initial_sfen_)
    Reset(other.Starting());
  size_t common = 0;
  while (common < moves_.size() && common < other.moves_.size() &&
         moves_[common] == other.moves_[common])
    ++common;
  Trim(static_cast<int>(common) + 1);
  for (size_t i = common; i < other.moves_.size(); ++i)
    Append(Move(other.moves_[i]));
  return *this;
}
void PositionHistory::Append(Move move) {
  auto &position = current_.board_.MutableShogi();
  const auto full = ::move16toMove(::Move(move.raw_data()), position);
  states_.emplace_back();
  position.doMove(full, states_.back());
  moves_.push_back(full);
}
void PositionHistory::Pop() {
  if (moves_.empty())
    throw std::out_of_range("Pop of empty shogi history");
  current_.board_.MutableShogi().undoMove(moves_.back());
  moves_.pop_back();
  states_.pop_back();
}
void PositionHistory::Trim(int size) {
  if (size < 1 || size > GetLength())
    throw std::out_of_range("Invalid shogi history trim");
  while (GetLength() > size)
    Pop();
}
} // namespace lczero
