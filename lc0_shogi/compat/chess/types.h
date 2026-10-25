// Shogi compatibility boundary for lc0 classic. GPL-3.0-or-later.
#pragma once
#include "move.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace lczero {
inline constexpr int kMaxLegalMoves = 594;
class Move {
public:
  constexpr Move() = default;
  explicit constexpr Move(uint16_t value) : value_(value) {}
  explicit Move(::Move move)
      : value_(static_cast<uint16_t>(move.proFromAndTo())) {}
  bool operator==(const Move &) const = default;
  void Flip() {} // All shogi moves use absolute coordinates.
  bool is_null() const { return value_ == 0; }
  uint16_t raw_data() const { return value_; }
  std::string ToString(bool = false) const { return ::Move(value_).toUSI(); }

private:
  uint16_t value_ = 0;
};
static_assert(sizeof(Move) == 2);
using MoveList = std::vector<Move>;
} // namespace lczero
