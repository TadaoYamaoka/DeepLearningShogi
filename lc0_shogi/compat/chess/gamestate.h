#pragma once
#include "chess/position.h"
namespace lczero {
struct GameState {
  Position startpos;
  std::vector<Move> moves;
};
} // namespace lczero
