#pragma once
#include "chess/position.h"
namespace lczero {
enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };
inline uint16_t MoveToNNIndex(Move move, int) { return move.raw_data(); }
} // namespace lczero
