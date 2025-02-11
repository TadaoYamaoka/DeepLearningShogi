/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include <algorithm>  // IWYU pragma: keep
#include <cstddef>

#include "sf_types.h"
#include "generateMoves.hpp"

namespace Stockfish {

using GenType = ::MoveType;

constexpr auto CAPTURES = ::MoveType::CapturePlusPro;
constexpr auto QUIETS = ::MoveType::NonCaptureMinusPro;
constexpr auto EVASIONS = ::MoveType::Evasion;
constexpr auto NON_EVASIONS = ::MoveType::NonEvasion;
constexpr auto LEGAL = ::MoveType::Legal;

template<GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    return generateMoves<Type>(moveList, pos);
}

}  // namespace Stockfish

#endif  // #ifndef MOVEGEN_H_INCLUDED
