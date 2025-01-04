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

#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "sf_types.h"

namespace Stockfish {

// Overloads of bitwise operators between a Bitboard and a Square for testing
// whether a given bit is set in a bitboard, and for setting and clearing bits.

inline Bitboard  operator&(Bitboard b, Square s) { return b & SetMaskBB[s]; }
inline Bitboard  operator|(Bitboard b, Square s) { return b | SetMaskBB[s]; }
inline Bitboard  operator^(Bitboard b, Square s) { return b ^ SetMaskBB[s]; }
inline Bitboard& operator|=(Bitboard& b, Square s) { b.setBit(s); return b; }
inline Bitboard& operator^=(Bitboard& b, Square s) { b.xorBit(s); return b; }

inline Bitboard operator&(Square s, Bitboard b) { return b & s; }
inline Bitboard operator|(Square s, Bitboard b) { return b | s; }
inline Bitboard operator^(Square s, Bitboard b) { return b ^ s; }


// Returns the bitboard of the least significant
// square of a non-zero bitboard. It is equivalent to square_bb(lsb(bb)).
inline Bitboard least_significant_square_bb(Bitboard b) {
    const u64 p0 = b.p(0);
    const u64 p1 = b.p(1);
    return (p0 != 0) ? Bitboard(p0 & -p0, 0) : Bitboard(0, p1 & -p1);
}

}  // namespace Stockfish

#endif  // #ifndef BITBOARD_H_INCLUDED
