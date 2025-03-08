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

#include "sf_evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>

#include "sf_position.h"
#include "sf_types.h"
#include "sf_usi.h"

#include "bitboard.hpp"

namespace Stockfish {

namespace {

constexpr Value PieceValue2[PIECE_NB] = {
    VALUE_ZERO, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue, GoldValue, 0, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue, VALUE_ZERO,
    VALUE_ZERO, -PawnValue, -LanceValue, -KnightValue, -SilverValue, -BishopValue, -RookValue, -GoldValue, 0, -ProPawnValue, -ProLanceValue, -ProKnightValue, -ProSilverValue, -HorseValue, -DragonValue };

constexpr Value HandPieceValue[HandPieceNum] =
{
    PawnValue, LanceValue, KnightValue, SilverValue, GoldValue, BishopValue, RookValue
};

}

    // Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Position& pos) {

    Value v = 0;

    Bitboard occupied_bb = pos.occupiedBB();
    FOREACH_BB(occupied_bb, Square sq, {
        const Piece pc = pos.piece(sq);

        if (pos.turn() == Black) {
            v += PieceValue2[pc];
        }
        else {
            v -= PieceValue2[pc];
        }
    });

    for (Color c = Black; c < ColorNum; ++c) {
        const Hand hand = pos.hand(c);
        const int sign = pos.turn() == c ? 1 : -1;
        for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
            auto num = hand.numOf(hp);
            v += num * HandPieceValue[hp] * sign;
        }
    }

    return v;
}

}  // namespace Stockfish
