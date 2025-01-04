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

#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <cassert>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>

#include "position.hpp"
#include "sf_types.h"

namespace Stockfish {

class TranspositionTable;


// Position class stores information regarding the board representation as
// pieces, side to move, hash keys, castling info, etc. Important methods are
// do_move() and undo_move(), used by the search to update node info when
// traversing the search tree.
class Position : public ::Position {
   public:
    Position()                           = default;
    Position(const Position&)            = delete;
    Position& operator=(const Position&) = delete;

    // SFEN string input/output
    std::string sfen() const { return toSFEN(); }

    // Position representation
    Bitboard pieces() const { return occupiedBB(); }
    Bitboard pieces(PieceType pt) const { return bbOf(pt); }
    Bitboard pieces(Color c) const { return bbOf(c); }
    Piece    piece_on(Square s) const { return piece(s); }

    // Checking
    Bitboard checkers() const { return checkersBB(); }

    // Attacks to/from a given square
    Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
    Bitboard attackers_to(Square s, Bitboard occupied) const { return attackersTo(s, occupied); }

    // Properties of moves
    bool  pseudo_legal(const Move m) const { return moveIsPseudoLegal(m); }
    bool  capture(Move m) const { return !m.is_drop() && piece_on(m.to_sq()) != NO_PIECE; }
    bool  capture_stage(Move m) const { return capture(m); }
    PieceType captured_piece_type() const { return st_->capturedPieceType; }
    Piece moved_piece(Move m) const { return movedPiece(m); }

    // Doing and undoing moves
    void do_move(Move m, StateInfo& newSt) { doMove(m, newSt); }
    void undo_move(Move m) { undoMove(m); }
    void do_null_move(StateInfo& newSt, const TranspositionTable& tt);
    void undo_null_move();

    // Static Exchange Evaluation
    bool see_ge(Move m, int threshold = 0) const;

    // Accessing hash keys
    Key key() const { return getKey(); }

    // Other properties of the position
    Color side_to_move() const { return turn(); }
    int   game_ply() const { return gamePly(); }
    RepetitionType is_draw(int ply) const { return isDraw(ply); }
};

}  // namespace Stockfish

#endif  // #ifndef POSITION_H_INCLUDED
