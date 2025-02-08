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

#include "sf_position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include "sf_bitboard.h"
#include "sf_misc.h"
#include "sf_tt.h"
#include "sf_usi.h"

using std::string;

namespace Stockfish {

void Position::init() {
    ::Position::initZobrist();
}

// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
void Position::update_slider_blockers(Color c) const {

    const Color us = c;
    const Color them = oppositeColor(us);

    st_->blockersForKing[c] = allZeroBB();
    st_->pinners[them] = allZeroBB();

    // pin する遠隔駒
    Bitboard snipers = bbOf(them);

    const Square ksq = kingSquare(us);

    // 障害物が無ければ玉に到達出来る駒のBitboardだけ残す。
    snipers &= (bbOf(Lance) & lanceAttackToEdge(us, ksq)) |
        (bbOf(Rook, Dragon) & rookAttackToEdge(ksq)) | (bbOf(Bishop, Horse) & bishopAttackToEdge(ksq));

    while (snipers) {
        const Square sq = snipers.firstOneFromSQ11();
        // pin する遠隔駒と玉の間にある駒の位置の Bitboard
        const Bitboard between = betweenBB(sq, ksq) & occupiedBB();

        // pin する遠隔駒と玉の間にある駒が1つで、かつ、引数の色のとき、その駒は(を) pin されて(して)いる。
        if (between
            && between.isOneBit<false>())
        {
            st_->blockersForKing[us] |= between;
            if (between.andIsAny(bbOf(us)))
                st_->pinners[them].setBit(sq);
        }
    }
}

// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {

    assert(m.is_ok());

    if (m.is_drop()) {
        return true;
    }
    else {
        Color  us = side_to_move();
        Square from = m.from();
        Square to = m.to();

        assert(color_of(moved_piece(m)) == us);
        assert(piece_on(kingSquare(us)) == make_piece(us, KING));

        // If the moving piece is a king, check whether the destination square is
        // attacked by the opponent.
        if (type_of(piece_on(from)) == KING)
            return !(attackers_to(to, pieces() ^ from) & pieces(~us));

        // A non-king move is legal if and only if it is not pinned or it
        // is moving along the ray towards or away from the king.
        return !(blockers_for_king(us) & from) || isAligned<false>(from, to, kingSquare(us));
    }
}

// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt, const TranspositionTable& tt) {

    assert(!checkers());
    assert(&newSt != st_);

    std::memcpy(&newSt, st_, sizeof(StateInfo));

    newSt.previous = st_;
    st_            = &newSt;

    st_->boardKey ^= zobTurn();
    prefetch(tt.first_entry(key()));

    st_->pliesFromNull = 0;

    turn_ = ~turn_;

    findCheckers();

    st_->hand = hand(turn_);

    st_->continuousCheck[~turn_] = 0;

    assert(isOK());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st_   = st_->previous;
    turn_ = ~turn_;
}


// Tests if the SEE (Static Exchange Evaluation)
// value of move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    assert(m.is_ok());

    Square from = m.from_sq(), to = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = (m.is_drop() ? 0 : PieceValue[piece_on(from)]) - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == side_to_move());
    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = side_to_move();
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to
        // the bitboard 'attackers' any X-ray attackers behind it.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= lanceAttack(stm, to, occupied) & (bbOf(LANCE, stm) | bbOf(ROOK, DRAGON));

            continue;
        }

        else if ((bb = stmAttackers & pieces(LANCE)))
        {
            if ((swap = LanceValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= lanceAttack(stm, to, occupied) & (bbOf(LANCE, stm) | bbOf(ROOK, DRAGON));

            continue;
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            continue;
        }

        else if ((bb = stmAttackers & pieces(PRO_PAWN)))
        {
            if ((swap = ProPawnValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(PRO_LANCE)))
        {
            if ((swap = ProLanceValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(PRO_KNIGHT)))
        {
            if ((swap = ProKnightValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(SILVER)))
        {
            if ((swap = SilverValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(PRO_SILVER)))
        {
            if ((swap = ProSilverValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(GOLD)))
        {
            if ((swap = GoldValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= bishopAttack(to, occupied) & bbOf(BISHOP, HORSE);

            continue;
        }


        else if ((bb = stmAttackers & pieces(HORSE)))
        {
            if ((swap = HorseValue - swap) < res)
                break;
        }
        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
        }

        else if ((bb = stmAttackers & pieces(DRAGON)))
        {
            if ((swap = DragonValue - swap) < res)
                break;
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;

        Square sq = bb.firstOneFromSQ11();
        occupied ^= sq;

        const Direction dir = squareRelation(to, sq);
        switch (dir) {
        case DirecFile:
        {
            const Bitboard rook_dragon = bbOf(ROOK, DRAGON);
            attackers |= (lanceAttack(stm, to, occupied) & (bbOf(LANCE, stm) | rook_dragon))
                | (lanceAttack(~stm, to, occupied) & (bbOf(LANCE, ~stm) | rook_dragon));
            break;
        }
        case DirecRank:
            attackers |= rookAttackRank(to, occupied) & bbOf(ROOK, DRAGON);
            break;
        case DirecDiagNESW:
        case DirecDiagNWSE:
            attackers |= bishopAttack(to, occupied) & bbOf(BISHOP, HORSE);
            break;
        default: UNREACHABLE;
        }
    }

    return bool(res);
}

// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {

    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);
}

}  // namespace Stockfish
