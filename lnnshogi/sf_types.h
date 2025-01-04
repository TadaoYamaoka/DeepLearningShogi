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

#ifndef TYPES_H_INCLUDED
    #define TYPES_H_INCLUDED

#include "move.hpp"

// When compiling with provided Makefile (e.g. for Linux and OSX), configuration
// is done automatically. To get started type 'make help'.
//
// When Makefile is not used (e.g. with Microsoft Visual Studio) some switches
// need to be set manually:
//
// -DNDEBUG      | Disable debugging mode. Always use this for release.
//
// -DNO_PREFETCH | Disable use of prefetch asm-instruction. You may need this to
//               | run on some very old machines.
//
// -DUSE_POPCNT  | Add runtime support for use of popcnt asm-instruction. Works
//               | only in 64-bit mode and requires hardware with popcnt support.
//
// -DUSE_PEXT    | Add runtime support for use of pext asm-instruction. Works
//               | only in 64-bit mode and requires hardware with pext support.

    #include <cassert>
    #include <cstdint>

    #if defined(_MSC_VER)
        // Disable some silly and noisy warnings from MSVC compiler
        #pragma warning(disable: 4127)  // Conditional expression is constant
        #pragma warning(disable: 4146)  // Unary minus operator applied to unsigned type
        #pragma warning(disable: 4800)  // Forcing value to bool 'true' or 'false'
    #endif

// Predefined macros hell:
//
// __GNUC__                Compiler is GCC, Clang or ICX
// __clang__               Compiler is Clang or ICX
// __INTEL_LLVM_COMPILER   Compiler is ICX
// _MSC_VER                Compiler is MSVC
// _WIN32                  Building on Windows (any)
// _WIN64                  Building on Windows 64 bit

    #if defined(__GNUC__) && (__GNUC__ < 9 || (__GNUC__ == 9 && __GNUC_MINOR__ <= 2)) \
      && defined(_WIN32) && !defined(__clang__)
        #define ALIGNAS_ON_STACK_VARIABLES_BROKEN
    #endif

    #define ASSERT_ALIGNED(ptr, alignment) assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0)

    #if defined(_WIN64) && defined(_MSC_VER)  // No Makefile used
        #include <intrin.h>                   // Microsoft header for _BitScanForward64()
        #define IS_64BIT
    #endif

    #if defined(USE_POPCNT) && defined(_MSC_VER)
        #include <nmmintrin.h>  // Microsoft header for _mm_popcnt_u64()
    #endif

    #if !defined(NO_PREFETCH) && defined(_MSC_VER)
        #include <xmmintrin.h>  // Microsoft header for _mm_prefetch()
    #endif

    #if defined(USE_PEXT)
        #include <immintrin.h>  // Header for _pext_u64() intrinsic
        #define pext(b, m) _pext_u64(b, m)
    #else
        #define pext(b, m) 0
    #endif

namespace Stockfish {

    #ifdef USE_POPCNT
constexpr bool HasPopCnt = true;
    #else
constexpr bool HasPopCnt = false;
    #endif

    #ifdef USE_PEXT
constexpr bool HasPext = true;
    #else
constexpr bool HasPext = false;
    #endif

    #ifdef IS_64BIT
constexpr bool Is64Bit = true;
    #else
constexpr bool Is64Bit = false;
    #endif

constexpr int MAX_MOVES = MaxLegalMoves;
constexpr int MAX_PLY   = 246;

constexpr auto WHITE = White;
constexpr auto BLACK = Black;
constexpr auto COLOR_NB = ColorNum;

enum CastlingRights {
    NO_CASTLING,
    WHITE_OO,
    WHITE_OOO = WHITE_OO << 1,
    BLACK_OO  = WHITE_OO << 2,
    BLACK_OOO = WHITE_OO << 3,

    KING_SIDE      = WHITE_OO | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,

    CASTLING_RIGHT_NB = 16
};

enum Bound {
    BOUND_NONE,
    BOUND_UPPER,
    BOUND_LOWER,
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

// Value is used as an alias for int, this is done to differentiate between a search
// value and any other integer value. The values used in search are always supposed
// to be in the range (-VALUE_NONE, VALUE_NONE] and should not exceed this range.
using Value = int;

constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_NONE     = 32002;
constexpr Value VALUE_INFINITE = 32001;

constexpr Value VALUE_MATE             = 32000;
constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

constexpr Value VALUE_TB                 = VALUE_MATE_IN_MAX_PLY - 1;
constexpr Value VALUE_TB_WIN_IN_MAX_PLY  = VALUE_TB - MAX_PLY;
constexpr Value VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY;


constexpr bool is_valid(Value value) { return value != VALUE_NONE; }

constexpr bool is_win(Value value) {
    assert(is_valid(value));
    return value >= VALUE_TB_WIN_IN_MAX_PLY;
}

constexpr bool is_loss(Value value) {
    assert(is_valid(value));
    return value <= VALUE_TB_LOSS_IN_MAX_PLY;
}

constexpr bool is_decisive(Value value) { return is_win(value) || is_loss(value); }

// In the code, we make the assumption that these values
// are such that non_pawn_material() can be used to uniquely
// identify the material on the board.
constexpr Value PawnValue = 90;
constexpr Value LanceValue = 315;
constexpr Value KnightValue = 405;
constexpr Value SilverValue = 495;
constexpr Value GoldValue = 540;
constexpr Value BishopValue = 855;
constexpr Value RookValue = 990;
constexpr Value ProPawnValue = 540;
constexpr Value ProLanceValue = 540;
constexpr Value ProKnightValue = 540;
constexpr Value ProSilverValue = 540;
constexpr Value HorseValue = 945;
constexpr Value DragonValue = 1395;
constexpr Value KingValue = 15000;

constexpr auto NO_PIECE_TYPE = PieceType::Occupied;
constexpr auto PAWN = PieceType::Pawn;
constexpr auto LANCE = PieceType::Lance;
constexpr auto KNIGHT = PieceType::Knight;
constexpr auto SILVER = PieceType::Silver;
constexpr auto BISHOP = PieceType::Bishop;
constexpr auto ROOK = PieceType::Rook;
constexpr auto GOLD = PieceType::Gold;
constexpr auto KING = PieceType::King;
constexpr auto PRO_PAWN = PieceType::ProPawn;
constexpr auto PRO_LANCE = PieceType::ProLance;
constexpr auto PRO_KNIGHT = PieceType::ProKnight;
constexpr auto PRO_SILVER = PieceType::ProSilver;
constexpr auto HORSE = PieceType::Horse;
constexpr auto DRAGON = PieceType::Dragon;

constexpr auto NO_PIECE = Piece::Empty;
constexpr auto PIECE_NB = Piece::PieceNone;

constexpr Value PieceValue[PIECE_NB] = {
  VALUE_ZERO, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue, GoldValue, KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue, VALUE_ZERO,
  VALUE_ZERO, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue, GoldValue, KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue };

using Depth = int;

enum : int {
    // The following DEPTH_ constants are used for transposition table entries
    // and quiescence search move generation stages. In regular search, the
    // depth stored in the transposition table is literal: the search depth
    // (effort) used to make the corresponding transposition table value. In
    // quiescence search, however, the transposition table entries only store
    // the current quiescence move generation stage (which should thus compare
    // lower than any regular search depth).
    DEPTH_QS = 0,
    // For transposition table entries where no searching at all was done
    // (whether regular or qsearch) we use DEPTH_UNSEARCHED, which should thus
    // compare lower than any quiescence or regular depth. DEPTH_ENTRY_OFFSET
    // is used only for the transposition table entry occupancy check (see tt.cpp),
    // and should thus be lower than DEPTH_UNSEARCHED.
    DEPTH_UNSEARCHED   = -2,
    DEPTH_ENTRY_OFFSET = -3
};

constexpr auto SQUARE_ZERO = SquareBegin;
constexpr auto SQUARE_NB = SquareNum;
constexpr auto SQ_NONE = SQUARE_NB;

constexpr auto FILE_1 = File1;
constexpr auto FILE_2 = File2;
constexpr auto FILE_3 = File3;
constexpr auto FILE_4 = File4;
constexpr auto FILE_5 = File5;
constexpr auto FILE_6 = File6;
constexpr auto FILE_7 = File7;
constexpr auto FILE_8 = File8;
constexpr auto FILE_9 = File9;
constexpr auto FILE_NB = FileNum;

constexpr auto RANK_1 = Rank1;
constexpr auto RANK_2 = Rank2;
constexpr auto RANK_3 = Rank3;
constexpr auto RANK_4 = Rank4;
constexpr auto RANK_5 = Rank5;
constexpr auto RANK_6 = Rank6;
constexpr auto RANK_7 = Rank7;
constexpr auto RANK_8 = Rank8;
constexpr auto RANK_9 = Rank9;
constexpr auto RANK_NB = RankNum;

// Toggle color
constexpr Color operator~(Color c) { return oppositeColor(c); }

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }

constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }

constexpr Piece make_piece(Color c, PieceType pt) { return colorAndPieceTypeToPiece(c, pt); }

constexpr PieceType type_of(Piece pc) { return pieceToPieceType(pc); }

inline Color color_of(Piece pc) {
    assert(pc != NO_PIECE);
    return Color(pc >> 3);
}

constexpr bool is_ok(Square s) { return s >= SQ11 && s <= SQ99; }

constexpr File file_of(Square s) { return makeFile(s); }

constexpr Rank rank_of(Square s) { return makeRank(s); }

constexpr Rank relative_rank(Color c, Rank r) { return c == BLACK ? r : (Rank)(8 - r); }


// Based on a congruential pseudo-random number generator
constexpr Key make_key(uint64_t seed) {
    return seed * 6364136223846793005ULL + 1442695040888963407ULL;
}


enum MoveType {
    NORMAL,
    PROMOTION  = PromoteFlag,
    DROP = 2 << 14,
};

// A move needs 16 bits to be stored
//
// bit  0- 5: destination square (from 0 to 63)
// bit  6-11: origin square (from 0 to 63)
// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
// NOTE: en passant bit is set only when a pawn can be captured
//
// Special cases are Move::none() and Move::null(). We can sneak these in because
// in any normal move the destination square and origin square are always different,
// but Move::none() and Move::null() have the same origin and destination square.

class Move : public ::Move {
   public:
    Move() = default;
    constexpr explicit Move(const u32 u) : ::Move(u) {}
    constexpr operator ::Move() const { return *this; }

    constexpr Square from_sq() const {
        assert(is_ok());
        return from();
    }

    constexpr Square to_sq() const {
        assert(is_ok());
        return to();
    }

    constexpr int is_drop() const { return isDrop(); }

    constexpr int is_promotion() const { return isPromotion(); }

    constexpr PieceType drop_type() const { return pieceTypeDropped(); }

    constexpr int from_to() const { return int(from_sq() + int(is_drop() ? (SQUARE_NB - 1) : 0)) * int(SQUARE_NB) + int(to_sq()); }

    constexpr MoveType type_of() const { return MoveType(((u16)(81 << 7) - (u16)(value() & 0x3f80)) & DROP | value() & PROMOTION); }

    constexpr bool is_ok() const { return isOK(); }

    static constexpr Move null() { return Move(MoveNull); }
    static constexpr Move none() { return Move(MoveNone); }

    constexpr bool operator==(const Move& m) const { return ::Move::operator==(m); }
    constexpr bool operator!=(const Move& m) const { return ::Move::operator!=(m); }

    constexpr explicit operator bool() const { return value() != 0; }

    constexpr std::uint32_t raw() const { return value(); }

    struct MoveHash {
        std::size_t operator()(const Move& m) const { return make_key(m.value()); }
    };
};

}  // namespace Stockfish

#endif  // #ifndef TYPES_H_INCLUDED
