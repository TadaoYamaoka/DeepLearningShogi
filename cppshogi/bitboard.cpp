/*
  Apery, a USI shogi playing engine derived from Stockfish, a UCI chess playing engine.
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Copyright (C) 2011-2018 Hiraoka Takuya

  Apery is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Apery is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.hpp"
#include "bitboard.hpp"

const Bitboard SetMaskBB[SquareNum] = {
    Bitboard(UINT64_C(1) <<  0,                 0),  // 0 , SQ11
    Bitboard(UINT64_C(1) <<  1,                 0),  // 1 , SQ12
    Bitboard(UINT64_C(1) <<  2,                 0),  // 2 , SQ13
    Bitboard(UINT64_C(1) <<  3,                 0),  // 3 , SQ14
    Bitboard(UINT64_C(1) <<  4,                 0),  // 4 , SQ15
    Bitboard(UINT64_C(1) <<  5,                 0),  // 5 , SQ16
    Bitboard(UINT64_C(1) <<  6,                 0),  // 6 , SQ17
    Bitboard(UINT64_C(1) <<  7,                 0),  // 7 , SQ18
    Bitboard(UINT64_C(1) <<  8,                 0),  // 8 , SQ19
    Bitboard(UINT64_C(1) <<  9,                 0),  // 9 , SQ21
    Bitboard(UINT64_C(1) << 10,                 0),  // 10, SQ22
    Bitboard(UINT64_C(1) << 11,                 0),  // 11, SQ23
    Bitboard(UINT64_C(1) << 12,                 0),  // 12, SQ24
    Bitboard(UINT64_C(1) << 13,                 0),  // 13, SQ25
    Bitboard(UINT64_C(1) << 14,                 0),  // 14, SQ26
    Bitboard(UINT64_C(1) << 15,                 0),  // 15, SQ27
    Bitboard(UINT64_C(1) << 16,                 0),  // 16, SQ28
    Bitboard(UINT64_C(1) << 17,                 0),  // 17, SQ29
    Bitboard(UINT64_C(1) << 18,                 0),  // 18, SQ31
    Bitboard(UINT64_C(1) << 19,                 0),  // 19, SQ32
    Bitboard(UINT64_C(1) << 20,                 0),  // 20, SQ33
    Bitboard(UINT64_C(1) << 21,                 0),  // 21, SQ34
    Bitboard(UINT64_C(1) << 22,                 0),  // 22, SQ35
    Bitboard(UINT64_C(1) << 23,                 0),  // 23, SQ36
    Bitboard(UINT64_C(1) << 24,                 0),  // 24, SQ37
    Bitboard(UINT64_C(1) << 25,                 0),  // 25, SQ38
    Bitboard(UINT64_C(1) << 26,                 0),  // 26, SQ39
    Bitboard(UINT64_C(1) << 27,                 0),  // 27, SQ41
    Bitboard(UINT64_C(1) << 28,                 0),  // 28, SQ42
    Bitboard(UINT64_C(1) << 29,                 0),  // 29, SQ43
    Bitboard(UINT64_C(1) << 30,                 0),  // 30, SQ44
    Bitboard(UINT64_C(1) << 31,                 0),  // 31, SQ45
    Bitboard(UINT64_C(1) << 32,                 0),  // 32, SQ46
    Bitboard(UINT64_C(1) << 33,                 0),  // 33, SQ47
    Bitboard(UINT64_C(1) << 34,                 0),  // 34, SQ48
    Bitboard(UINT64_C(1) << 35,                 0),  // 35, SQ49
    Bitboard(UINT64_C(1) << 36,                 0),  // 36, SQ51
    Bitboard(UINT64_C(1) << 37,                 0),  // 37, SQ52
    Bitboard(UINT64_C(1) << 38,                 0),  // 38, SQ53
    Bitboard(UINT64_C(1) << 39,                 0),  // 39, SQ54
    Bitboard(UINT64_C(1) << 40,                 0),  // 40, SQ55
    Bitboard(UINT64_C(1) << 41,                 0),  // 41, SQ56
    Bitboard(UINT64_C(1) << 42,                 0),  // 42, SQ57
    Bitboard(UINT64_C(1) << 43,                 0),  // 43, SQ58
    Bitboard(UINT64_C(1) << 44,                 0),  // 44, SQ59
    Bitboard(UINT64_C(1) << 45,                 0),  // 45, SQ61
    Bitboard(UINT64_C(1) << 46,                 0),  // 46, SQ62
    Bitboard(UINT64_C(1) << 47,                 0),  // 47, SQ63
    Bitboard(UINT64_C(1) << 48,                 0),  // 48, SQ64
    Bitboard(UINT64_C(1) << 49,                 0),  // 49, SQ65
    Bitboard(UINT64_C(1) << 50,                 0),  // 50, SQ66
    Bitboard(UINT64_C(1) << 51,                 0),  // 51, SQ67
    Bitboard(UINT64_C(1) << 52,                 0),  // 52, SQ68
    Bitboard(UINT64_C(1) << 53,                 0),  // 53, SQ69
    Bitboard(UINT64_C(1) << 54,                 0),  // 54, SQ71
    Bitboard(UINT64_C(1) << 55,                 0),  // 55, SQ72
    Bitboard(UINT64_C(1) << 56,                 0),  // 56, SQ73
    Bitboard(UINT64_C(1) << 57,                 0),  // 57, SQ74
    Bitboard(UINT64_C(1) << 58,                 0),  // 58, SQ75
    Bitboard(UINT64_C(1) << 59,                 0),  // 59, SQ76
    Bitboard(UINT64_C(1) << 60,                 0),  // 60, SQ77
    Bitboard(UINT64_C(1) << 61,                 0),  // 61, SQ78
    Bitboard(UINT64_C(1) << 62,                 0),  // 62, SQ79
    Bitboard(                0, UINT64_C(1) <<  0),  // 63, SQ81
    Bitboard(                0, UINT64_C(1) <<  1),  // 64, SQ82
    Bitboard(                0, UINT64_C(1) <<  2),  // 65, SQ83
    Bitboard(                0, UINT64_C(1) <<  3),  // 66, SQ84
    Bitboard(                0, UINT64_C(1) <<  4),  // 67, SQ85
    Bitboard(                0, UINT64_C(1) <<  5),  // 68, SQ86
    Bitboard(                0, UINT64_C(1) <<  6),  // 69, SQ87
    Bitboard(                0, UINT64_C(1) <<  7),  // 70, SQ88
    Bitboard(                0, UINT64_C(1) <<  8),  // 71, SQ89
    Bitboard(                0, UINT64_C(1) <<  9),  // 72, SQ91
    Bitboard(                0, UINT64_C(1) << 10),  // 73, SQ92
    Bitboard(                0, UINT64_C(1) << 11),  // 74, SQ93
    Bitboard(                0, UINT64_C(1) << 12),  // 75, SQ94
    Bitboard(                0, UINT64_C(1) << 13),  // 76, SQ95
    Bitboard(                0, UINT64_C(1) << 14),  // 77, SQ96
    Bitboard(                0, UINT64_C(1) << 15),  // 78, SQ97
    Bitboard(                0, UINT64_C(1) << 16),  // 79, SQ98
    Bitboard(                0, UINT64_C(1) << 17)   // 80, SQ99
};

const Bitboard FileMask[FileNum] = {
    File1Mask, File2Mask, File3Mask, File4Mask, File5Mask, File6Mask, File7Mask, File8Mask, File9Mask
};

const Bitboard RankMask[RankNum] = {
    Rank1Mask, Rank2Mask, Rank3Mask, Rank4Mask, Rank5Mask, Rank6Mask, Rank7Mask, Rank8Mask, Rank9Mask
};

const Bitboard InFrontMask[ColorNum][RankNum] = {
    { InFrontOfRank1Black, InFrontOfRank2Black, InFrontOfRank3Black, InFrontOfRank4Black, InFrontOfRank5Black, InFrontOfRank6Black, InFrontOfRank7Black, InFrontOfRank8Black, InFrontOfRank9Black },
    { InFrontOfRank1White, InFrontOfRank2White, InFrontOfRank3White, InFrontOfRank4White, InFrontOfRank5White, InFrontOfRank6White, InFrontOfRank7White, InFrontOfRank8White, InFrontOfRank9White }
};

// これらは一度値を設定したら二度と変更しない。
// 本当は const 化したい。
Bitboard LanceAttack[ColorNum][SquareNum][128];
Bitboard RookAttackRankToMask[SquareNum][2];
Bitboard256 BishopAttackToMask[SquareNum][2];

Bitboard KingAttack[SquareNum];
Bitboard GoldAttack[ColorNum][SquareNum];
Bitboard SilverAttack[ColorNum][SquareNum];
Bitboard KnightAttack[ColorNum][SquareNum];
Bitboard PawnAttack[ColorNum][SquareNum];

Bitboard BetweenBB[SquareNum][SquareNum];

Bitboard RookAttackToEdge[SquareNum];
Bitboard BishopAttackToEdge[SquareNum];
Bitboard LanceAttackToEdge[ColorNum][SquareNum];

Bitboard GoldCheckTable[ColorNum][SquareNum];
Bitboard SilverCheckTable[ColorNum][SquareNum];
Bitboard KnightCheckTable[ColorNum][SquareNum];
Bitboard LanceCheckTable[ColorNum][SquareNum];
Bitboard PawnCheckTable[ColorNum][SquareNum];
Bitboard BishopCheckTable[ColorNum][SquareNum];
Bitboard HorseCheckTable[ColorNum][SquareNum];

Bitboard Neighbor5x5Table[SquareNum]; // 25 近傍
