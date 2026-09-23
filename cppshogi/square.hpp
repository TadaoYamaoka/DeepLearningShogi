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

#ifndef APERY_SQUARE_HPP
#define APERY_SQUARE_HPP

#include "overloadEnumOperators.hpp"
#include "common.hpp"
#include "color.hpp"

// 盤面を [0, 80] の整数の index で表す
// Bitboard のビットが縦に並んでいて、
// 0 ビット目から順に、以下の位置と対応させる。
// SQ11 = 1一, SQ19 = 1九, SQ99 = 9九
enum Square {
    SQ11, SQ12, SQ13, SQ14, SQ15, SQ16, SQ17, SQ18, SQ19,
    SQ21, SQ22, SQ23, SQ24, SQ25, SQ26, SQ27, SQ28, SQ29,
    SQ31, SQ32, SQ33, SQ34, SQ35, SQ36, SQ37, SQ38, SQ39,
    SQ41, SQ42, SQ43, SQ44, SQ45, SQ46, SQ47, SQ48, SQ49,
    SQ51, SQ52, SQ53, SQ54, SQ55, SQ56, SQ57, SQ58, SQ59,
    SQ61, SQ62, SQ63, SQ64, SQ65, SQ66, SQ67, SQ68, SQ69,
    SQ71, SQ72, SQ73, SQ74, SQ75, SQ76, SQ77, SQ78, SQ79,
    SQ81, SQ82, SQ83, SQ84, SQ85, SQ86, SQ87, SQ88, SQ89,
    SQ91, SQ92, SQ93, SQ94, SQ95, SQ96, SQ97, SQ98, SQ99,
    SquareNum, // = 81
    SquareBegin = 0,
    SquareNoLeftNum = SQ61,
    B_hand_pawn   = SquareNum     + -1,
    B_hand_lance  = B_hand_pawn   + 18,
    B_hand_knight = B_hand_lance  +  4,
    B_hand_silver = B_hand_knight +  4,
    B_hand_gold   = B_hand_silver +  4,
    B_hand_bishop = B_hand_gold   +  4,
    B_hand_rook   = B_hand_bishop +  2,
    W_hand_pawn   = B_hand_rook   +  2,
    W_hand_lance  = W_hand_pawn   + 18,
    W_hand_knight = W_hand_lance  +  4,
    W_hand_silver = W_hand_knight +  4,
    W_hand_gold   = W_hand_silver +  4,
    W_hand_bishop = W_hand_gold   +  4,
    W_hand_rook   = W_hand_bishop +  2,
    SquareHandNum = W_hand_rook   +  3
};
OverloadEnumOperators(Square);

// 筋
enum File {
    File1, File2, File3, File4, File5, File6, File7, File8, File9, FileNum,
    FileNoLeftNum = File6,
    FileBegin = 0,
    // 画面表示など順序が決まっているループに使う。
    // ループで <, > を使わない事で、レイアウトを変えても変更点が少なくて済む。
    FileDeltaE = -1, FileDeltaW = 1,
    File1Wall = File1 + FileDeltaE, // File1 の右の壁の位置。
    File9Wall = File9 + FileDeltaW, // File9 の左の壁の位置。
};
OverloadEnumOperators(File);
inline int abs(const File f) { return std::abs(static_cast<int>(f)); }

// 段
enum Rank {
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9, RankNum,
    RankBegin = 0,
    // 画面表示など順序が決まっているループに使う。
    // ループで <, > を使わない事で、レイアウトを変えても変更点が少なくて済む。
    RankDeltaN = -1, RankDeltaS = 1,
    Rank1Wall = Rank1 + RankDeltaN, // Rank1 の上の壁の位置。
    Rank9Wall = Rank9 + RankDeltaS, // Rank9 の下の壁の位置。
};
OverloadEnumOperators(Rank);
inline int abs(const Rank r) { return std::abs(static_cast<int>(r)); }

// 先手のときは BRANK, 後手のときは WRANK より target が前の段にあるなら true を返す。
template <Color US, Rank BRANK, Rank WRANK>
inline bool isInFrontOf(const Rank target) { return (US == Black ? (target < BRANK) : (WRANK < target)); }

template <Color US, Rank BRANK, Rank WRANK>
inline bool isBehind(const Rank target) { return (US == Black ? (BRANK < target) : (target < WRANK)); }

template <Color US, File BFILE, File WFILE>
inline bool isLeftOf(const File target) { return (US == Black ? (BFILE < target) : (target < WFILE)); }

template <Color US, File BFILE, File WFILE>
inline bool isRightOf(const File target) { return (US == Black ? (target < BFILE) : (WFILE < target)); }

enum SquareDelta {
    DeltaNothing = 0, // 同一の Square にあるとき
    DeltaN = -1, DeltaE = -9, DeltaS = 1, DeltaW = 9,
    DeltaNE = DeltaN + DeltaE,
    DeltaSE = DeltaS + DeltaE,
    DeltaSW = DeltaS + DeltaW,
    DeltaNW = DeltaN + DeltaW
};
OverloadEnumOperators(SquareDelta);

inline Square operator + (const Square lhs, const SquareDelta rhs) { return lhs + static_cast<Square>(rhs); }
inline void operator += (Square& lhs, const SquareDelta rhs) { lhs = lhs + static_cast<Square>(rhs); }
inline Square operator - (const Square lhs, const SquareDelta rhs) { return lhs - static_cast<Square>(rhs); }
inline void operator -= (Square& lhs, const SquareDelta rhs) { lhs = lhs - static_cast<Square>(rhs); }

inline bool isInFile(const File f) { return (0 <= f) && (f < FileNum); }
inline bool isInRank(const Rank r) { return (0 <= r) && (r < RankNum); }
// s が Square の中に含まれているか判定
inline bool isInSquare(const Square s) { return (0 <= s) && (s < SquareNum); }
// File, Rank のどちらかがおかしいかも知れない時は、
// こちらを使う。
// こちらの方が遅いが、どんな File, Rank にも対応している。
inline bool isInSquare(const File f, const Rank r) { return isInFile(f) && isInRank(r); }

// 速度が必要な場面で使用するなら、テーブル引きの方が有効だと思う。
inline constexpr Square makeSquare(const File f, const Rank r) {
    return static_cast<Square>(static_cast<int>(f) * 9 + static_cast<int>(r));
}

const Rank SquareToRank[SquareNum] = {
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9,
    Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Rank9
};

const File SquareToFile[SquareNum] = {
    File1, File1, File1, File1, File1, File1, File1, File1, File1,
    File2, File2, File2, File2, File2, File2, File2, File2, File2,
    File3, File3, File3, File3, File3, File3, File3, File3, File3,
    File4, File4, File4, File4, File4, File4, File4, File4, File4,
    File5, File5, File5, File5, File5, File5, File5, File5, File5,
    File6, File6, File6, File6, File6, File6, File6, File6, File6,
    File7, File7, File7, File7, File7, File7, File7, File7, File7,
    File8, File8, File8, File8, File8, File8, File8, File8, File8,
    File9, File9, File9, File9, File9, File9, File9, File9, File9
};

// 速度が必要な場面で使用する。
inline Rank makeRank(const Square s) {
    assert(isInSquare(s));
    return SquareToRank[s];
}
inline File makeFile(const Square s) {
    assert(isInSquare(s));
    return SquareToFile[s];
}

// 位置関係、方向
// ボナンザそのまま
// でもあまり使わないので普通の enum と同様に 0 から順に値を付けて行けば良いと思う。
enum Direction {
    DirecMisc     = Binary<  0>::value, // 縦、横、斜めの位置に無い場合
    DirecFile     = Binary< 10>::value, // 縦
    DirecRank     = Binary< 11>::value, // 横
    DirecDiagNESW = Binary<100>::value, // 右上から左下
    DirecDiagNWSE = Binary<101>::value, // 左上から右下
    DirecCross    = Binary< 10>::value, // 縦、横
    DirecDiag     = Binary<100>::value, // 斜め
};
OverloadEnumOperators(Direction);

// 2つの位置関係のテーブル
extern Direction SquareRelation[SquareNum][SquareNum];
inline Direction squareRelation(const Square sq1, const Square sq2) { return SquareRelation[sq1][sq2]; }

// 何かの駒で一手で行ける位置関係についての距離のテーブル。桂馬の位置は距離1とする。
extern int SquareDistance[SquareNum][SquareNum];
inline int squareDistance(const Square sq1, const Square sq2) { return SquareDistance[sq1][sq2]; }

// from, to, ksq が 縦横斜めの同一ライン上にあれば true を返す。
template <bool FROM_KSQ_NEVER_BE_DIRECMISC>
inline bool isAligned(const Square from, const Square to, const Square ksq) {
    const Direction direc = squareRelation(from, ksq);
    if (FROM_KSQ_NEVER_BE_DIRECMISC) {
        assert(direc != DirecMisc);
        return (direc == squareRelation(from, to));
    }
    else
        return (direc != DirecMisc && direc == squareRelation(from, to));
}

inline char fileToCharUSI(const File f) { return '1' + f; }
// todo: アルファベットが辞書順に並んでいない処理系があるなら対応すること。
inline char rankToCharUSI(const Rank r) {
    static_assert('a' + 1 == 'b', "");
    static_assert('a' + 2 == 'c', "");
    static_assert('a' + 3 == 'd', "");
    static_assert('a' + 4 == 'e', "");
    static_assert('a' + 5 == 'f', "");
    static_assert('a' + 6 == 'g', "");
    static_assert('a' + 7 == 'h', "");
    static_assert('a' + 8 == 'i', "");
    return 'a' + r;
}
inline std::string squareToStringUSI(const Square sq) {
    const Rank r = makeRank(sq);
    const File f = makeFile(sq);
    const char ch[] = {fileToCharUSI(f), rankToCharUSI(r), '\0'};
    return std::string(ch);
}

inline char fileToCharCSA(const File f) { return '1' + f; }
inline char rankToCharCSA(const Rank r) { return '1' + r; }
inline std::string squareToStringCSA(const Square sq) {
    const Rank r = makeRank(sq);
    const File f = makeFile(sq);
    const char ch[] = {fileToCharCSA(f), rankToCharCSA(r), '\0'};
    return std::string(ch);
}

inline File charCSAToFile(const char c) { return static_cast<File>(c - '1'); }
inline Rank charCSAToRank(const char c) { return static_cast<Rank>(c - '1'); }
inline File charUSIToFile(const char c) { return static_cast<File>(c - '1'); }
inline Rank charUSIToRank(const char c) { return static_cast<Rank>(c - 'a'); }

// 後手の位置を先手の位置へ変換
inline constexpr Square inverse(const Square sq) { return SquareNum - 1 - sq; }
// 左右変換
inline constexpr File inverse(const File f) { return FileNum - 1 - f; }
// 上下変換
inline constexpr Rank inverse(const Rank r) { return RankNum - 1 - r; }
// Square の左右だけ変換
inline Square inverseFile(const Square sq) { return makeSquare(inverse(makeFile(sq)), makeRank(sq)); }

inline constexpr Square inverseIfWhite(const Color c, const Square sq) { return (c == Black ? sq : inverse(sq)); }

inline bool canPromote(const Color c, const Rank fromOrToRank) {
#if 1
    static_assert(Black == 0, "");
    static_assert(Rank1 == 0, "");
    return static_cast<bool>(0x1c00007u & (1u << ((c << 4) + fromOrToRank)));
#else
    // 同じ意味。
    return (c == Black ? isInFrontOf<Black, Rank4, Rank6>(fromOrToRank) : isInFrontOf<White, Rank4, Rank6>(fromOrToRank));
#endif
}
// 移動元、もしくは移動先の升sqを与えたときに、そこが成れるかどうかを判定する。
inline bool canPromote(const Color c, const Square fromOrTo) {
    return canPromote(c, makeRank(fromOrTo));
}
// 移動元と移動先の升を与えて、成れるかどうかを判定する。
// (移動元か移動先かのどちらかが敵陣であれば成れる)
inline bool canPromote(const Color c, const Square from, const Square to)
{
    return canPromote(c, from) || canPromote(c, to);
}

inline bool isOpponentField(const Color c, const Rank r) {
    return canPromote(c, r);
}

// 以下は、Aperyにはなかった処理
// やねうら王から移植した

// --------------------
//   壁つきの升表現
// --------------------

// 長い利きを更新するときにある升からある方向に駒にぶつかるまでずっと利きを更新していきたいことがあるが、
// sqの升が盤外であるかどうかを判定する簡単な方法がない。そこで、Squareの表現を拡張して盤外であることを検出
// できるようにする。

// bit 0..7   : Squareと同じ意味
// bit 8      : Squareからのborrow用に1にしておく
// bit 9..13  : いまの升から盤外まで何升右に升があるか(ここがマイナスになるとborrowでbit13が1になる)
// bit 14..18 : いまの升から盤外まで何升上に(略
// bit 19..23 : いまの升から盤外まで何升下に(略
// bit 24..28 : いまの升から盤外まで何升左に(略
enum SquareWithWall : int32_t {
    // 相対移動するときの差分値
    SQWW_R = DeltaE - (1 << 9) + (1 << 24), SQWW_U = DeltaN - (1 << 14) + (1 << 19), SQWW_D = -int(SQWW_U), SQWW_L = -int(SQWW_R),
    SQWW_RU = int(SQWW_R) + int(SQWW_U), SQWW_RD = int(SQWW_R) + int(SQWW_D), SQWW_LU = int(SQWW_L) + int(SQWW_U), SQWW_LD = int(SQWW_L) + int(SQWW_D),

    // SQ_11の地点に対応する値(他の升はこれ相対で事前に求めテーブルに格納)
    SQWW_11 = SQ11 | (1 << 8) /* bit8 is 1 */ | (0 << 9) /*右に0升*/ | (0 << 14) /*上に0升*/ | (8 << 19) /*下に8升*/ | (8 << 24) /*左に8升*/,

    // SQWW_RIGHTなどを足して行ったときに盤外に行ったときのborrow bitの集合
    SQWW_BORROW_MASK = (1 << 13) | (1 << 18) | (1 << 23) | (1 << 28),
};
OverloadEnumOperators(SquareWithWall);

// 型変換。下位8bit == Square
constexpr Square sqww_to_sq(SquareWithWall sqww) { return Square(sqww & 0xff); }

extern SquareWithWall sqww_table[SquareNum + 1];

// 型変換。Square型から。
static SquareWithWall to_sqww(Square sq) { return sqww_table[sq]; }

// 盤内か。壁(盤外)だとfalseになる。
constexpr bool is_ok(SquareWithWall sqww) { return (sqww & SQWW_BORROW_MASK) == 0; }

// --------------------
//        方角
// --------------------

// Long Effect Libraryの一部。これは8近傍、24近傍の利きを直列化したり方角を求めたりするライブラリ。
namespace Effect8
{
    // 方角を表す。遠方駒の利きや、玉から見た方角を表すのに用いる。
    // bit0..右上、bit1..右、bit2..右下、bit3..上、bit4..下、bit5..左上、bit6..左、bit7..左下
    // 同時に複数のbitが1であることがありうる。
    enum Directions : uint8_t {
        DIRECTIONS_ZERO = 0, DIRECTIONS_RU = 1, DIRECTIONS_R = 2, DIRECTIONS_RD = 4,
        DIRECTIONS_U = 8, DIRECTIONS_D = 16, DIRECTIONS_LU = 32, DIRECTIONS_L = 64, DIRECTIONS_LD = 128,
        DIRECTIONS_CROSS = DIRECTIONS_U | DIRECTIONS_D | DIRECTIONS_R | DIRECTIONS_L,
        DIRECTIONS_DIAG = DIRECTIONS_RU | DIRECTIONS_RD | DIRECTIONS_LU | DIRECTIONS_LD,
    };

    // sq1にとってsq2がどのdirectionにあるか。
    // "Direction"ではなく"Directions"を返したほうが、縦横十字方向や、斜め方向の位置関係にある場合、
    // DIRECTIONS_CROSSやDIRECTIONS_DIAGのような定数が使えて便利。
    extern Directions direc_table[SquareNum + 1][SquareNum + 1];
    static Directions directions_of(Square sq1, Square sq2) { return direc_table[sq1][sq2]; }

    // Directionsをpopしたもの。複数の方角を同時に表すことはない。
    // おまけで桂馬の移動も追加しておく。
    enum Direct {
        DIRECT_RU, DIRECT_R, DIRECT_RD, DIRECT_U, DIRECT_D, DIRECT_LU, DIRECT_L, DIRECT_LD,
        DIRECT_NB, DIRECT_ZERO = 0, DIRECT_RUU = 8, DIRECT_LUU, DIRECT_RDD, DIRECT_LDD, DIRECT_NB_PLUS4
    };
    OverloadEnumOperators(Direct);

    // ある方角の反対の方角(180度回転させた方角)を得る。
    constexpr Direct operator~(Direct d) {
        // Directの定数値を変更したら、この関数はうまく動作しない。
        static_assert(Effect8::DIRECT_R == 1, "");
        static_assert(Effect8::DIRECT_L == 6, "");
        // DIRECT_RUUなどは引数に渡してはならない。
        return Direct(7 - d);
    }

    // DirectからDirectionsへの逆変換
    constexpr Directions to_directions(Direct d) { return Directions(1 << d); }

    constexpr bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB_PLUS4; }

    // DirectをSquareWithWall型の差分値で表現したもの。
    constexpr SquareWithWall DirectToDeltaWW_[DIRECT_NB] = { SQWW_RU,SQWW_R,SQWW_RD,SQWW_U,SQWW_D,SQWW_LU,SQWW_L,SQWW_LD, };
    constexpr SquareWithWall DirectToDeltaWW(Direct d) { return DirectToDeltaWW_[d]; }
}

#endif // #ifndef APERY_SQUARE_HPP
