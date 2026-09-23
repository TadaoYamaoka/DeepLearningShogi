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

#ifndef APERY_BITBOARD_HPP
#define APERY_BITBOARD_HPP

#include "common.hpp"
#include "square.hpp"
#include "color.hpp"

class Bitboard;
extern const Bitboard SetMaskBB[SquareNum];

class Bitboard {
public:
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
    Bitboard& operator = (const Bitboard& rhs) {
        _mm_store_si128(&this->m_, rhs.m_);
        return *this;
    }
    Bitboard(const Bitboard& bb) {
        _mm_store_si128(&this->m_, bb.m_);
    }
#endif
    Bitboard() {}
    Bitboard(const u64 v0, const u64 v1) {
        this->p_[0] = v0;
        this->p_[1] = v1;
    }
    u64 p(const int index) const { return p_[index]; }
    void set(const int index, const u64 val) { p_[index] = val; }
    u64 merge() const { return this->p(0) | this->p(1); }
    explicit operator bool() const {
#ifdef HAVE_SSE4
        return !(_mm_testz_si128(this->m_, _mm_set1_epi8(static_cast<char>(0xffu))));
#else
        return (this->merge() ? true : false);
#endif
    }
    bool isAny() const { return static_cast<bool>(*this); }
    // これはコードが見難くなるけど仕方ない。
    bool andIsAny(const Bitboard& bb) const {
#ifdef HAVE_SSE4
        return !(_mm_testz_si128(this->m_, bb.m_));
#else
        return (*this & bb).isAny();
#endif
    }
    Bitboard operator ~ () const {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        Bitboard tmp;
        _mm_store_si128(&tmp.m_, _mm_andnot_si128(this->m_, _mm_set1_epi8(static_cast<char>(0xffu))));
        return tmp;
#else
        return Bitboard(~this->p(0), ~this->p(1));
#endif
    }
    Bitboard operator &= (const Bitboard& rhs) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_and_si128(this->m_, rhs.m_));
#else
        this->p_[0] &= rhs.p(0);
        this->p_[1] &= rhs.p(1);
#endif
        return *this;
    }
    Bitboard operator |= (const Bitboard& rhs) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_or_si128(this->m_, rhs.m_));
#else
        this->p_[0] |= rhs.p(0);
        this->p_[1] |= rhs.p(1);
#endif
        return *this;
    }
    Bitboard operator ^= (const Bitboard& rhs) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_xor_si128(this->m_, rhs.m_));
#else
        this->p_[0] ^= rhs.p(0);
        this->p_[1] ^= rhs.p(1);
#endif
        return *this;
    }
    Bitboard operator <<= (const int i) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_slli_epi64(this->m_, i));
#else
        this->p_[0] <<= i;
        this->p_[1] <<= i;
#endif
        return *this;
    }
    Bitboard operator >>= (const int i) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_srli_epi64(this->m_, i));
#else
        this->p_[0] >>= i;
        this->p_[1] >>= i;
#endif
        return *this;
    }
    Bitboard operator & (const Bitboard& rhs) const { return Bitboard(*this) &= rhs; }
    Bitboard operator | (const Bitboard& rhs) const { return Bitboard(*this) |= rhs; }
    Bitboard operator ^ (const Bitboard& rhs) const { return Bitboard(*this) ^= rhs; }
    Bitboard operator << (const int i) const { return Bitboard(*this) <<= i; }
    Bitboard operator >> (const int i) const { return Bitboard(*this) >>= i; }
    bool operator == (const Bitboard& rhs) const {
#ifdef HAVE_SSE4
        return (_mm_testc_si128(_mm_cmpeq_epi8(this->m_, rhs.m_), _mm_set1_epi8(static_cast<char>(0xffu))) ? true : false);
#else
        return (this->p(0) == rhs.p(0)) && (this->p(1) == rhs.p(1));
#endif
    }
    bool operator != (const Bitboard& rhs) const { return !(*this == rhs); }
    // これはコードが見難くなるけど仕方ない。
    Bitboard andEqualNot(const Bitboard& bb) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        _mm_store_si128(&this->m_, _mm_andnot_si128(bb.m_, this->m_));
#else
        (*this) &= ~bb;
#endif
        return *this;
    }
    // これはコードが見難くなるけど仕方ない。
    Bitboard notThisAnd(const Bitboard& bb) const {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        Bitboard temp;
        _mm_store_si128(&temp.m_, _mm_andnot_si128(this->m_, bb.m_));
        return temp;
#else
        return ~(*this) & bb;
#endif
    }
    bool isSet(const Square sq) const {
        assert(isInSquare(sq));
        return andIsAny(SetMaskBB[sq]);
    }
    void setBit(const Square sq) { *this |= SetMaskBB[sq]; }
    void clearBit(const Square sq) { andEqualNot(SetMaskBB[sq]); }
    void xorBit(const Square sq) { (*this) ^= SetMaskBB[sq]; }
    void xorBit(const Square sq1, const Square sq2) { (*this) ^= (SetMaskBB[sq1] | SetMaskBB[sq2]); }
    // Bitboard の right 側だけの要素を調べて、最初に 1 であるマスの index を返す。
    // そのマスを 0 にする。
    // Bitboard の right 側が 0 でないことを前提にしている。
    FORCE_INLINE Square firstOneRightFromSQ11() {
        const Square sq = static_cast<Square>(firstOneFromLSB(this->p(0)));
        // LSB 側の最初の 1 の bit を 0 にする
        this->p_[0] &= this->p(0) - 1;
        return sq;
    }
    // Bitboard の left 側だけの要素を調べて、最初に 1 であるマスの index を返す。
    // そのマスを 0 にする。
    // Bitboard の left 側が 0 でないことを前提にしている。
    FORCE_INLINE Square firstOneLeftFromSQ81() {
        const Square sq = static_cast<Square>(firstOneFromLSB(this->p(1)) + 63);
        // LSB 側の最初の 1 の bit を 0 にする
        this->p_[1] &= this->p(1) - 1;
        return sq;
    }
    // Bitboard を SQ11 から SQ99 まで調べて、最初に 1 であるマスの index を返す。
    // そのマスを 0 にする。
    // Bitboard が allZeroBB() でないことを前提にしている。
    // VC++ の _BitScanForward() は入力が 0 のときに 0 を返す仕様なので、
    // 最初に 0 でないか判定するのは少し損。
    FORCE_INLINE Square firstOneFromSQ11() {
        if (this->p(0))
            return firstOneRightFromSQ11();
        return firstOneLeftFromSQ81();
    }
    // 返す位置を 0 にしないバージョン。
    FORCE_INLINE Square constFirstOneRightFromSQ11() const { return static_cast<Square>(firstOneFromLSB(this->p(0))); }
    FORCE_INLINE Square constFirstOneLeftFromSQ81() const { return static_cast<Square>(firstOneFromLSB(this->p(1)) + 63); }
    FORCE_INLINE Square constFirstOneFromSQ11() const {
        if (this->p(0))
            return constFirstOneRightFromSQ11();
        return constFirstOneLeftFromSQ81();
    }
    // Bitboard の 1 の bit を数える。
    // Crossover は、merge() すると 1 である bit が重なる可能性があるなら true
    template <bool Crossover = true>
    int popCount() const { return (Crossover ? count1s(p(0)) + count1s(p(1)) : count1s(merge())); }
    // bit が 1 つだけ立っているかどうかを判定する。
    // Crossover は、merge() すると 1 である bit が重なる可能性があるなら true
    template <bool Crossover = true>
    bool isOneBit() const {
#if defined (HAVE_SSE42)
        return (this->popCount<Crossover>() == 1);
#else
        if (!isAny())
            return false;
        else if (this->p(0))
            return !((this->p(0) & (this->p(0) - 1)) | this->p(1));
        return !(this->p(1) & (this->p(1) - 1));
#endif
    }
    // byte単位で入れ替えたBitboardを返す。
    // 飛車の利きの右方向と角の利きの右上、右下方向を求める時に使う。
    Bitboard byteReverse() const {
#if defined (HAVE_SSE4)
        const __m128i shuffle = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        Bitboard b0;
        b0.m_ = _mm_shuffle_epi8(m_, shuffle);
        return b0;
#else
        Bitboard b0;
        b0.p_[0] = bswap64(p_[1]);
        b0.p_[1] = bswap64(p_[0]);
        return b0;
#endif
    }
    // SSE2のunpackを実行して返す。
    static void unpack(const Bitboard& hiIn, const Bitboard& loIn, Bitboard& hiOut, Bitboard& loOut) {
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
        hiOut.m_ = _mm_unpackhi_epi64(loIn.m_, hiIn.m_);
        loOut.m_ = _mm_unpacklo_epi64(loIn.m_, hiIn.m_);
#else
        hiOut.p_[0] = loIn.p_[1];
        hiOut.p_[1] = hiIn.p_[1];

        loOut.p_[0] = loIn.p_[0];
        loOut.p_[1] = hiIn.p_[0];
#endif
    }
    // 2組のBitboardを、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
    // 128bit整数とみなして1引き算したBitboardを返す。
    static void decrement(const Bitboard& hiIn, const Bitboard& loIn, Bitboard& hiOut, Bitboard& loOut)
    {
#if defined (HAVE_SSE42)
        // loが0の時だけ1減算するときにhiからの桁借りが生じるので、
        // hi += (lo == 0) ? -1 : 0;
        // みたいな処理で良い。
        hiOut.m_ = _mm_add_epi64(hiIn.m_, _mm_cmpeq_epi64(loIn.m_, _mm_setzero_si128()));

        //  1減算する
        loOut.m_ = _mm_add_epi64(loIn.m_, _mm_set1_epi64x(-1LL));
#else
        // bool型はtrueだと(暗黙の型変換で)1だとみなされる。
        hiOut.p_[0] = hiIn.p_[0] - (loIn.p_[0] == 0);
        hiOut.p_[1] = hiIn.p_[1] - (loIn.p_[1] == 0);

        loOut.p_[0] = loIn.p_[0] - 1;
        loOut.p_[1] = loIn.p_[1] - 1;
#endif
    }

    // for debug
    void printBoard() const {
        std::cout << "   A  B  C  D  E  F  G  H  I\n";
        for (Rank r = Rank1; r != Rank9Wall; r += RankDeltaS) {
            std::cout << (9 - r);
            for (File f = File9; f != File1Wall; f += FileDeltaE)
                std::cout << (this->isSet(makeSquare(f, r)) ? "  X" : "  .");
            std::cout << "\n";
        }

        std::cout << std::endl;
    }

    // 指定した位置が Bitboard のどちらの u64 変数の要素か
    static int part(const Square sq) { return static_cast<int>(SQ79 < sq); }

private:
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
    union {
        u64 p_[2];
        __m128i m_;
    };
#else
    u64 p_[2];  // p_[0] : 先手から見て、1一から7九までを縦に並べたbit. 63bit使用. right と呼ぶ。
                // p_[1] : 先手から見て、8一から1九までを縦に並べたbit. 18bit使用. left  と呼ぶ。
#endif

    friend class Bitboard256;
};

// Bitboard 2つを256bit registerで扱う。
// Qugiyの角の利きに使用する。
// cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal_210518.pdf
// やねうら王の実装を参考にした
class Bitboard256 {
public:
    Bitboard256() {}
#if defined (HAVE_AVX2)
    Bitboard256& operator = (const Bitboard256& rhs) { _mm256_store_si256(&this->m_, rhs.m_); return *this; }
    Bitboard256(const Bitboard256& bb) { _mm256_store_si256(&this->m_, bb.m_); }

    // 同じBitboardを2つに複製し、それをBitboard256とする。
    Bitboard256(const Bitboard& b1) { m_ = _mm256_broadcastsi128_si256(b1.m_); }

    // 2つのBitboardを合わせたBitboard256を作る。
    Bitboard256(const Bitboard& b1, const Bitboard& b2) {
        // m = _mm256_set_epi64x(b2.p[1],b2.p[0],b1.p[1],b1.p[0]);
        m_ = _mm256_castsi128_si256(b1.m_);        // 256bitにcast(上位は0)。これはcompiler向けの命令。
        m_ = _mm256_inserti128_si256(m_, b2.m_, 1); // 上位128bitにb2.mを代入
    }
#else
    Bitboard256(const Bitboard& b1, const Bitboard& b2) { p_[0] = b1.p_[0]; p_[1] = b1.p_[1]; p_[2] = b2.p_[0]; p_[3] = b2.p_[1]; }
    Bitboard256(const Bitboard& b1) { p_[0] = p_[2] = b1.p_[0]; p_[1] = p_[3] = b1.p_[1]; }
#endif
    Bitboard256 operator &= (const Bitboard256& rhs) {
#if defined (HAVE_AVX2)
        _mm256_store_si256(&this->m_, _mm256_and_si256(this->m_, rhs.m_));
#else
        this->p_[0] &= rhs.p_[0];
        this->p_[1] &= rhs.p_[1];
        this->p_[2] &= rhs.p_[2];
        this->p_[3] &= rhs.p_[3];
#endif
        return *this;
    }
    Bitboard256 operator |= (const Bitboard256& rhs) {
#if defined (HAVE_AVX2)
        _mm256_store_si256(&this->m_, _mm256_or_si256(this->m_, rhs.m_));
#else
        this->p_[0] |= rhs.p_[0];
        this->p_[1] |= rhs.p_[1];
        this->p_[2] |= rhs.p_[2];
        this->p_[3] |= rhs.p_[3];
#endif
        return *this;
    }
    Bitboard256 operator ^= (const Bitboard256& rhs) {
#if defined (HAVE_AVX2)
        _mm256_store_si256(&this->m_, _mm256_xor_si256(this->m_, rhs.m_));
#else
        this->p_[0] ^= rhs.p_[0];
        this->p_[1] ^= rhs.p_[1];
        this->p_[2] ^= rhs.p_[2];
        this->p_[3] ^= rhs.p_[3];
#endif
        return *this;
    }
    Bitboard256 operator & (const Bitboard256& rhs) const { return Bitboard256(*this) &= rhs; }
    Bitboard256 operator | (const Bitboard256& rhs) const { return Bitboard256(*this) |= rhs; }
    Bitboard256 operator ^ (const Bitboard256& rhs) const { return Bitboard256(*this) ^= rhs; }
    // byte単位で入れ替えたBitboardを返す。
    // 角の利きの右上、右下方向を求める時に使う。
    Bitboard256 byteReverse() const {
#if defined (HAVE_AVX2)
        const __m256i shuffle = _mm256_set_epi8
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        Bitboard256 b0;
        b0.m_ = _mm256_shuffle_epi8(m_, shuffle);
        return b0;
#else
        Bitboard256 b0;
        b0.p_[0] = bswap64(p_[3]);
        b0.p_[1] = bswap64(p_[2]);
        b0.p_[2] = bswap64(p_[1]);
        b0.p_[3] = bswap64(p_[0]);
        return b0;
#endif
    }
    // 保持している2つの盤面を重ね合わせた(OR)Bitboardを返す。
    Bitboard merge() const
    {
#if defined (HAVE_AVX2)
        Bitboard b;
        b.m_ = _mm_or_si128(_mm256_castsi256_si128(m_), _mm256_extracti128_si256(m_, 1));
        return b;
#else
        Bitboard b;
        b.p_[0] = p_[0] | p_[2];
        b.p_[1] = p_[1] | p_[3];
        return b;
#endif
    }
    // SSE2のunpackを実行して返す。
    static void unpack(const Bitboard256& hiIn, const Bitboard256& loIn, Bitboard256& hiOut, Bitboard256& loOut) {
#if defined (HAVE_AVX2)
        hiOut.m_ = _mm256_unpackhi_epi64(loIn.m_, hiIn.m_);
        loOut.m_ = _mm256_unpacklo_epi64(loIn.m_, hiIn.m_);
#else
        hiOut.p_[0] = loIn.p_[1];
        hiOut.p_[1] = hiIn.p_[1];
        hiOut.p_[2] = loIn.p_[3];
        hiOut.p_[3] = hiIn.p_[3];

        loOut.p_[0] = loIn.p_[0];
        loOut.p_[1] = hiIn.p_[0];
        loOut.p_[2] = loIn.p_[2];
        loOut.p_[3] = hiIn.p_[2];
#endif
    }
    // 2組のBitboard256を、それぞれ64bitのhi×2とlo×2と見たときに(unpackするとそうなる)
    // 128bit整数とみなして1引き算したBitboardを返す。
    static void decrement(const Bitboard256& hiIn, const Bitboard256& loIn, Bitboard256& hiOut, Bitboard256& loOut)
    {
#if defined (HAVE_AVX2)

        // loが0の時だけ1減算するときにhiからの桁借りが生じるので、
        // hi += (lo == 0) ? -1 : 0;
        // みたいな処理で良い。
        hiOut.m_ = _mm256_add_epi64(hiIn.m_, _mm256_cmpeq_epi64(loIn.m_, _mm256_setzero_si256()));

        //  1減算する
        loOut.m_ = _mm256_add_epi64(loIn.m_, _mm256_set1_epi64x(-1LL));
#else
        // bool型はtrueだと(暗黙の型変換で)1だとみなされる。
        hiOut.p_[0] = hiIn.p_[0] - (loIn.p_[0] == 0);
        hiOut.p_[1] = hiIn.p_[1] - (loIn.p_[1] == 0);
        hiOut.p_[2] = hiIn.p_[2] - (loIn.p_[2] == 0);
        hiOut.p_[3] = hiIn.p_[3] - (loIn.p_[3] == 0);

        loOut.p_[0] = loIn.p_[0] - 1;
        loOut.p_[1] = loIn.p_[1] - 1;
        loOut.p_[2] = loIn.p_[2] - 1;
        loOut.p_[3] = loIn.p_[3] - 1;
#endif
    }

private:
#if defined (HAVE_SSE2) || defined (HAVE_SSE4)
    union {
        u64 p_[4];
        __m256i m_;
    };
#else
    u64 p_[4];
#endif
};

inline Bitboard setMaskBB(const Square sq) { return SetMaskBB[sq]; }

// 実際に使用する部分の全て bit が立っている Bitboard
inline Bitboard allOneBB() { return Bitboard(UINT64_C(0x7fffffffffffffff), UINT64_C(0x000000000003ffff)); }
inline Bitboard allZeroBB() { return Bitboard(0, 0); }

// 指定した位置の属する file の bit を shift し、
// index を求める為に使用する。
const int Slide[SquareNum] = {
    1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
    10, 10, 10, 10, 10, 10, 10, 10, 10,
    19, 19, 19, 19, 19, 19, 19, 19, 19,
    28, 28, 28, 28, 28, 28, 28, 28, 28,
    37, 37, 37, 37, 37, 37, 37, 37, 37,
    46, 46, 46, 46, 46, 46, 46, 46, 46,
    55, 55, 55, 55, 55, 55, 55, 55, 55,
    1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ,
    10, 10, 10, 10, 10, 10, 10, 10, 10
};

const Bitboard File1Mask = Bitboard(UINT64_C(0x1ff) << (9 * 0), 0);
const Bitboard File2Mask = Bitboard(UINT64_C(0x1ff) << (9 * 1), 0);
const Bitboard File3Mask = Bitboard(UINT64_C(0x1ff) << (9 * 2), 0);
const Bitboard File4Mask = Bitboard(UINT64_C(0x1ff) << (9 * 3), 0);
const Bitboard File5Mask = Bitboard(UINT64_C(0x1ff) << (9 * 4), 0);
const Bitboard File6Mask = Bitboard(UINT64_C(0x1ff) << (9 * 5), 0);
const Bitboard File7Mask = Bitboard(UINT64_C(0x1ff) << (9 * 6), 0);
const Bitboard File8Mask = Bitboard(0, 0x1ff << (9 * 0));
const Bitboard File9Mask = Bitboard(0, 0x1ff << (9 * 1));

const Bitboard Rank1Mask = Bitboard(UINT64_C(0x40201008040201) << 0, 0x201 << 0);
const Bitboard Rank2Mask = Bitboard(UINT64_C(0x40201008040201) << 1, 0x201 << 1);
const Bitboard Rank3Mask = Bitboard(UINT64_C(0x40201008040201) << 2, 0x201 << 2);
const Bitboard Rank4Mask = Bitboard(UINT64_C(0x40201008040201) << 3, 0x201 << 3);
const Bitboard Rank5Mask = Bitboard(UINT64_C(0x40201008040201) << 4, 0x201 << 4);
const Bitboard Rank6Mask = Bitboard(UINT64_C(0x40201008040201) << 5, 0x201 << 5);
const Bitboard Rank7Mask = Bitboard(UINT64_C(0x40201008040201) << 6, 0x201 << 6);
const Bitboard Rank8Mask = Bitboard(UINT64_C(0x40201008040201) << 7, 0x201 << 7);
const Bitboard Rank9Mask = Bitboard(UINT64_C(0x40201008040201) << 8, 0x201 << 8);

extern const Bitboard FileMask[FileNum];
extern const Bitboard RankMask[RankNum];
extern const Bitboard InFrontMask[ColorNum][RankNum];

inline Bitboard fileMask(const File f) { return FileMask[f]; }
template <File F> inline Bitboard fileMask() {
    static_assert(FileBegin <= F && F < FileNum, "");
    return (F == File1 ? File1Mask
            : F == File2 ? File2Mask
            : F == File3 ? File3Mask
            : F == File4 ? File4Mask
            : F == File5 ? File5Mask
            : F == File6 ? File6Mask
            : F == File7 ? File7Mask
            : F == File8 ? File8Mask
            : /*F == File9 ?*/ File9Mask);
}

inline Bitboard rankMask(const Rank r) { return RankMask[r]; }
template <Rank R> inline Bitboard rankMask() {
    static_assert(RankBegin <= R && R < RankNum, "");
    return (R == Rank1 ? Rank1Mask
            : R == Rank2 ? Rank2Mask
            : R == Rank3 ? Rank3Mask
            : R == Rank4 ? Rank4Mask
            : R == Rank5 ? Rank5Mask
            : R == Rank6 ? Rank6Mask
            : R == Rank7 ? Rank7Mask
            : R == Rank8 ? Rank8Mask
            : /*R == Rank9 ?*/ Rank9Mask);
}

// 直接テーブル引きすべきだと思う。
inline Bitboard squareFileMask(const Square sq) {
    const File f = makeFile(sq);
    return fileMask(f);
}

// 直接テーブル引きすべきだと思う。
inline Bitboard squareRankMask(const Square sq) {
    const Rank r = makeRank(sq);
    return rankMask(r);
}

const Bitboard InFrontOfRank1Black = allZeroBB();
const Bitboard InFrontOfRank2Black = rankMask<Rank1>();
const Bitboard InFrontOfRank3Black = InFrontOfRank2Black | rankMask<Rank2>();
const Bitboard InFrontOfRank4Black = InFrontOfRank3Black | rankMask<Rank3>();
const Bitboard InFrontOfRank5Black = InFrontOfRank4Black | rankMask<Rank4>();
const Bitboard InFrontOfRank6Black = InFrontOfRank5Black | rankMask<Rank5>();
const Bitboard InFrontOfRank7Black = InFrontOfRank6Black | rankMask<Rank6>();
const Bitboard InFrontOfRank8Black = InFrontOfRank7Black | rankMask<Rank7>();
const Bitboard InFrontOfRank9Black = InFrontOfRank8Black | rankMask<Rank8>();

const Bitboard InFrontOfRank9White = allZeroBB();
const Bitboard InFrontOfRank8White = rankMask<Rank9>();
const Bitboard InFrontOfRank7White = InFrontOfRank8White | rankMask<Rank8>();
const Bitboard InFrontOfRank6White = InFrontOfRank7White | rankMask<Rank7>();
const Bitboard InFrontOfRank5White = InFrontOfRank6White | rankMask<Rank6>();
const Bitboard InFrontOfRank4White = InFrontOfRank5White | rankMask<Rank5>();
const Bitboard InFrontOfRank3White = InFrontOfRank4White | rankMask<Rank4>();
const Bitboard InFrontOfRank2White = InFrontOfRank3White | rankMask<Rank3>();
const Bitboard InFrontOfRank1White = InFrontOfRank2White | rankMask<Rank2>();

inline Bitboard inFrontMask(const Color c, const Rank r) { return InFrontMask[c][r]; }
template <Color C, Rank R> inline Bitboard inFrontMask() {
    static_assert(C == Black || C == White, "");
    static_assert(RankBegin <= R && R < RankNum, "");
    return (C == Black ? (R == Rank1 ? InFrontOfRank1Black
                          : R == Rank2 ? InFrontOfRank2Black
                          : R == Rank3 ? InFrontOfRank3Black
                          : R == Rank4 ? InFrontOfRank4Black
                          : R == Rank5 ? InFrontOfRank5Black
                          : R == Rank6 ? InFrontOfRank6Black
                          : R == Rank7 ? InFrontOfRank7Black
                          : R == Rank8 ? InFrontOfRank8Black
                          : /*R == Rank9 ?*/ InFrontOfRank9Black)
            : (R == Rank1 ? InFrontOfRank1White
               : R == Rank2 ? InFrontOfRank2White
               : R == Rank3 ? InFrontOfRank3White
               : R == Rank4 ? InFrontOfRank4White
               : R == Rank5 ? InFrontOfRank5White
               : R == Rank6 ? InFrontOfRank6White
               : R == Rank7 ? InFrontOfRank7White
               : R == Rank8 ? InFrontOfRank8White
               : /*R == Rank9 ?*/ InFrontOfRank9White));
}
// 敵陣を表現するBitboard。
const Bitboard EnemyField[ColorNum] = { rankMask<Rank1>() | rankMask<Rank2>() | rankMask<Rank3>() , rankMask<Rank7>() | rankMask<Rank8>() | rankMask<Rank9>() };
inline const Bitboard enemyField(Color c) { return EnemyField[c]; }

// メモリ節約をせず、無駄なメモリを持っている。
extern Bitboard LanceAttack[ColorNum][SquareNum][128];
extern Bitboard RookAttackRankToMask[SquareNum][2];
extern Bitboard256 BishopAttackToMask[SquareNum][2];

extern Bitboard KingAttack[SquareNum];
extern Bitboard GoldAttack[ColorNum][SquareNum];
extern Bitboard SilverAttack[ColorNum][SquareNum];
extern Bitboard KnightAttack[ColorNum][SquareNum];
extern Bitboard PawnAttack[ColorNum][SquareNum];

extern Bitboard BetweenBB[SquareNum][SquareNum];

extern Bitboard RookAttackToEdge[SquareNum];
extern Bitboard BishopAttackToEdge[SquareNum];
extern Bitboard LanceAttackToEdge[ColorNum][SquareNum];

extern Bitboard GoldCheckTable[ColorNum][SquareNum];
extern Bitboard SilverCheckTable[ColorNum][SquareNum];
extern Bitboard KnightCheckTable[ColorNum][SquareNum];
extern Bitboard LanceCheckTable[ColorNum][SquareNum];
extern Bitboard PawnCheckTable[ColorNum][SquareNum];
extern Bitboard BishopCheckTable[ColorNum][SquareNum];
extern Bitboard HorseCheckTable[ColorNum][SquareNum];

extern Bitboard Neighbor5x5Table[SquareNum]; // 25 近傍

// todo: 香車の筋がどこにあるか先に分かっていれば、Bitboard の片方の変数だけを調べれば良くなる。
inline Bitboard lanceAttack(const Color c, const Square sq, const Bitboard& occupied) {
    const int part = Bitboard::part(sq);
    const int index = (occupied.p(part) >> Slide[sq]) & 127;
    return LanceAttack[c][sq][index];
}
// 飛車の縦だけの利き。香車の利きを使い、index を共通化することで高速化している。
inline Bitboard rookAttackFile(const Square sq, const Bitboard& occupied) {
    const int part = Bitboard::part(sq);
    const int index = (occupied.p(part) >> Slide[sq]) & 127;
    return LanceAttack[Black][sq][index] | LanceAttack[White][sq][index];
}
// 飛車の横だけの利き
// cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal_210518.pdf
inline Bitboard rookAttackRank(const Square sq, const Bitboard& occupied) {
    Bitboard hi, lo, t1, t0;

    const Bitboard mask_lo = RookAttackRankToMask[sq][0];
    const Bitboard mask_hi = RookAttackRankToMask[sq][1];

    // occupiedを逆順にする
    Bitboard rocc = occupied.byteReverse();

    // roccとoccを2枚並べて、その上位u64をhi、下位u64をloに集める。
    // occ側は(先手から見て)左方向への利き、roccは右方向への利き。
    Bitboard::unpack(rocc, occupied, hi, lo);

    // 飛車の横方向の利きでmask
    hi &= mask_hi;
    lo &= mask_lo;

    // 1減算することにより、利きが通るマスまでが変化する。
    Bitboard::decrement(hi, lo, t1, t0);

    // 減算して変化したマスを抽出してmask
    t1 = (t1 ^ hi) & mask_hi;
    t0 = (t0 ^ lo) & mask_lo;

    // unpackしていたものを元の状態に戻す(unpackの逆変換はunpack)
    Bitboard::unpack(t1, t0, hi, lo);

    // byte_reverseして元の状態に戻して、重ね合わせる。
    // hiの方には、右方向の利き、loは左方向の利きが得られている。
    return hi.byteReverse() | lo;
}
inline Bitboard rookAttack(const Square sq, const Bitboard& occupied) {
    return rookAttackRank(sq, occupied) | rookAttackFile(sq, occupied);
}
// 角の利き
// cf. https://www.apply.computer-shogi.org/wcsc31/appeal/Qugiy/appeal_210518.pdf
inline Bitboard bishopAttack(const Square sq, const Bitboard& occupied) {
    const Bitboard256 mask_lo = BishopAttackToMask[sq][0];
    const Bitboard256 mask_hi = BishopAttackToMask[sq][1];

    // occupiedを2枚並べたBitboard256を用意する。
    const Bitboard256 occ2(occupied);

    // occupiedを(byte単位で)左右反転させたBitboardを2枚並べたBitboard256を用意する。
    const Bitboard256 rocc2(occupied.byteReverse());

    Bitboard256 hi, lo, t1, t0;
    Bitboard256::unpack(rocc2, occ2, hi, lo);

    hi &= mask_hi;
    lo &= mask_lo;

    Bitboard256::decrement(hi, lo, t1, t0);

    // xorで変化した升を抽出して、step effectでmaskすれば完成
    t1 = (t1 ^ hi) & mask_hi;
    t0 = (t0 ^ lo) & mask_lo;

    // unpackしていたものを元の状態に戻す(unpackの逆変換はunpack)
    Bitboard256::unpack(t1, t0, hi, lo);

    // byte_reverseして元の状態に戻して、重ね合わせる。
    // hiの方には、右方向の利き、loは左方向の利きが得られている。
    return (hi.byteReverse() | lo).merge();
}
inline Bitboard goldAttack(const Color c, const Square sq) { return GoldAttack[c][sq]; }
inline Bitboard silverAttack(const Color c, const Square sq) { return SilverAttack[c][sq]; }
inline Bitboard knightAttack(const Color c, const Square sq) { return KnightAttack[c][sq]; }
inline Bitboard pawnAttack(const Color c, const Square sq) { return PawnAttack[c][sq]; }

// Bitboard で直接利きを返す関数。
// 1段目には歩は存在しないので、1bit シフトで別の筋に行くことはない。
// ただし、from に歩以外の駒の Bitboard を入れると、別の筋のビットが立ってしまうことがあるので、
// 別の筋のビットが立たないか、立っても問題ないかを確認して使用すること。
template <Color US> inline Bitboard pawnAttack(const Bitboard& from) { return (US == Black ? (from >> 1) : (from << 1)); }
inline Bitboard pawnAttack(Color c, const Bitboard& from) { return (c == Black ? (from >> 1) : (from << 1)); }
inline Bitboard kingAttack(const Square sq) { return KingAttack[sq]; }
inline Bitboard dragonAttack(const Square sq, const Bitboard& occupied) { return rookAttack(sq, occupied) | kingAttack(sq); }
inline Bitboard horseAttack(const Square sq, const Bitboard& occupied) { return bishopAttack(sq, occupied) | kingAttack(sq); }
inline Bitboard queenAttack(const Square sq, const Bitboard& occupied) { return rookAttack(sq, occupied) | bishopAttack(sq, occupied); }

// sq1, sq2 の間(sq1, sq2 は含まない)のビットが立った Bitboard
inline Bitboard betweenBB(const Square sq1, const Square sq2) { return BetweenBB[sq1][sq2]; }
inline Bitboard rookAttackToEdge(const Square sq) { return RookAttackToEdge[sq]; }
inline Bitboard bishopAttackToEdge(const Square sq) { return BishopAttackToEdge[sq]; }
inline Bitboard lanceAttackToEdge(const Color c, const Square sq) { return LanceAttackToEdge[c][sq]; }
inline Bitboard dragonAttackToEdge(const Square sq) { return rookAttackToEdge(sq) | kingAttack(sq); }
inline Bitboard horseAttackToEdge(const Square sq) { return bishopAttackToEdge(sq) | kingAttack(sq); }
inline Bitboard goldCheckTable(const Color c, const Square sq) { return GoldCheckTable[c][sq]; }
inline Bitboard silverCheckTable(const Color c, const Square sq) { return SilverCheckTable[c][sq]; }
inline Bitboard knightCheckTable(const Color c, const Square sq) { return KnightCheckTable[c][sq]; }
inline Bitboard lanceCheckTable(const Color c, const Square sq) { return LanceCheckTable[c][sq]; }
inline Bitboard pawnCheckTable(const Color c, const Square sq) { return PawnCheckTable[c][sq]; }
inline Bitboard bishopCheckTable(const Color c, const Square sq) { return BishopCheckTable[c][sq]; }
inline Bitboard horseCheckTable(const Color c, const Square sq) { return HorseCheckTable[c][sq]; }

inline Bitboard neighbor5x5Table(const Square sq) { return Neighbor5x5Table[sq]; }
// todo: テーブル引きを検討
inline Bitboard rookStepAttacks(const Square sq) { return goldAttack(Black, sq) & goldAttack(White, sq); }
// todo: テーブル引きを検討
inline Bitboard bishopStepAttacks(const Square sq) { return silverAttack(Black, sq) & silverAttack(White, sq); }
// 前方3方向の位置のBitboard
inline Bitboard goldAndSilverAttacks(const Color c, const Square sq) { return goldAttack(c, sq) & silverAttack(c, sq); }

// Bitboard の全ての bit に対して同様の処理を行う際に使用するマクロ
// xxx に処理を書く。
// xxx には template 引数を 2 つ以上持つクラスや関数は () でくくらないと使えない。
// これはマクロの制約。
// 同じ処理のコードが 2 箇所で生成されるため、コードサイズが膨れ上がる。
// その為、あまり多用すべきでないかも知れない。
#define FOREACH_BB(bb, sq, xxx)                 \
    do {                                        \
        while (bb.p(0)) {                       \
            sq = bb.firstOneRightFromSQ11();    \
            xxx;                                \
        }                                       \
        while (bb.p(1)) {                       \
            sq = bb.firstOneLeftFromSQ81();     \
            xxx;                                \
        }                                       \
    } while (false)

template <typename T> FORCE_INLINE void foreachBB(Bitboard& bb, Square& sq, T t) {
    while (bb.p(0)) {
        sq = bb.firstOneRightFromSQ11();
        t(0);
    }
    while (bb.p(1)) {
        sq = bb.firstOneLeftFromSQ81();
        t(1);
    }
}

#endif // #ifndef APERY_BITBOARD_HPP
