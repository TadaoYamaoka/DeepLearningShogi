#include <numeric>
#include <algorithm>
#include <type_traits>

#include "cppshogi.h"

inline void set_features1(features1_t features1, const Color c, const int f1idx, const Square sq)
{
	features1[c][f1idx][sq] = _one;
}
inline void set_features1(packed_features1_t packed_features1, const Color c, const int f1idx, const Square sq)
{
	const int idx = MAX_FEATURES1_NUM * (int)SquareNum * (int)c + (int)SquareNum * f1idx + sq;
	packed_features1[idx >> 3] |= (1 << (idx & 7));
}

inline void set_features2(features2_t features2, const Color c, const int f2idx, const u32 num)
{
	std::fill_n(features2[MAX_PIECES_IN_HAND_SUM * (int)c + f2idx], (int)SquareNum * num, _one);
}
inline void set_features2(packed_features2_t packed_features2, const Color c, const int f2idx, const u32 num)
{
	for (u32 i = 0; i < num; ++i) {
		const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx + i;
		packed_features2[idx >> 3] |= (1 << (idx & 7));
	}
}

inline void set_features2(features2_t features2, const int f2idx)
{
	std::fill_n(features2[f2idx], SquareNum, _one);
}
inline void set_features2(packed_features2_t packed_features2, const int f2idx)
{
	packed_features2[f2idx >> 3] |= (1 << (f2idx & 7));
}

inline void add_attacks(Bitboard (&counts)[MAX_ATTACK_NUM], const Bitboard& attacks) {
	// 下の段を更新する前に、利きが重なるマスを上の段へ加える。
#if defined(HAVE_AVX2)
	static_assert(MAX_ATTACK_NUM == 3 && sizeof(Bitboard) == 16,
		"AVX2 attack counts require three 128-bit boards");
	const __m128i attack = _mm_set_epi64x(attacks.p(1), attacks.p(0));
	const __m256i both = _mm256_broadcastsi128_si256(attack);
	const __m256i lower = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&counts[0]));
	const __m256i upper = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&counts[1]));
	// 旧[0,1]から[1,2]を同時に更新する。読み込みは書き込みより先に行う。
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(&counts[1]),
		_mm256_or_si256(upper, _mm256_and_si256(lower, both)));
#else
	for (int i = MAX_ATTACK_NUM - 1; i > 0; --i)
		counts[i] |= counts[i - 1] & attacks;
#endif
	counts[0] |= attacks;
}

template <Color turn>
inline void set_attack_features(features1_t features1, const Color c, const int f1idx, Bitboard attacks) {
	FOREACH_BB(attacks, Square to, {
		if (turn == White) to = SQ99 - to;
		set_features1(features1, c, f1idx, to);
	});
}

template <Color turn>
inline void set_attack_features(packed_features1_t features1, const Color c, const int f1idx, Bitboard attacks) {
	u64 low, high;
	if constexpr (turn == White) {
		// 各バイト内のビットを反転してから、バイト順も反転する
#if defined(HAVE_SSE4)
		const __m128i value = _mm_set_epi64x(attacks.p(1), attacks.p(0));
		const __m128i mask = _mm_set1_epi8(0x0f);
		const __m128i reverse_nibble = _mm_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
		const __m128i lo = _mm_shuffle_epi8(_mm_slli_epi16(reverse_nibble, 4), _mm_and_si128(value, mask));
		const __m128i hi = _mm_shuffle_epi8(reverse_nibble, _mm_and_si128(_mm_srli_epi16(value, 4), mask));
		const __m128i reversed = _mm_shuffle_epi8(_mm_or_si128(lo, hi),
			_mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
		const u64 reversed_low = static_cast<u64>(_mm_cvtsi128_si64(reversed));
		const u64 reversed_high = static_cast<u64>(_mm_extract_epi64(reversed, 1));
		low = (reversed_high << 17) | (reversed_low >> 46);
		high = reversed_high >> 47;
#else
		const Bitboard mask1(0x5555555555555555ULL, 0x5555555555555555ULL);
		const Bitboard mask2(0x3333333333333333ULL, 0x3333333333333333ULL);
		const Bitboard mask4(0x0f0f0f0f0f0f0f0fULL, 0x0f0f0f0f0f0f0f0fULL);
		attacks = ((attacks >> 1) & mask1) | ((attacks & mask1) << 1);
		attacks = ((attacks >> 2) & mask2) | ((attacks & mask2) << 2);
		attacks = ((attacks >> 4) & mask4) | ((attacks & mask4) << 4);
		attacks = attacks.byteReverse();
		low = (attacks.p(1) << 17) | (attacks.p(0) >> 46);
		high = attacks.p(1) >> 47;
#endif
	}
	else {
		// Bitboardの63ビット＋18ビットを、連続した81ビットに詰める
		low = attacks.p(0) | (attacks.p(1) << 63);
		high = attacks.p(1) >> 1;
	}

	const int idx = MAX_FEATURES1_NUM * (int)SquareNum * (int)c + (int)SquareNum * f1idx;
	const int shift = idx & 7;
	if (shift != 0) {
		high = (high << shift) | (low >> (64 - shift));
		low <<= shift;
	}
	// 境界バイトの隣接特徴量や、呼び出し前に立っていたビットは保持する
	char* out = features1 + (idx >> 3);
	for (int i = 0; i < 8; ++i)
		out[i] |= static_cast<char>((low >> (8 * i)) & 0xff);
	for (int i = 0; i < 3; ++i)
		out[8 + i] |= static_cast<char>((high >> (8 * i)) & 0xff);
}

template <typename T1>
FORCE_INLINE Bitboard feature_attacks(const PieceType pt, const Color c, const Square sq, const Bitboard& occupied) {
    // 圧縮形式では、既存の駒別利き生成をその場で展開する。
    // 同条件の計測で圧縮形式は高速化したが、floatの先手では遅くなったため、floatは従来の呼び出しを保つ。
    if constexpr (std::is_same_v<T1, packed_features1_t>) {
        switch (pt) {
        case Occupied:  return allZeroBB();
        case Pawn:      return pawnAttack(c, sq);
        case Lance:     return lanceAttack(c, sq, occupied);
        case Knight:    return knightAttack(c, sq);
        case Silver:    return silverAttack(c, sq);
        case Bishop:    return bishopAttack(sq, occupied);
        case Rook:      return rookAttack(sq, occupied);
        case Gold:
        case ProPawn:
        case ProLance:
        case ProKnight:
        case ProSilver: return goldAttack(c, sq);
        case King:      return kingAttack(sq);
        case Horse:     return horseAttack(sq, occupied);
        case Dragon:    return dragonAttack(sq, occupied);
        default:        UNREACHABLE; return allOneBB();
        }
    } else {
        return Position::attacksFrom(pt, c, sq, occupied);
    }
}

// make input features
template <Color turn, typename T1 = features1_t, typename T2 = features2_t>
inline void make_input_features(const Position& position, T1 features1, T2 features2) {
#if defined(HAVE_SSE4)
	constexpr bool bulk_attacks = std::is_same_v<T1, packed_features1_t>;
#else
	// 後手の一括書き込みには、高速なビット反転が必要。
	constexpr bool bulk_attacks = turn == Black && std::is_same_v<T1, packed_features1_t>;
#endif
	const Bitboard occupied_bb = position.occupiedBB();

	// 歩と歩以外に分ける
	Bitboard pawns_bb = position.bbOf(Pawn);
	Bitboard without_pawns_bb = occupied_bb & ~pawns_bb;
	// 利き数集計用
	Bitboard attack_num[ColorNum][MAX_ATTACK_NUM];
	std::memset(attack_num, 0, sizeof(attack_num));

	// 歩以外
	FOREACH_BB(without_pawns_bb, Square sq, {
		const Piece pc = position.piece(sq);
		const PieceType pt = pieceToPieceType(pc);
		Color c = pieceToColor(pc);
		Bitboard attacks = feature_attacks<T1>(pt, c, sq, occupied_bb);

		// 後手の場合、色を反転し、盤面を180度回転
		if (turn == White) {
			c = oppositeColor(c);
			sq = SQ99 - sq;
		}

		// 駒の配置
		set_features1(features1, c, pt - 1, sq);

		// 利き数は盤面を反転せず、1枚以上・2枚以上・3枚以上に分けて集計する。
		add_attacks(attack_num[c], attacks);

		// 飛車・角・龍・馬の利きを、圧縮特徴量へまとめて書き込む。
		if constexpr (bulk_attacks) {
			if (pt == Bishop || pt == Rook || pt == Horse || pt == Dragon) {
				set_attack_features<turn>(features1, c, PIECETYPE_NUM + pt - 1, attacks);
				continue;
			}
		}

		FOREACH_BB(attacks, Square to, {
			// 後手の場合、盤面を180度回転
			if (turn == White) to = SQ99 - to;

			// 駒の利き
			set_features1(features1, c, PIECETYPE_NUM + pt - 1, to);
		});
	});

	for (Color c = Black; c < ColorNum; ++c) {
		// 後手の場合、色を反転
		const Color c2 = turn == Black ? c : oppositeColor(c);

		// 歩
		Bitboard pawns_bb2 = pawns_bb & position.bbOf(c2);
		// 歩の利き先は重ならないため、全ての歩をまとめて加算できる。
		const Bitboard pawn_attacks = c2 == Black ? pawns_bb2 >> 1 : pawns_bb2 << 1;
		add_attacks(attack_num[c], pawn_attacks);
		const SquareDelta pawnDelta = c == Black ? DeltaN : DeltaS;
		FOREACH_BB(pawns_bb2, Square sq, {
			// 後手の場合、盤面を180度回転
			if (turn == White) sq = SQ99 - sq;

			// 駒の配置
			set_features1(features1, c, Pawn - 1, sq);

			// 駒の利き
			const Square to = sq + pawnDelta; // 1マス先
			set_features1(features1, c, PIECETYPE_NUM + Pawn - 1, to);
		});

		// 集計した利き数を特徴量に変換する。
		for (int i = 0; i < MAX_ATTACK_NUM; ++i) {
			set_attack_features<turn>(features1, c, PIECETYPE_NUM + PIECETYPE_NUM + i, attack_num[c][i]);
		}

		// 持ち駒
		const Hand hand = position.hand(c);
		const u32 numHPawn = hand.numOf(HPawn);
		set_features2(features2, c2, 0, std::min(numHPawn, (u32)MAX_HPAWN_NUM));
		const u32 numHLance = hand.numOf(HLance);
		set_features2(features2, c2, MAX_HPAWN_NUM, numHLance);
		const u32 numHKnight = hand.numOf(HKnight);
		set_features2(features2, c2, MAX_HPAWN_NUM + MAX_HLANCE_NUM, numHKnight);
		const u32 numHSilver = hand.numOf(HSilver);
		set_features2(features2, c2, MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM, numHSilver);
		const u32 numHGold = hand.numOf(HGold);
		set_features2(features2, c2, MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM, numHGold);
		const u32 numHBishop = hand.numOf(HBishop);
		set_features2(features2, c2, MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM, numHBishop);
		const u32 numHRook = hand.numOf(HRook);
		set_features2(features2, c2, MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM, numHRook);

#ifdef NYUGYOKU_FEATURES
		// 入玉宣言
		// 敵陣のマスク
		const Bitboard opponentsField = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());

		// 玉が敵陣三段目以内に入っている
		int kingCount = 0;
		if (position.bbOf(King, c).andIsAny(opponentsField)) {
			set_features2(features2, MAX_FEATURES2_HAND_NUM + 1 + (int)c2 * MAX_FEATURES2_NYUGYOKU_NUM);
			kingCount = 1;
		}

		// 敵陣三段目以内の駒(10枚までの残り枚数)
		const int ownPiecesCount = (position.bbOf(c) & opponentsField).popCount() - kingCount;
		const int restOppFieldNum = 10 - ownPiecesCount;
		if (restOppFieldNum < static_cast<int>(MAX_NYUGYOKU_OPP_FIELD)) {
			set_features2(features2, MAX_FEATURES2_HAND_NUM + 1 + (int)c2 * MAX_FEATURES2_NYUGYOKU_NUM + 1 + std::max(0, restOppFieldNum));
		}

		// 点数(先手28点、後手27点までの残り枚数)
		const int ownBigPiecesCount = (position.bbOf(Rook, Dragon, Bishop, Horse) & opponentsField & position.bbOf(c)).popCount();
		const int ownSmallPiecesCount = ownPiecesCount - ownBigPiecesCount;
		const int val = ownSmallPiecesCount
			+ numHPawn + numHLance + numHKnight
			+ numHSilver + numHGold
			+ (ownBigPiecesCount + numHBishop + numHRook) * 5;
		const int restPoint = (c == Black ? 28 : 27) - val;
		if (restPoint < static_cast<int>(MAX_NYUGYOKU_SCORE)) {
			set_features2(features2, MAX_FEATURES2_HAND_NUM + 1 + (int)c2 * MAX_FEATURES2_NYUGYOKU_NUM + 1 + MAX_NYUGYOKU_OPP_FIELD + std::max(0, restPoint));
		}
#endif
    }

	// is check
	if (position.inCheck()) {
		set_features2(features2, MAX_FEATURES2_HAND_NUM);
	}
}

void make_input_features(const Position& position, features1_t features1, features2_t features2) {
	position.turn() == Black ? make_input_features<Black>(position, features1, features2) : make_input_features<White>(position, features1, features2);
}

void make_input_features(const Position& position, packed_features1_t packed_features1, packed_features2_t packed_features2) {
	position.turn() == Black ?
		make_input_features<Black, packed_features1_t, packed_features2_t>(position, packed_features1, packed_features2) :
		make_input_features<White, packed_features1_t, packed_features2_t>(position, packed_features1, packed_features2);
}

constexpr MOVE_DIRECTION get_move_direction(const int dir_x, const int dir_y) {
	if (dir_y < 0 && dir_x == 0) {
		return UP;
	}
	else if (dir_y == -2 && dir_x == -1) {
		return UP2_LEFT;
	}
	else if (dir_y == -2 && dir_x == 1) {
		return UP2_RIGHT;
	}
	else if (dir_y < 0 && dir_x < 0) {
		return UP_LEFT;
	}
	else if (dir_y < 0 && dir_x > 0) {
		return UP_RIGHT;
	}
	else if (dir_y == 0 && dir_x < 0) {
		return LEFT;
	}
	else if (dir_y == 0 && dir_x > 0) {
		return RIGHT;
	}
	else if (dir_y > 0 && dir_x == 0) {
		return DOWN;
	}
	else if (dir_y > 0 && dir_x < 0) {
		return DOWN_LEFT;
	}
	else /* if (dir_y > 0 && dir_x > 0) */ {
		return DOWN_RIGHT;
	}
}

namespace {
struct MoveLabelTable {
	u16 labels[SquareNum][SquareNum]{};

	constexpr MoveLabelTable() {
		for (int from = 0; from < SquareNum; ++from) {
			for (int to = 0; to < SquareNum; ++to) {
				labels[from][to] = static_cast<u16>(SquareNum *
					get_move_direction(from / 9 - to / 9, to % 9 - from % 9) + to);
			}
		}
	}
};

constexpr MoveLabelTable move_label_table;
}

// make move
int make_move_label(const u16 move16, const Color color) {
	// see: move.hpp : 30
	// xxxxxxxx x1111111  移動先
	// xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
	// x1xxxxxx xxxxxxxx  1 なら成り
	u16 to_sq = move16 & 0b1111111;
	u16 from_sq = (move16 >> 7) & 0b1111111;
	assert(to_sq < SquareNum);

	if (from_sq < SquareNum) {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
			from_sq = (u16)SQ99 - from_sq;
		}

		return move_label_table.labels[from_sq][to_sq] +
			// promote
			((move16 & 0b100000000000000) ? 9 * 9 * 10 : 0);
	}
	// 持ち駒の場合
	else {
		// 白の場合、盤面を180度回転
		if (color == White) {
			to_sq = (u16)SQ99 - to_sq;
		}
		const int hand_piece = from_sq - (int)SquareNum;
		const int move_direction_label = MOVE_DIRECTION_NUM + hand_piece;
		return 9 * 9 * move_direction_label + to_sq;
	}
}
