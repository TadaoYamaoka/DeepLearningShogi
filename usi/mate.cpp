#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "mate.h"

const constexpr size_t MaxCheckMoves = 73;

// 詰み探索用のMovePicker
template <bool or_node, bool INCHECK>
class MovePicker {
public:
	explicit MovePicker(const Position& pos) {
		if (or_node) {
			last_ = generateMoves<Check>(moveList_, pos);
			if (INCHECK) {
				// 自玉が王手の場合、逃げる手かつ王手をかける手を生成
				ExtMove* curr = moveList_;
				while (curr != last_) {
					if (!pos.moveIsPseudoLegal<false>(curr->move))
						curr->move = (--last_)->move;
					else
						++curr;
				}
			}
		}
		else {
			last_ = generateMoves<Evasion>(moveList_, pos);
			// 玉の移動による自殺手と、pinされている駒の移動による自殺手を削除
			ExtMove* curr = moveList_;
			const Bitboard pinned = pos.pinnedBB();
			while (curr != last_) {
				if (!pos.pseudoLegalMoveIsLegal<false, false>(curr->move, pinned))
					curr->move = (--last_)->move;
				else
					++curr;
			}
		}
		assert(size() <= MaxCheckMoves);
	}
	size_t size() const { return static_cast<size_t>(last_ - moveList_); }
	ExtMove* begin() { return &moveList_[0]; }
	ExtMove* end() { return last_; }
	bool empty() const { return size() == 0; }

private:
	ExtMove moveList_[MaxCheckMoves];
	ExtMove* last_;
};

enum PieceTypeCheck
{
	PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO, // 不成りのまま王手になるところ(成れる場合は含まず)
	PIECE_TYPE_CHECK_PAWN_WITH_PRO, // 成りで王手になるところ
	PIECE_TYPE_CHECK_LANCE,
	PIECE_TYPE_CHECK_KNIGHT,
	PIECE_TYPE_CHECK_SILVER,
	PIECE_TYPE_CHECK_GOLD,
	PIECE_TYPE_CHECK_BISHOP,
	PIECE_TYPE_CHECK_ROOK,
	PIECE_TYPE_CHECK_PRO_BISHOP,
	PIECE_TYPE_CHECK_PRO_ROOK,
	PIECE_TYPE_CHECK_NON_SLIDER, // 王手になる非遠方駒の移動元

	PIECE_TYPE_CHECK_NB,
	PIECE_TYPE_CHECK_ZERO = 0,
};
OverloadEnumOperators(PieceTypeCheck);

// 王手になる候補の駒の位置を示すBitboard
Bitboard CHECK_CAND_BB[SquareNum + 1][PIECE_TYPE_CHECK_NB][ColorNum];

// 玉周辺の利きを求めるときに使う、玉周辺に利きをつける候補の駒を表すBB
// COLORのところは王手する側の駒
Bitboard CHECK_AROUND_BB[SquareNum + 1][Promoted][ColorNum];

// 移動により王手になるbitboardを返す。
// us側が王手する。sq_king = 敵玉の升。pc = 駒
inline Bitboard check_cand_bb(Color us, PieceTypeCheck pc, Square sq_king)
{
	return CHECK_CAND_BB[sq_king][pc][us];
}

// 敵玉8近傍の利きに関係する自駒の候補のbitboardを返す。ここになければ玉周辺に利きをつけない。
// pt = PAWN～HDK
inline Bitboard check_around_bb(Color us, PieceType pt, Square sq_king)
{
	return CHECK_AROUND_BB[sq_king][pt - 1][us];
}

// sq1に対してsq2の升の延長上にある次の升を得る。
// 隣接していないか、盤外になるときはSQUARE_NB
// テーブルサイズを小さくしておきたいのでu8にしておく。
/*Square*/ u8 NextSquare[SquareNum + 1][SquareNum + 1];
inline Square nextSquare(Square sq1, Square sq2) { return (Square)NextSquare[sq1][sq2]; }

// CHECK_CAND_BB、CHECK_AROUND_BBの初期化
void init_check_bb()
{
	for (PieceTypeCheck p = PIECE_TYPE_CHECK_ZERO; p < PIECE_TYPE_CHECK_NB; ++p)
		for (Square sq = SQ11; sq < SquareNum; ++sq)
			for (Color c = Black; c < ColorNum; ++c)
			{
				Bitboard bb = allZeroBB(), tmp = allZeroBB();
				Square to;

				// 敵陣
				Bitboard enemyBB = enemyField(c);

				switch ((int)p)
				{
				case PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO:
					// 歩が不成りで王手になるところだけ。

					bb = pawnAttack(~c, sq) & ~enemyBB;
					if (!bb)
						break;
					to = bb.firstOneFromSQ11();
					bb = pawnAttack(~c, to);
					break;

				case PIECE_TYPE_CHECK_PAWN_WITH_PRO:

					bb = goldAttack(~c, sq) & enemyBB;
					bb = pawnAttack(~c, bb);
					break;

				case PIECE_TYPE_CHECK_LANCE:

					// 成りによるものもあるからな..候補だけ列挙しておくか。
					bb = lanceAttackToEdge(~c, sq);
					if (enemyBB ^ setMaskBB(sq))
					{
						// 敵陣なので成りで王手できるから、sqより下段の香も足さないと。
						if (makeFile(sq) != File1)
							bb |= lanceAttackToEdge(~c, sq + DeltaE);
						if (makeFile(sq) != File9)
							bb |= lanceAttackToEdge(~c, sq + DeltaW);
					}

					break;

				case PIECE_TYPE_CHECK_KNIGHT:

					// 敵玉から桂の桂にある駒
					tmp = knightAttack(~c, sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= knightAttack(~c, to);
					}
					// 成って王手(金)になる移動元
					tmp = goldAttack(~c, sq) & enemyBB;
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= knightAttack(~c, to);
					}
					break;

				case PIECE_TYPE_CHECK_SILVER:

					// 敵玉から銀の銀にある駒。
					tmp = silverAttack(~c, sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= silverAttack(~c, to);
					}
					// 成って王手の場合、敵玉から金の銀にある駒
					tmp = goldAttack(~c, sq) & enemyBB;
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= silverAttack(~c, to);
					}
					// あと4段目の玉に3段目から成っての王手。玉のひとつ下の升とその斜めおよび、
					// 玉のひとつ下の升の2つとなりの升
					{
						Rank r = (c == Black ? Rank4 : Rank6);
						if (r == makeRank(sq))
						{
							r = (c == Black ? Rank3 : Rank7);
							to = makeSquare(makeFile(sq), r);
							bb |= setMaskBB(to);
							bb |= bishopStepAttacks(to);

							// 2升隣。
							if (makeFile(to) >= File3)
								bb |= setMaskBB(to + DeltaE * 2);
							if (makeFile(to) <= File7)
								bb |= setMaskBB(to + DeltaW * 2);
						}

						// 5段目の玉に成りでのバックアタック的な..
						if (makeRank(sq) == Rank5)
							bb |= knightAttack(c, sq);
					}
					break;

				case PIECE_TYPE_CHECK_GOLD:
					// 敵玉から金の金にある駒
					tmp = goldAttack(~c, sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= goldAttack(~c, to);
					}
					break;

					// この4枚、どうせいないときもあるわけで、効果に乏しいので要らないのでは…。
				case PIECE_TYPE_CHECK_BISHOP:
				case PIECE_TYPE_CHECK_PRO_BISHOP:
				case PIECE_TYPE_CHECK_ROOK:
				case PIECE_TYPE_CHECK_PRO_ROOK:
					// 王の8近傍の8近傍(24近傍)か、王の3列、3行か。結構の範囲なのでこれ無駄になるな…。
					break;

					// 非遠方駒の合体bitboard。ちょっとぐらい速くなるんだろう…。
				case PIECE_TYPE_CHECK_NON_SLIDER:
					bb = CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_GOLD][c]
						| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_KNIGHT][c]
						| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_SILVER][c]
						| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_PAWN_WITH_NO_PRO][c]
						| CHECK_CAND_BB[sq][PIECE_TYPE_CHECK_PAWN_WITH_PRO][c];
					break;
				}
				bb &= ~setMaskBB(sq); // sqの地点邪魔なので消しておく。
				CHECK_CAND_BB[sq][p][c] = bb;
			}


	for (PieceType p = Pawn; p <= King; ++p)
		for (Square sq = SQ11; sq < SquareNum; ++sq)
			for (Color c = Black; c < ColorNum; ++c)
			{
				Bitboard bb = allZeroBB(), tmp = allZeroBB();
				Square to;

				switch (p)
				{
				case Pawn:
					// これ用意するほどでもないんだな
					// 一応、用意するコード書いておくか..
					bb = pawnAttack(c, bb);
					// →　このシフトでp[0]の63bit目に来るとまずいので..
					bb &= allOneBB(); // ALL_BBでand取っておく。
					break;

				case Lance:
					// 香で玉8近傍の利きに関与するのは…。玉と同じ段より攻撃側の陣にある香だけか..
					bb = lanceAttackToEdge(~c, sq);
					if (makeFile(sq) != File1)
						bb |= lanceAttackToEdge(~c, sq + DeltaE) | setMaskBB(sq + DeltaE);
					if (makeFile(sq) != File9)
						bb |= lanceAttackToEdge(~c, sq + DeltaW) | setMaskBB(sq + DeltaW);
					break;

				case Knight:
					// 桂は玉8近傍の逆桂か。
					tmp = kingAttack(sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= knightAttack(~c, to);
					}
					break;

				case Silver:
					// 同じく
					tmp = kingAttack(sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= silverAttack(~c, to);
					}
					break;

				case Gold:
					// 同じく
					tmp = kingAttack(sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= goldAttack(~c, to);
					}
					break;

				case Bishop:
					// 同じく
					tmp = kingAttack(sq);
					while (tmp)
					{
						to = tmp.firstOneFromSQ11();
						bb |= bishopAttackToEdge(to);
					}
					break;

				case ROOK:
					// 同じく
					tmp = kingEffect(sq);
					while (tmp)
					{
						to = tmp.pop();
						bb |= rookStepEffect(to);
					}
					break;

					// HDK相当
				case KING:
					// 同じく
					tmp = kingEffect(sq);
					while (tmp)
					{
						to = tmp.pop();
						bb |= kingEffect(to);
					}
					break;

				default:
					UNREACHABLE;
				}

				bb &= ~Bitboard(sq); // sqの地点邪魔なので消しておく。
										// CHECK_CAND_BBとは並び順を変えたので注意。
				CHECK_AROUND_BB[sq][p - 1][c] = bb;
			}

	// NextSquareの初期化
	// Square NextSquare[SQUARE_NB][SQUARE_NB];
	// sq1に対してsq2の升の延長上にある次の升を得る。
	// 隣接していないか、盤外になるときはSQUARE_NB

	for (auto s1 : SQ)
		for (auto s2 : SQ)
		{
			Square next_sq = SQ_NB;

			// 隣接していなくてもok。縦横斜かどうかだけ判定すべし。
			if (queenStepEffect(s1) & s2)
			{
				File vf = File(sgn(file_of(s2) - file_of(s1)));
				Rank vr = Rank(sgn(rank_of(s2) - rank_of(s1)));

				File s3f = file_of(s2) + vf;
				Rank s3r = rank_of(s2) + vr;
				// 盤面の範囲外に出ていないかのテスト
				if (is_ok(s3f) && is_ok(s3r))
					next_sq = s3f | s3r;
			}
			NextSquare[s1][s2] = next_sq;
		}

}

// NonSliderの利きのみ列挙
template <Color US>
Bitboard AttacksAroundKingNonSlider(Position pos) {
	Square sq_king = pos.kingSquare(US);
	Color Them = ~US;
	Square from;
	Bitboard bb;

	// 歩は普通でいい
	Bitboard sum = pos.attacksFrom<Pawn>(Them, pos.bbOf(Them, PAWN));

	// ほとんどのケースにおいて候補になる駒はなく、whileで回らずに抜けると期待している。
	bb = pos.bbOf(Them, KNIGHT) & check_around_bb(Them, KNIGHT, sq_king);
	while (bb)
	{
		from = bb.pop();
		sum |= knightEffect(Them, from);
	}
	bb = pos.pieces(Them, SILVER) & check_around_bb(Them, SILVER, sq_king);
	while (bb)
	{
		from = bb.pop();
		sum |= silverEffect(Them, from);
	}
	bb = pos.pieces(Them, GOLDS) & check_around_bb(Them, GOLD, sq_king);
	while (bb)
	{
		from = bb.pop();
		sum |= goldEffect(Them, from);
	}
	bb = pos.pieces(Them, HDK) & check_around_bb(Them, KING, sq_king);
	while (bb)
	{
		from = bb.pop();
		sum |= kingEffect(from);
	}
	return sum;
}

// 3手詰めチェック
// 手番側が王手でないこと
template <bool INCHECK>
FORCE_INLINE bool mateMoveIn3Ply(Position& pos)
{
	// OR節点

	StateInfo si;
	StateInfo si2;

	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, INCHECK>(pos))
	{
		const Move& m = ml.move;

		pos.doMove(m, si, ci, true);

		// 千日手のチェック
		if (pos.isDraw(16) == RepetitionWin) {
			// 受け側の反則勝ち
			pos.undoMove(m);
			continue;
		}

		// この局面ですべてのevasionを試す
		MovePicker<false, false> move_picker2(pos);

		if (move_picker2.size() == 0) {
			// 1手で詰んだ
			pos.undoMove(m);
			return true;
		}

		const CheckInfo ci2(pos);
		for (const auto& move : move_picker2)
		{
			const Move& m2 = move.move;

			// この指し手で逆王手になるなら、不詰めとして扱う
			if (pos.moveGivesCheck(m2, ci2))
				goto NEXT_CHECK;

			pos.doMove(m2, si2, ci2, false);

			if (!pos.mateMoveIn1Ply()) {
				// 詰んでないので、m2で詰みを逃れている。
				pos.undoMove(m2);
				goto NEXT_CHECK;
			}

			pos.undoMove(m2);
		}

		// すべて詰んだ
		pos.undoMove(m);
		return true;

	NEXT_CHECK:;
		pos.undoMove(m);
	}
	return false;
}

// 奇数手詰めチェック
// 詰ます手を返すバージョン
template <bool INCHECK>
Move mateMoveInOddPlyReturnMove(Position& pos, const int depth) {
	// OR節点

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, INCHECK>(pos)) {
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, true);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionLose: // 相手が負け
		{
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return ml.move;
		}
		case RepetitionDraw:
		case RepetitionWin: // 相手が勝ち
		case RepetitionSuperior: // 相手が駒得
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionInferior: break; // 相手が駒損
		default: UNREACHABLE;
		}

		//std::cout << ml.move().toUSI() << std::endl;
		// 偶数手詰めチェック
		if (mateMoveInEvenPly(pos, depth - 1)) {
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return ml.move;
		}

		pos.undoMove(ml.move);
	}
	return Move::moveNone();
}
template Move mateMoveInOddPlyReturnMove<true>(Position& pos, const int depth);
template Move mateMoveInOddPlyReturnMove<false>(Position& pos, const int depth);

// 奇数手詰めチェック
template <bool INCHECK = false>
bool mateMoveInOddPly(Position& pos, const int depth)
{
	// OR節点

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, INCHECK>(pos)) {
		//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, true);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionLose: // 相手が負け
		{
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return true;
		}
		case RepetitionDraw:
		case RepetitionWin: // 相手の勝ち
		case RepetitionSuperior: // 相手が駒得
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionInferior: break; // 相手が駒損
		default: UNREACHABLE;
		}

		// 王手の場合
		// 偶数手詰めチェック
		if (mateMoveInEvenPly(pos, depth - 1)) {
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return true;
		}

		pos.undoMove(ml.move);
	}
	return false;
}

// 偶数手詰めチェック
// 手番側が王手されていること
bool mateMoveInEvenPly(Position& pos, const int depth)
{
	// AND節点

	// すべてのEvasionについて
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<false, false>(pos)) {
		//std::cout << depth << " : " << pos.toSFEN() << " : " << ml.move.toUSI() << std::endl;
		const bool givesCheck = pos.moveGivesCheck(ml.move, ci);

		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move, state, ci, givesCheck);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionWin: // 自分が勝ち
		{
			pos.undoMove(ml.move);
			continue;
		}
		case RepetitionDraw:
		case RepetitionLose: // 自分が負け
		case RepetitionInferior: // 自分が駒損
		{
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move);
			return false;
		}
		case RepetitionSuperior: break; // 自分が駒得
		default: UNREACHABLE;
		}

		if (depth == 4) {
			// 3手詰めかどうか
			if (givesCheck ? !mateMoveIn3Ply<true>(pos) : !mateMoveIn3Ply<false>(pos)) {
				// 3手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move);
				return false;
			}
		}
		else {
			// 奇数手詰めかどうか
			if (givesCheck ? !mateMoveInOddPly<true>(pos, depth - 1) : !mateMoveInOddPly<false>(pos, depth - 1)) {
				// 偶数手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move);
				return false;
			}
		}

		pos.undoMove(ml.move);
	}
	return true;
}
