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
				const Bitboard pinned = pos.pinnedBB();
				while (curr != last_) {
					if (!pos.pseudoLegalMoveIsEvasion(curr->move, pinned))
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
// 手番側が王手でないこと
// 詰ます手を返すバージョン
Move mateMoveInOddPlyReturnMove(Position& pos, const int depth) {
	// OR節点

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : MovePicker<true, false>(pos)) {
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

// 奇数手詰めチェック
// 手番側が王手でないこと
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
