﻿#pragma once

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

template <int depth> bool mateMoveInEvenPly(Position& pos);

// 詰み探索用のMovePicker
namespace ns_mate {
	const constexpr size_t MaxCheckMoves = 91;

	template <bool or_node, bool INCHECK>
	class MovePicker {
	public:
		explicit MovePicker(const Position& pos) {
			if (or_node) {
				last_ = generateMoves<CheckAll>(moveList_, pos);
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
}

// 3手詰めチェック
template <bool INCHECK>
FORCE_INLINE bool mateMoveIn3Ply(Position& pos, const int draw_ply = INT_MAX)
{
	// OR節点

	// 最大手数チェック
	if (pos.gamePly() + 2 > draw_ply)
		return false;

	StateInfo si;
	StateInfo si2;

	const CheckInfo ci(pos);
	for (const auto& ml : ns_mate::MovePicker<true, INCHECK>(pos))
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
		ns_mate::MovePicker<false, false> move_picker2(pos);

		if (move_picker2.size() == 0) {
			// 1手で詰んだ
			pos.undoMove(m);
			return true;
		}

		// 最大手数チェック
		if (pos.gamePly() + 3 > draw_ply) {
			pos.undoMove(m);
			continue;
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
template <int depth, bool INCHECK>
Move mateMoveInOddPlyReturnMove(Position& pos, const int draw_ply = INT_MAX) {
	// OR節点

	// 最大手数チェック
	if (pos.gamePly() + 2 > draw_ply)
		return Move::moveNone();

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : ns_mate::MovePicker<true, INCHECK>(pos)) {
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
		if (mateMoveInEvenPly<depth - 1>(pos, draw_ply)) {
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return ml.move;
		}

		pos.undoMove(ml.move);
	}
	return Move::moveNone();
}

// 奇数手詰めチェック
template <int depth, bool INCHECK = false>
bool mateMoveInOddPly(Position& pos, const int draw_ply = INT_MAX)
{
	// OR節点

	// 最大手数チェック
	if (pos.gamePly() + 2 > draw_ply)
		return false;

	// すべての合法手について
	const CheckInfo ci(pos);
	for (const auto& ml : ns_mate::MovePicker<true, INCHECK>(pos)) {
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
		if (mateMoveInEvenPly<depth - 1>(pos, draw_ply)) {
			// 詰みが見つかった時点で終了
			pos.undoMove(ml.move);
			return true;
		}

		pos.undoMove(ml.move);
	}
	return false;
}

// 3手詰めの特殊化
template <> FORCE_INLINE bool mateMoveInOddPly<3, false>(Position& pos, const int draw_ply) { return mateMoveIn3Ply<false>(pos, draw_ply); }
template <> FORCE_INLINE bool mateMoveInOddPly<3, true>(Position& pos, const int draw_ply) { return mateMoveIn3Ply<true>(pos, draw_ply); }

// 偶数手詰めチェック
// 手番側が王手されていること
template <int depth>
bool mateMoveInEvenPly(Position& pos, const int draw_ply = INT_MAX)
{
	// AND節点

	// すべてのEvasionについて
	const CheckInfo ci(pos);
	for (const auto& ml : ns_mate::MovePicker<false, false>(pos)) {
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

		// 奇数手詰めかどうか
		if (givesCheck ? !mateMoveInOddPly<depth - 1, true>(pos, draw_ply) : !mateMoveInOddPly<depth - 1, false>(pos, draw_ply)) {
			// 偶数手詰めでない場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move);
			return false;
		}

		pos.undoMove(ml.move);
	}
	return true;
}
