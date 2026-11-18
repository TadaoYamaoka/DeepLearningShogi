#pragma once

#include <cassert>
#include <climits>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

template <int depth> bool mateMoveInEvenPly(Position& pos, const int draw_ply = INT_MAX);

// 詰み探索用の指し手picker
namespace ns_mate {
	const constexpr size_t MaxCheckMoves = 91;

	// 攻め側。王手生成とdoMove()で同じCheckInfoを共有する。
	template <bool INCHECK>
	class CheckMovePicker {
	public:
		CheckMovePicker(const Position& pos, const CheckInfo& ci) {
			last_ = generateCheckAllMoves(moveList_, pos, ci);
			if (INCHECK) {
				ExtMove* curr = moveList_;
				// 玉以外の候補が現れるまでは、遮断・捕獲先を求めない。
				while (curr != last_ && curr->move.pieceTypeFrom() == King) {
					if (kingMoveIsEvasion(pos, curr->move))
						++curr;
					else
						curr->move = (--last_)->move;
				}
				if (curr != last_) {
					const Bitboard checkers = pos.checkersBB();
					Bitboard remaining = checkers;
					const Square checker = remaining.firstOneFromSQ11();
					const Bitboard nonKingTargets = remaining ? allZeroBB() : betweenBB(checker, pos.kingSquare(pos.turn())) | checkers;
					do {
						const Move move = curr->move;
						const bool evasion = move.pieceTypeFrom() == King
							? kingMoveIsEvasion(pos, move)
							: nonKingTargets.isSet(move.to());
						if (evasion)
							++curr;
						else
							curr->move = (--last_)->move;
					} while (curr != last_);
				}
			}
			assert(static_cast<size_t>(last_ - moveList_) <= MaxCheckMoves);
		}
		size_t size() const { return static_cast<size_t>(last_ - moveList_); }
		ExtMove* begin() { return &moveList_[0]; }
		ExtMove* end() { return last_; }
		bool empty() const { return size() == 0; }

	private:
		static bool kingMoveIsEvasion(const Position& pos, Move move) {
			Bitboard occupied = pos.occupiedBB();
			occupied.clearBit(move.from());
			return !pos.attackersToIsAny(oppositeColor(pos.turn()), move.to(), occupied);
		}

		ExtMove moveList_[MaxCheckMoves];
		ExtMove* last_;
	};

	// 受け側。玉以外の応手は、従来の探索順で必要になってから生成する。
	class EvasionMovePicker {
	public:
		explicit EvasionMovePicker(const Position& pos)
			: curr_(moveList_), last_(generateKingEvasions(moveList_, pos)) {}

		// posは必ずこのpickerを構築した局面に戻してから呼び出す。
		// pinnedには同じ局面で計算したCheckInfo::pinned等を渡す。
		// 合法手が残っていなければmoveNone()を返す。
		Move nextLegal(const Position& pos, const Bitboard& pinned) {
			if (nonKingEvasionsPending_) {
				const bool hasKingMove = curr_ != last_;
				if (hasKingMove && kingMoveIsLegal(pos, curr_->move))
					return (curr_++)->move;

				// 非合法な玉の手は候補列全体の末尾と交換する。玉の手を使い切った場合もここで追記する。
				Bitboard checkers = pos.checkersBB();
				const Square checker = checkers.firstOneFromSQ11();
				if (!checkers)
					last_ = generateNonKingEvasions(last_, pos, checker);
				nonKingEvasionsPending_ = false;
				if (hasKingMove)
					curr_->move = (--last_)->move;
			}

			while (curr_ != last_) {
				if (!generatedMoveIsLegal(pos, curr_->move, pinned))
					curr_->move = (--last_)->move;
				else
					return (curr_++)->move;
			}
			return Move::moveNone();
		}

	private:
		static bool kingMoveIsLegal(const Position& pos, const Move move) {
			return !pos.attackersToIsAny(oppositeColor(pos.turn()), move.to());
		}

		// 回避手生成済みの候補なので、移動元の駒種は指し手から取得できる。
		static bool generatedMoveIsLegal(const Position& pos, const Move move, const Bitboard& pinned) {
			if (move.isDrop())
				return true;
			assert(move.pieceTypeFrom() == pieceToPieceType(pos.piece(move.from())));
			if (move.pieceTypeFrom() == King)
				return kingMoveIsLegal(pos, move);
			return !pos.isPinnedIllegal(move.from(), move.to(), pos.kingSquare(pos.turn()), pinned);
		}

		ExtMove moveList_[MaxLegalMoves];
		ExtMove* curr_;
		ExtMove* last_;
		bool nonKingEvasionsPending_ = true;
	};

	enum class MateRepetitionAction {
		Continue,
		Success,
		Failure
	};

	// 攻め側の王手後。現在の手番は受け側。
	FORCE_INLINE MateRepetitionAction repetitionAfterCheck(const Position& pos) {
		switch (pos.isDraw(16)) {
		case NotRepetition:
		case RepetitionInferior:
			return MateRepetitionAction::Continue;
		case RepetitionLose:
			return MateRepetitionAction::Success;
		case RepetitionDraw:
		case RepetitionWin:
		case RepetitionSuperior:
			return MateRepetitionAction::Failure;
		default:
			UNREACHABLE;
			return MateRepetitionAction::Failure;
		}
	}

	// 受け側の着手後。現在の手番は攻め側。
	FORCE_INLINE MateRepetitionAction repetitionAfterEvasion(const Position& pos) {
		switch (pos.isDraw(16)) {
		case NotRepetition:
		case RepetitionSuperior:
			return MateRepetitionAction::Continue;
		case RepetitionWin:
			return MateRepetitionAction::Success;
		case RepetitionDraw:
		case RepetitionLose:
		case RepetitionInferior:
			return MateRepetitionAction::Failure;
		default:
			UNREACHABLE;
			return MateRepetitionAction::Failure;
		}
	}
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
	for (const auto& ml : ns_mate::CheckMovePicker<INCHECK>(pos, ci))
	{
		const Move& m = ml.move;

		pos.doMove(m, si, ci, true);

		// RepetitionLoseだけは、盤面上の詰みを調べる前でも攻め側の成功になる。
		// 受け側のcontinuousCheckが4未満ならRepetitionLoseは起こらないため、
		// 反復判定を「盤面上で詰みが見つかった時」まで遅延する。
		bool defer_check_repetition = false;
		if (si.continuousCheck[pos.turn()] >= 4) {
			const ns_mate::MateRepetitionAction repetition = ns_mate::repetitionAfterCheck(pos);
			if (repetition == ns_mate::MateRepetitionAction::Success) {
				pos.undoMove(m);
				return true;
			}
			if (repetition == ns_mate::MateRepetitionAction::Failure) {
				pos.undoMove(m);
				continue;
			}
		}
		else {
			defer_check_repetition = true;
		}

		// 最初の合法な受けだけを取得する。残りの合法性判定は遅延する。
		ns_mate::EvasionMovePicker move_picker2(pos);
		const Bitboard pinned2 = pos.pinnedBB();
		Move m2 = move_picker2.nextLegal(pos, pinned2);

		if (!m2) {
			// 盤面上は1手で詰んでいる。遅延していた反復判定があれば、ここでだけ実行する。
			if (defer_check_repetition &&
				ns_mate::repetitionAfterCheck(pos) == ns_mate::MateRepetitionAction::Failure) {
				pos.undoMove(m);
				continue;
			}
			pos.undoMove(m);
			return true;
		}

		// 最大手数チェック
		if (pos.gamePly() + 3 > draw_ply) {
			pos.undoMove(m);
			continue;
		}

		// 受けなし・手数制限で終了する場合はCheckInfo全体を構築しない。
		// 最初の合法性判定に使ったpin情報を再利用する。
		const CheckInfo ci2(pos, pinned2);
		do {

			// この指し手で逆王手になるなら、不詰めとして扱う
			if (pos.moveGivesCheck(m2, ci2))
				goto NEXT_CHECK;

			pos.doMove(m2, si2, ci2, false);

			if (!pos.mateMoveIn1Ply()) {
				// 詰んでないので、m2で詰みを逃れている。
				// m2は非王手なので、反復判定でこの失敗が成功へ反転することはない。
				pos.undoMove(m2);
				goto NEXT_CHECK;
			}

			// 盤面上の1手詰めが見つかった場合だけ反復履歴を走査する。
			// m2は非王手なのでRepetitionWinは発生しないが、分類は一般の偶数手探索と揃える。
			const ns_mate::MateRepetitionAction repetition = ns_mate::repetitionAfterEvasion(pos);
			if (repetition == ns_mate::MateRepetitionAction::Failure) {
				pos.undoMove(m2);
				goto NEXT_CHECK;
			}

			pos.undoMove(m2);
		} while ((m2 = move_picker2.nextLegal(pos, ci2.pinned)));

		// 盤面上ではすべての受けに対して詰んだ。
		// m後の反復判定を遅延していた場合だけ、成功を確定する直前に走査する。
		if (defer_check_repetition &&
			ns_mate::repetitionAfterCheck(pos) == ns_mate::MateRepetitionAction::Failure) {
			pos.undoMove(m);
			continue;
		}

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
	for (const auto& ml : ns_mate::CheckMovePicker<INCHECK>(pos, ci)) {
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
	for (const auto& ml : ns_mate::CheckMovePicker<INCHECK>(pos, ci)) {
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
bool mateMoveInEvenPly(Position& pos, const int draw_ply)
{
	// AND節点

	// 受け手の合法性は取り出すときに判定し、ci.pinnedを共有する。
	const CheckInfo ci(pos);
	ns_mate::EvasionMovePicker move_picker(pos);
	for (Move m = move_picker.nextLegal(pos, ci.pinned); m;
		m = move_picker.nextLegal(pos, ci.pinned)) {
		//std::cout << depth << " : " << pos.toSFEN() << " : " << m.toUSI() << std::endl;
		const bool givesCheck = pos.moveGivesCheck(m, ci);

		// 1手動かす
		StateInfo state;
		pos.doMove(m, state, ci, givesCheck);

		// 千日手チェック
		switch (pos.isDraw(16)) {
		case NotRepetition: break;
		case RepetitionWin: // 自分が勝ち
		{
			pos.undoMove(m);
			continue;
		}
		case RepetitionDraw:
		case RepetitionLose: // 自分が負け
		case RepetitionInferior: // 自分が駒損
		{
			// 詰みが見つからなかった時点で終了
			pos.undoMove(m);
			return false;
		}
		case RepetitionSuperior: break; // 自分が駒得
		default: UNREACHABLE;
		}

		// 奇数手詰めかどうか
		if (givesCheck ? !mateMoveInOddPly<depth - 1, true>(pos, draw_ply) : !mateMoveInOddPly<depth - 1, false>(pos, draw_ply)) {
			// 偶数手詰めでない場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(m);
			return false;
		}

		pos.undoMove(m);
	}
	return true;
}
