#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "mate.h"

// 奇数手詰めチェック
// 手番側が王手でないこと
// 詰ます手を返すバージョン
Move mateMoveInOddPlyReturnMove(Position& pos, int depth) {
	// OR節点

	// すべての合法手について
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move(), state);

		// 王手かどうか
		if (pos.inCheck()) {
			//std::cout << ml.move().toUSI() << std::endl;
			// 王手の場合
			// 偶数手詰めチェック
			if (mateMoveInEvenPly(pos, depth - 1)) {
				// 詰みが見つかった時点で終了
				pos.undoMove(ml.move());
				return ml.move();
			}
		}

		pos.undoMove(ml.move());
	}
	return Move::moveNone();
}

// 奇数手詰めチェック
// 手番側が王手でないこと
bool mateMoveInOddPly(Position& pos, int depth)
{
	// OR節点

	// すべての合法手について
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move(), state);

		// 王手かどうか
		if (pos.inCheck()) {
			//std::cout << ml.move().toUSI() << std::endl;
			// 王手の場合
			// 偶数手詰めチェック
			if (mateMoveInEvenPly(pos, depth - 1)) {
				// 詰みが見つかった時点で終了
				pos.undoMove(ml.move());
				return true;
			}
		}

		pos.undoMove(ml.move());
	}
	return false;
}

// 偶数手詰めチェック
// 手番側が王手されていること
bool mateMoveInEvenPly(Position& pos, int depth)
{
	// AND節点

	// すべてのEvasionについて
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		//std::cout << " " << ml.move().toUSI() << std::endl;
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move(), state);

		// 王手かどうか
		if (pos.inCheck()) {
			// 王手の場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move());
			return false;
		}

		if (depth == 4) {
			// 3手詰めかどうか
			if (!mateMoveIn3Ply(pos)) {
				// 3手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move());
				return false;
			}
		}
		else {
			// 奇数手詰めかどうか
			if (!mateMoveInOddPly(pos, depth - 1)) {
				// 偶数手詰めでない場合
				// 詰みが見つからなかった時点で終了
				pos.undoMove(ml.move());
				return false;
			}
		}

		pos.undoMove(ml.move());
	}
	return true;
}

// 3手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn3Ply(Position& pos)
{
	// OR節点

	// すべての合法手について
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move(), state);

		// 王手かどうか
		if (pos.inCheck()) {
			//std::cout << ml.move().toUSI() << std::endl;
			// 王手の場合
			// 2手詰めチェック
			if (mateMoveIn2Ply(pos)) {
				// 詰みが見つかった時点で終了
				pos.undoMove(ml.move());
				return true;
			}
		}

		pos.undoMove(ml.move());
	}
	return false;
}

// 2手詰めチェック
// 手番側が王手されていること
bool mateMoveIn2Ply(Position& pos)
{
	// AND節点

	// すべてのEvasionについて
	for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
		//std::cout << " " << ml.move().toUSI() << std::endl;
		// 1手動かす
		StateInfo state;
		pos.doMove(ml.move(), state);

		// 王手かどうか
		if (pos.inCheck()) {
			// 王手の場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move());
			return false;
		}

		// 1手詰めかどうか
		if (pos.mateMoveIn1Ply() == Move::moveNone()) {
			// 1手詰めでない場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move());
			return false;
		}

		pos.undoMove(ml.move());
	}
	return true;
}