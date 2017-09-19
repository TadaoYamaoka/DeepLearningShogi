#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"

#include "mate.h"

// 7手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn7Ply(Position& pos)
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
			// 6手詰めチェック
			if (mateMoveIn6Ply(pos)) {
				// 詰みが見つかった時点で終了
				pos.undoMove(ml.move());
				return true;
			}
		}

		pos.undoMove(ml.move());
	}
	return false;
}

// 6手詰めチェック
// 手番側が王手されていること
bool mateMoveIn6Ply(Position& pos)
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

		// 5手詰めかどうか
		if (!mateMoveIn5Ply(pos)) {
			// 5手詰めでない場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move());
			return false;
		}

		pos.undoMove(ml.move());
	}
	return true;
}

// 5手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn5Ply(Position& pos)
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
			// 4手詰めチェック
			if (mateMoveIn4Ply(pos)) {
				// 詰みが見つかった時点で終了
				pos.undoMove(ml.move());
				return true;
			}
		}

		pos.undoMove(ml.move());
	}
	return false;
}

// 4手詰めチェック
// 手番側が王手されていること
bool mateMoveIn4Ply(Position& pos)
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

		// 3手詰めかどうか
		if (!mateMoveIn3Ply(pos)) {
			// 3手詰めでない場合
			// 詰みが見つからなかった時点で終了
			pos.undoMove(ml.move());
			return false;
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