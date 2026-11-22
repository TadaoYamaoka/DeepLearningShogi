#pragma once

#include <memory>
#include "mate.h"

// UCTワーカーが単独で所有する。盤面上の応手で証明できた5手不詰めだけを記録する。
class Mate5RefutationCache {
public:
	// 攻め玉が王手されていない末端からの探索専用。
	bool mate(Position& pos, int draw_ply) {
		return mate(pos, draw_ply, [] {});
	}

	template <class OnSearch>
	bool mate(Position& pos, int draw_ply, OnSearch onSearch) {
		assert(!pos.inCheck());
		assert(pos.continuousCheck(oppositeColor(pos.turn())) == 0);
#if defined(BAN_BLACK_REPETITION) || defined(BAN_WHITE_REPETITION)
		onSearch();
		return mateMoveInOddPly<5>(pos, draw_ply);
#else
		if (pos.gamePly() + 2 > draw_ply)
			return false;

		const CheckInfo ci(pos);
		ns_mate::CheckMovePicker<false> checks(pos, ci);
		// 内側の3手探索は王手後（根元から3手目）にさらに3手分の余裕を確認するため、再利用は残り6手以上に限る。
		bool reusable = checks.size() >= 4 && draw_ply - pos.gamePly() >= 6;
		Key* slot = nullptr;
		if (reusable) {
			const Key key = pos.getKey();
			slot = &entries_[(key >> 1) % Capacity];
			// 64ビットキーの衝突による見逃しを許容する。
			if (*slot == key)
				return false;
		}

		// 探索中に反復か外側の逆王手があれば記録しない。残る不詰めは各王手に盤上の反証がある。
		// 入口で受け方の連続王手は0。非王手の応手の後も0なので、攻め方に有利な反復判定に必要な4手に届かない。
		onSearch();
		const bool found = ns_mate::searchChecks<5>(pos, draw_ply, ci, checks, [&] { reusable = false; });
		if (!found && reusable)
			*slot = pos.getKey();
		return found;
#endif
	}

private:
#if !defined(BAN_BLACK_REPETITION) && !defined(BAN_WHITE_REPETITION)
	static constexpr size_t Capacity = 1u << 19;
	// キー0は枠0、キー2は枠1を引く。枠0だけ2で初期化すれば空枠にヒットしない。
	std::unique_ptr<Key[]> entries_{new Key[Capacity]{2}};
#endif
};
