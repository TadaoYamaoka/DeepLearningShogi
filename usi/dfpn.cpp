#include <unordered_set>

#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "dfpn.h"

using namespace std;
using namespace ns_dfpn;

int64_t DfPn::HASH_SIZE_MB = 2048;
uint32_t DfPn::kMaxDepth = 30;
const constexpr uint32_t REPEAT = UINT_MAX;
const constexpr size_t MaxCheckMoves = 73;

// --- 詰み将棋探索

void DfPn::dfpn_stop(const bool stop)
{
	this->stop = stop;
}

// 詰将棋エンジン用のMovePicker
template <bool or_node>
class MovePicker {
public:
	explicit MovePicker(const Position& pos) {
		if (or_node) {
			last_ = generateMoves<Check>(moveList_, pos);
			if (pos.inCheck()) {
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

// 置換表
TranspositionTable::~TranspositionTable() {
	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}
}

TTEntry& TranspositionTable::LookUp(const Key key, const Hand hand, const uint16_t depth) {
	auto& entries = tt[key & clusters_mask];
	uint32_t hash_high = key >> 32;
	return LookUpDirect(entries, hash_high, hand, depth);
}

TTEntry& TranspositionTable::LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth) {
	int max_pn = 1;
	int max_dn = 1;
	// 検索条件に合致するエントリを返す
	for (size_t i = 0; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
		TTEntry& entry = entries.entries[i];
		if (generation != entry.generation) {
			// 空のエントリが見つかった場合
			entry.hash_high = hash_high;
			entry.depth = depth;
			entry.hand = hand;
			entry.pn = max_pn;
			entry.dn = max_dn;
			entry.generation = generation;
			entry.num_searched = 0;
			return entry;
		}

		if (hash_high == entry.hash_high && generation == entry.generation) {
			if (hand == entry.hand && depth == entry.depth) {
				// keyが合致するエントリを見つけた場合
				// 残りのエントリに優越関係を満たす局面があり証明済みの場合、それを返す
				for (i++; i < sizeof(entries.entries) / sizeof(TTEntry); i++) {
					TTEntry& entry_rest = entries.entries[i];
					if (generation != entry_rest.generation) break;
					if (hash_high == entry_rest.hash_high) {
						if (entry_rest.pn == 0) {
							if (hand.isEqualOrSuperior(entry_rest.hand) && entry_rest.num_searched != REPEAT) {
								entry_rest.generation = generation;
								return entry_rest;
							}
						}
						else if (entry_rest.dn == 0) {
							if (entry_rest.hand.isEqualOrSuperior(hand) && entry_rest.num_searched != REPEAT) {
								entry_rest.generation = generation;
								return entry_rest;
							}
						}
					}
				}
				return entry;
			}
			// 優越関係を満たす局面に証明済みの局面がある場合、それを返す
			if (entry.pn == 0) {
				if (hand.isEqualOrSuperior(entry.hand) && entry.num_searched != REPEAT) {
					return entry;
				}
			}
			else if (entry.dn == 0) {
				if (entry.hand.isEqualOrSuperior(hand) && entry.num_searched != REPEAT) {
					return entry;
				}
			}
			else if (entry.hand.isEqualOrSuperior(hand)) {
				if (entry.pn > max_pn) max_pn = entry.pn;
			}
			else if (hand.isEqualOrSuperior(entry.hand)) {
				if (entry.dn > max_dn) max_dn = entry.dn;
			}
		}
	}

	//cout << "hash entry full" << endl;
	// 合致するエントリが見つからなかったので
	// 古いエントリをつぶす
	TTEntry* best_entry = nullptr;
	uint32_t best_num_searched = UINT_MAX;
	TTEntry* best_entry_include_mate = nullptr;
	uint32_t best_num_searched_include_mate = UINT_MAX;
	for (auto& entry : entries.entries) {
		if (entry.pn != 0) {
			if (best_num_searched > entry.num_searched) {
				best_entry = &entry;
				best_num_searched = entry.num_searched;
			}
		}
		else {
			if (best_num_searched_include_mate > entry.num_searched) {
				best_entry_include_mate = &entry;
				best_num_searched_include_mate = entry.num_searched;
			}
		}
	}
	if (best_entry == nullptr)
		best_entry = best_entry_include_mate;

	best_entry->hash_high = hash_high;
	best_entry->hand = hand;
	best_entry->depth = depth;
	best_entry->pn = 1;
	best_entry->dn = 1;
	best_entry->generation = generation;
	best_entry->num_searched = 0;
	return *best_entry;
}

template <bool or_node>
TTEntry& TranspositionTable::LookUp(const Position& n, const uint16_t depth) {
	return LookUp(n.getBoardKey(), or_node ? n.hand(n.turn()) : n.hand(oppositeColor(n.turn())), depth);
}

// moveを指した後の子ノードのキーを返す
template <bool or_node>
void TranspositionTable::GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand) {
	// 手駒は常に先手の手駒で表す
	if (or_node) {
		hand = n.hand(n.turn());
		if (move.isDrop()) {
			hand.minusOne(move.handPieceDropped());
		}
		else {
			const Piece to_pc = n.piece(move.to());
			if (to_pc != Empty) {
				const PieceType pt = pieceToPieceType(to_pc);
				hand.plusOne(pieceTypeToHandPiece(pt));
			}
		}
	}
	else {
		hand = n.hand(oppositeColor(n.turn()));
	}
	Key key = n.getBoardKeyAfter(move);
	entries = &tt[key & clusters_mask];
	hash_high = key >> 32;
}

// moveを指した後の子ノードの置換表エントリを返す
template <bool or_node>
TTEntry& TranspositionTable::LookUpChildEntry(const Position& n, const Move move, const uint16_t depth) {
	Cluster* entries;
	uint32_t hash_high;
	Hand hand;
	GetChildFirstEntry<or_node>(n, move, entries, hash_high, hand);
	return LookUpDirect(*entries, hash_high, hand, depth + 1);
}

void TranspositionTable::Resize(int64_t hash_size_mb) {
	if (hash_size_mb == 16) {
		hash_size_mb = 4096;
	}
	int64_t new_num_clusters = 1LL << msb((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
	if (new_num_clusters == num_clusters) {
		return;
	}

	num_clusters = new_num_clusters;

	if (tt_raw) {
		std::free(tt_raw);
		tt_raw = nullptr;
		tt = nullptr;
	}

	tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
	tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
	clusters_mask = num_clusters - 1;
}

void TranspositionTable::NewSearch() {
	++generation;
	if (generation == 0) generation = 1;
}

static const constexpr int kInfinitePnDn = 100000000;

// 王手の指し手が近接王手か
FORCE_INLINE bool moveGivesNeighborCheck(const Position& pos, const Move& move)
{
	const Color them = oppositeColor(pos.turn());
	const Square ksq = pos.kingSquare(them);

	const Square to = move.to();

	// 敵玉の8近傍
	if (pos.attacksFrom<King>(ksq).isSet(to))
		return true;

	// 桂馬による王手
	if (move.pieceTypeTo() == Lance)
		return true;

	return false;
}

// 反証駒を計算(持っている持ち駒を最大数にする(後手の持ち駒を加える))
FORCE_INLINE u32 dp(const Hand& us, const Hand& them) {
	u32 dp = 0;
	u32 pawn = us.exists<HPawn>(); if (pawn > 0) dp += pawn + them.exists<HPawn>();
	u32 lance = us.exists<HLance>(); if (lance > 0) dp += lance + them.exists<HLance>();
	u32 knight = us.exists<HKnight>(); if (knight > 0) dp += knight + them.exists<HKnight>();
	u32 silver = us.exists<HSilver>(); if (silver > 0) dp += silver + them.exists<HSilver>();
	u32 gold = us.exists<HGold>(); if (gold > 0) dp += gold + them.exists<HGold>();
	u32 bishop = us.exists<HBishop>(); if (bishop > 0) dp += bishop + them.exists<HBishop>();
	u32 rook = us.exists<HRook>(); if (rook > 0) dp += rook + them.exists<HRook>();
	return dp;
}

template <bool or_node>
void DfPn::dfpn_inner(Position& n, int thpn, int thdn/*, bool inc_flag*/, uint16_t depth, int64_t& searchedNode) {
	auto& entry = transposition_table.LookUp<or_node>(n, depth);

	if (depth > kMaxDepth) {
		entry.pn = kInfinitePnDn;
		entry.dn = 0;
		return;
	}

	// if (n is a terminal node) { handle n and return; }
	MovePicker<or_node> move_picker(n);
	if (move_picker.empty()) {
		// nが先端ノード

		if (or_node) {
			// 自分の手番でここに到達した場合は王手の手が無かった、
			entry.pn = kInfinitePnDn;
			entry.dn = 0;

			// 反証駒
			// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
			entry.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));
		}
		else {
			// 相手の手番でここに到達した場合は王手回避の手が無かった、
			// 1手詰めを行っているため、ここに到達することはない
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
		}

		return;
	}

	// 新規節点で固定深さの探索を併用
	if (entry.num_searched == 0) {
		if (or_node) {
			if (!n.inCheck()) {
				// 3手詰みチェック
				Color us = n.turn();
				Color them = oppositeColor(us);

				StateInfo si;
				StateInfo si2;

				const CheckInfo ci(n);
				for (const auto& ml : move_picker)
				{
					const Move& m = ml.move;

					n.doMove(m, si, ci, true);

					// 千日手のチェック
					if (n.isDraw(16) == RepetitionWin) {
						// 受け側の反則勝ち
						n.undoMove(m);
						continue;
					}

					auto& entry2 = transposition_table.LookUp<false>(n, depth + 1);

					// この局面ですべてのevasionを試す
					MovePicker<false> move_picker2(n);

					if (move_picker2.size() == 0) {
						// 1手で詰んだ
						n.undoMove(m);

						entry2.pn = 0;
						entry2.dn = kInfinitePnDn;

						entry.pn = 0;
						entry.dn = kInfinitePnDn;

						// 証明駒を初期化
						entry.hand.set(0);

						// 打つ手ならば証明駒に加える
						if (m.isDrop()) {
							entry.hand.plusOne(m.handPieceDropped());
						}
						// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
						if (!moveGivesNeighborCheck(n, m))
							entry.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));

						return;
					}

					const CheckInfo ci2(n);
					for (const auto& move : move_picker2)
					{
						const Move& m2 = move.move;

						// この指し手で逆王手になるなら、不詰めとして扱う
						if (n.moveGivesCheck(m2, ci2))
							goto NEXT_CHECK;

						n.doMove(m2, si2, ci2, false);

						if (n.mateMoveIn1Ply()) {
							auto& entry1 = transposition_table.LookUp<true>(n, depth + 2);
							entry1.pn = 0;
							entry1.dn = kInfinitePnDn;
						}
						else {
							// 詰んでないので、m2で詰みを逃れている。
							n.undoMove(m2);
							goto NEXT_CHECK;
						}

						n.undoMove(m2);
					}

					// すべて詰んだ
					n.undoMove(m);

					entry2.pn = 0;
					entry2.dn = kInfinitePnDn;

					entry.pn = 0;
					entry.dn = kInfinitePnDn;

					return;

				NEXT_CHECK:;
					n.undoMove(m);

					if (entry2.num_searched == 0) {
						entry2.num_searched = 1;
						entry2.pn = move_picker2.size();
						entry2.dn = move_picker2.size();
					}
				}
			}
		}
		else {
			// 2手読みチェック
			StateInfo si2;
			// この局面ですべてのevasionを試す
			const CheckInfo ci2(n);
			for (const auto& move : move_picker)
			{
				const Move& m2 = move.move;

				// この指し手で逆王手になるなら、不詰めとして扱う
				if (n.moveGivesCheck(m2, ci2))
					goto NO_MATE;

				n.doMove(m2, si2, ci2, false);

				if (const Move move = n.mateMoveIn1Ply()) {
					auto& entry1 = transposition_table.LookUp<true>(n, depth + 1);
					entry1.pn = 0;
					entry1.dn = kInfinitePnDn;

					// 証明駒を初期化
					entry1.hand.set(0);

					// 打つ手ならば証明駒に加える
					if (move.isDrop()) {
						entry1.hand.plusOne(move.handPieceDropped());
					}
					// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
					if (!moveGivesNeighborCheck(n, move))
						entry1.hand.setPP(n.hand(n.turn()), n.hand(oppositeColor(n.turn())));
				}
				else {
					// 詰んでないので、m2で詰みを逃れている。
					// 不詰みチェック
					// 王手がない場合
					MovePicker<true> move_picker2(n);
					if (move_picker2.empty()) {
						auto& entry1 = transposition_table.LookUp<true>(n, depth + 1);
						entry1.pn = kInfinitePnDn;
						entry1.dn = 0;
						// 反証駒
						// 持っている持ち駒を最大数にする(後手の持ち駒を加える)
						entry1.hand.set(dp(n.hand(n.turn()), n.hand(oppositeColor(n.turn()))));

						n.undoMove(m2);

						entry.pn = kInfinitePnDn;
						entry.dn = 0;
						// 子局面の反証駒を設定
						// 打つ手ならば、反証駒から削除する
						if (m2.isDrop()) {
							entry.hand = entry1.hand;
							entry.hand.minusOne(m2.handPieceDropped());
						}
						// 先手の駒を取る手ならば、反証駒に追加する
						else {
							const Piece to_pc = n.piece(m2.to());
							if (to_pc != Empty) {
								const PieceType pt = pieceToPieceType(to_pc);
								const HandPiece hp = pieceTypeToHandPiece(pt);
								if (entry.hand.numOf(hp) > entry1.hand.numOf(hp)) {
									entry.hand = entry1.hand;
									entry.hand.plusOne(hp);
								}
							}
						}
						return;
					}
					n.undoMove(m2);
					goto NO_MATE;
				}

				n.undoMove(m2);
			}

			// すべて詰んだ
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			return;

		NO_MATE:;
		}
	}

	// 千日手のチェック
	switch (n.isDraw(16)) {
	case RepetitionWin:
		//cout << "RepetitionWin" << endl;
		// 連続王手の千日手による勝ち
		if (or_node) {
			// ここは通らないはず
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
		}
		else {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
		}
		return;

	case RepetitionLose:
		//cout << "RepetitionLose" << endl;
		// 連続王手の千日手による負け
		if (or_node) {
			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			entry.num_searched = REPEAT;
		}
		else {
			// ここは通らないはず
			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			entry.num_searched = REPEAT;
		}
		return;

	case RepetitionDraw:
		//cout << "RepetitionDraw" << endl;
		// 普通の千日手
		// ここは通らないはず
		entry.pn = kInfinitePnDn;
		entry.dn = 0;
		entry.num_searched = REPEAT;
		return;
	}

	// 子局面のハッシュエントリをキャッシュ
	struct TTKey {
		TranspositionTable::Cluster* entries;
		uint32_t hash_high;
		Hand hand;
	} ttkeys[MaxCheckMoves];

	for (const auto& move : move_picker) {
		auto& ttkey = ttkeys[&move - move_picker.begin()];
		transposition_table.GetChildFirstEntry<or_node>(n, move, ttkey.entries, ttkey.hash_high, ttkey.hand);
	}

	while (searchedNode < maxSearchNode && !stop) {
		++entry.num_searched;

		Move best_move;
		int thpn_child;
		int thdn_child;

		// expand and compute pn(n) and dn(n);
		if (or_node) {
			// ORノードでは最も証明数が小さい = 玉の逃げ方の個数が少ない = 詰ましやすいノードを選ぶ
			int best_pn = kInfinitePnDn;
			int second_best_pn = kInfinitePnDn;
			int best_dn = 0;
			uint32_t best_num_search = UINT_MAX;

			entry.pn = kInfinitePnDn;
			entry.dn = 0;
			// 子局面の反証駒の積集合
			u32 pawn = UINT_MAX;
			u32 lance = UINT_MAX;
			u32 knight = UINT_MAX;
			u32 silver = UINT_MAX;
			u32 gold = UINT_MAX;
			u32 bishop = UINT_MAX;
			u32 rook = UINT_MAX;
			bool first = true;
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, depth + 1);
				if (child_entry.pn == 0) {
					// 詰みの場合
					//cout << n.toSFEN() << " or" << endl;
					//cout << bitset<32>(entry.hand.value()) << endl;
					entry.pn = 0;
					entry.dn = kInfinitePnDn;
					// 子局面の証明駒を設定
					// 打つ手ならば、証明駒に追加する
					if (move.move.isDrop()) {
						const HandPiece hp = move.move.handPieceDropped();
						if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
							entry.hand = child_entry.hand;
							entry.hand.plusOne(move.move.handPieceDropped());
						}
					}
					// 後手の駒を取る手ならば、証明駒から削除する
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {
							entry.hand = child_entry.hand;
							const PieceType pt = pieceToPieceType(to_pc);
							const HandPiece hp = pieceTypeToHandPiece(pt);
							if (entry.hand.exists(hp))
								entry.hand.minusOne(hp);
						}
					}
					//cout << bitset<32>(entry.hand.value()) << endl;
					break;
				}
				else if (entry.dn == 0) {
					if (child_entry.dn == 0) {
						const Hand& child_dp = child_entry.hand;
						// 歩
						const u32 child_pawn = child_dp.exists<HPawn>();
						if (child_pawn < pawn) pawn = child_pawn;
						// 香車
						const u32 child_lance = child_dp.exists<HLance>();
						if (child_lance < lance) lance = child_lance;
						// 桂馬
						const u32 child_knight = child_dp.exists<HKnight>();
						if (child_knight < knight) knight = child_knight;
						// 銀
						const u32 child_silver = child_dp.exists<HSilver>();
						if (child_silver < silver) silver = child_silver;
						// 金
						const u32 child_gold = child_dp.exists<HGold>();
						if (child_gold < gold) gold = child_gold;
						// 角
						const u32 child_bishop = child_dp.exists<HBishop>();
						if (child_bishop < bishop) bishop = child_bishop;
						// 飛車
						const u32 child_rook = child_dp.exists<HRook>();
						if (child_rook < rook) rook = child_rook;
					}
				}
				entry.pn = std::min(entry.pn, child_entry.pn);
				entry.dn += child_entry.dn;

				if (child_entry.pn < best_pn ||
					child_entry.pn == best_pn && best_num_search > child_entry.num_searched) {
					second_best_pn = best_pn;
					best_pn = child_entry.pn;
					best_dn = child_entry.dn;
					best_move = move;
					best_num_search = child_entry.num_searched;
				}
				else if (child_entry.pn < second_best_pn) {
					second_best_pn = child_entry.pn;
				}
			}
			entry.dn = std::min(entry.dn, kInfinitePnDn);
			if (entry.dn == 0) {
				// 不詰みの場合
				//cout << n.hand(n.turn()).value() << "," << entry.hand.value() << ",";
				// 先手が一枚も持っていない種類の先手の持ち駒を反証駒から削除する
				u32 curr_pawn = entry.hand.template exists<HPawn>(); if (curr_pawn == 0) pawn = 0; else if (pawn < curr_pawn) pawn = curr_pawn;
				u32 curr_lance = entry.hand.template exists<HLance>(); if (curr_lance == 0) lance = 0; else if (lance < curr_lance) lance = curr_lance;
				u32 curr_knight = entry.hand.template exists<HKnight>(); if (curr_knight == 0) knight = 0; else if (knight < curr_knight) knight = curr_knight;
				u32 curr_silver = entry.hand.template exists<HSilver>(); if (curr_silver == 0) silver = 0; else if (silver < curr_silver) silver = curr_silver;
				u32 curr_gold = entry.hand.template exists<HGold>(); if (curr_gold == 0) gold = 0; else if (gold < curr_gold) gold = curr_gold;
				u32 curr_bishop = entry.hand.template exists<HBishop>(); if (curr_bishop == 0) bishop = 0; else if (bishop < curr_bishop) bishop = curr_bishop;
				u32 curr_rook = entry.hand.template exists<HRook>(); if (curr_rook == 0) rook = 0; else if (rook < curr_rook) rook = curr_rook;
				// 反証駒に子局面の証明駒の積集合を設定
				entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
				//cout << entry.hand.value() << endl;
			}
			else {
				thpn_child = std::min(thpn, second_best_pn + 1);
				thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
			}
		}
		else {
			// ANDノードでは最も反証数の小さい = 王手の掛け方の少ない = 不詰みを示しやすいノードを選ぶ
			int best_dn = kInfinitePnDn;
			int second_best_dn = kInfinitePnDn;
			int best_pn = 0;
			uint32_t best_num_search = UINT_MAX;

			entry.pn = 0;
			entry.dn = kInfinitePnDn;
			// 子局面の証明駒の和集合
			u32 pawn = 0;
			u32 lance = 0;
			u32 knight = 0;
			u32 silver = 0;
			u32 gold = 0;
			u32 bishop = 0;
			u32 rook = 0;
			bool all_mate = true;
			for (const auto& move : move_picker) {
				auto& ttkey = ttkeys[&move - move_picker.begin()];
				const auto& child_entry = transposition_table.LookUpDirect(*ttkey.entries, ttkey.hash_high, ttkey.hand, depth + 1);
				if (all_mate) {
					if (child_entry.pn == 0) {
						const Hand& child_pp = child_entry.hand;
						// 歩
						const u32 child_pawn = child_pp.exists<HPawn>();
						if (child_pawn > pawn) pawn = child_pawn;
						// 香車
						const u32 child_lance = child_pp.exists<HLance>();
						if (child_lance > lance) lance = child_lance;
						// 桂馬
						const u32 child_knight = child_pp.exists<HKnight>();
						if (child_knight > knight) knight = child_knight;
						// 銀
						const u32 child_silver = child_pp.exists<HSilver>();
						if (child_silver > silver) silver = child_silver;
						// 金
						const u32 child_gold = child_pp.exists<HGold>();
						if (child_gold > gold) gold = child_gold;
						// 角
						const u32 child_bishop = child_pp.exists<HBishop>();
						if (child_bishop > bishop) bishop = child_bishop;
						// 飛車
						const u32 child_rook = child_pp.exists<HRook>();
						if (child_rook > rook) rook = child_rook;
					}
					else
						all_mate = false;
				}
				if (child_entry.dn == 0) {
					// 不詰みの場合
					entry.pn = kInfinitePnDn;
					entry.dn = 0;
					// 子局面の反証駒を設定
					// 打つ手ならば、反証駒から削除する
					if (move.move.isDrop()) {
						const HandPiece hp = move.move.handPieceDropped();
						if (entry.hand.numOf(hp) < child_entry.hand.numOf(hp)) {
							entry.hand = child_entry.hand;
							entry.hand.minusOne(hp);
						}
					}
					// 先手の駒を取る手ならば、反証駒に追加する
					else {
						const Piece to_pc = n.piece(move.move.to());
						if (to_pc != Empty) {
							const PieceType pt = pieceToPieceType(to_pc);
							const HandPiece hp = pieceTypeToHandPiece(pt);
							if (entry.hand.numOf(hp) > child_entry.hand.numOf(hp)) {
								entry.hand = child_entry.hand;
								entry.hand.plusOne(hp);
							}
						}
					}
					break;
				}
				entry.pn += child_entry.pn;
				entry.dn = std::min(entry.dn, child_entry.dn);

				if (child_entry.dn < best_dn ||
					child_entry.dn == best_dn && best_num_search > child_entry.num_searched) {
					second_best_dn = best_dn;
					best_dn = child_entry.dn;
					best_pn = child_entry.pn;
					best_move = move;
				}
				else if (child_entry.dn < second_best_dn) {
					second_best_dn = child_entry.dn;
				}
			}
			entry.pn = std::min(entry.pn, kInfinitePnDn);
			if (entry.pn == 0) {
				// 詰みの場合
				//cout << n.toSFEN() << " and" << endl;
				//cout << bitset<32>(entry.hand.value()) << endl;
				// 証明駒に子局面の証明駒の和集合を設定
				u32 curr_pawn = entry.hand.template exists<HPawn>(); if (pawn > curr_pawn) pawn = curr_pawn;
				u32 curr_lance = entry.hand.template exists<HLance>(); if (lance > curr_lance) lance = curr_lance;
				u32 curr_knight = entry.hand.template exists<HKnight>(); if (knight > curr_knight) knight = curr_knight;
				u32 curr_silver = entry.hand.template exists<HSilver>(); if (silver > curr_silver) silver = curr_silver;
				u32 curr_gold = entry.hand.template exists<HGold>(); if (gold > curr_gold) gold = curr_gold;
				u32 curr_bishop = entry.hand.template exists<HBishop>(); if (bishop > curr_bishop) bishop = curr_bishop;
				u32 curr_rook = entry.hand.template exists<HRook>(); if (rook > curr_rook) rook = curr_rook;
				entry.hand.set(pawn | lance | knight | silver | gold | bishop | rook);
				//cout << bitset<32>(entry.hand.value()) << endl;
				// 後手が一枚も持っていない種類の先手の持ち駒を証明駒に設定する
				if (!(n.checkersBB() & n.attacksFrom<King>(n.kingSquare(n.turn())) || n.checkersBB() & n.attacksFrom<Knight>(n.turn(), n.kingSquare(n.turn()))))
					entry.hand.setPP(n.hand(oppositeColor(n.turn())), n.hand(n.turn()));
				//cout << bitset<32>(entry.hand.value()) << endl;
			}
			else {
				thpn_child = std::min(thpn - entry.pn + best_pn, kInfinitePnDn);
				thdn_child = std::min(thdn, second_best_dn + 1);
			}
		}

		// if (pn(n) >= thpn || dn(n) >= thdn)
		//   break; // termination condition is satisfied
		if (entry.pn >= thpn || entry.dn >= thdn) {
			break;
		}

		StateInfo state_info;
		//cout << n.toSFEN() << "," << best_move.toUSI() << endl;
		n.doMove(best_move, state_info);
		++searchedNode;
		dfpn_inner<!or_node>(n, thpn_child, thdn_child/*, inc_flag*/, depth + 1, searchedNode);
		n.undoMove(best_move);
	}
}

// 詰みの手返す
Move DfPn::dfpn_move(Position& pos) {
	MovePicker<true> move_picker(pos);
	Move mate1ply = pos.mateMoveIn1Ply();
	if (mate1ply || move_picker.empty()) {
		if (mate1ply) {
			return mate1ply;
		}
	}

	for (const auto& move : move_picker) {
		const auto& child_entry = transposition_table.LookUpChildEntry<true>(pos, move, 0);
		if (child_entry.pn == 0) {
			return move;
		}
	}

	return Move::moveNull();
}

void DfPn::init()
{
	transposition_table.Resize(HASH_SIZE_MB);
}

// 詰将棋探索のエントリポイント
bool DfPn::dfpn(Position& r) {
	// 自玉に王手がかかっていないこと

	// キャッシュの世代を進める
	transposition_table.NewSearch();

	searchedNode = 0;
	dfpn_inner<true>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, 0, searchedNode);
	const auto& entry = transposition_table.LookUp<true>(r, 0);

	//cout << searchedNode << endl;

	/*std::vector<Move> moves;
	std::unordered_set<Key> visited;
	dfs(true, r, moves, visited);
	for (Move& move : moves)
	cout << move.toUSI() << " ";
	cout << endl;*/

	return entry.pn == 0;
}

// 詰将棋探索のエントリポイント
bool DfPn::dfpn_andnode(Position& r) {
	// 自玉に王手がかかっていること

	// キャッシュの世代を進める
	transposition_table.NewSearch();

	searchedNode = 0;
	dfpn_inner<false>(r, kInfinitePnDn, kInfinitePnDn/*, false*/, 0, searchedNode);
	const auto& entry = transposition_table.LookUp<false>(r, 0);

	return entry.pn == 0;
}