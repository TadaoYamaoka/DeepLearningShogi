#ifdef MAKE_BOOK
#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"

#include "cppshogi.h"
#include "fastmath.h"
#include "UctSearch.h"
#include "Message.h"
#include "dfpn.h"
#include "make_book.h"
#include "USIBookEngine.h"
#include <numeric>

constexpr u32 VALUE_EVALED = 0x4000000;

// 特定局面の評価値を置き換える
extern std::map<Key, Score> book_key_eval_map;

struct book_iddfs_child_node_t {
	book_iddfs_child_node_t() : move(moveNone()), score(ScoreZero) {}
	book_iddfs_child_node_t(const Move move)
		: move(move), score(ScoreZero) {}
	// ムーブコンストラクタ
	book_iddfs_child_node_t(book_iddfs_child_node_t&& o) noexcept
		: move(o.move), score(o.score) {}
	// ムーブ代入演算子
	book_iddfs_child_node_t& operator=(book_iddfs_child_node_t&& o) noexcept {
		move = o.move;
		score = o.score;
		return *this;
	}

	// メモリ節約のため、moveの最上位バイトで評価済みの状態を表す
	bool IsEvaled() const { return move.value() & VALUE_EVALED; }
	void SetEvaled() { move |= Move(VALUE_EVALED); }

	Move move;
	Score score; // 評価値
};

struct book_iddfs_node_t {
	book_iddfs_node_t()
		: move_count(NOT_EXPANDED), win(0), child_num(0) {}

	// 子ノード作成
	book_iddfs_node_t* CreateChildNode(int i) {
		return (child_nodes[i] = std::make_unique<book_iddfs_node_t>()).get();
	}
	// 子ノード一つのみで初期化する
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<book_iddfs_child_node_t[]>(1);
		child[0].move = move;
	}
	// 候補手の展開
	bool ExpandNode(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, book_iddfs_child_node_t* parent) {
		const Key key = Book::bookKey(pos);
		const auto itr = outMap.find(key);
		if (itr == outMap.end()) {
			return false;
		}

		const auto& entries = itr->second;
		Score trusted_score = entries[0].score;
		struct ChildEntry {
			Move move;
			Score score;
			bool is_evaled;
		};

		std::vector<ChildEntry> child_entries;
		child_entries.reserve(entries.size());
		for (const auto& entry : entries) {
			const Move move = move16toMove(Move(entry.fromToPro), pos);
			Score score;
			bool is_evaled;
			// 訪問回数が少ない評価値は信頼しない
			if (entry.score < trusted_score)
				trusted_score = entry.score;
			StateInfo state;
			pos.doMove(move, state);
			const Key key = Book::bookKey(pos);
			// 特定局面の評価値を置き換える
			if (book_key_eval_map.size() > 0 && book_key_eval_map.find(key) != book_key_eval_map.end()) {
				score = -book_key_eval_map[key];
				is_evaled = true;
			}
			else {
				// 千日手
				switch (pos.isDraw()) {
				case RepetitionDraw:
					score = pos.turn() == Black ? draw_score_white : draw_score_black;
					is_evaled = true;
					break;
				case RepetitionWin:
					score = -ScoreMaxEvaluate;
					is_evaled = true;
					break;
				case RepetitionLose:
					score = ScoreMaxEvaluate;
					is_evaled = true;
					break;
				default:
				{
					const auto itr_next = outMap.find(key);
					if (itr_next == outMap.end()) {
						score = trusted_score;
						is_evaled = true;
					}
					else {
						// 1手先の評価値を使用する
						score = -itr_next->second[0].score;
						is_evaled = false;
					}
				}
				}
			}
			pos.undoMove(move);
			child_entries.emplace_back() = { move, score, is_evaled };
		}
		for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
			const Move& move = ml.move();
			const u16 move16 = (u16)(move.value());
			if (std::any_of(entries.begin(), entries.end(), [move16](const BookEntry& entry) { return entry.fromToPro == move16; }))
				continue;
			const auto itr_next = outMap.find(Book::bookKeyAfter(pos, key, move));
			if (itr_next == outMap.end())
				continue;
			Score score;
			bool is_evaled;
			StateInfo state;
			pos.doMove(move, state);
			// 特定局面の評価値を置き換える
			if (book_key_eval_map.size() > 0 && book_key_eval_map.find(key) != book_key_eval_map.end()) {
				score = -book_key_eval_map[key];
				is_evaled = true;
			}
			else {
				// 千日手
				switch (pos.isDraw()) {
				case RepetitionDraw:
					score = pos.turn() == Black ? draw_score_white : draw_score_black;
					is_evaled = true;
					break;
				case RepetitionWin:
					score = -ScoreMaxEvaluate;
					is_evaled = true;
					break;
				case RepetitionLose:
					score = ScoreMaxEvaluate;
					is_evaled = true;
					break;
				default:
					score = -itr_next->second[0].score;
					is_evaled = false;
				}
			}
			pos.undoMove(move);
			child_entries.emplace_back() = { move, score, is_evaled };
		}
		child = std::make_unique<book_iddfs_child_node_t[]>(child_entries.size());
		auto* child_node = child.get();
		bool is_all_evaled = true;
		Score max_score = -ScoreInfinite;
		for (auto& entry : child_entries) {
			child_node->move = entry.move;
			child_node->score = entry.score;
			if (entry.is_evaled) {
				max_score = std::max(max_score, entry.score);
				child_node->SetEvaled();
			}
			else {
				is_all_evaled = false;
			}
			++child_node;
		}
		if (is_all_evaled) {
			parent->score = -max_score;
			parent->SetEvaled();
		}
		child_num = (short)child_entries.size();
		move_count = 0;
		return true;
	}
	// 子ノードへのポインタ配列の初期化
	void InitChildNodes() {
		child_nodes = std::make_unique<std::unique_ptr<book_iddfs_node_t>[]>(child_num);
	}
	// 1つを除くすべての子を削除する
	// 1つも見つからない場合、新しいノードを作成する
	// 残したノードを返す
	book_iddfs_node_t* ReleaseChildrenExceptOne(const Move move);

	std::atomic<int> move_count;
	std::atomic<double> win;
	short child_num;
	std::unique_ptr<book_iddfs_child_node_t[]> child;
	std::unique_ptr<std::unique_ptr<book_iddfs_node_t>[]> child_nodes;
};

NodeGarbageCollector<book_iddfs_node_t> gBookNodeGc;

book_iddfs_node_t* book_iddfs_node_t::ReleaseChildrenExceptOne(const Move move) {
	if (child_num > 0 && child_nodes) {
		// 一つを残して削除する
		bool found = false;
		for (int i = 0; i < child_num; ++i) {
			auto& uct_child = child[i];
			auto& child_node = child_nodes[i];
			if (uct_child.move == move) {
				found = true;
				if (!child_node) {
					// 新しいノードを作成する
					child_node = std::make_unique<book_iddfs_node_t>();
				}
				// 0番目の要素に移動する
				if (i != 0) {
					child[0] = std::move(uct_child);
					child_nodes[0] = std::move(child_node);
				}
			}
			else {
				// 子ノードを削除（ガベージコレクタに追加）
				if (child_node)
					gBookNodeGc.AddToGcQueue(std::move(child_node));
			}
		}

		if (found) {
			// 子ノードを一つにする
			child_num = 1;
			return child_nodes[0].get();
		}
		else {
			// 合法手に不成を生成していないため、ノードが存在しても見つからない場合がある
			// 子ノードが見つからなかった場合、新しいノードを作成する
			CreateSingleChildNode(move);
			InitChildNodes();
			return (child_nodes[0] = std::make_unique<book_iddfs_node_t>()).get();
		}
	}
	else {
		// 子ノード未展開、または子ノードへのポインタ配列が未初期化の場合
		CreateSingleChildNode(move);
		// 子ノードへのポインタ配列を初期化する
		InitChildNodes();
		return (child_nodes[0] = std::make_unique<book_iddfs_node_t>()).get();
	}
}


class BookNodeTree {
public:
	~BookNodeTree() { DeallocateTree(); }
	// ツリー内の位置を設定し、ツリーの再利用を試みる
	// 新しい位置が古い位置と同じゲームであるかどうかを返す（いくつかの着手動が追加されている）
	// 位置が完全に異なる場合、または以前よりも短い場合は、falseを返す
	bool ResetToPosition(const std::vector<Move>& moves) {
		if (!gamebegin_node_) {
			gamebegin_node_ = std::make_unique<book_iddfs_node_t>();
			current_head_ = gamebegin_node_.get();
		}

		book_iddfs_node_t* old_head = current_head_;
		book_iddfs_node_t* prev_head = nullptr;
		current_head_ = gamebegin_node_.get();
		bool seen_old_head = (gamebegin_node_.get() == old_head);
		for (const auto& move : moves) {
			prev_head = current_head_;
			// current_head_に着手を追加する
			current_head_ = current_head_->ReleaseChildrenExceptOne(move);
			if (old_head == current_head_) seen_old_head = true;
		}

		// MakeMoveは兄弟が存在しないことを保証する 
		// ただし、古いヘッドが現れない場合は、以前に検索された位置の祖先である位置がある可能性があることを意味する
		// つまり、古い子が以前にトリミングされていても、current_head_は古いデータを保持する可能性がある
		// その場合、current_head_をリセットする必要がある
		if (!seen_old_head && current_head_ != old_head) {
			if (prev_head) {
				assert(prev_head->child_num == 1);
				auto& prev_uct_child_node = prev_head->child_nodes[0];
				gBookNodeGc.AddToGcQueue(std::move(prev_uct_child_node));
				prev_uct_child_node = std::make_unique<book_iddfs_node_t>();
				current_head_ = prev_uct_child_node.get();
			}
			else {
				// 開始局面に戻った場合
				DeallocateTree();
			}
		}
		return seen_old_head;
	}
	book_iddfs_node_t* GetCurrentHead() const { return current_head_; }
	void DeallocateTree() {
		// gamebegin_node_.reset（）と同じだが、実際の割り当て解除はGCスレッドで行われる
		gBookNodeGc.AddToGcQueue(std::move(gamebegin_node_));
		gamebegin_node_ = std::make_unique<book_iddfs_node_t>();
		current_head_ = gamebegin_node_.get();
	}

private:
	// 探索を開始するノード
	book_iddfs_node_t* current_head_ = nullptr;
	// ゲーム木のルートノード
	std::unique_ptr<book_iddfs_node_t> gamebegin_node_;
} book_tree_iddfs;



std::tuple<Score, bool, bool> book_search(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, Score alpha, const Score beta, const bool beta_is_evaled, const Score score, book_iddfs_child_node_t& parent, book_iddfs_node_t* current, const int depth, const int depth_limit) {
	const Key key = Book::bookKey(pos);

	// 特定局面の評価値を置き換える
	if (book_key_eval_map.size() > 0 && book_key_eval_map.find(key) != book_key_eval_map.end()) {
		//std::cout << pos.toSFEN() << std::endl;
		return { -book_key_eval_map[key], true, true };
	}

	const auto itr = outMap.find(key);
	if (itr == outMap.end()) {
		// エントリがない場合、自身の評価値を返す
		return { score, true, true };
	}

	// 深さ制限
	if (depth == depth_limit) {
		return { score, false, false };
	}

	// ルートノードの展開
	current->ExpandNode(pos, outMap, &parent);

	book_iddfs_child_node_t* child = current->child.get();
	const int child_num = current->child_num;

	// 評価済みチェック
	if (parent.IsEvaled()) {
		auto best_child = std::max_element(child, child + child_num, [child](const book_iddfs_child_node_t& l, const book_iddfs_child_node_t& r) {
			return l.score < r.score;
			});
		return { -best_child->score, true, true };
	}

	// 子ノードへのポインタ配列が初期化されていない場合、初期化する
	if (!current->child_nodes) current->InitChildNodes();

	// ソート
	std::vector<size_t> indices(child_num);
	std::iota(indices.begin(), indices.end(), 0);
	std::stable_sort(indices.begin(), indices.end(), [child](const size_t l, const size_t r) {
		return child[l].score > child[r].score;
		});

	bool is_evaled_all = true;
	bool is_true_evaled_all = true;
	bool alpha_is_evaled = beta_is_evaled;
	for (int i = 0; i < child_num; i++) {
		Score value;
		bool is_evaled, is_true_evaled;
		if (child[indices[i]].IsEvaled()) {
			value = child[indices[i]].score;
			is_evaled = true;
			is_true_evaled = true;
		}
		else {
			book_iddfs_node_t* child_node = (current->child_nodes[indices[i]]) ? current->child_nodes[indices[i]].get() : current->CreateChildNode(indices[i]);
			const Move move = child[indices[i]].move;
			StateInfo state;
			pos.doMove(move, state);
			std::tie(value, is_evaled, is_true_evaled) = book_search(pos, outMap, -beta, -alpha, alpha_is_evaled, child[indices[i]].score, child[indices[i]], child_node, depth + 1, depth_limit);
			pos.undoMove(move);
		}
		//std::cout << depth_limit << "\t" << depth << "\t" << child[indices[i]].move.toUSI() << "\t" << value << "\t" << is_evaled << std::endl;

		if (value > alpha) {
			alpha = value;
			alpha_is_evaled = true;
		}
		if (alpha >= beta) {
			parent.score = -alpha;
			if (is_true_evaled && beta_is_evaled) {
				parent.SetEvaled();
			}
			return { -alpha, is_evaled, is_true_evaled && beta_is_evaled };
		}

		is_evaled_all &= is_evaled;
		is_true_evaled_all &= is_true_evaled;
	}

	// βカットされなかった場合
	parent.score = -alpha;
	if (is_true_evaled_all) {
		parent.SetEvaled();
	}
	return { -alpha, is_evaled_all, is_true_evaled_all };
}


std::tuple<int, u16, Score> book_search_iddfs(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves) {
	book_tree_iddfs.ResetToPosition(moves);
	book_iddfs_node_t* current_root = book_tree_iddfs.GetCurrentHead();

	// ルートノードの展開
	book_iddfs_child_node_t parent;
	if (current_root->child_num == 0) {
		current_root->ExpandNode(pos, outMap, &parent);
	}

	const int child_num = current_root->child_num;
	book_iddfs_child_node_t* child = current_root->child.get();

	// 評価済みチェック
	if (parent.IsEvaled()) {
		auto best_child = std::max_element(child, child + child_num, [child](const book_iddfs_child_node_t& l, const book_iddfs_child_node_t& r) {
			return l.score < r.score;
			});
		const int best_index = (int)(best_child - child);
		return { best_index, (u16)best_child->move.value(), best_child->score };
	}

	// 子ノードへのポインタ配列が初期化されていない場合、初期化する
	if (!current_root->child_nodes) current_root->InitChildNodes();


	const Key key = Book::bookKey(pos);
	int best = 0;
	Score alpha = -ScoreInfinite;

	for (int depth = 1; depth < 256; ++depth) {
		// ソート
		std::vector<size_t> indices(child_num);
		std::iota(indices.begin(), indices.end(), 0);
		std::stable_sort(indices.begin(), indices.end(), [child](const size_t l, const size_t r) {
			return child[l].score > child[r].score;
			});

		best = 0;
		alpha = -ScoreInfinite;
		bool is_evaled_all = true;
		bool alpha_is_evaled = false;

		for (int i = 0; i < child_num; i++) {
			Score value;
			bool is_evaled;
			bool is_true_evaled;
			if (child[indices[i]].IsEvaled()) {
				value = child[indices[i]].score;
				is_evaled = true;
				is_true_evaled = true;
			}
			else {
				book_iddfs_node_t* child_node = (current_root->child_nodes[indices[i]]) ? current_root->child_nodes[indices[i]].get() : current_root->CreateChildNode(indices[i]);
				const Move move = child[indices[i]].move;
				StateInfo state;
				pos.doMove(move, state);
				std::tie(value, is_evaled, is_true_evaled) = book_search(pos, outMap, -ScoreInfinite, -alpha, alpha_is_evaled, child[indices[i]].score, child[indices[i]], child_node, 0, depth);
				pos.undoMove(move);
			}
			//std::cout << depth << "\t" << -1 << "\t" << child[indices[i]].move.toUSI() << "\t" << value << "\t" << is_evaled << std::endl;

			if (value > alpha) {
				best = indices[i];
				alpha = value;
				alpha_is_evaled = is_evaled;
			}
			is_evaled_all &= is_evaled;
		}

		if (is_evaled_all) break;
	}
	return { best, (u16)child[best].move.value(), alpha};
}

void make_book_iddfs(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves) {
	make_book_inner(pos, limits, bookMap, outMap, count, depth, isBlack, moves, book_search_iddfs);
}

#endif