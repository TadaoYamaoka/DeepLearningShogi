#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

struct uct_node_t {
	uct_node_t()
		: move_count(0), win(0.0f), evaled(false), nnrate(0.0f), value_win(0.0f), visited_nnrate(0.0f), child_num(0) {}

	uct_node_t& operator=(uct_node_t&& o) noexcept {
		move = o.move;
		move_count = o.move_count.load();
		win = o.win.load();
		evaled = o.evaled.load();
		nnrate = o.nnrate;
		value_win = o.value_win.load();
		visited_nnrate = o.visited_nnrate.load();
		child_num = o.child_num;
		child = std::move(o.child);
		return *this;
	}

	// 子ノード一つのみで初期化する
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<uct_node_t[]>(1);
		child[0].move = move;
	}
	// 候補手の展開
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child_num = ml.size();
		child = std::make_unique<uct_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
	}

	// 1つを除くすべての子を削除する
	// 1つも見つからない場合、新しいノードを作成する
	// 残したノードを返す
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	void Lock() {
		mtx.lock();
	}
	void UnLock() {
		mtx.unlock();
	}

	Move move;                    // 着手する座標
	std::atomic<int> move_count;  // 探索回数
	std::atomic<float> win;       // 価値の合計
	std::atomic<bool> evaled;     // 評価済か
	float nnrate;                 // ポリシーネットワークの確率
	std::atomic<float> value_win; // バリューネットワークの価値
	std::atomic<float> visited_nnrate;
	int child_num;                       // 子ノードの数
	std::unique_ptr<uct_node_t[]> child; // 子ノード

	std::mutex mtx;
};

class NodeTree {
public:
	~NodeTree() { DeallocateTree(); }
	// ツリー内の位置を設定し、ツリーの再利用を試みる
	// 新しい位置が古い位置と同じゲームであるかどうかを返す（いくつかの着手動が追加されている）
	// 位置が完全に異なる場合、または以前よりも短い場合は、falseを返す
	bool ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves);
	uct_node_t* GetCurrentHead() const { return current_head_; }

private:
	void DeallocateTree();
	// 探索を開始するノード
	uct_node_t* current_head_ = nullptr;
	// ゲーム木のルートノード
	std::unique_ptr<uct_node_t> gamebegin_node_;
	// 以前の局面
	Key history_starting_pos_key_;
};
