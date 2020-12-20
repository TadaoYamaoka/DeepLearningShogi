#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

class MutexPool {
public:
	MutexPool(uint32_t n) : num(n), mutexes(std::make_unique<std::mutex[]>(n)) {
		if ((n & (n - 1))) {
			// nが2の冪でない場合、最も上位にある1であるビットのみを残した値とする
			n = n | (n >> 1);
			n = n | (n >> 2);
			n = n | (n >> 4);
			n = n | (n >> 8);
			n = n | (n >> 16);
			num = n ^ (n >> 1);
		}
	}
	uint32_t GetIndex(const uint64_t key) const {
		return ((key & 0xffffffff) ^ ((key >> 32) & 0xffffffff)) & (num - 1);
	}
	std::mutex& operator[] (const uint32_t idx) {
		assert(idx < num);
		return mutexes[idx];
	}

private:
	uint32_t num;
	std::unique_ptr<std::mutex[]> mutexes;
};
extern MutexPool mutex_pool;

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
		mutex_idx = o.mutex_idx;
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
		mutex_idx = mutex_pool.GetIndex(pos->getKey());
	}

	// 1つを除くすべての子を削除する
	// 1つも見つからない場合、新しいノードを作成する
	// 残したノードを返す
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	void Lock() {
		mutex_pool[mutex_idx].lock();
	}
	void UnLock() {
		mutex_pool[mutex_idx].unlock();
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
	uint32_t mutex_idx;
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
