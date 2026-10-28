#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

#ifdef WIN_TYPE_DOUBLE
typedef double WinType;
#else
typedef float WinType;
#endif

#ifdef LEARN
template <typename T>
using atomic_t = T;
#else
template <typename T>
using atomic_t = std::atomic<T>;
#endif

// 詰み探索で詰みの場合の定数
constexpr u32 VALUE_WIN = 0x1000000;
constexpr u32 VALUE_LOSE = 0x2000000;
// 千日手の場合のvalue_winの定数
constexpr u32 VALUE_DRAW = 0x4000000;

// ノード未展開を表す定数
constexpr int NOT_EXPANDED = -1;

struct uct_node_t;
struct child_node_t {
private:
	// 着手と勝敗フラグを同じ32ビット値に保持する。
	atomic_t<u32> move_;
#ifndef LEARN
	static_assert(std::atomic<u32>::is_always_lock_free, "Move flags must be lock-free");
#endif

	void AddMoveFlags(const u32 bits) {
#ifdef LEARN
		move_ |= bits;
#else
		// フラグは単調に増えるため、設定済みならRMWを繰り返さない。
		if ((GetMove().value() & bits) != bits)
			move_.fetch_or(bits, std::memory_order_relaxed);
#endif
	}

public:
	child_node_t(const Move move = Move::moveNone())
		: move_(move.value()), nnrate(0.0f), move_count(0), win((WinType)0) {}
	// ムーブコンストラクタ
	child_node_t(child_node_t&& o) noexcept
		: move_(o.GetMove().value()), nnrate(o.nnrate), move_count((int)o.move_count), win((WinType)o.win) {}
	// ムーブ代入演算子
	child_node_t& operator=(child_node_t&& o) noexcept {
		SetMove(o.GetMove());
		move_count = (int)o.move_count;
		win = (WinType)o.win;
		nnrate = (float)o.nnrate;
		return *this;
	}

	// 従来どおり、上位ビットの勝敗フラグも含めて返す。
	Move GetMove() const {
#ifdef LEARN
		return Move(move_);
#else
		return Move(move_.load(std::memory_order_relaxed));
#endif
	}
	// 初期化・探索停止中のツリー再利用時のみ、着手全体を設定する。
	void SetMove(const Move move) {
#ifdef LEARN
		move_ = move.value();
#else
		move_.store(move.value(), std::memory_order_relaxed);
#endif
	}
	bool IsWin() const { return GetMove().value() & VALUE_WIN; }
	void SetWin() { AddMoveFlags(VALUE_WIN); }
	bool IsLose() const { return GetMove().value() & VALUE_LOSE; }
	void SetLose() { AddMoveFlags(VALUE_LOSE); }
	bool IsDraw() const { return GetMove().value() & VALUE_DRAW; }
	void SetDraw() { AddMoveFlags(VALUE_DRAW); }

	float nnrate;                // ニューラルネットワークでのレート
	atomic_t<int> move_count; // 探索回数
	atomic_t<WinType> win;    // 勝った回数
};

struct uct_node_t {
	uct_node_t()
		: move_count(NOT_EXPANDED), win(0), visited_nnrate(0.0f), child_num(0) {}

	// 子ノード作成
	uct_node_t* CreateChildNode(int i) {
		return (child_nodes[i] = std::make_unique<uct_node_t>()).get();
	}
	// 子ノード一つのみで初期化する
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<child_node_t[]>(1);
		child[0].SetMove(move);
	}
	// 候補手の展開
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->SetMove(ml.move());
		child_num = (short)ml.size();
	}
	// 子ノードへのポインタ配列の初期化
	void InitChildNodes() {
		child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(child_num);
	}

	// 1つを除くすべての子を削除する
	// 1つも見つからない場合、新しいノードを作成する
	// 残したノードを返す
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	bool IsEvaled() const { return move_count != NOT_EXPANDED; }
	void SetEvaled() { move_count = 0; }

	atomic_t<int> move_count;
	atomic_t<WinType> win;
	atomic_t<float> visited_nnrate;
	short child_num;                       // 子ノードの数
	std::unique_ptr<child_node_t[]> child; // 子ノードの情報
	std::unique_ptr<std::unique_ptr<uct_node_t>[]> child_nodes; // 子ノードへのポインタ配列
};

class NodeTree {
public:
	~NodeTree() = default;
	// ツリー内の位置を設定し、ツリーの再利用を試みる
	// 新しい位置が古い位置と同じゲームであるかどうかを返す（いくつかの着手動が追加されている）
	// 位置が完全に異なる場合、または以前よりも短い場合は、falseを返す
	bool ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves);
	uct_node_t* GetCurrentHead() const { return current_head_; }
	void DeallocateTree();

private:
	// 探索を開始するノード
	uct_node_t* current_head_ = nullptr;
	// ゲーム木のルートノード
	std::unique_ptr<uct_node_t> gamebegin_node_;
	// 以前の局面
	Key history_starting_pos_key_;
};

// Boltzmann distribution
void set_softmax_temperature(const float temperature);
void softmax_temperature_with_normalize(child_node_t* child_node, const int child_num);
