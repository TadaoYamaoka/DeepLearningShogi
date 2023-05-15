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
	child_node_t() : move_count(0), win((WinType)0), nnrate(0.0f) {}
	child_node_t(const Move move)
		: move(move), move_count(0), win((WinType)0), nnrate(0.0f) {}
	// ムーブコンストラクタ
	child_node_t(child_node_t&& o) noexcept
		: move(o.move), move_count((int)o.move_count), win((WinType)o.win), nnrate(o.nnrate) {}
	// ムーブ代入演算子
	child_node_t& operator=(child_node_t&& o) noexcept {
		move = o.move;
		move_count = (int)o.move_count;
		win = (WinType)o.win;
		nnrate = (float)o.nnrate;
		return *this;
	}

	// メモリ節約のため、moveの最上位バイトでWin/Lose/Drawの状態を表す
	bool IsWin() const { return move.value() & VALUE_WIN; }
	void SetWin() { move |= Move(VALUE_WIN); }
	bool IsLose() const { return move.value() & VALUE_LOSE; }
	void SetLose() { move |= Move(VALUE_LOSE); }
	bool IsDraw() const { return move.value() & VALUE_DRAW; }
	void SetDraw() { move |= Move(VALUE_DRAW); }

	Move move;                   // 着手する座標
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
		child[0].move = move;
	}
	// 候補手の展開
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
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
	~NodeTree() { DeallocateTree(); }
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

// Periodicity of garbage collection, milliseconds.
constexpr int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
template <typename T=uct_node_t>
class NodeGarbageCollector {
public:
	NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

	// Takes ownership of a subtree, to dispose it in a separate thread when
	// it has time.
	void AddToGcQueue(std::unique_ptr<T> node) {
		if (!node) return;
		std::lock_guard<std::mutex> lock(gc_mutex_);
		subtrees_to_gc_.emplace_back(std::move(node));
	}

	~NodeGarbageCollector() {
		// Flips stop flag and waits for a worker thread to stop.
		stop_.store(true);
		gc_thread_.join();
	}

private:
	void GarbageCollect() {
		while (!stop_.load()) {
			// Node will be released in destructor when mutex is not locked.
			std::unique_ptr<T> node_to_gc;
			{
				// Lock the mutex and move last subtree from subtrees_to_gc_ into
				// node_to_gc.
				std::lock_guard<std::mutex> lock(gc_mutex_);
				if (subtrees_to_gc_.empty()) return;
				node_to_gc = std::move(subtrees_to_gc_.back());
				subtrees_to_gc_.pop_back();
			}
		}
	}

	void Worker() {
		while (!stop_.load()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
			GarbageCollect();
		};
	}

	mutable std::mutex gc_mutex_;
	std::vector<std::unique_ptr<T>> subtrees_to_gc_;

	// When true, Worker() should stop and exit.
	std::atomic<bool> stop_{ false };
	std::thread gc_thread_;
};

// Boltzmann distribution
void set_softmax_temperature(const float temperature);
void softmax_temperature_with_normalize(child_node_t* child_node, const int child_num);
