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

// 詰み探索で詰みの場合の定数
//  値に意味はないがDebugMessage表示で勝ちの場合にnnrateが正の値になっている方がよいため、子ノードの勝ちを負の値とする
constexpr float VALUE_WIN = -FLT_MAX;
constexpr float VALUE_LOSE = FLT_MAX;
// 千日手の場合のvalue_winの定数
constexpr float VALUE_DRAW = FLT_MAX / 2;

struct uct_node_t;
struct child_node_t {
	child_node_t() : move_count(0), win(0.0f), nnrate(0.0f) {}
	child_node_t(const Move move)
		: move(move), move_count(0), win(0.0f), nnrate(0.0f) {}
	// ムーブコンストラクタ
	child_node_t(child_node_t&& o) noexcept
		: move(o.move), move_count(0), win(0.0f), nnrate(0.0f) {}
	// ムーブ代入演算子
	child_node_t& operator=(child_node_t&& o) noexcept {
		move = o.move;
		move_count = (int)o.move_count;
		win = (float)o.win;
		nnrate = (float)o.nnrate;
		return *this;
	}

	// メモリ節約のため、nnrateでWin/Lose/Drawの状態を表す
	bool IsWin() const { return nnrate == VALUE_WIN; }
	void SetWin() { nnrate = VALUE_WIN; }
	bool IsLose() const { return nnrate == VALUE_LOSE; }
	void SetLose() { nnrate = VALUE_LOSE; }
	bool IsDraw() const { return nnrate == VALUE_DRAW; }
	void SetDraw() { nnrate = VALUE_DRAW; }

	Move move;                   // 着手する座標
	std::atomic<int> move_count; // 探索回数
	std::atomic<WinType> win;    // 勝った回数
	float nnrate;                // ニューラルネットワークでのレート
};

struct uct_node_t {
	uct_node_t()
		: move_count(-1), win(0), visited_nnrate(0.0f), child_num(0) {}

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
		child_num = (short)ml.size();
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
	}
	// 子ノードへのポインタ配列の初期化
	void InitChildNodes() {
		child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(child_num);
	}

	// 1つを除くすべての子を削除する
	// 1つも見つからない場合、新しいノードを作成する
	// 残したノードを返す
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	bool IsEvaled() const { return move_count != -1; }
	void SetEvaled() { move_count = 0; }

	std::atomic<int> move_count;
	std::atomic<WinType> win;
	std::atomic<float> visited_nnrate;
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

private:
	void DeallocateTree();
	// 探索を開始するノード
	uct_node_t* current_head_ = nullptr;
	// ゲーム木のルートノード
	std::unique_ptr<uct_node_t> gamebegin_node_;
	// 以前の局面
	Key history_starting_pos_key_;
};
