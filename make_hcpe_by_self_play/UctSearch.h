#pragma once

#include "move.hpp"

// 候補手の最大数(盤上全体)
constexpr int UCT_CHILD_MAX = 593;

constexpr unsigned int NOT_EXPANDED = -1; // 未展開のノードのインデックス

struct child_node_t {
	Move move;          // 着手する座標
	int move_count;     // 探索回数
	float win;          // 勝った回数
	unsigned int index; // インデックス
};

struct uct_node_t {
	int move_count;
	float win;
	bool evaled;      // 評価済か
	bool draw;        // 千日手の可能性あり
	float value_win;
	float visited_nnrate;
	int child_num;                      // 子ノードの数
	child_node_t child[UCT_CHILD_MAX];  // 子ノードの情報
	Key key;
};
