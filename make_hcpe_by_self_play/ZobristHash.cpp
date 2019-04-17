#include <cstdlib>
#include <cmath>
#include <iostream>

#include "ZobristHash.h"

using namespace std;

////////////////////////////////////
//  ハッシュテーブルのサイズの設定  //
////////////////////////////////////
void
UctHash::SetHashSize(const unsigned int new_size)
{
	if (!(new_size & (new_size - 1))) {
		uct_hash_size = new_size;
		uct_hash_limit = new_size * 9 / 10;
	}
	else {
		cerr << "Hash size must be 2 ^ n" << endl;
		for (int i = 1; i <= 20; i++) {
			cerr << "2^" << i << ":" << (1 << i) << endl;
		}
		exit(1);
	}

}


//////////////////////////////////
//  UCTノードのハッシュの初期化  //
//////////////////////////////////
UctHash::UctHash(const unsigned int hash_size)
{
	SetHashSize(hash_size);

	node_hash = new node_hash_t[hash_size];

	if (node_hash == nullptr) {
		cerr << "Cannot allocate memory" << endl;
		exit(1);
	}

	used = 0;
	enough_size = true;

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		node_hash[i].id = NOT_USE;
	}
}


//////////////////////////////////////
//  UCTノードのハッシュ情報のクリア  //
/////////////////////////////////////
void
UctHash::ClearUctHash(const int id)
{
	enough_size = true;

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		if (node_hash[i].id == id) {
			node_hash[i].id = NOT_USE;
			used--;
		}
	}
}

///////////////////////
//  古いデータの削除  //
///////////////////////
void
UctHash::delete_hash_recursively(Position &pos, const uct_node_t* uct_node, const unsigned int index, const int id) {
	node_hash[index].id = id;
	used++;

	const child_node_t *child_node = uct_node[index].child;
	for (int i = 0; i < uct_node[index].child_num; i++) {
		if (child_node[i].index != NOT_EXPANDED && node_hash[child_node[i].index].id == NOT_USE) {
			StateInfo st;
			pos.doMove(child_node[i].move, st);
			delete_hash_recursively(pos, uct_node, child_node[i].index, id);
			pos.undoMove(child_node[i].move);
		}
	}
}

void
UctHash::DeleteOldHash(const Position* pos, const uct_node_t* uct_node, const int id)
{
	// 現在の局面をルートとする局面以外を削除する
	unsigned int root = FindSameHashIndex(pos->getKey(), pos->gamePly(), id);

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		if (node_hash[i].id == id) {
			node_hash[i].id = NOT_USE;
			used--;
		}
	}

	if (root != uct_hash_size) {
		// 盤面のコピー
		Position pos_copy(*pos);
		delete_hash_recursively(pos_copy, uct_node, root, id);
	}

	enough_size = true;
}

//////////////////////////////////////
//  未使用のインデックスを探して返す  //
//////////////////////////////////////
unsigned int
UctHash::SearchEmptyIndex(const unsigned long long hash, const Color color, const int moves, const int id)
{
	const unsigned int key = TransHash(hash);
	unsigned int i = key;

	do {
		if (node_hash[i].id == NOT_USE) {
			node_hash[i].id = id;
			node_hash[i].hash = hash;
			node_hash[i].moves = moves;
			node_hash[i].color = color;
			used++;
			if (used > uct_hash_limit)
				enough_size = false;
			return i;
		}
		i++;
		if (i >= uct_hash_size) i = 0;
	} while (i != key);

	return uct_hash_size;
}


////////////////////////////////////////////
//  ハッシュ値に対応するインデックスを返す  //
////////////////////////////////////////////
unsigned int
UctHash::FindSameHashIndex(const unsigned long long hash, const int moves, const int id) const
{
	const unsigned int key = TransHash(hash);
	unsigned int i = key;

	do {
		if (node_hash[i].id == NOT_USE) {
			return uct_hash_size;
		}
		else if (node_hash[i].hash == hash &&
			node_hash[i].moves == moves &&
			node_hash[i].id == id) {
			return i;
		}
		i++;
		if (i >= uct_hash_size) i = 0;
	} while (i != key);

	return uct_hash_size;
}
