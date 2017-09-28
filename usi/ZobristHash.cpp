#include <cstdlib>
#include <cmath>
#include <iostream>

#include "ZobristHash.h"
#include "UctSearch.h"

using namespace std;

node_hash_t *node_hash;
static unsigned int used;

unsigned int uct_hash_size = UCT_HASH_SIZE;
unsigned int uct_hash_limit = UCT_HASH_SIZE * 9 / 10;

bool enough_size;

////////////////////////////////////
//  ハッシュテーブルのサイズの設定  //
////////////////////////////////////
void
SetHashSize(const unsigned int new_size)
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


/////////////////////////
//  インデックスの取得  //
/////////////////////////
unsigned int
TransHash(const unsigned long long hash)
{
	return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (uct_hash_size - 1);
}


//////////////////////////////////
//  UCTノードのハッシュの初期化  //
//////////////////////////////////
void
InitializeUctHash(void)
{
	node_hash = (node_hash_t *)malloc(sizeof(node_hash_t) * uct_hash_size);

	if (node_hash == NULL) {
		cerr << "Cannot allocate memory" << endl;
		exit(1);
	}

	used = 0;
	enough_size = true;

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		node_hash[i].flag = false;
		node_hash[i].hash = 0;
		node_hash[i].color = 0;
		node_hash[i].moves = 0;
	}
}


//////////////////////////////////////
//  UCTノードのハッシュ情報のクリア  //
/////////////////////////////////////
void
ClearUctHash(void)
{
	used = 0;
	enough_size = true;

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		node_hash[i].flag = false;
		node_hash[i].hash = 0;
		node_hash[i].color = 0;
		node_hash[i].moves = 0;
	}
}


///////////////////////
//  古いデータの削除  //
///////////////////////
void delete_hash_recursively(Position &pos, const unsigned int index) {
	node_hash[index].flag = true;
	used++;

	child_node_t *child_node = uct_node[index].child;
	for (int i = 0; i < uct_node[index].child_num; i++) {
		StateInfo st;
		pos.doMove(child_node[i].move, st);
		delete_hash_recursively(pos, child_node[i].index);
		pos.undoMove(child_node[i].move);
	}
}

void
DeleteOldHash(const Position* pos)
{
	// 現在の局面をルートとする局面以外を削除する
	used = 0;

	for (unsigned int i = 0; i < uct_hash_size; i++) {
		node_hash[i].flag = false;
	}

	unsigned int root = FindSameHashIndex(pos->getKey(), pos->turn(), pos->gamePly());
	if (root != uct_hash_size) {
		// 盤面のコピー
		Position pos_copy(*pos);
		delete_hash_recursively(pos_copy, root);
	}

	enough_size = true;
}


//////////////////////////////////////
//  未使用のインデックスを探して返す  //
//////////////////////////////////////
unsigned int
SearchEmptyIndex(const unsigned long long hash, const int color, const int moves)
{
	const unsigned int key = TransHash(hash);
	unsigned int i = key;

	do {
		if (!node_hash[i].flag) {
			node_hash[i].flag = true;
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
FindSameHashIndex(const unsigned long long hash, const int color, const int moves)
{
	const unsigned int key = TransHash(hash);
	unsigned int i = key;

	do {
		if (!node_hash[i].flag) {
			return uct_hash_size;
		}
		else if (node_hash[i].hash == hash &&
			node_hash[i].color == color &&
			node_hash[i].moves == moves) {
			return i;
		}
		i++;
		if (i >= uct_hash_size) i = 0;
	} while (i != key);

	return uct_hash_size;
}


bool
CheckRemainingHashSize(void)
{
	return enough_size;
}

// ハッシュ使用率を取得(単位はパーミル(全体を1000とした値))
int GetUctHashUsageRate()
{
	return 1000 * used / uct_hash_size;
}
