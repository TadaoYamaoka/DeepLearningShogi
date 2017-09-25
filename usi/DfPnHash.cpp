#include <cstdlib>
#include <cmath>
#include <iostream>

#include "DfPnHash.h"

using namespace std;

node_dfpn_hash_t *node_dfpn_hash;
static unsigned int used;
static int oldest_move;

unsigned int dfpn_hash_size = DFPN_HASH_SIZE;
unsigned int dfpn_hash_limit = DFPN_HASH_SIZE * 9 / 10;

static bool enough_dfpn_hash_size;

////////////////////////////////////
//  ハッシュテーブルのサイズの設定  //
////////////////////////////////////
void
SetDfPnHashSize(const unsigned int new_size)
{
	if (!(new_size & (new_size - 1))) {
		dfpn_hash_size = new_size;
		dfpn_hash_limit = new_size * 9 / 10;
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
TransDfPnHash(const unsigned long long hash)
{
	return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (dfpn_hash_size - 1);
}


//////////////////////////////////
//  DfPnノードのハッシュの初期化  //
//////////////////////////////////
void
InitializeDfPnHash(void)
{
	node_dfpn_hash = (node_dfpn_hash_t *)malloc(sizeof(node_dfpn_hash_t) * dfpn_hash_size);

	if (node_dfpn_hash == NULL) {
		cerr << "Cannot allocate memory" << endl;
		exit(1);
	}

	oldest_move = 1;
	used = 0;
	enough_dfpn_hash_size = true;

	for (unsigned int i = 0; i < dfpn_hash_size; i++) {
		node_dfpn_hash[i].flag = false;
		node_dfpn_hash[i].hash = 0;
		node_dfpn_hash[i].color = 0;
		node_dfpn_hash[i].moves = 0;
	}
}


//////////////////////////////////////
//  DfPnノードのハッシュ情報のクリア  //
/////////////////////////////////////
void
ClearDfPnHash(void)
{
	used = 0;
	enough_dfpn_hash_size = true;

	for (unsigned int i = 0; i < dfpn_hash_size; i++) {
		node_dfpn_hash[i].flag = false;
		node_dfpn_hash[i].hash = 0;
		node_dfpn_hash[i].color = 0;
		node_dfpn_hash[i].moves = 0;
	}
}


///////////////////////
//  古いデータの削除  //
///////////////////////
void
DeleteOldDfPnHash(const Position* pos)
{
	while (oldest_move < pos->gamePly()) {
		for (unsigned int i = 0; i < dfpn_hash_size; i++) {
			if (node_dfpn_hash[i].flag && node_dfpn_hash[i].moves == oldest_move) {
				node_dfpn_hash[i].flag = false;
				node_dfpn_hash[i].hash = 0;
				node_dfpn_hash[i].color = 0;
				node_dfpn_hash[i].moves = 0;
				used--;
			}
		}
		oldest_move++;
	}

	enough_dfpn_hash_size = true;
}


//////////////////////////////////////
//  未使用のインデックスを探して返す  //
//////////////////////////////////////
unsigned int
SearchEmptyDfPnHashIndex(const unsigned long long hash, const int color, const int moves)
{
	const unsigned int key = TransDfPnHash(hash);
	unsigned int i = key;

	do {
		if (!node_dfpn_hash[i].flag) {
			node_dfpn_hash[i].flag = true;
			node_dfpn_hash[i].hash = hash;
			node_dfpn_hash[i].color = color;
			node_dfpn_hash[i].moves = moves;
			used++;
			if (used > dfpn_hash_limit)
				enough_dfpn_hash_size = false;
			return i;
		}
		i++;
		if (i >= dfpn_hash_size) i = 0;
	} while (i != key);

	return dfpn_hash_size;
}


////////////////////////////////////////////
//  ハッシュ値に対応するインデックスを返す  //
////////////////////////////////////////////
unsigned int
FindSameDfPnHashIndex(const unsigned long long hash, const int color)
{
	const unsigned int key = TransDfPnHash(hash);
	unsigned int i = key;

	do {
		if (!node_dfpn_hash[i].flag) {
			return dfpn_hash_size;
		}
		else if (node_dfpn_hash[i].hash == hash &&
			node_dfpn_hash[i].color == color) {
			return i;
		}
		i++;
		if (i >= dfpn_hash_size) i = 0;
	} while (i != key);

	return dfpn_hash_size;
}


bool
CheckRemainingDfPnHashSize(void)
{
	return enough_dfpn_hash_size;
}

