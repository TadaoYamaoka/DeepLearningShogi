#pragma once

#include "position.hpp"

const unsigned int DFPN_HASH_SIZE = 2097152;

struct node_dfpn_hash_t {
	unsigned long long hash;
	int color;
	int moves;
	bool flag;
};

//  DfPn用ハッシュテーブル
extern node_dfpn_hash_t *node_dfpn_hash;

//  DfPn用ハッシュテーブルのサイズ
extern unsigned int dfpn_hash_size;

//  ハッシュテーブルのサイズの設定
void SetDfPnHashSize(const unsigned int new_size);

//  DfPnノードのハッシュの初期化
void InitializeDfPnHash(void);

//  DfPnノードのハッシュ情報のクリア
void ClearDfPnHash(void);

//  古いデータの削除
void DeleteOldDfPnHash(const Position *pos);

//  未使用のインデックスを探す
unsigned int SearchEmptyDfPnHashIndex(const unsigned long long hash, const int color, const int moves);

//  ハッシュ値に対応するインデックスを返す
unsigned int FindSameDfPnHashIndex(const unsigned long long hash, const int color);

//  ハッシュ表が埋まっていないか確認
bool CheckRemainingDfPnHashSize(void);
