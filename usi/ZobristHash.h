#pragma once

#include "position.hpp"

constexpr unsigned int NOT_FOUND = -1;

struct node_hash_t {
	unsigned long long hash;
	Color color;
	int moves;
	bool flag;
};

class UctHash
{
public:
	UctHash() {};
	UctHash(const unsigned int hash_size) { Init(hash_size); }
	UctHash(UctHash&& o) :
		uct_hash_size(o.uct_hash_size),
		uct_hash_limit(o.uct_hash_limit),
		node_hash(o.node_hash),
		used(o.used),
		enough_size(o.enough_size) {
		o.node_hash = nullptr;
	}
	~UctHash() {
		delete[] node_hash;
	}

	void Init(const unsigned int hash_size);

	// UCTノードのハッシュ情報のクリア
	void ClearUctHash();

	// 古いデータの削除
	void DeleteOldHash(const Position* pos);
	void DeleteBeforHash(const Position *pos);

	// 未使用のインデックスを探す
	unsigned int SearchEmptyIndex(const unsigned long long hash, const Color color, const int moves);

	// ハッシュ値に対応するインデックスを返す
	unsigned int FindSameHashIndex(const unsigned long long hash, const int moves) const;

	//  ハッシュ表が埋まっていないか確認
	bool CheckRemainingHashSize(void) const { return enough_size; }

	// ハッシュ使用率を取得(単位はパーミル(全体を1000とした値))
	int GetUctHashUsageRate() const { return (int)(1000.0 * used / uct_hash_size); }

	// 残りのハッシュ数を取得
	int GetRestUctHash() const { return uct_hash_limit - used; }

	// ノードを返す
	const node_hash_t& operator [](const size_t i) { return node_hash[i]; }

private:
	//  UCT用ハッシュテーブルのサイズ
	unsigned int uct_hash_size;
	unsigned int uct_hash_limit;

	//  UCT用ハッシュテーブル
	node_hash_t* node_hash;
	unsigned int used;
	bool enough_size;

	//  ハッシュテーブルのサイズの設定
	void SetHashSize(const unsigned int new_size);

	//  インデックスの取得
	unsigned int TransHash(const unsigned long long hash) const {
		return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (uct_hash_size - 1);
	}

	// 古いハッシュの削除
	void delete_hash_recursively(const unsigned int index);
};
