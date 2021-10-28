#pragma once

#include <atomic>

// 置換表
namespace ns_dfpn {
	struct TTEntry {
		// ハッシュの上位32ビット
		uint32_t hash_high;
		Hand hand; // 手駒（常に先手の手駒）
		int pn;
		int dn;
		uint16_t depth;
		uint16_t generation;
		uint32_t num_searched;
	};

	struct TranspositionTable {
		struct Cluster {
			TTEntry entries[256];
		};

		~TranspositionTable();

		TTEntry& LookUp(const Key key, const Hand hand, const uint16_t depth);

		TTEntry& LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth);

		template <bool or_node>
		TTEntry& LookUp(const Position& n);

		// moveを指した後の子ノードのキーを返す
		template <bool or_node>
		void GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand);

		// moveを指した後の子ノードの置換表エントリを返す
		template <bool or_node>
		TTEntry& LookUpChildEntry(const Position& n, const Move move);

		void Resize(int64_t hash_size_mb);

		void NewSearch();

		void* tt_raw = nullptr;
		Cluster* tt = nullptr;
		int64_t num_clusters = 0;
		int64_t clusters_mask = 0;
		uint16_t generation = 0;
	};
}

class DfPn
{
public:
	void init();
	bool dfpn(Position& r);
	bool dfpn_andnode(Position& r);
	void dfpn_stop(const bool stop);
	Move dfpn_move(Position& pos);
	std::tuple<std::string, int, Move> get_pv(Position& pos);

	static void set_hashsize(const uint64_t size) {
		HASH_SIZE_MB = size;
	}
	static void set_draw_ply(const int draw_ply) {
		// WCSCのルールでは、最大手数で詰ました場合は勝ちになるため+1する
		DfPn::draw_ply = draw_ply + 1;
	}
	void set_maxdepth(const int depth) {
		kMaxDepth = depth;
	}
	void set_max_search_node(const int64_t max_search_node) {
		maxSearchNode = max_search_node;
	}

	int64_t searchedNode = 0;
private:
	template <bool or_node>
	void dfpn_inner(Position& n, const int thpn, const int thdn/*, bool inc_flag*/, const uint16_t maxDepth, int64_t& searchedNode);
	template<bool or_node>
	int get_pv_inner(Position& pos, std::vector<Move>& pv);

	ns_dfpn::TranspositionTable transposition_table;
	std::atomic<bool> stop;
	int64_t maxSearchNode = 2097152;

	int kMaxDepth = 31;
	static int64_t HASH_SIZE_MB;
	static int draw_ply;
};
