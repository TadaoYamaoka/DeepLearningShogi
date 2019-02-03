#pragma once

#include <atomic>

// �u���\
namespace ns_dfpn {
	struct TTEntry {
		// �n�b�V���̏��32�r�b�g
		uint32_t hash_high;
		Hand hand; // ���i��ɐ��̎��j
		int pn;
		int dn;
		uint16_t depth;
		uint16_t generation;
		int num_searched;
	};

	struct TranspositionTable {
		struct Cluster {
			TTEntry entries[256];
		};

		~TranspositionTable();

		TTEntry& LookUp(const Key key, const Hand hand, const uint16_t depth);

		TTEntry& LookUpDirect(Cluster& entries, const uint32_t hash_high, const Hand hand, const uint16_t depth);

		template <bool or_node>
		TTEntry& LookUp(const Position& n, const uint16_t depth);

		// move���w������̎q�m�[�h�̃L�[��Ԃ�
		template <bool or_node>
		void GetChildFirstEntry(const Position& n, const Move move, Cluster*& entries, uint32_t& hash_high, Hand& hand);

		// move���w������̎q�m�[�h�̒u���\�G���g����Ԃ�
		template <bool or_node>
		TTEntry& LookUpChildEntry(const Position& n, const Move move, const uint16_t depth);

		void Resize();

		void NewSearch();

		void* tt_raw = nullptr;
		Cluster* tt = nullptr;
		int64_t num_clusters = 0;
		int64_t clusters_mask = 0;
		uint16_t generation = 0;
	};

	class DfPn
	{
	public:
		void init();
		bool dfpn(Position& r);
		bool dfpn_andnode(Position& r);
		void dfpn_stop();
		void set_maxdepth(uint32_t depth);
		Move dfpn_move(Position& pos);

	private:
		template <bool or_node>
		void dfpn_inner(Position& n, int thpn, int thdn/*, bool inc_flag*/, uint16_t depth, int64_t& searchedNode);

		TranspositionTable transposition_table;
		uint32_t kMaxDepth = 30;
		int64_t searchedNode = 0;
		std::atomic<bool> stop;
	};
}