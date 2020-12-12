#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

struct uct_node_t {
	uct_node_t()
		: move_count(0), win(0.0f), evaled(false), nnrate(0.0f), value_win(0.0f), visited_nnrate(0.0f), child_num(0) {}

	uct_node_t& operator=(uct_node_t&& o) noexcept {
		move = o.move;
		move_count = o.move_count.load();
		win = o.win.load();
		evaled = o.evaled.load();
		nnrate = o.nnrate;
		value_win = o.value_win.load();
		visited_nnrate = o.visited_nnrate.load();
		child_num = o.child_num;
		child = std::move(o.child);
		return *this;
	}

	// �q�m�[�h��݂̂ŏ���������
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<uct_node_t[]>(1);
		child[0].move = move;
	}
	// ����̓W�J
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child_num = ml.size();
		child = std::make_unique<uct_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
	}

	// 1���������ׂĂ̎q���폜����
	// 1��������Ȃ��ꍇ�A�V�����m�[�h���쐬����
	// �c�����m�[�h��Ԃ�
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	void Lock() {
		mtx.lock();
	}
	void UnLock() {
		mtx.unlock();
	}

	Move move;                    // ���肷����W
	std::atomic<int> move_count;  // �T����
	std::atomic<float> win;       // ���l�̍��v
	std::atomic<bool> evaled;     // �]���ς�
	float nnrate;                 // �|���V�[�l�b�g���[�N�̊m��
	std::atomic<float> value_win; // �o�����[�l�b�g���[�N�̉��l
	std::atomic<float> visited_nnrate;
	int child_num;                       // �q�m�[�h�̐�
	std::unique_ptr<uct_node_t[]> child; // �q�m�[�h

	std::mutex mtx;
};

class NodeTree {
public:
	~NodeTree() { DeallocateTree(); }
	// �c���[���̈ʒu��ݒ肵�A�c���[�̍ė��p�����݂�
	// �V�����ʒu���Â��ʒu�Ɠ����Q�[���ł��邩�ǂ�����Ԃ��i�������̒��蓮���ǉ�����Ă���j
	// �ʒu�����S�ɈقȂ�ꍇ�A�܂��͈ȑO�����Z���ꍇ�́Afalse��Ԃ�
	bool ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves);
	uct_node_t* GetCurrentHead() const { return current_head_; }

private:
	void DeallocateTree();
	// �T�����J�n����m�[�h
	uct_node_t* current_head_ = nullptr;
	// �Q�[���؂̃��[�g�m�[�h
	std::unique_ptr<uct_node_t> gamebegin_node_;
	// �ȑO�̋ǖ�
	Key history_starting_pos_key_;
};
