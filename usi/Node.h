#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

struct uct_node_t;
struct child_node_t {
	child_node_t() : move_count(0), win(0.0f), nnrate(0.0f) {}
	child_node_t(const Move move)
		: move(move), move_count(0), win(0.0f), nnrate(0.0f) {}
	// ���[�u�R���X�g���N�^
	child_node_t(child_node_t&& o) noexcept
		: move(o.move), move_count(0), win(0.0f), nnrate(0.0f), node(std::move(o.node)) {}
	// ���[�u������Z�q
	child_node_t& operator=(child_node_t&& o) noexcept {
		move = o.move;
		move_count = (int)o.move_count;
		win = (float)o.win;
		nnrate = (float)o.nnrate;
		node = std::move(o.node);
		return *this;
	}

	// �m�[�h�̓W�J
	uct_node_t* ExpandNode(const Position* pos);

	Move move;                   // ���肷����W
	std::atomic<int> move_count; // �T����
	std::atomic<float> win;      // ��������
	float nnrate;                // �j���[�����l�b�g���[�N�ł̃��[�g
	std::unique_ptr<uct_node_t> node; // �q�m�[�h�ւ̃|�C���^
};

struct uct_node_t {
	uct_node_t()
		: move_count(0), win(0.0f), evaled(false), value_win(0.0f), visited_nnrate(0.0f), child_num(0) {}
	// ���@��̈ꗗ�ŏ���������
	uct_node_t(MoveList<Legal>& ml)
		: move_count(0), win(0.0f), evaled(false), value_win(0.0f), visited_nnrate(0.0f),
		child_num(ml.size()), child(std::make_unique<child_node_t[]>(ml.size())) {
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
	}

	// �q�m�[�h��݂̂ŏ���������
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<child_node_t[]>(1);
		child[0].move = move;
	}
	// ���@��̈ꗗ�ŏ���������
	void CreateChildNode(MoveList<Legal>& ml) {
		child_num = ml.size();
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
	}

	// 1���������ׂĂ̎q���폜����
	// 1��������Ȃ��ꍇ�A�V�����m�[�h���쐬����
	// �c�����m�[�h��Ԃ�
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	void Lock() {
		while (lock.test_and_set(std::memory_order_acquire))
			;
	}
	void UnLock() {
		lock.clear(std::memory_order_release);
	}

	std::atomic<int> move_count;
	std::atomic<float> win;
	std::atomic<bool> evaled;      // �]���ς�
	std::atomic<float> value_win;
	std::atomic<float> visited_nnrate;
	int child_num;                         // �q�m�[�h�̐�
	std::unique_ptr<child_node_t[]> child; // �q�m�[�h�̏��

	std::atomic_flag lock = ATOMIC_FLAG_INIT;
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
