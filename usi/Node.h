#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"
#include "xoshiro128.h"

class MutexPool {
public:
	MutexPool(uint32_t n) : num(n), mutexes(std::make_unique<std::mutex[]>(n)) {
		if ((n & (n - 1))) {
			std::cerr << "Warning: Mutex pool size must be 2 ^ n" << std::endl;
			// n��2�̙p�łȂ��ꍇ�A�ł���ʂɂ���1�ł���r�b�g�݂̂��c�����l�Ƃ���
			n = n | (n >> 1);
			n = n | (n >> 2);
			n = n | (n >> 4);
			n = n | (n >> 8);
			n = n | (n >> 16);
			num = n ^ (n >> 1);
		}
	}
	uint32_t GetIndex() {
		return rnd.next() & (num - 1);
	}
	std::mutex& operator[] (const uint32_t idx) {
		assert(idx < num);
		return mutexes[idx];
	}

private:
	uint32_t num;
	std::unique_ptr<std::mutex[]> mutexes;
	Xoshiro128 rnd;
};
extern MutexPool mutex_pool;

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

	// �q�m�[�h�쐬
	uct_node_t* CreateChildNode() {
		node = std::make_unique<uct_node_t>();
		return node.get();
	}

	Move move;                   // ���肷����W
	std::atomic<int> move_count; // �T����
	std::atomic<float> win;      // ��������
	float nnrate;                // �j���[�����l�b�g���[�N�ł̃��[�g
	std::unique_ptr<uct_node_t> node; // �q�m�[�h�ւ̃|�C���^
};

struct uct_node_t {
	uct_node_t()
		: move_count(0), win(0.0f), evaled(false), value_win(0.0f), visited_nnrate(0.0f), child_num(0) {}

	// �q�m�[�h��݂̂ŏ���������
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<child_node_t[]>(1);
		child[0].move = move;
	}
	// ����̓W�J
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child_num = ml.size();
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
		mutex_idx = mutex_pool.GetIndex();
	}

	// 1���������ׂĂ̎q���폜����
	// 1��������Ȃ��ꍇ�A�V�����m�[�h���쐬����
	// �c�����m�[�h��Ԃ�
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	void Lock() {
		mutex_pool[mutex_idx].lock();
	}
	void UnLock() {
		mutex_pool[mutex_idx].unlock();
	}

	std::atomic<int> move_count;
	std::atomic<float> win;
	std::atomic<bool> evaled;      // �]���ς�
	std::atomic<float> value_win;
	std::atomic<float> visited_nnrate;
	int child_num;                         // �q�m�[�h�̐�
	std::unique_ptr<child_node_t[]> child; // �q�m�[�h�̏��

	uint32_t mutex_idx;
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
