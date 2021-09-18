#pragma once

#include <atomic>
#include <vector>
#include <memory>

#include "cppshogi.h"

#ifdef WIN_TYPE_DOUBLE
typedef double WinType;
#else
typedef float WinType;
#endif

#ifdef LEARN
template <typename T>
using atomic_t = T;
#else
template <typename T>
using atomic_t = std::atomic<T>;
#endif

// �l�ݒT���ŋl�݂̏ꍇ�̒萔
constexpr u32 VALUE_WIN = 0x1000000;
constexpr u32 VALUE_LOSE = 0x2000000;
// �����̏ꍇ��value_win�̒萔
constexpr u32 VALUE_DRAW = 0x4000000;

// �m�[�h���W�J��\���萔
constexpr int NOT_EXPANDED = -1;

struct uct_node_t;
struct child_node_t {
	child_node_t() : move_count(0), win(0.0f), nnrate(0.0f) {}
	child_node_t(const Move move)
		: move(move), move_count(0), win(0.0f), nnrate(0.0f) {}
	// ���[�u�R���X�g���N�^
	child_node_t(child_node_t&& o) noexcept
		: move(o.move), move_count((int)o.move_count), win((float)o.win), nnrate(o.nnrate) {}
	// ���[�u������Z�q
	child_node_t& operator=(child_node_t&& o) noexcept {
		move = o.move;
		move_count = (int)o.move_count;
		win = (float)o.win;
		nnrate = (float)o.nnrate;
		return *this;
	}

	// �������ߖ�̂��߁Amove�̍ŏ�ʃo�C�g��Win/Lose/Draw�̏�Ԃ�\��
	bool IsWin() const { return move.value() & VALUE_WIN; }
	void SetWin() { move |= Move(VALUE_WIN); }
	bool IsLose() const { return move.value() & VALUE_LOSE; }
	void SetLose() { move |= Move(VALUE_LOSE); }
	bool IsDraw() const { return move.value() & VALUE_DRAW; }
	void SetDraw() { move |= Move(VALUE_DRAW); }

	Move move;                   // ���肷����W
	float nnrate;                // �j���[�����l�b�g���[�N�ł̃��[�g
	atomic_t<int> move_count; // �T����
	atomic_t<WinType> win;    // ��������
};

struct uct_node_t {
	uct_node_t()
		: move_count(NOT_EXPANDED), win(0), visited_nnrate(0.0f), child_num(0) {}

	// �q�m�[�h�쐬
	uct_node_t* CreateChildNode(int i) {
		return (child_nodes[i] = std::make_unique<uct_node_t>()).get();
	}
	// �q�m�[�h��݂̂ŏ���������
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<child_node_t[]>(1);
		child[0].move = move;
	}
	// ����̓W�J
	void ExpandNode(const Position* pos) {
		MoveList<Legal> ml(*pos);
		child = std::make_unique<child_node_t[]>(ml.size());
		auto* child_node = child.get();
		for (; !ml.end(); ++ml) child_node++->move = ml.move();
		child_num = (short)ml.size();
	}
	// �q�m�[�h�ւ̃|�C���^�z��̏�����
	void InitChildNodes() {
		child_nodes = std::make_unique<std::unique_ptr<uct_node_t>[]>(child_num);
	}

	// 1���������ׂĂ̎q���폜����
	// 1��������Ȃ��ꍇ�A�V�����m�[�h���쐬����
	// �c�����m�[�h��Ԃ�
	uct_node_t* ReleaseChildrenExceptOne(const Move move);

	bool IsEvaled() const { return move_count != NOT_EXPANDED; }
	void SetEvaled() { move_count = 0; }

	atomic_t<int> move_count;
	atomic_t<WinType> win;
	atomic_t<float> visited_nnrate;
	short child_num;                       // �q�m�[�h�̐�
	std::unique_ptr<child_node_t[]> child; // �q�m�[�h�̏��
	std::unique_ptr<std::unique_ptr<uct_node_t>[]> child_nodes; // �q�m�[�h�ւ̃|�C���^�z��
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

// Boltzmann distribution
void set_softmax_temperature(const float temperature);
void softmax_temperature_with_normalize(child_node_t* child_node, const int child_num);
