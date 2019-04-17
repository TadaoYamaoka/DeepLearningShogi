#pragma once

#include "move.hpp"

// ����̍ő吔(�Տ�S��)
constexpr int UCT_CHILD_MAX = 593;

constexpr unsigned int NOT_EXPANDED = -1; // ���W�J�̃m�[�h�̃C���f�b�N�X

struct child_node_t {
	Move move;          // ���肷����W
	int move_count;     // �T����
	float win;          // ��������
	unsigned int index; // �C���f�b�N�X
	float nnrate;       // �j���[�����l�b�g���[�N�ł̃��[�g
};

struct uct_node_t {
	int move_count;
	float win;
	bool evaled;      // �]���ς�
	bool draw;        // �����̉\������
	float value_win;
	int child_num;                      // �q�m�[�h�̐�
	child_node_t child[UCT_CHILD_MAX];  // �q�m�[�h�̏��
};
