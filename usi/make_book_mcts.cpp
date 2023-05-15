#ifdef MAKE_BOOK
#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"

#include "cppshogi.h"
#include "fastmath.h"
#include "UctSearch.h"
#include "Message.h"
#include "dfpn.h"
#include "make_book.h"
#include "USIBookEngine.h"


constexpr float VALUE_NAN = -FLT_MAX;
constexpr u32 VALUE_EVALED = 0x4000000;

extern float beta;

int book_mcts_playouts = 10000000;
int book_mcts_threads = 32;
bool book_mcts_debug = false;

constexpr uint64_t MUTEX_NUM = 65536; // must be 2^n
std::mutex book_mutexes[MUTEX_NUM];
inline std::mutex& GetPositionMutex(const Position& pos)
{
	return book_mutexes[pos.getKey() & (MUTEX_NUM - 1)];
}

inline Score value_to_score(const float value) {
	return (Score)int(-logf(1.0f / value - 1.0f) * 756.0f);
}

struct book_child_node_t {
	book_child_node_t() : move(moveNone()), move_count(0), win(0.0), prob(0.0f), value(VALUE_NAN) {}
	book_child_node_t(const Move move)
		: move(move), move_count(0), win(0.0), prob(0.0f), value(VALUE_NAN) {}
	// ���[�u�R���X�g���N�^
	book_child_node_t(book_child_node_t&& o) noexcept
		: move(o.move), move_count((int)o.move_count), win((double)o.win), prob(o.prob), value(o.value) {}
	// ���[�u������Z�q
	book_child_node_t& operator=(book_child_node_t&& o) noexcept {
		move = o.move;
		value = o.value;
		move_count = (int)o.move_count;
		win = (double)o.win;
		prob = (float)o.prob;
		return *this;
	}

	// �������ߖ�̂��߁Amove�̍ŏ�ʃo�C�g�ŕ]���ς݂̏�Ԃ�\��
	bool IsEvaled() const { return move.value() & VALUE_EVALED; }
	void SetEvaled() { move |= Move(VALUE_EVALED); }

	Move move;
	float prob; // ���O�m��
	float value; // ���l
	std::atomic<int> move_count;
	std::atomic<double> win;
};

struct book_uct_node_t {
	book_uct_node_t()
		: move_count(NOT_EXPANDED), win(0), child_num(0) {}

	// �q�m�[�h�쐬
	book_uct_node_t* CreateChildNode(int i) {
		return (child_nodes[i] = std::make_unique<book_uct_node_t>()).get();
	}
	// �q�m�[�h��݂̂ŏ���������
	void CreateSingleChildNode(const Move move) {
		child_num = 1;
		child = std::make_unique<book_child_node_t[]>(1);
		child[0].move = move;
	}
	// ����̓W�J
	bool ExpandNode(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, book_child_node_t* parent) {
		const Key key = Book::bookKey(pos);
		const auto itr = outMap.find(key);
		if (itr == outMap.end()) {
			__debugbreak();
			return false;
		}

		const auto& entries = itr->second;
		Score trusted_score = entries[0].score;
		struct ChildEntry {
			Move move;
			Score score;
			float value;
			bool is_evaled;
			float prob;
		};
		const auto softmax_temperature_with_normalize = [](std::vector<ChildEntry>& child_entries) {
			const auto child_num = child_entries.size();
			float max = 0.0f;
			for (int i = 0; i < child_num; i++) {
				float& x = child_entries[i].prob;
				x *= beta / 756.0f;
				if (x > max) {
					max = x;
				}
			}
			// �I�[�o�[�t���[��h�~���邽�ߍő�l�ň���
			float sum = 0.0f;
			for (int i = 0; i < child_num; i++) {
				float& x = child_entries[i].prob;
				x = expf(x - max);
				sum += x;
			}
			// normalize
			for (int i = 0; i < child_num; i++) {
				float& x = child_entries[i].prob;
				x /= sum;
			}
		};

		std::vector<ChildEntry> child_entries;
		child_entries.reserve(entries.size());
		for (const auto& entry : entries) {
			const Move move = move16toMove(Move(entry.fromToPro), pos);
			Score score;
			float value;
			bool is_evaled;
			// �K��񐔂����Ȃ��]���l�͐M�����Ȃ�
			if (entry.score < trusted_score)
				trusted_score = entry.score;
			// �����
			StateInfo state;
			pos.doMove(move, state);
			switch (pos.isDraw()) {
			case RepetitionDraw:
				score = pos.turn() == Black ? draw_score_white : draw_score_black;
				value = score_to_value(score);
				is_evaled = true;
				break;
			case RepetitionWin:
				score = -ScoreMaxEvaluate;
				value = 0.0f;
				is_evaled = true;
				break;
			case RepetitionLose:
				score = ScoreMaxEvaluate;
				value = 1.0f;
				is_evaled = true;
				break;
			default:
			{
				const auto itr_next = outMap.find(Book::bookKey(pos));
				if (itr_next == outMap.end()) {
					score = trusted_score;
					is_evaled = true;
				}
				else {
					// 1���̕]���l���g�p����
					score = -itr_next->second[0].score;
					is_evaled = false;
				}
				value = score_to_value(score);
			}
			}
			pos.undoMove(move);
			child_entries.emplace_back() = { move, score, value, is_evaled, (float)score };
		}
		for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
			const Move& move = ml.move();
			const u16 move16 = (u16)(move.value());
			if (std::any_of(entries.begin(), entries.end(), [move16](const BookEntry& entry) { return entry.fromToPro == move16; }))
				continue;
			const auto itr_next = outMap.find(Book::bookKeyAfter(pos, key, move));
			if (itr_next == outMap.end())
				continue;
			Score score;
			float value;
			bool is_evaled;
			// �����
			StateInfo state;
			pos.doMove(move, state);
			switch (pos.isDraw()) {
			case RepetitionDraw:
				score = pos.turn() == Black ? draw_score_white : draw_score_black;
				value = score_to_value(score);
				is_evaled = true;
				break;
			case RepetitionWin:
				score = -ScoreMaxEvaluate;
				value = 0.0f;
				is_evaled = true;
				break;
			case RepetitionLose:
				score = ScoreMaxEvaluate;
				value = 1.0f;
				is_evaled = true;
				break;
			default:
				score = -itr_next->second[0].score;
				value = score_to_value(score);
				is_evaled = false;
			}
			pos.undoMove(move);
			child_entries.emplace_back() = { move, score, value, is_evaled, (float)score };
		}
		softmax_temperature_with_normalize(child_entries);
		//for (auto entry : child_entries) std::cout << entry.prob << std::endl;
		child = std::make_unique<book_child_node_t[]>(child_entries.size());
		auto* child_node = child.get();
		bool is_all_evaled = true;
		float max_value = -FLT_MAX;
		for (auto& entry : child_entries) {
			child_node->move = entry.move;
			child_node->prob = entry.prob;
			child_node->value = entry.value;
			if (entry.is_evaled) {
				max_value = std::max(max_value, entry.value);
				child_node->SetEvaled();
			}
			else {
				is_all_evaled = false;
			}
			++child_node;
		}
		if (is_all_evaled) {
			parent->value = 1.0f - max_value;
			parent->SetEvaled();
		}
		child_num = (short)child_entries.size();
		move_count = 0;
		return true;
	}
	// �q�m�[�h�ւ̃|�C���^�z��̏�����
	void InitChildNodes() {
		child_nodes = std::make_unique<std::unique_ptr<book_uct_node_t>[]>(child_num);
	}
	// 1���������ׂĂ̎q���폜����
	// 1��������Ȃ��ꍇ�A�V�����m�[�h���쐬����
	// �c�����m�[�h��Ԃ�
	book_uct_node_t* ReleaseChildrenExceptOne(const Move move);

	std::atomic<int> move_count;
	std::atomic<double> win;
	short child_num;
	std::unique_ptr<book_child_node_t[]> child;
	std::unique_ptr<std::unique_ptr<book_uct_node_t>[]> child_nodes;
};

NodeGarbageCollector<book_uct_node_t> gNodeGc;

book_uct_node_t* book_uct_node_t::ReleaseChildrenExceptOne(const Move move) {
	if (child_num > 0 && child_nodes) {
		// ����c���č폜����
		bool found = false;
		for (int i = 0; i < child_num; ++i) {
			auto& uct_child = child[i];
			auto& child_node = child_nodes[i];
			if (uct_child.move == move) {
				found = true;
				if (!child_node) {
					// �V�����m�[�h���쐬����
					child_node = std::make_unique<book_uct_node_t>();
				}
				// 0�Ԗڂ̗v�f�Ɉړ�����
				if (i != 0) {
					child[0] = std::move(uct_child);
					child_nodes[0] = std::move(child_node);
				}
			}
			else {
				// �q�m�[�h���폜�i�K�x�[�W�R���N�^�ɒǉ��j
				if (child_node)
					gNodeGc.AddToGcQueue(std::move(child_node));
			}
		}

		if (found) {
			// �q�m�[�h����ɂ���
			child_num = 1;
			return child_nodes[0].get();
		}
		else {
			// ���@��ɕs���𐶐����Ă��Ȃ����߁A�m�[�h�����݂��Ă�������Ȃ��ꍇ������
			// �q�m�[�h��������Ȃ������ꍇ�A�V�����m�[�h���쐬����
			CreateSingleChildNode(move);
			InitChildNodes();
			return (child_nodes[0] = std::make_unique<book_uct_node_t>()).get();
		}
	}
	else {
		// �q�m�[�h���W�J�A�܂��͎q�m�[�h�ւ̃|�C���^�z�񂪖��������̏ꍇ
		CreateSingleChildNode(move);
		// �q�m�[�h�ւ̃|�C���^�z�������������
		InitChildNodes();
		return (child_nodes[0] = std::make_unique<book_uct_node_t>()).get();
	}
}


class BookNodeTree {
public:
	~BookNodeTree() { DeallocateTree(); }
	// �c���[���̈ʒu��ݒ肵�A�c���[�̍ė��p�����݂�
	// �V�����ʒu���Â��ʒu�Ɠ����Q�[���ł��邩�ǂ�����Ԃ��i�������̒��蓮���ǉ�����Ă���j
	// �ʒu�����S�ɈقȂ�ꍇ�A�܂��͈ȑO�����Z���ꍇ�́Afalse��Ԃ�
	bool ResetToPosition(const std::vector<Move>& moves) {
		if (!gamebegin_node_) {
			gamebegin_node_ = std::make_unique<book_uct_node_t>();
			current_head_ = gamebegin_node_.get();
		}

		book_uct_node_t* old_head = current_head_;
		book_uct_node_t* prev_head = nullptr;
		current_head_ = gamebegin_node_.get();
		bool seen_old_head = (gamebegin_node_.get() == old_head);
		for (const auto& move : moves) {
			prev_head = current_head_;
			// current_head_�ɒ����ǉ�����
			current_head_ = current_head_->ReleaseChildrenExceptOne(move);
			if (old_head == current_head_) seen_old_head = true;
		}

		// MakeMove�͌Z�킪���݂��Ȃ����Ƃ�ۏ؂��� 
		// �������A�Â��w�b�h������Ȃ��ꍇ�́A�ȑO�Ɍ������ꂽ�ʒu�̑c��ł���ʒu������\�������邱�Ƃ��Ӗ�����
		// �܂�A�Â��q���ȑO�Ƀg���~���O����Ă��Ă��Acurrent_head_�͌Â��f�[�^��ێ�����\��������
		// ���̏ꍇ�Acurrent_head_�����Z�b�g����K�v������
		if (!seen_old_head && current_head_ != old_head) {
			if (prev_head) {
				assert(prev_head->child_num == 1);
				auto& prev_uct_child_node = prev_head->child_nodes[0];
				gNodeGc.AddToGcQueue(std::move(prev_uct_child_node));
				prev_uct_child_node = std::make_unique<book_uct_node_t>();
				current_head_ = prev_uct_child_node.get();
			}
			else {
				// �J�n�ǖʂɖ߂����ꍇ
				DeallocateTree();
			}
		}
		return seen_old_head;
	}
	book_uct_node_t* GetCurrentHead() const { return current_head_; }
	void DeallocateTree() {
		// gamebegin_node_.reset�i�j�Ɠ��������A���ۂ̊��蓖�ĉ�����GC�X���b�h�ōs����
		gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
		gamebegin_node_ = std::make_unique<book_uct_node_t>();
		current_head_ = gamebegin_node_.get();
	}

private:
	// �T�����J�n����m�[�h
	book_uct_node_t* current_head_ = nullptr;
	// �Q�[���؂̃��[�g�m�[�h
	std::unique_ptr<book_uct_node_t> gamebegin_node_;
} book_tree;


int select_max_ucb_child(book_child_node_t* parent, book_uct_node_t* current)
{
	const book_child_node_t* uct_child = current->child.get();
	const int child_num = current->child_num;
	int max_child = 0;
	const int sum = current->move_count;
	const double sum_win = current->win;
	float q, u, max_value;
	int child_evaled_count = 0;
	float child_max_value = -FLT_MAX;

	max_value = -FLT_MAX;

	const float sqrt_sum = sqrtf(static_cast<const float>(sum));
	const float c = FastLog((sum + c_base + 1.0f) / c_base) + c_init;
	const float parent_q = sum_win > 0 ? (float)(sum_win / sum) : 0.0f;
	const float init_u = sum == 0 ? 1.0f : sqrt_sum;

	// UCB�l�ő�̎�����߂�
	for (int i = 0; i < child_num; i++) {
		const double win = uct_child[i].win;
		const int move_count = uct_child[i].move_count;

		if (uct_child[i].IsEvaled()) {
			child_evaled_count++;
			child_max_value = std::max(child_max_value, uct_child[i].value);
			q = uct_child[i].value; // �m�肵���l
			u = sqrt_sum / (1 + move_count);
		}
		else if (move_count == 0) {
			// ���T���̃m�[�h�̉��l�ɁA�e�m�[�h�̉��l���g�p����
			q = parent_q;
			u = init_u;
		}
		else {
			q = (float)(win / move_count);
			u = sqrt_sum / (1 + move_count);
		}

		const float prob = uct_child[i].prob;

		const float ucb_value = q + c * u * prob;

		if (ucb_value > max_value) {
			max_value = ucb_value;
			max_child = i;
		}
	}

	if (child_evaled_count == child_num) {
		// �q�m�[�h�����ׂĕ]���ς݂̂��߁A�e�m�[�h���X�V
		parent->value = 1.0f - child_max_value;
		parent->SetEvaled();
	}

	return max_child;
}


float uct_search(Position& pos, book_child_node_t* parent, book_uct_node_t* current, const std::unordered_map<Key, std::vector<BookEntry> >& outMap) {
	float result;
	book_child_node_t* uct_child = current->child.get();

	// ���݌��Ă���m�[�h�����b�N
	auto& mutex = GetPositionMutex(pos);
	mutex.lock();
	// �q�m�[�h�ւ̃|�C���^�z�񂪏���������Ă��Ȃ��ꍇ�A����������
	if (!current->child_nodes) current->InitChildNodes();
	// UCB�l�ő�̎�����߂�
	const unsigned int next_index = select_max_ucb_child(parent, current);
	// �]���ς�
	if (uct_child[next_index].IsEvaled()) {
		mutex.unlock();
		result = uct_child[next_index].value;

		// �T�����ʂ̍X�V
		atomic_fetch_add(&current->win, (double)result);
		current->move_count++;
		atomic_fetch_add(&uct_child[next_index].win, (double)result);
		uct_child[next_index].move_count++;

		return 1.0f - result;
	}
	// �I�񂾎�𒅎�
	StateInfo st;
	pos.doMove(uct_child[next_index].move, st);

	// Virtual Loss�����Z
	current->move_count += VIRTUAL_LOSS;
	uct_child[next_index].move_count += VIRTUAL_LOSS;
	// �m�[�h�̓W�J�̊m�F
	if (!current->child_nodes[next_index]) {
		result = uct_child[next_index].value;

		const Key key = Book::bookKey(pos);
		const auto itr = outMap.find(key);
		if (itr == outMap.end()) {
			uct_child[next_index].SetEvaled();
		}
		else {
			// �m�[�h�̍쐬
			book_uct_node_t* child_node = current->CreateChildNode(next_index);

			// �����W�J����
			child_node->ExpandNode(pos, outMap, &uct_child[next_index]);
		}
		// ���݌��Ă���m�[�h�̃��b�N������
		mutex.unlock();
	}
	else {
		// ���݌��Ă���m�[�h�̃��b�N������
		mutex.unlock();

		book_uct_node_t* next_node = current->child_nodes[next_index].get();

		// ��Ԃ����ւ���1��[���ǂ�
		result = uct_search(pos, &uct_child[next_index], next_node, outMap);
	}

	// �T�����ʂ̔��f
	atomic_fetch_add(&current->win, (double)result);
	if constexpr (VIRTUAL_LOSS != 1) current->move_count += 1 - VIRTUAL_LOSS;
	atomic_fetch_add(&uct_child[next_index].win, (double)result);
	if constexpr (VIRTUAL_LOSS != 1) uct_child[next_index].move_count += 1 - VIRTUAL_LOSS;

	return 1.0f - result;
}

// �K��񐔂��ő�̎q�m�[�h��I��
int book_select_max_child_node(const book_child_node_t* parent, const book_uct_node_t* uct_node)
{
	const auto child = uct_node->child.get();
	const int child_num = uct_node->child_num;

	if (parent->IsEvaled()) {
		// �q�m�[�h�����ׂĕ]���ς݂̏ꍇ�A���l���ő�̎��I��
		float child_max_value = -FLT_MAX;
		int child_evaled_max_index = 0;

		for (int i = 0; i < child_num; ++i) {
			const float value = child[i].value;
			if (value > child_max_value) {
				child_max_value = value;
				child_evaled_max_index = i;
			}
		}
		return child_evaled_max_index;
	}
	else {
		// �K��񐔂��ő�̎��I��
		int max_move_count = -1;
		int max_index = 0;
		for (int i = 0; i < child_num; ++i) {
			const int move_count = child[i].move_count;
			if (move_count > max_move_count) {
				max_move_count = move_count;
				max_index = i;
			}
		}
		return max_index;
	}
}

std::string get_pv(const book_uct_node_t* root_uct_node, const unsigned int best_root_child_index) {
	auto best_uct_child = &root_uct_node->child[best_root_child_index];
	const Move move = best_uct_child->move;
	std::string pv = move.toUSI();
	const book_uct_node_t* best_node = root_uct_node;
	unsigned int best_index = best_root_child_index;
	while (best_node->child_nodes && best_node->child_nodes[best_index]) {
		best_node = best_node->child_nodes[best_index].get();
		if (!best_node || best_node->child_num == 0)
			break;

		// �ő�̎q�m�[�h
		best_index = book_select_max_child_node(best_uct_child, best_node);
		best_uct_child = &best_node->child[best_index];

		const auto best_move = best_uct_child->move;
		pv += " ";
		pv += best_move.toUSI();
	}
	return pv;
}

void reset_to_position(const std::vector<Move>& moves) {
	book_tree.ResetToPosition(moves);
}

const BookEntry& parallel_uct_search(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves) {
	reset_to_position(moves);
	book_uct_node_t* current_root = book_tree.GetCurrentHead();
	book_child_node_t parent;
	// ���[�g�m�[�h�̓W�J
	if (current_root->child_num == 0) {
		current_root->ExpandNode(pos, outMap, &parent);
	}

	int playout_count = current_root->move_count;
	std::atomic<bool> interruption = false;
	static BookEntry tmp; // entries�ɂȂ��v�f��Ԃ��ꍇ�Astatic�ϐ��Ɋi�[����

	#pragma omp parallel num_threads(book_mcts_threads)
	while (playout_count < book_mcts_playouts && !interruption) {
		// �Ֆʂ̃R�s�[
		Position pos_copy(pos);

		// 1��v���C�A�E�g����
		uct_search(pos_copy, &parent, current_root, outMap);

		#pragma omp atomic
		playout_count++;

		#pragma omp master
		{
			// �ł��؂�̊m�F
			if (parent.IsEvaled()) {
				interruption = true;
			}
			else {
				int max_searched = 0, second_searched = 0;
				int max_index = 0, second_index = 0;

				const book_child_node_t* uct_child = current_root->child.get();
				const int child_num = current_root->child_num;
				for (int i = 0; i < child_num; i++) {
					if (uct_child[i].move_count > max_searched) {
						second_searched = max_searched;
						second_index = max_index;
						max_searched = uct_child[i].move_count;
						max_index = i;
					}
					else if (uct_child[i].move_count > second_searched) {
						second_searched = uct_child[i].move_count;
						second_index = i;
					}
				}
				// �c��̒T���Ŏ��P�肪�őP��𒴂���\�����Ȃ��ꍇ�͑ł��؂�
				const int rest_po = book_mcts_playouts - playout_count;
				if (max_searched - second_searched > rest_po) {
					interruption = true;
				}
			}
		}
	}
	const int selected_index = book_select_max_child_node(&parent, current_root);
	const auto child = current_root->child.get();

	if (book_mcts_debug) {
		std::cout << book_pos_cmd;
		for (Move move : moves) {
			std::cout << " " << move.toUSI();
		}
		std::cout << std::endl;
		std::cout << "playout: " << playout_count << " select: " << selected_index << " value: " << child[selected_index].win / child[selected_index].move_count << " cp: " << value_to_score(child[selected_index].win / child[selected_index].move_count) << " evaled: " << child[selected_index].IsEvaled() << " " << child[selected_index].value << " pv: " << get_pv(current_root, selected_index) << std::endl;
		for (int i = 0; i < current_root->child_num; ++i) {
			std::cout << i << ": " << child[i].move.toUSI() << " count: " << child[i].move_count << " value: " << child[i].win / child[i].move_count << " evaled: " << child[i].IsEvaled() << " " << child[i].value << " prob: " << child[i].prob << std::endl;
		}
	}
		
	if (selected_index < entries.size()) {
		return entries[selected_index];
	}
	else {
		const float value = child[selected_index].value;
		tmp.score = value_to_score(value);
		tmp.fromToPro = (u16)child[selected_index].move.value();
		return tmp;
	}
}

void make_book_mcts(Position& pos, LimitsType& limits, std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves) {
	make_book_inner(pos, limits, bookMap, outMap, count, depth, isBlack, moves, parallel_uct_search);
}

#endif