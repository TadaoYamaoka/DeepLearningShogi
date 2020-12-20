#include "Node.h"

#include <thread>
#include <mutex>

MutexPool mutex_pool(1048576);

// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
public:
    NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

    // Takes ownership of a subtree, to dispose it in a separate thread when
    // it has time.
    void AddToGcQueue(uct_node_t* node) {
        if (!node->child) return;
        std::lock_guard<std::mutex> lock(gc_mutex_);
        subtrees_to_gc_.emplace_back(std::move(node->child));
    }

    ~NodeGarbageCollector() {
        // Flips stop flag and waits for a worker thread to stop.
        stop_.store(true);
        gc_thread_.join();
    }

private:
    void GarbageCollect() {
        while (!stop_.load()) {
            // Node will be released in destructor when mutex is not locked.
            std::unique_ptr<uct_node_t[]> node_to_gc;
            {
                // Lock the mutex and move last subtree from subtrees_to_gc_ into
                // node_to_gc.
                std::lock_guard<std::mutex> lock(gc_mutex_);
                if (subtrees_to_gc_.empty()) return;
                node_to_gc = std::move(subtrees_to_gc_.back());
                subtrees_to_gc_.pop_back();
            }
        }
    }

    void Worker() {
        while (!stop_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
            GarbageCollect();
        };
    }

    mutable std::mutex gc_mutex_;
    std::vector<std::unique_ptr<uct_node_t[]>> subtrees_to_gc_;

    // When true, Worker() should stop and exit.
    std::atomic<bool> stop_{ false };
    std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;


/////////////////////////////////////////////////////////////////////////
// uct_node_t
/////////////////////////////////////////////////////////////////////////

uct_node_t* uct_node_t::ReleaseChildrenExceptOne(const Move move)
{
    // ����c���č폜����
    bool found = false;
    for (int i = 0; i < child_num; ++i) {
        auto& uct_child = child[i];
        if (uct_child.move == move) {
            found = true;
            // 0�Ԗڂ̗v�f�Ɉړ�����
            if (i != 0)
                child[0] = std::move(uct_child);
        }
        else {
            // �q�m�[�h���폜�i�K�x�[�W�R���N�^�ɒǉ��j
            gNodeGc.AddToGcQueue(&uct_child);
        }
    }

    if (found) {
        // �q�m�[�h����ɂ���
        child_num = 1;
        return &child[0];
    }
    else {
        // �q�m�[�h��������Ȃ������ꍇ�A�V�����m�[�h���쐬����
        CreateSingleChildNode(move);
        return &child[0];
    }
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

bool NodeTree::ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves) {
    int no_capture_ply;
    int full_moves;
    if (gamebegin_node_ && history_starting_pos_key_ != starting_pos_key) {
        // ���S�ɈقȂ�ʒu
        DeallocateTree();
    }

    if (!gamebegin_node_) {
        gamebegin_node_ = std::make_unique<uct_node_t>();
        current_head_ = gamebegin_node_.get();
    }

    history_starting_pos_key_ = starting_pos_key;

    uct_node_t* old_head = current_head_;
    uct_node_t* prev_head = nullptr;
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
            uct_node_t* prev_uct_child = &prev_head->child[0];
            gNodeGc.AddToGcQueue(prev_uct_child);
            current_head_ = prev_uct_child;
            current_head_->move_count = 0;
            current_head_->win = 0;
            current_head_->evaled = false;
            current_head_->visited_nnrate = 0;
            current_head_->child_num = 0;
        }
        else {
            // �J�n�ǖʂɖ߂����ꍇ
            DeallocateTree();
        }
    }
    return seen_old_head;
}

void NodeTree::DeallocateTree() {
    // gamebegin_node_.reset�i�j�Ɠ��������A���ۂ̊��蓖�ĉ�����GC�X���b�h�ōs����
    gNodeGc.AddToGcQueue(gamebegin_node_.get());
    gamebegin_node_ = std::make_unique<uct_node_t>();
    current_head_ = gamebegin_node_.get();
}
