#include "Node.h"

#include <thread>
#include <mutex>

// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
public:
    NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

    // Takes ownership of a subtree, to dispose it in a separate thread when
    // it has time.
    void AddToGcQueue(std::unique_ptr<uct_node_t> node) {
        if (!node) return;
        std::lock_guard<std::mutex> lock(gc_mutex_);
        subtrees_to_gc_.emplace_back(std::move(node));
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
            std::unique_ptr<uct_node_t> node_to_gc;
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
    std::vector<std::unique_ptr<uct_node_t>> subtrees_to_gc_;

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
                    child_node = std::make_unique<uct_node_t>();
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
            return (child_nodes[0] = std::make_unique<uct_node_t>()).get();
        }
    }
    else {
        // �q�m�[�h���W�J�A�܂��͎q�m�[�h�ւ̃|�C���^�z�񂪖��������̏ꍇ
        CreateSingleChildNode(move);
        // �q�m�[�h�ւ̃|�C���^�z�������������
        InitChildNodes();
        return (child_nodes[0] = std::make_unique<uct_node_t>()).get();
    }
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

bool NodeTree::ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves) {
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
            auto& prev_uct_child_node = prev_head->child_nodes[0];
            gNodeGc.AddToGcQueue(std::move(prev_uct_child_node));
            prev_uct_child_node = std::make_unique<uct_node_t>();
            current_head_ = prev_uct_child_node.get();
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
    gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
    gamebegin_node_ = std::make_unique<uct_node_t>();
    current_head_ = gamebegin_node_.get();
}
