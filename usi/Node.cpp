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
        // 一つを残して削除する
        bool found = false;
        for (int i = 0; i < child_num; ++i) {
            auto& uct_child = child[i];
            auto& child_node = child_nodes[i];
            if (uct_child.move == move) {
                found = true;
                if (!child_node) {
                    // 新しいノードを作成する
                    child_node = std::make_unique<uct_node_t>();
                }
                // 0番目の要素に移動する
                if (i != 0) {
                    child[0] = std::move(uct_child);
                    child_nodes[0] = std::move(child_node);
                }
            }
            else {
                // 子ノードを削除（ガベージコレクタに追加）
                if (child_node)
                    gNodeGc.AddToGcQueue(std::move(child_node));
            }
        }

        if (found) {
            // 子ノードを一つにする
            child_num = 1;
            return child_nodes[0].get();
        }
        else {
            // 合法手に不成を生成していないため、ノードが存在しても見つからない場合がある
            // 子ノードが見つからなかった場合、新しいノードを作成する
            CreateSingleChildNode(move);
            InitChildNodes();
            return (child_nodes[0] = std::make_unique<uct_node_t>()).get();
        }
    }
    else {
        // 子ノード未展開、または子ノードへのポインタ配列が未初期化の場合
        CreateSingleChildNode(move);
        // 子ノードへのポインタ配列を初期化する
        InitChildNodes();
        return (child_nodes[0] = std::make_unique<uct_node_t>()).get();
    }
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

bool NodeTree::ResetToPosition(const Key starting_pos_key, const std::vector<Move>& moves) {
    if (gamebegin_node_ && history_starting_pos_key_ != starting_pos_key) {
        // 完全に異なる位置
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
        // current_head_に着手を追加する
        current_head_ = current_head_->ReleaseChildrenExceptOne(move);
        if (old_head == current_head_) seen_old_head = true;
    }

    // MakeMoveは兄弟が存在しないことを保証する 
    // ただし、古いヘッドが現れない場合は、以前に検索された位置の祖先である位置がある可能性があることを意味する
    // つまり、古い子が以前にトリミングされていても、current_head_は古いデータを保持する可能性がある
    // その場合、current_head_をリセットする必要がある
    if (!seen_old_head && current_head_ != old_head) {
        if (prev_head) {
            assert(prev_head->child_num == 1);
            auto& prev_uct_child_node = prev_head->child_nodes[0];
            gNodeGc.AddToGcQueue(std::move(prev_uct_child_node));
            current_head_ = new uct_node_t();
        }
        else {
            // 開始局面に戻った場合
            DeallocateTree();
        }
    }
    return seen_old_head;
}

void NodeTree::DeallocateTree() {
    // gamebegin_node_.reset（）と同じだが、実際の割り当て解除はGCスレッドで行われる
    gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
    gamebegin_node_ = std::make_unique<uct_node_t>();
    current_head_ = gamebegin_node_.get();
}

// Boltzmann distribution
// see: Reinforcement Learning : An Introduction 2.3.SOFTMAX ACTION SELECTION
constexpr float default_softmax_temperature = 1.0f;
float beta = 1.0f / default_softmax_temperature;
void set_softmax_temperature(const float temperature) {
    beta = 1.0f / temperature;
}

void softmax_temperature_with_normalize(child_node_t* child_node, const int child_num) {
    // apply beta exponent to probabilities(in log space)
    float max = 0.0f;
    for (int i = 0; i < child_num; i++) {
        float& x = child_node[i].nnrate;
        x *= beta;
        if (x > max) {
            max = x;
        }
    }
    // オーバーフローを防止するため最大値で引く
    float sum = 0.0f;
    for (int i = 0; i < child_num; i++) {
        float& x = child_node[i].nnrate;
        x = expf(x - max);
        sum += x;
    }
    // normalize
    for (int i = 0; i < child_num; i++) {
        float& x = child_node[i].nnrate;
        x /= sum;
    }
}
