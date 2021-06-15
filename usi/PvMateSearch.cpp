#ifdef PV_MATE_SEARCH
#include "UctSearch.h"
#include "PvMateSearch.h"

// ゲーム木
extern std::unique_ptr<NodeTree> tree;
extern const Position* pos_root;

std::set<child_node_t*> PvMateSearcher::searched;
std::mutex PvMateSearcher::mtx_searched;

void PvMateSearcher::SearchInner(Position& pos, uct_node_t* uct_node)
{
	// 停止
	if (stop) return;

	// 未展開の場合、終了する
	if (!uct_node->IsEvaled() || !uct_node->child) {
		std::this_thread::yield();
		return;
	}

	child_node_t* uct_child = uct_node->child.get();

	// 訪問回数が最大の子ノードを選択
	const auto next_index = select_max_child_node(uct_node);

	// 詰みの場合、終了する
	if (uct_child[next_index].IsWin() || uct_child[next_index].IsLose()) {
		std::this_thread::yield();
		return;
	}

	// 選んだ手を着手
	StateInfo st;
	const auto move = uct_child[next_index].move;
	pos.doMove(move, st);

	// 停止
	if (stop) return;

	// 未探索の場合、詰み探索する
	mtx_searched.lock();
	// pos.getKey()は、経路によって詰みの結果が異なるため、ポインタをキーとする
	if (searched.emplace(&uct_child[next_index]).second) {
		mtx_searched.unlock();
		// 詰みの場合、ノードを更新
		if (dfpn.dfpn(pos)) {
			uct_child[next_index].SetWin();
		}
		else if (stop) {
			// 途中で停止された場合、未探索にする
			std::lock_guard<std::mutex> lock(mtx_searched);
			searched.erase(&uct_child[next_index]);
		}
		// 探索中にPVが変わっている可能性があるため、ルートに戻る
	}
	else {
		mtx_searched.unlock();
		// 展開済みの場合、PV上の次の手へ
		if (uct_node->child_nodes && uct_node->child_nodes[next_index])
			SearchInner(pos, uct_node->child_nodes[next_index].get());
		else
			std::this_thread::yield();
	}
}

void PvMateSearcher::Run()
{
#ifdef THREAD_POOL
	if (th == nullptr) {
		th = new std::thread([&]() {
			while (!term_th) {
				// 停止になるまで繰り返す
				while (!stop) {
					// 盤面のコピー
					Position pos_copy(*pos_root);
					// PV上の詰み探索
					SearchInner(pos_copy, tree->GetCurrentHead());
				}

				std::unique_lock<std::mutex> lk(mtx_th);
				ready_th = false;
				cond_th.notify_all();

				// スレッドを停止しないで待機する
				cond_th.wait(lk, [this] { return ready_th || term_th; });
			}
			});
	}
	else {
		// スレッドを再開する
		std::unique_lock<std::mutex> lk(mtx_th);
		ready_th = true;
		cond_th.notify_all();
	}
#else
	th = new std::thread([&]() {
		// 停止になるまで繰り返す
		while (!stop) {
			// 盤面のコピー
			Position pos_copy(*pos_root);
			// PV上の詰み探索
			SearchInner(pos_copy, tree->GetCurrentHead());
		}
	});
#endif
}

void PvMateSearcher::Stop(const bool stop)
{
	dfpn.dfpn_stop(stop);
	this->stop = stop;
}

void PvMateSearcher::Join()
{
#ifdef THREAD_POOL
	std::unique_lock<std::mutex> lk(mtx_th);
	if (ready_th && !term_th)
		cond_th.wait(lk, [this] { return !ready_th || term_th; });
#else
	th->join();
	delete th;
#endif
}

#ifdef THREAD_POOL
// スレッドを終了
void PvMateSearcher::Term() {
	{
		std::unique_lock<std::mutex> lk(mtx_th);
		term_th = true;
		ready_th = false;
		cond_th.notify_all();
	}
	th->join();
	delete th;
}
#endif
#endif
