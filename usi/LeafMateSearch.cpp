#include "LeafMateSearch.h"
#include "mate.h"

#include <deque>

extern int draw_ply;
extern bool debug_message;

struct LeafMateRequest
{
	std::unique_ptr<Position> pos;
	StateListPtr states;
	child_node_t* uct_child;

	LeafMateRequest(std::unique_ptr<Position>& pos, StateListPtr& states, child_node_t* uct_child) noexcept :
		pos(std::move(pos)), states(std::move(states)), uct_child(uct_child) {}
	LeafMateRequest(LeafMateRequest&& o) noexcept :
		pos(std::move(o.pos)), states(std::move(o.states)), uct_child(o.uct_child) {}
};

std::deque<LeafMateRequest> leaf_mate_requests;
std::mutex leaf_mate_mtx;
std::atomic<bool> stop;

void QueuingLeafMateRequest(std::unique_ptr<Position>& pos, StateListPtr& states, child_node_t* uct_child)
{
	std::lock_guard<std::mutex> lock(leaf_mate_mtx);
	leaf_mate_requests.push_back({ pos, states, uct_child });
}

class LeafMateSearcher
{
public:
	LeafMateSearcher() :
#ifdef THREAD_POOL
		ready_th(true),
		term_th(false),
#endif
		th(nullptr)
	{
	}
	LeafMateSearcher(LeafMateSearcher&& o) noexcept : th(o.th) {} // 未使用
	void Run();
	void Join();
#ifdef THREAD_POOL
	void Term();
#endif

private:
	void SearchOne();

	std::thread* th;

#ifdef THREAD_POOL
	// スレッドプール用
	std::mutex mtx_th;
	std::condition_variable cond_th;
	bool ready_th;
	bool term_th;
#endif
};

void LeafMateSearcher::Run()
{
#ifdef THREAD_POOL
	if (th == nullptr) {
		th = new std::thread([&]() {
			while (!term_th) {
				// 停止になるまで繰り返す
				while (!stop) {
					SearchOne();
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
			SearchOne();
		}
		});
#endif
}

void LeafMateSearcher::SearchOne()
{
	leaf_mate_mtx.lock();
	if (leaf_mate_requests.empty()) {
		leaf_mate_mtx.unlock();
		return;
	}
	auto req = std::move(leaf_mate_requests.front());
	leaf_mate_requests.pop_front();
	leaf_mate_mtx.unlock();

	// 詰みチェック
	if (!req.pos->inCheck()) {
		if (mateMoveInOddPly<MATE_SEARCH_DEPTH, false>(*req.pos.get(), draw_ply)) {
			req.uct_child->SetWin();
		}
	}
	else {
		if (mateMoveInOddPly<MATE_SEARCH_DEPTH, true>(*req.pos.get(), draw_ply)) {
			req.uct_child->SetWin();
		}
	}
}

void LeafMateSearcher::Join()
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
void LeafMateSearcher::Term() {
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

std::vector<LeafMateSearcher> searchers;

void InitLeafMateSearch(const int threads)
{
	searchers.resize(threads);
}

void RunLeafMateSearch()
{
	stop = false;
	for (auto& searcher : searchers) {
		searcher.Run();
	}
}

void JoinLeafMateSearch()
{
	for (auto& searcher : searchers) {
		searcher.Join();
	}
	// 未完了の状態にする
	for (int i = 0; i < leaf_mate_requests.size(); i++)
	{
		leaf_mate_requests[i].uct_child->SetQueuing();
	}
	if (debug_message)
		std::cout << "leaf_mate_requests remaining size: " << leaf_mate_requests.size() << std::endl;
	leaf_mate_requests.clear();
}

void StopLeafMateSearch()
{
	stop = true;
}
