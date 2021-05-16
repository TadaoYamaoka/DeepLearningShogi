#pragma once

#include "position.hpp"
#include "Node.h"
#include "dfpn.h"

class PvMateSearcher
{
public:
	static void Clear() {
		searched.clear();
	}

	PvMateSearcher(const int depth, const int nodes) :
#ifdef THREAD_POOL
		ready_th(true),
		term_th(false),
#endif
		th(nullptr)
	{
		dfpn.init();
		dfpn.set_maxdepth(depth);
		dfpn.set_max_search_node(nodes);
	}
	PvMateSearcher(PvMateSearcher&& o) noexcept : th(o.th) {} // 未使用
	void Run();
	void Stop();
	void Join();
#ifdef THREAD_POOL
	void Term();
#endif

private:
	void SearchInner(Position& pos, uct_node_t* uct_node);

	// 探索済みノード
	static std::set<child_node_t*> searched;
	static std::mutex mtx_searched;

	std::thread* th;
	DfPn dfpn;
	std::atomic<bool> stop;

#ifdef THREAD_POOL
	// スレッドプール用
	std::mutex mtx_th;
	std::condition_variable cond_th;
	bool ready_th;
	bool term_th;
#endif
};