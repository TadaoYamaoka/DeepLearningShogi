#pragma once

#include "position.hpp"
#include "move.hpp"
#include "search.hpp"

#include <utility>
#include <string>
#include <thread>
#include <mutex>
#include <future>
#include <atomic>
#include <boost/process.hpp>

struct USIPonderResult
{
	std::string bestMove;
	std::string info;

	USIPonderResult() {}
	USIPonderResult(USIPonderResult&& o) noexcept : bestMove(std::move(o.bestMove)), info(std::move(o.info)) {}
	USIPonderResult& operator=(USIPonderResult&& o) noexcept {
		bestMove = std::move(o.bestMove);
		info = std::move(o.info);
		return *this;
	}
};

class USIPonderEngine
{
public:
	USIPonderEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options);
	USIPonderEngine(USIPonderEngine&& o) noexcept {} // not use
	~USIPonderEngine();
	void WaitInit();
	void GoPonderAsync(const std::string& usi_position, const LimitsType& limits);
	void Join();
	void Stop();
	void Quit();
	USIPonderResult Ponderhit();
	const std::string& GetUsiPosition() const { return usi_position; }
	bool IsLiving() { return living; }

private:
	boost::process::opstream ops;
	boost::process::ipstream ips;
	boost::process::child proc;
	std::unique_ptr<std::thread> th;
	std::promise<USIPonderResult> promise_ponder;
	std::future<USIPonderResult> future_ponder;
	std::string usi_position;
	int btime;
	int wtime;
	int binc;
	int winc;
	int byoyomi;
	std::atomic<bool> living = true;

	void Init(const std::vector<std::pair<std::string, std::string>>& options);
	USIPonderResult GoPonder();
	void WaitStarted();

	// スレッドプール用
	std::mutex mtx_th;
	std::condition_variable cond_th;
	bool ready_th;
	std::atomic<bool> term_th;
	bool started = false;
	bool job_pending = false;
};
