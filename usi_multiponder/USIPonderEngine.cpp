#include "USIPonderEngine.h"

#include "usi.hpp"

#include <exception>
#include <thread>
#include <boost/iostreams/copy.hpp>

USIPonderEngine::USIPonderEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options) :
#ifdef THREAD_POOL
	ready_th(true),
	term_th(false),
#endif
	proc(path, boost::process::std_in < ops, boost::process::std_out > ips, boost::process::start_dir(path.substr(0, path.find_last_of("\\/"))))
{
	th.reset(new std::thread([this, options] { Init(options); }));
}

USIPonderEngine::~USIPonderEngine()
{
	if (th) {
#ifdef THREAD_POOL
		// スレッドを終了
		{
			std::unique_lock<std::mutex> lk(mtx_th);
			term_th = true;
			ready_th = false;
			cond_th.notify_all();
		}
#endif
		if (th->joinable())
			th->join();
	}
	proc.wait();
}

void USIPonderEngine::Init(const std::vector<std::pair<std::string, std::string>>& options)
{
	for (const auto& option : options) {
		const std::string option_line = "setoption name " + option.first + " value " + option.second + "\n";
		ops.write(option_line.c_str(), option_line.size());
	}

	ops << "isready" << std::endl;

	std::string line;
	bool is_ok = false;
	while (proc.running() && std::getline(ips, line)) {
		if (line.substr(0, line.find_last_not_of("\r") + 1) == "readyok") {
			is_ok = true;
			break;
		}
	}
	if (!is_ok)
		throw std::runtime_error("expected readyok");

	ops << "usinewgame" << std::endl;
}

void USIPonderEngine::WaitInit()
{
	th->join();
	th.release();
}

std::ostream& operator<<(std::ostream& os, const Move& move)
{
	os << move.toUSI();
	return os;
}

USIPonderResult USIPonderEngine::GoPonder()
{
	ops << "position " << usi_position << std::endl;

	ops << "go ponder btime " << btime << " wtime " << wtime << " binc " << binc << " winc " << winc;
	if (byoyomi > 0)
		ops << " byoyomi " << byoyomi;
	ops << std::endl;

	std::string line;
	USIPonderResult result;
	bool is_ok = false;
	while (proc.running() && std::getline(ips, line)) {
		if (line.substr(0, 9) == "bestmove ") {
			is_ok = true;
			break;
		}
		result.info = std::move(line);
	}
	if (!is_ok) {
		living = false;
		return std::move(result);
	}

	auto end = line.find_first_of(" \r", 9 + 3);
	if (end == std::string::npos)
		end = line.size();

	result.bestMove = line.substr(9, end - 9);

	return std::move(result);
}

void USIPonderEngine::GoPonderAsync(const std::string& usi_position, const LimitsType& limits)
{
	promise_ponder = std::promise<USIPonderResult>();
	future_ponder = promise_ponder.get_future();
	this->usi_position = std::move(usi_position);
	this->btime = limits.time[Black];
	this->wtime = limits.time[White];
	this->binc = limits.inc[Black];
	this->winc = limits.inc[White];
	this->byoyomi = limits.moveTime;
#ifdef THREAD_POOL
	if (!th) {
		th.reset(new std::thread([&]() {
			while (!term_th) {
				promise_ponder.set_value(GoPonder());

				std::unique_lock<std::mutex> lk(mtx_th);
				ready_th = false;
				cond_th.notify_all();

				// スレッドを停止しないで待機する
				cond_th.wait(lk, [this] { return ready_th || term_th; });
			}
		}));
	}
	else {
		// スレッドを再開する
		std::unique_lock<std::mutex> lk(mtx_th);
		ready_th = true;
		cond_th.notify_all();
	}
#else
	th.reset(new std::thread([this] {
		promise_ponder.set_value(GoPonder());
	}));
#endif
}

void USIPonderEngine::Join() {
#ifdef THREAD_POOL
	std::unique_lock<std::mutex> lk(mtx_th);
	if (ready_th && !term_th)
		cond_th.wait(lk, [this] { return !ready_th || term_th; });
#else
	th->join();
	th.release();
#endif
}

void USIPonderEngine::Stop()
{
	ops << "stop" << std::endl;
}

void USIPonderEngine::Quit()
{
	ops << "quit" << std::endl;
}

USIPonderResult USIPonderEngine::Ponderhit()
{
	ops << "ponderhit" << std::endl;
	return std::move(future_ponder.get());
}

