#pragma once

#include "position.hpp"
#include "move.hpp"

#include <utility>
#include <string>
#include <thread>
#include <mutex>
#include <boost/process.hpp>

inline Move moveResign() { return Move(-1); }
inline Move moveWin() { return Move(-2); }
inline Move moveAbort() { return Move(-3); }

struct USIEngineResult {
	Move move;
	int score;

	USIEngineResult() : move(Move::moveNone()), score(0) {}
	USIEngineResult(const Move& move, const int& score) : move(move), score(score) {}
};

class USIEngine
{
public:
	USIEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options, const int num);
	USIEngine(USIEngine&& o) {} // not use
	~USIEngine();
	USIEngineResult Think(const Position& pos, const std::string& usi_position, const int byoyomi);
	void ThinkAsync(const int id, const Position& pos, const std::string& usi_position, const int byoyomi);
	const USIEngineResult& ThinkDone(const int id) const { return results[id]; }
	void WaitThinking() {
		if (t) {
			std::lock_guard<std::mutex> lock(mtx);
		}
	}

private:
	boost::process::opstream ops;
	boost::process::ipstream ips;
	boost::process::child proc;
	std::thread* t = nullptr;
	std::mutex mtx;
	std::vector<USIEngineResult> results;
};
