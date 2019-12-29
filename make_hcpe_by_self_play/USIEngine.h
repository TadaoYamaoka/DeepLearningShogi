#pragma once

#include "position.hpp"
#include "move.hpp"

#include <vector>
#include <utility>
#include <string>
#include <boost/process.hpp>

inline Move moveResign() { return Move(-1); }
inline Move moveWin() { return Move(-2); }

class USIEngine
{
public:
	USIEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options);
	USIEngine(USIEngine&& o) {} // not use
	~USIEngine();
	Move Think(const Position& pos, const std::string& usi_position, const int byoyomi);
	void ThinkAsync(const Position& pos, const std::string& usi_position, const int byoyomi);
	Move ThinkDone() { return moveDone; }

private:
	boost::process::opstream ops;
	boost::process::ipstream ips;
	boost::process::child proc;
	std::thread* t = nullptr;
	Move moveDone;
};
