#include "USIEngine.h"

#include "usi.hpp"

#include <exception>
#include <thread>
#include <boost/iostreams/copy.hpp>

USIEngine::USIEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options) :
	proc(path, boost::process::std_in < ops, boost::process::std_out > ips, boost::process::start_dir(path.substr(0, path.find_last_of("\\/"))))
{
	for (const auto& option : options) {
		const std::string option_line = "setoption name " + option.first + " value " + option.second + "\n";
		ops.write(option_line.c_str(), option_line.size());
	}

	ops << "isready" << std::endl;

	std::string line;
	std::getline(ips, line);
	if (line.substr(0, line.find_last_not_of("\r") + 1) != "readyok")
		throw std::runtime_error("expected readyok");

	ops << "usinewgame" << std::endl;
}

USIEngine::~USIEngine()
{
	if (t) {
		t->join();
		delete t;
	}
	ops << "quit" << std::endl;
	proc.wait();
}

std::ostream& operator<<(std::ostream& os, const Move& move)
{
	os << move.toUSI();
	return os;
}

Move USIEngine::Think(const Position& pos, const std::string& usi_position, const int byoyomi)
{
	ops << usi_position << std::endl;

	ops << "go btime 0 wtime 0 byoyomi " << byoyomi << std::endl;

	std::string line;
	bool is_ok = false;
	while (proc.running() && std::getline(ips, line)) {
		if (line.substr(0, 9) == "bestmove ") {
			is_ok = true;
			break;
		}
	}
	if (!is_ok)
		throw std::runtime_error("expected bestmove");

	auto end = line.find_first_of(" \r", 9 + 3);
	if (end == std::string::npos)
		end = line.size();

	const auto moveStr = line.substr(9, end - 9);
	if (moveStr == "resign")
		return moveResign();
	if (moveStr == "win")
		return moveWin();
	return usiToMove(pos, moveStr);
}

void USIEngine::ThinkAsync(const Position& pos, const std::string& usi_position, const int byoyomi)
{
	if (t) {
		t->join();
		delete t;
	}
	moveDone = Move::moveNone();
	t = new std::thread([this, &pos, &usi_position, byoyomi]() {
		moveDone = this->Think(pos, usi_position, byoyomi);
	});
}
