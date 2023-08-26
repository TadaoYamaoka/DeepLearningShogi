#ifdef MAKE_BOOK
#include "USIBookEngine.h"

#include "usi.hpp"

#include <exception>
#include <boost/iostreams/copy.hpp>

USIBookEngine::USIBookEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options, const bool random_nodes) :
	proc(path, boost::process::std_in < ops, boost::process::std_out > ips, boost::process::start_dir(path.substr(0, path.find_last_of("\\/")))),
	random_nodes(random_nodes), gen(std::chrono::system_clock::now().time_since_epoch().count()), gamma1(0.5, 1.0), gamma2(9.0, 1.0)
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

USIBookEngine::~USIBookEngine()
{
	ops << "quit" << std::endl;
	proc.wait();
}

USIBookResult USIBookEngine::Go(const std::string& book_pos_cmd, const std::vector<Move>& moves, const int nodes)
{
	const int nodes_ = random_nodes ? (int)((generate_beta() * 9.5 + 0.5) * nodes) : nodes;
	ops << book_pos_cmd;
	for (Move move : moves) {
		ops << " " << move.toUSI();
	}
	ops << std::endl;

	ops << "go nodes " << nodes_ << std::endl;

	std::string line;
	USIBookResult result;
	bool is_ok = false;
	while (proc.running() && std::getline(ips, line)) {
		if (line.substr(0, 9) == "bestmove ") {
			is_ok = true;
			break;
		}
		result.info = std::move(line);
	}
	if (!is_ok) {
		throw std::runtime_error("expected bestmove");
	}

	auto end = line.find_first_of(" \r", 9 + 3);
	if (end == std::string::npos)
		end = line.size();

	result.bestMove = line.substr(9, end - 9);

	return std::move(result);
}
#endif