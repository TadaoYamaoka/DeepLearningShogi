#pragma once
#ifdef MAKE_BOOK
#include "position.hpp"
#include "move.hpp"

#include <string>
#include <boost/process.hpp>

struct USIBookResult
{
	std::string bestMove;
	std::string info;

	USIBookResult() {}
	USIBookResult(USIBookResult&& o) noexcept : bestMove(std::move(o.bestMove)), info(std::move(o.info)) {}
	USIBookResult& operator=(USIBookResult&& o) noexcept {
		bestMove = std::move(o.bestMove);
		info = std::move(o.info);
		return *this;
	}
};

class USIBookEngine
{
public:
	USIBookEngine(const std::string path, const std::vector<std::pair<std::string, std::string>>& options, const bool random_nodes);
	USIBookEngine(USIBookEngine&& o) noexcept {} // not use
	~USIBookEngine();
	USIBookResult Go(const std::string& book_pos_cmd, const std::vector<Move>& moves, const int nodes);

private:
	boost::process::opstream ops;
	boost::process::ipstream ips;
	boost::process::child proc;

	bool random_nodes;
	std::mt19937_64 gen;
	std::gamma_distribution<double> gamma1;
	std::gamma_distribution<double> gamma2;

	double generate_beta() {
		const double x = gamma1(gen);
		const double y = gamma2(gen);

		return x / (x + y);
	}
};
#endif
