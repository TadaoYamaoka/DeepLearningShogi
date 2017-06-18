#if defined POLICY_ONLY
// 方策のみ
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python.hpp>

namespace py = boost::python;

int main()
{
	Py_Initialize();

	py::object dlshogi_ns = py::import("dlshogi.usi").attr("__dict__");

	py::object dlshogi_usi_main = dlshogi_ns["main"];
	dlshogi_usi_main();

	Py_Finalize();

	return 0;
}
#else
// 探索あり
#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "movePicker.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "tt.hpp"

#include "UctSearch.h"

struct MySearcher : Searcher {
	STATIC void doUSICommandLoop(int argc, char* argv[]);
};
void go_uct(Position& pos, std::istringstream& ssCmd);

int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	auto s = std::unique_ptr<MySearcher>(new MySearcher);

	// 各種初期化
	InitializeUctSearch();
	InitializeSearchSetting();
	InitializeUctHash();

	s->init();
	s->doUSICommandLoop(argc, argv);
	s->threads.exit();
}

void MySearcher::doUSICommandLoop(int argc, char* argv[]) {
	bool evalTableIsRead = false;
	Position pos(DefaultStartPositionSFEN, threads.main(), thisptr);

	std::string cmd;
	std::string token;

	for (int i = 1; i < argc; ++i)
		cmd += std::string(argv[i]) + " ";

	do {
		if (argc == 1 && !std::getline(std::cin, cmd))
			cmd = "quit";

		std::istringstream ssCmd(cmd);

		ssCmd >> std::skipws >> token;

		if (token == "quit" || token == "stop" || token == "ponderhit" || token == "gameover") {
		}
		else if (token == "go") go_uct(pos, ssCmd);
		else if (token == "position") setPosition(pos, ssCmd);
		else if (token == "usinewgame"); // isready で準備は出来たので、対局開始時に特にする事はない。
		else if (token == "usi") std::cout << "id name " << std::string(options["Engine_Name"])
			<< "\nid author Tadao Yamaoka"
			<< "\n" << options
			<< "\nusiok" << std::endl;
		else if (token == "isready") { // 対局開始前の準備。
			// 詰みの探索にしか使用しないためオプション上書き
			/*const std::string overwrite_options[] = {
				"name Eval_Dir value H:\\src\\elmo_for_learn\\bin\\20161007", // 暫定
				"name Threads value 1",
				"name MultiPV value 1",
				"name Max_Random_Score_Diff value 0" };
			for (auto& str : overwrite_options) {
				std::istringstream is(str);
				setOption(is);
			}

			tt.clear();
			threads.main()->previousScore = ScoreInfinite;
			if (!evalTableIsRead) {
				// 一時オブジェクトを生成して Evaluator::init() を呼んだ直後にオブジェクトを破棄する。
				// 評価関数の次元下げをしたデータを格納する分のメモリが無駄な為、
				std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
				evalTableIsRead = true;
			}*/

			std::cout << "readyok" << std::endl;
		}
		else if (token == "setoption") setOption(ssCmd);
	} while (token != "quit" && argc == 1);

	threads.main()->waitForSearchFinished();
}

void go_uct(Position& pos, std::istringstream& ssCmd) {
	LimitsType limits;
	std::string token;

	limits.startTime.restart();

	while (ssCmd >> token) {
		if (token == "ponder") limits.ponder = true;
		else if (token == "btime") ssCmd >> limits.time[Black];
		else if (token == "wtime") ssCmd >> limits.time[White];
		else if (token == "binc") ssCmd >> limits.inc[Black];
		else if (token == "winc") ssCmd >> limits.inc[White];
		else if (token == "infinite") limits.infinite = true;
		else if (token == "byoyomi" || token == "movetime") ssCmd >> limits.moveTime;
		else if (token == "mate") ssCmd >> limits.mate;
		else if (token == "depth") ssCmd >> limits.depth;
		else if (token == "nodes") ssCmd >> limits.nodes;
		else if (token == "searchmoves") {
			while (ssCmd >> token)
				limits.searchmoves.push_back(usiToMove(pos, token));
		}
	}
	if (limits.moveTime != 0)
		limits.moveTime -= pos.searcher()->options["Byoyomi_Margin"];
	else if (pos.searcher()->options["Time_Margin"] != 0)
		limits.time[pos.turn()] -= pos.searcher()->options["Time_Margin"];

	// 詰みの探索用
	/*limits.depth = static_cast<Depth>(6);
	pos.searcher()->alpha = -ScoreMaxEvaluate;
	pos.searcher()->beta = ScoreMaxEvaluate;
	pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);*/

	// 詰みのチェック
	Move mateMove = pos.mateMoveIn1Ply();
	if (mateMove != Move::moveNone()) {
		std::cout << "info score mate 1" << std::endl;
		std::cout << "bestmove " << mateMove.toUSI() << std::endl;
	}
	else {
		// UCTによる探索
		Move move = UctSearchGenmove(&pos);
		if (move == Move::moveNone()) {
			std::cout << "bestmove resign" << std::endl;
		}
		else {
			std::cout << "bestmove " << move.toUSI() << std::endl;
		}
	}

	// 探索待ち
	/*pos.searcher()->threads.main()->waitForSearchFinished();

	Score score = pos.searcher()->threads.main()->rootMoves[0].score;
	Move move2 = pos.searcher()->threads.main()->rootMoves[0].pv[0];

	std::cout << "score:" << score << std::endl;
	std::cout << "move:" << move2.toUSI() << std::endl;*/


}

#endif