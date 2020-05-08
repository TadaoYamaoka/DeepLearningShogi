#include "init.hpp"
#include "position.hpp"
#include "usi.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"

#include "cppshogi.h"
#include "UctSearch.h"
#include "Message.h"
#include "dfpn.h"

#include <signal.h>

extern std::ostream& operator << (std::ostream& os, const OptionsMap& om);

struct MySearcher : Searcher {
	STATIC void doUSICommandLoop(int argc, char* argv[]);
};
void go_uct(Position& pos, std::istringstream& ssCmd, const std::string& posCmd);
bool nyugyoku(const Position& pos);

ns_dfpn::DfPn dfpn;
int dfpn_min_search_millisecs = 300;

volatile sig_atomic_t stopflg = false;

void sigint_handler(int signum)
{
	stopflg = true;
}

int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	//HuffmanCodedPos::init();
	auto s = std::unique_ptr<MySearcher>(new MySearcher);

	s->init();
	s->doUSICommandLoop(argc, argv);
}

void MySearcher::doUSICommandLoop(int argc, char* argv[]) {
	bool evalTableIsRead = false;
	Position pos(DefaultStartPositionSFEN, thisptr);
	std::string posCmd("startpos");

	std::string cmd;
	std::string token;
	std::thread th;

	for (int i = 1; i < argc; ++i)
		cmd += std::string(argv[i]) + " ";

	do {
		if (argc == 1 && !std::getline(std::cin, cmd))
			cmd = "quit";

		//std::cout << "info string " << cmd << std::endl;
		std::istringstream ssCmd(cmd);

		ssCmd >> std::skipws >> token;

		if (token == "quit") {
			StopUctSearch();
			TerminateUctSearch();
			if (th.joinable())
				th.join();
			FinalizeUctSearch();
		}
		else if (token == "gameover") {
			StopUctSearch();
			GameOver();
		}
		else if (token == "stop") {
			StopUctSearch();
			if (th.joinable())
				th.join();
			// 無視されるがbestmoveを返す
			std::cout << "bestmove resign" << std::endl;
		}
		else if (token == "ponderhit" || token == "go") {
			// ponderの探索を停止
			StopUctSearch();
			if (th.joinable())
				th.join();
			th = std::thread([&pos, tmpCmd = cmd.substr((size_t)ssCmd.tellg() + 1), &posCmd] {
				std::istringstream ssCmd(tmpCmd);
				go_uct(pos, ssCmd, posCmd);
			});
		}
		else if (token == "position") {
			// 探索中にsetPositionを行うとStateInfoが壊れるため、探索を停止してから変更する必要があるため、
			// 文字列の保存のみ行う
			posCmd = cmd.substr((size_t)ssCmd.tellg() + 1);
		}
		else if (token == "usinewgame"); // isready で準備は出来たので、対局開始時に特にする事はない。
		else if (token == "usi") std::cout << "id name " << std::string(options["Engine_Name"])
			<< "\nid author Tadao Yamaoka"
			<< "\n" << options
			<< "\nusiok" << std::endl;
		else if (token == "isready") { // 対局開始前の準備。
			static bool initialized = false;
			if (!initialized) {
				// 各種初期化
				InitializeUctSearch(options["UCT_NodeLimit"]);
				const std::string model_paths[max_gpu] = { options["DNN_Model"], options["DNN_Model2"], options["DNN_Model3"], options["DNN_Model4"], options["DNN_Model5"], options["DNN_Model6"], options["DNN_Model7"], options["DNN_Model8"] };
				SetModelPath(model_paths);
				const int new_thread[max_gpu] = { options["UCT_Threads"], options["UCT_Threads2"], options["UCT_Threads3"], options["UCT_Threads4"], options["UCT_Threads5"], options["UCT_Threads6"], options["UCT_Threads7"], options["UCT_Threads8"] };
				const int new_policy_value_batch_maxsize[max_gpu] = { options["DNN_Batch_Size"], options["DNN_Batch_Size2"], options["DNN_Batch_Size3"], options["DNN_Batch_Size4"], options["DNN_Batch_Size5"], options["DNN_Batch_Size6"], options["DNN_Batch_Size7"], options["DNN_Batch_Size8"] };
				SetThread(new_thread, new_policy_value_batch_maxsize);

				if (options["Mate_Root_Search"] > 0) {
					ns_dfpn::DfPn::set_hashsize(options["DfPn_Hash"]);
					dfpn.init();
				}
			}
			else {
				NewGame();
			}
			initialized = true;

			// 詰み探索用
			if (options["Mate_Root_Search"] > 0) {
				ns_dfpn::DfPn::set_maxdepth(options["Mate_Root_Search"]);
			}

			// オプション設定
			set_softmax_temperature(options["Softmax_Temperature"] / 100.0f);
			SetResignThreshold(options["Resign_Threshold"]);
			SetDrawPly(options["Draw_Ply"]);
			SetDrawValue(options["Draw_Value_Black"], options["Draw_Value_White"]);
			dfpn_min_search_millisecs = options["DfPn_Min_Search_Millisecs"];
			c_init = options["C_init"] / 100.0f;
			c_base = options["C_base"];
			c_fpu_reduction = options["C_fpu_reduction"] / 100.0f;
			c_init_root = options["C_init_root"] / 100.0f;
			c_base_root = options["C_base_root"];
			c_fpu_reduction_root = options["C_fpu_reduction_root"] / 100.0f;
			SetReuseSubtree(options["ReuseSubtree"]);

			// 初回探索をキャッシュ
			SEARCH_MODE search_mode = GetMode();
			Position pos_tmp(DefaultStartPositionSFEN, thisptr);
			SetMode(CONST_PLAYOUT_MODE);
			SetPlayout(1);
			InitializeSearchSetting();
			Move ponder;
			UctSearchGenmove(&pos_tmp, pos.getKey(), {}, ponder);
			SetPlayout(CONST_PLAYOUT); // 元に戻す
			SetMode(search_mode); // 元に戻す

			// 固定プレイアウトモード
			if (options["Const_Playout"] > 0) {
				SetMode(CONST_PLAYOUT_MODE);
				SetPlayout(options["Const_Playout"]);
			}

			InitializeSearchSetting();

			// PonderingMode
			if (GetMode() != CONST_PLAYOUT_MODE)
				SetPonderingMode(options["USI_Ponder"] && !options["Stochastic_Ponder"]);

			// DebugMessageMode
			SetDebugMessageMode(options["DebugMessage"]);

			std::cout << "readyok" << std::endl;
		}
		else if (token == "setoption") setOption(ssCmd);
	} while (token != "quit" && argc == 1);

	if (th.joinable())
		th.join();
}

void go_uct(Position& pos, std::istringstream& ssCmd, const std::string& posCmd) {
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
	if (limits.moveTime != 0) {
		limits.moveTime -= pos.searcher()->options["Byoyomi_Margin"];
		SetConstTime(limits.moveTime / 1000.0);
	}
	else if (pos.searcher()->options["Time_Margin"] != 0)
		limits.time[pos.turn()] -= pos.searcher()->options["Time_Margin"];

	// 持ち時間設定
	if (limits.time[pos.turn()] > 0)
		SetRemainingTime(limits.time[pos.turn()] / 1000.0, pos.turn());
	SetIncTime(limits.inc[pos.turn()] / 1000.0, pos.turn());

	// 探索開始局面設定
	Key starting_pos_key;
	std::vector<Move> moves;
	{
		std::istringstream ssPosCmd(posCmd);
		std::string token;
		std::string sfen;

		ssPosCmd >> token;

		if (token == "startpos") {
			sfen = DefaultStartPositionSFEN;
			ssPosCmd >> token; // "moves" が入力されるはず。
		}
		else if (token == "sfen") {
			while (ssPosCmd >> token && token != "moves")
				sfen += token + " ";
		}
		else
			return;

		pos.set(sfen);
		pos.searcher()->states = StateListPtr(new std::deque<StateInfo>(1));

		starting_pos_key = pos.getKey();

		Ply currentPly = pos.gamePly();
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.searcher()->states->push_back(StateInfo());
			pos.doMove(move, pos.searcher()->states->back());
			++currentPly;
			moves.emplace_back(move);
		}
		pos.setStartPosPly(currentPly);
	}

	Move ponderMove = Move::moveNone();

	// Book使用
	static Book book;
	if (!limits.ponder && pos.searcher()->options["OwnBook"]) {
		const std::tuple<Move, Score> bookMoveScore = book.probe(pos, pos.searcher()->options["Book_File"], pos.searcher()->options["Best_Book_Move"]);
		if (std::get<0>(bookMoveScore)) {
			std::cout << "info"
				<< " score cp " << std::get<1>(bookMoveScore)
				<< " pv " << std::get<0>(bookMoveScore).toUSI()
				<< std::endl;

			auto move = std::get<0>(bookMoveScore);
			std::cout << "bestmove " << move.toUSI() << std::endl;

			// 確率的なPonderの場合、相手局面から探索を継続する
			if (pos.searcher()->options["USI_Ponder"] && pos.searcher()->options["Stochastic_Ponder"]) {
				StateInfo st;
				pos.doMove(move, st);
				pos.setStartPosPly(pos.gamePly() + 1);

				moves.emplace_back(move);
				UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove, true);
			}
			return;
		}
	}

	// 入玉勝ちかどうかを判定
	if (nyugyoku(pos)) {
		std::cout << "bestmove win" << std::endl;
		return;
	}

	// 詰みの探索用
	std::unique_ptr<std::thread> t;
	dfpn.dfpn_stop(false);
	std::atomic<bool> dfpn_done(false);
	bool mate = false;
	Position pos_copy(pos);
	if (!limits.ponder && pos.searcher()->options["Mate_Root_Search"] > 0) {
		t.reset(new std::thread([&pos_copy, &mate, &dfpn_done]() {
			if (!pos_copy.inCheck()) {
				mate = dfpn.dfpn(pos_copy);
				if (mate)
					StopUctSearch();
				dfpn_done = true;
			}
		}));
	}

	// UCTによる探索
	Move move = UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove, limits.ponder);

	// Ponderの場合、結果を返さない
	if (limits.ponder)
		return;

	// 詰み探索待ち
	if (pos.searcher()->options["Mate_Root_Search"] > 0) {
		// 最小詰み探索時間の間待つ
		const int time_limit = GetTimeLimit();
		while (!dfpn_done) {
			const auto elapse = limits.startTime.elapsed();
			if (elapse >= time_limit ||
				elapse >= dfpn_min_search_millisecs)
				break;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		dfpn.dfpn_stop(true);
		t->join();
		if (mate) {
			// 詰み
			Move move2 = dfpn.dfpn_move(pos);
			// PV表示
			std::cout << "info score mate + pv " << move2.toUSI();
			std::cout << std::endl;
			std::cout << "bestmove " << move2.toUSI() << std::endl;
			return;
		}
	}

	if (move == Move::moveNone()) {
		std::cout << "bestmove resign" << std::endl;
		return;
	}
	std::cout << "bestmove " << move.toUSI();
	// 確率的なPonderの場合、ponderを返さない
	if (pos.searcher()->options["USI_Ponder"] && pos.searcher()->options["Stochastic_Ponder"]) {
		std::cout << std::endl;

		// 相手局面から探索を継続する
		StateInfo st;
		pos.doMove(move, st);
		pos.setStartPosPly(pos.gamePly() + 1);

		moves.emplace_back(move);
		UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove, true);
	}
	else if (ponderMove != Move::moveNone())
		std::cout << " ponder " << ponderMove.toUSI() << std::endl;
	else
		std::cout << std::endl;
}
