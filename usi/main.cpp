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

#include <future>

#ifdef MULTI_PONDER
#include "../usi_multiponder/USIPonderEngine.h"
std::atomic<bool> need_multi_ponder = false;
constexpr size_t nohit = (size_t)-1;
#endif

extern std::ostream& operator << (std::ostream& os, const OptionsMap& om);

struct MySearcher : Searcher {
	static void doUSICommandLoop(int argc, char* argv[]);
#ifdef MAKE_BOOK
	static void makeBook(std::istringstream& ssCmd, const std::string& posCmd);
	static void makeMinMaxBook(std::istringstream& ssCmd, const std::string& posCmd);
	static void makeMctsBook(std::istringstream& ssCmd, const std::string& posCmd);
	static void mergeBook(std::istringstream& ssCmd, const std::string& posCmd);
	static void makeBookPosition(std::istringstream& ssCmd, const std::string& posCmd);
	static void makeBookPositions(std::istringstream& ssCmd);
	static void outpuNoneConnectPositions(std::istringstream& ssCmd);
	static void evalPositionsWithUsiEngine(std::istringstream& ssCmd, const std::string& posCmd);
	static void diffEval(std::istringstream& ssCmd, const std::string& posCmd);
	static void deleteOverMaxEval(std::istringstream& ssCmd);
#endif
	static Key starting_pos_key;
	static std::vector<Move> moves;
	static std::promise<std::pair<Move, Move>> promise;
	static std::future<std::pair<Move, Move>> future;
	static void setPositionAndLimits(Position& pos, std::istringstream& ssCmd, const std::string& posCmd);
	static void goUct(Position& pos);
#ifndef MULTI_PONDER
	static void getAndPrintBestMove();
#else
	static std::pair<Move, Move> getAndPrintBestMove();
#endif
};

Key MySearcher::starting_pos_key;
std::vector<Move> MySearcher::moves;
std::promise<std::pair<Move, Move>> MySearcher::promise;
std::future<std::pair<Move, Move>> MySearcher::future;

DfPn dfpn;
int dfpn_min_search_millisecs = 300;

int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	//HuffmanCodedPos::init();
	auto s = std::unique_ptr<MySearcher>(new MySearcher);

	s->init();
	s->doUSICommandLoop(argc, argv);

	// リソースの破棄はOSに任せてすぐに終了する
	std::quick_exit(0);
}

void MySearcher::doUSICommandLoop(int argc, char* argv[]) {
	bool evalTableIsRead = false;
	Position pos(DefaultStartPositionSFEN, thisptr);
	std::string posCmd("startpos");

	std::string cmd;
	std::string token;
	std::thread th;

#ifdef MULTI_PONDER
	std::vector<Move> multi_ponder_moves;
	std::vector<size_t> multi_ponder_engine_index;
	std::vector<USIPonderEngine> usi_ponder_engines;
#endif

	for (int i = 1; i < argc; ++i)
		cmd += std::string(argv[i]) + " ";

	do {
		if (argc == 1 && !std::getline(std::cin, cmd))
			cmd = "quit";

		std::istringstream ssCmd(cmd);
		token.clear();

		ssCmd >> std::skipws >> token;

		if (token == "quit") {
#ifdef MULTI_PONDER
			for (size_t i = 0; i < usi_ponder_engines.size(); i++) {
				usi_ponder_engines[i].Quit();
			}
#endif
			StopUctSearch();
			TerminateUctSearch();
			if (th.joinable())
				th.join();
			FinalizeUctSearch();
		}
		else if (token == "gameover") {
			StopUctSearch();
			if (th.joinable())
				th.join();
			GameOver();
		}
		else if (token == "stop") {
			StopUctSearch();
			if (th.joinable())
				th.join();
			if (limits.ponder) {
				// 無視されるがbestmoveを返す
				std::cout << "bestmove resign" << std::endl;
			}
		}
		else if (token == "go") {
#ifdef MULTI_PONDER
			USIPonderResult ponderResult;
			size_t ponderhit_i = nohit;
			if (need_multi_ponder) {
				for (size_t i : multi_ponder_engine_index) {
					if (usi_ponder_engines[i].IsLiving()) {
						if (posCmd == usi_ponder_engines[i].GetUsiPosition())
							ponderhit_i = i;
						else
							usi_ponder_engines[i].Stop();
					}
				}
				if (ponderhit_i != nohit ) {
					std::cout << "info string multiponder " << ponderhit_i + 1 << " ponderhit" << std::endl;
					ponderResult = usi_ponder_engines[ponderhit_i].Ponderhit();
				}
				for (size_t i : multi_ponder_engine_index) {
					if (usi_ponder_engines[i].IsLiving())
						usi_ponder_engines[i].Join();
				}
			}

			ResetMultiPonder();
			need_multi_ponder = false;
			multi_ponder_moves.clear();
			multi_ponder_engine_index.clear();
#endif
			// ponderの探索を停止
			StopUctSearch();
			if (th.joinable())
				th.join();
			setPositionAndLimits(pos, ssCmd, posCmd);
			InitUctSearchStop();
			// ponderhitで探索を継続するため、bestmoveはfutureで受け取る
			promise = std::promise<std::pair<Move, Move>>();
			future = promise.get_future();
#ifdef MULTI_PONDER
			if (ponderResult.bestMove != "") {
				// multi ponderがponderhitの場合
				std::cout << ponderResult.info << "\n";
				if (ponderResult.bestMove == "resign")
					promise.set_value({ Move::moveResign(), Move::moveNone() });
				else if (ponderResult.bestMove == "win")
					promise.set_value({ Move::moveWin(), Move::moveNone() });
				else {
					// 相手局面から探索を継続する
					Move move = usiToMove(pos, ponderResult.bestMove);
					need_multi_ponder = true;
					promise.set_value({ move, Move::moveNone() });
					th = std::thread([&pos, move] {
						StateInfo st;
						pos.doMove(move, st);

						Move ponderMove;
						moves.emplace_back(move);
						SetPondering(true);
						UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove);
					});
				}
			}
			else
			{
				th = std::thread([&pos] {
					goUct(pos);
				});
			}
#else
			th = std::thread([&pos] {
				goUct(pos);
			});
#endif
			if (!limits.ponder) {
#ifndef MULTI_PONDER
				getAndPrintBestMove();
#else
				const auto bestMove = getAndPrintBestMove();
				if (need_multi_ponder) {
					// ルートノードが展開されるのを待つ
					WaitPrepareMultiPonder();

					// 訪問回数・方策順にソートした指し手を取得
					const bool ret = GetMultiPonderMove(multi_ponder_moves, options["Multi_Ponder"]);

					if (ret && multi_ponder_moves.size() > 0) {
						std::string pos_ss(posCmd);
						if (posCmd.find("moves") == posCmd.npos) {
							pos_ss += " moves";
						}
						pos_ss += " " + bestMove.first.toUSI();
						// USIエンジンにgo ponderを送信
						std::cout << "info string multiponder";
						for (size_t i = 0, j = 0; i < multi_ponder_moves.size(); i++) {
							size_t ponder_i = i;
							// ponderhitしたエンジンに1番目の候補手を割り当てる
							if (ponderhit_i != nohit) {
								if (i == 0) {
									ponder_i = ponderhit_i;
								}
								else if (i <= ponderhit_i) {
									ponder_i = i - 1;
								}
							}
							multi_ponder_engine_index.emplace_back(ponder_i);
							if (usi_ponder_engines[ponder_i].IsLiving()) {
								const std::string ponder_move = std::move(multi_ponder_moves[j++].toUSI());
								std::cout << " " << ponder_i + 1 << " " << ponder_move;
								usi_ponder_engines[ponder_i].GoPonderAsync(pos_ss + " " + ponder_move, limits);
							}
						}
						std::cout << "\n";
					}
					else {
						need_multi_ponder = false;
					}
				}
#endif
			}
		}
		else if (token == "ponderhit") {
			// go ponderの探索をそのまま継続し、ponderingの状態のみ変更する
			SetPondering(false);
			getAndPrintBestMove();
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
				// モデルファイル存在チェック
				{
					bool is_err = false;
					for (int i = 0; i < max_gpu; ++i) {
						if (model_paths[i] != "") {
							std::ifstream ifs(model_paths[i].c_str());
							if (!ifs.is_open()) {
								std::cout << model_paths[i] << " file not found" << std::endl;
								is_err = true;
								break;
							}
						}
					}
					if (is_err)
						break;
				}
				SetModelPath(model_paths);
				// モデルの.iniファイルにしたがってデフォルトパラメータ変更
				std::ifstream is = std::ifstream(model_paths[0] + ".ini");
				while (is) {
					std::string line;
					is >> line;
					if (line != "") {
						const auto pos = line.find_first_of('=');
						const auto name = line.substr(0, pos);
						if (options[name].isDefault()) {
							options[name] = line.substr(pos + 1);
							std::cout << "info string " << name << "=" << options[name] << std::endl;
						}
					}
				}
				const int new_thread[max_gpu] = { options["UCT_Threads"], options["UCT_Threads2"], options["UCT_Threads3"], options["UCT_Threads4"], options["UCT_Threads5"], options["UCT_Threads6"], options["UCT_Threads7"], options["UCT_Threads8"] };
				const int new_policy_value_batch_maxsize[max_gpu] = { options["DNN_Batch_Size"], options["DNN_Batch_Size2"], options["DNN_Batch_Size3"], options["DNN_Batch_Size4"], options["DNN_Batch_Size5"], options["DNN_Batch_Size6"], options["DNN_Batch_Size7"], options["DNN_Batch_Size8"] };
				SetThread(new_thread, new_policy_value_batch_maxsize);

				if (options["Mate_Root_Search"] > 0) {
					DfPn::set_hashsize(options["DfPn_Hash"]);
					dfpn.init();
				}

#ifdef PV_MATE_SEARCH
				// PVの詰み探索の設定
				if (options["PV_Mate_Search_Threads"] > 0) {
					SetPvMateSearch(options["PV_Mate_Search_Threads"], options["PV_Mate_Search_Depth"], options["PV_Mate_Search_Nodes"]);
				}
#endif
#ifdef MULTI_PONDER
				// デストラクタが実行されないようにreserveする
				usi_ponder_engines.reserve(options["Multi_Ponder"]);
				for (int i = 0; i < options["Multi_Ponder"]; i++) {
					std::vector<std::pair<std::string, std::string>> usi_engine_options;
					std::string engine_options = options["Multi_Ponder_Engine" + std::to_string(i + 1) + "_Options"];
					// usiToCsa.rbでは,が使えないため#でも区切ることができるようにする
					std::replace(engine_options.begin(), engine_options.end(), '#', ',');
					std::istringstream ss(engine_options);
					std::string field;
					while (std::getline(ss, field, ',')) {
						const auto p = field.find_first_of(":");
						usi_engine_options.emplace_back(field.substr(0, p), field.substr(p + 1));
					}
					usi_ponder_engines.emplace_back((std::string)options["Multi_Ponder_Engine" + std::to_string(i + 1)], usi_engine_options);
				}
				for (size_t i = 0; i < usi_ponder_engines.size(); i++) {
					usi_ponder_engines[i].WaitInit();
				}
#endif
			}
			initialized = true;
#ifdef MULTI_PONDER
			need_multi_ponder = false;
#endif

			NewGame();

			// 詰み探索用
			if (options["Mate_Root_Search"] > 0) {
				dfpn.set_maxdepth(options["Mate_Root_Search"]);
				const int draw_ply = options["Draw_Ply"];
				if (draw_ply > 0)
					DfPn::set_draw_ply(draw_ply);
			}

			// オプション設定
			set_softmax_temperature(options["Softmax_Temperature"] / 100.0f);
			SetResignThreshold(options["Resign_Threshold"]);
			SetDrawPly(options["Draw_Ply"]);
			SetDrawValue(options["Draw_Value_Black"], options["Draw_Value_White"]);
			dfpn_min_search_millisecs = options["DfPn_Min_Search_Millisecs"];
			c_init = options["C_init"] / 100.0f;
			c_base = (float)options["C_base"];
			c_fpu_reduction = options["C_fpu_reduction"] / 100.0f;
			c_init_root = options["C_init_root"] / 100.0f;
			c_base_root = (float)options["C_base_root"];
			c_fpu_reduction_root = options["C_fpu_reduction_root"] / 100.0f;
			SetReuseSubtree(options["ReuseSubtree"]);
			SetPvInterval(options["PV_Interval"]);
			SetMultiPV(options["MultiPV"]);
			SetEvalCoef(options["Eval_Coef"]);
			SetRandomMove(options["Random_Ply"], options["Random_Temperature"], options["Random_Temperature_Drop"], options["Random_Cutoff"], options["Random_Cutoff_Drop"]);
			SetRandomMove2(options["Random2_Ply"], options["Random2_Probability"], options["Random2_Temperature"], options["Random2_Cutoff"], options["Random2_Value_Limit"]);

			// DebugMessageMode
			SetDebugMessageMode(options["DebugMessage"]);

			// 初回探索をキャッシュ
			Position pos_tmp(DefaultStartPositionSFEN, thisptr);
			LimitsType limits;
			limits.startTime.restart();
			limits.nodes = 1;

			SetLimits(limits);
			Move ponder;
			UctSearchGenmove(&pos_tmp, pos_tmp.getKey(), {}, ponder);

			// 固定プレイアウトモード
			SetConstPlayout(options["Const_Playout"]);

			// PonderingMode
			SetPonderingMode(options["USI_Ponder"] && !options["Stochastic_Ponder"]);

			std::cout << "readyok" << std::endl;
		}
		else if (token == "setoption") {
			setOption(ssCmd);
			SetMultiPV(options["MultiPV"]);
		}
#ifdef MAKE_BOOK
		else if (token == "make_book") makeBook(ssCmd, posCmd);
		else if (token == "make_minmax_book") makeMinMaxBook(ssCmd, posCmd);
		else if (token == "make_mcts_book") makeMctsBook(ssCmd, posCmd);
		else if (token == "merge_book") mergeBook(ssCmd, posCmd);
		else if (token == "make_book_position") makeBookPosition(ssCmd, posCmd);
		else if (token == "make_book_positions") makeBookPositions(ssCmd);
		else if (token == "output_none_connect_positions") outpuNoneConnectPositions(ssCmd);
		else if (token == "eval_positions_with_usi_engine") evalPositionsWithUsiEngine(ssCmd, posCmd);
		else if (token == "diff_eval") diffEval(ssCmd, posCmd);
		else if (token == "delete_over_max_eval") deleteOverMaxEval(ssCmd);
#endif
	} while (token != "quit" && argc == 1);

	if (th.joinable())
		th.join();
}

void MySearcher::setPositionAndLimits(Position& pos, std::istringstream& ssCmd, const std::string& posCmd) {
	std::string token;

	limits.startTime.restart();

	// 探索開始局面設定
	// 持ち時間設定よりも前に実行が必要
	moves.clear();
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
		states = StateListPtr(new std::deque<StateInfo>(1));

		starting_pos_key = pos.getKey();

		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.doMove(move, states->emplace_back());
			moves.emplace_back(move);
		}
	}

	// limitsをクリアして再設定
	limits.nodes = limits.time[Black] = limits.time[White] = limits.inc[Black] = limits.inc[White] = limits.movesToGo = limits.moveTime = limits.mate = limits.infinite = limits.ponder = 0;
	ssCmd >> token;
	if (token == "ponder") {
		limits.ponder = true;
	}
	do {
		if (token == "btime") ssCmd >> limits.time[Black];
		else if (token == "wtime") ssCmd >> limits.time[White];
		else if (token == "binc") ssCmd >> limits.inc[Black];
		else if (token == "winc") ssCmd >> limits.inc[White];
		else if (token == "infinite") limits.infinite = true;
		else if (token == "byoyomi" || token == "movetime") ssCmd >> limits.moveTime;
		else if (token == "mate") ssCmd >> limits.mate;
		else if (token == "nodes") ssCmd >> limits.nodes;
	} while (ssCmd >> token);
	if (limits.moveTime != 0) {
		limits.moveTime -= options["Byoyomi_Margin"];
	}
	else if (options["Time_Margin"] != 0) {
		limits.time[pos.turn()] -= options["Time_Margin"];
	}

	SetLimits(&pos, limits);
}

void MySearcher::goUct(Position& pos) {
	Move ponderMove = Move::moveNone();

	// Book使用
	static Book book;
	if (options["OwnBook"]) {
		const std::tuple<Move, Score> bookMoveScore = book.probe(pos, options["Book_File"], options["Best_Book_Move"], options["Book_Consider_Draw"]);
		if (std::get<0>(bookMoveScore)) {
			std::cout << "info"
				<< " score cp " << std::get<1>(bookMoveScore)
				<< " pv " << std::get<0>(bookMoveScore).toUSI()
				<< std::endl;

#ifdef MULTI_PONDER
			if (options["USI_Ponder"] && options["Stochastic_Ponder"])
				need_multi_ponder = true;
#endif
			auto move = std::get<0>(bookMoveScore);
			promise.set_value({ move, Move::moveNone() });

			// 確率的なPonderの場合、相手局面から探索を継続する
			if (options["USI_Ponder"] && options["Stochastic_Ponder"]) {
				StateInfo st;
				pos.doMove(move, st);

				moves.emplace_back(move);
				SetPondering(true);
				UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove);
			}
			return;
		}
	}

	// 入玉勝ちかどうかを判定
	if (nyugyoku(pos)) {
		promise.set_value({ Move::moveWin(), Move::moveNone() });
		return;
	}

	// 詰みの探索用
	std::unique_ptr<std::thread> t;
	dfpn.dfpn_stop(false);
	std::atomic<bool> dfpn_done(false);
	bool mate = false;
	const uint32_t mate_depth = options["Mate_Root_Search"];
	Position pos_copy(pos);
	if (mate_depth > 0) {
		t.reset(new std::thread([&pos_copy, &mate, &dfpn_done]() {
			mate = dfpn.dfpn(pos_copy);
			if (mate)
				StopUctSearch();
			dfpn_done = true;
		}));
	}

	// UCTによる探索
	Move move = UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove);

	// 詰み探索待ち
	if (mate_depth > 0) {
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
			std::string mate_pv;
			int mate_depth;
			Move mate_move;
			std::tie(mate_pv, mate_depth, mate_move) = dfpn.get_pv(pos);
			// PV表示
			if (mate_depth > 0)
				std::cout << "info score mate "<< mate_depth << " pv " << mate_pv;
			else
				std::cout << "info score mate + pv " << mate_pv;
			std::cout << std::endl;
			promise.set_value({ mate_move, Move::moveNone() });
			return;
		}
	}

	if (move == Move::moveNone()) {
		promise.set_value({ Move::moveResign(), Move::moveNone() });
		return;
	}
	// 確率的なPonderの場合、ponderを返さない
	if (!IsUctSearchStoped() && options["USI_Ponder"] && options["Stochastic_Ponder"]) {
#ifdef MULTI_PONDER
		need_multi_ponder = true;
		limits.time[pos.turn()] -= std::min(limits.time[pos.turn()], (limits.startTime.elapsed() + 999) / 1000 * 1000);
#endif
		promise.set_value({ move, Move::moveNone() });

		// 相手局面から探索を継続する
		StateInfo st;
		pos.doMove(move, st);

		moves.emplace_back(move);
		SetPondering(true);
		UctSearchGenmove(&pos, starting_pos_key, moves, ponderMove);
	}
	else {
		promise.set_value({ move, ponderMove });
	}
}

#ifndef MULTI_PONDER
void MySearcher::getAndPrintBestMove() {
	std::thread([] {
		const auto bestMove = future.get();
		std::cout << "bestmove ";
		if (bestMove.first == Move::moveResign()) {
			std::cout << "resign";
		}
		else if (bestMove.first == Move::moveWin()) {
			std::cout << "win";
		}
		else {
			std::cout << bestMove.first.toUSI();
		}
		if (bestMove.second != Move::moveNone()) {
			std::cout << " ponder " << bestMove.second.toUSI();
		}
		std::cout << std::endl;
	}).detach();
}
#else
std::pair<Move, Move> MySearcher::getAndPrintBestMove() {
	auto bestMove = future.get();
	std::cout << "bestmove ";
	if (bestMove.first == Move::moveResign()) {
		std::cout << "resign";
	}
	else if (bestMove.first == Move::moveWin()) {
		std::cout << "win";
	}
	else {
		std::cout << bestMove.first.toUSI();
	}
	if (bestMove.second != Move::moveNone()) {
		std::cout << " ponder " << bestMove.second.toUSI();
	}
	std::cout << std::endl;

	return bestMove;
}
#endif

#ifdef MAKE_BOOK
#include "make_book.h"
#include <filesystem>

class BookLock {
public:
	BookLock(const std::string filename, const bool use_book_lock) : filename(filename), use_book_lock(use_book_lock) {
		if (use_book_lock) {
			std::error_code ec;
			while (!std::filesystem::create_directory(filename + ".lock", ec)) {
				std::cout << filename << " locked" << std::endl;
				std::this_thread::sleep_for(std::chrono::seconds(3));
			}
		}
	}
	~BookLock() {
		if (use_book_lock) {
			if (!std::filesystem::remove(filename + ".lock"))
				std::quick_exit(1);
		}
	}
private:
	std::string filename;
	bool use_book_lock;
};

// 定跡作成
void MySearcher::makeBook(std::istringstream& ssCmd, const std::string& posCmd) {
	// isreadyを先に実行しておくこと。

	std::string bookFileName;
	std::string outFileName;
	int playoutNum;
	int limitTrialNum;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;
	ssCmd >> playoutNum;
	ssCmd >> limitTrialNum;

	// プレイアウト数固定
	LimitsType limits;
	limits.nodes = playoutNum;

	// 保存間隔
	const int save_book_interval = options["Save_Book_Interval"];

	// 1定跡作成ごとのスリープ時間(ガベージコレクションが間に合わない場合に設定する)
	make_book_sleep = options["Make_Book_Sleep"];

	// 探索打ち切りを使用する
	use_interruption = options["Use_Interruption"];

	// 評価値の閾値
	book_eval_threshold = options["Book_Eval_Threshold"];

	// 訪問回数の閾値(1000分率)
	book_visit_threshold = options["Book_Visit_Threshold"] / 1000.0;

	book_cutoff = options["Book_Cutoff"] / 1000.0f;

	// 訪問回数に応じてランダムに選択する際の温度パラメータ
	book_reciprocal_temperature = 1000.0 / options["Book_Temperature"];

	// 相手定跡の最善手を選択する
	book_best_move = options["Book_Best_Move"];

	// 相手定跡の評価値閾値
	book_eval_diff = (Score)(int)options["Book_Eval_Diff"];

	// 先手、後手どちらの定跡を作成するか("black":先手、"white":後手、それ以外:両方)
	const Color make_book_color = std::string(options["Make_Book_Color"]) == "black" ? Black : std::string(options["Make_Book_Color"]) == "white" ? White : ColorNum;

	// 定期的にマージする定跡ファイル
	const std::string merge_file = options["Book_Merge_File"];

	// マージ時にロックする
	const bool use_book_lock = options["Use_Book_Lock"];

	// MinMaxで選ぶ確率
	book_minmax_prob = options["Book_MinMax_Prob"] / 1000.0;
	book_minmax_prob_opp = options["Book_MinMax_Prob_Opp"] / 1000.0;

	// MinMaxのために相手定跡の手番でも探索する
	make_book_for_minmax = options["Make_Book_For_MinMax"];

	// 千日手の評価値
	const auto book_draw_value_black = (float)options["Book_Draw_Value_Black"] / 1000.0f;
	const auto book_draw_value_white = (float)options["Book_Draw_Value_White"] / 1000.0f;
	draw_score_black = Score(-logf(1.0f / book_draw_value_black - 1.0f) * eval_coef);
	draw_score_white = Score(-logf(1.0f / book_draw_value_white - 1.0f) * eval_coef);

	// 相手定跡から外れた場合USIエンジンを使う
	init_usi_book_engine(options["USI_Book_Engine"], options["USI_Book_Engine_Options"], options["USI_Book_Engine_Nodes"], options["USI_Book_Engine_Prob"] / 1000.0, options["USI_Book_Engine_Nodes_Own"], options["USI_Book_Engine_Prob_Own"] / 1000.0);

	// αβ探索で特定局面の評価値を置き換える
	init_book_key_eval_map(options["Book_Key_Eval_Map"]);

	// αβ探索の代わりにMCTSを使う
	const bool book_use_mcts = options["Book_Use_Mcts"];
	book_mcts_playouts = options["Book_Mcts_Playouts"];
	book_mcts_threads = options["Book_Mcts_Threads"];
	book_mcts_temperature = options["Book_Mcts_Temperature"] / 100.0f;
	book_mcts_debug = options["Book_Mcts_Debug"];

	SetReuseSubtree(options["ReuseSubtree"]);

	// outFileが存在するときは追加する
	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	book_pos_cmd = "position " + posCmd;
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

		if (token != "moves")
			book_pos_cmd += " moves";
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.doMove(move, pos.searcher()->states->emplace_back());
		}
	}
	book_starting_pos_key = pos.getKey();

	// 定跡読み込み
	bookMap.clear();
	read_book(bookFileName, bookMap);

	// 定跡マージ
	int merged = 0;
	if (merge_file != "") {
		BookLock book_lock(merge_file, use_book_lock);
		merged += merge_book(outMap, merge_file);
	}

	int black_num = 0;
	int white_num = 0;
	size_t prev_num = outMap.size();
	std::vector<Move> moves;
	const auto make_book_search = book_use_mcts ? make_book_mcts : make_book_alpha_beta;
	for (int trial = 0; trial < limitTrialNum;) {
		// 進捗状況表示
		std::cout << trial << "/" << limitTrialNum << " (" << int((double)trial / limitTrialNum * 100) << "%)" << std::endl;

		// 先手番
		if (make_book_color == Black || make_book_color == ColorNum) {
			int count = 0;
			moves.clear();
			// 探索
			Position pos_copy(pos);
			make_book_search(pos_copy, limits, bookMap, outMap, count, 0, true, moves);
			black_num += count;
			trial++;
		}

		// 後手番
		if (make_book_color == White || make_book_color == ColorNum) {
			int count = 0;
			moves.clear();
			// 探索
			Position pos_copy(pos);
			make_book_search(pos_copy, limits, bookMap, outMap, count, 0, false, moves);
			white_num += count;
			trial++;
		}

		// 完了時およびSave_Book_Intervalごとに途中経過を保存
		if (outMap.size() > prev_num && (trial % save_book_interval == 0 || trial >= limitTrialNum))
		{
			// 定跡マージ
			if (merge_file != "") {
				BookLock book_lock(merge_file, use_book_lock);
				merged += merge_book(outMap, merge_file);
			}

			prev_num = outMap.size();
			{
				BookLock book_lock(outFileName, use_book_lock);
				saveOutmap(outFileName, outMap);
			}
		}
	}

	// 結果表示
	std::cout << "input\t" << input_num << std::endl;
	std::cout << "black\t" << black_num << std::endl;
	std::cout << "white\t" << white_num << std::endl;
	std::cout << "sum\t" << black_num + white_num << std::endl;
	std::cout << "entries\t" << outMap.size() << std::endl;
	std::cout << "merged entries\t" << merged << std::endl;
}

void saveBookMapMinMax(const std::string& outFileName, const std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax) {
	// キーをソート
	std::set<Key> keySet;
	for (auto& elem : bookMapMinMax) {
		keySet.emplace(elem.first);
	}

	std::ofstream ofs(outFileName.c_str(), std::ios::binary);
	for (const Key key : keySet) {
		const auto itr = bookMapMinMax.find(key);
		const auto& elem = *itr;
		BookEntry entry = { elem.first, elem.second.move16, 1, elem.second.score };
		ofs.write(reinterpret_cast<char*>(&entry), sizeof(BookEntry));
	}
}

void MySearcher::makeMinMaxBook(std::istringstream& ssCmd, const std::string& posCmd) {
	std::string bookFileName;
	std::string outFileName;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;

	// 先手、後手どちらの定跡を作成するか("black":先手、"white":後手、それ以外:両方)
	const Color make_book_color = std::string(options["Make_Book_Color"]) == "black" ? Black : std::string(options["Make_Book_Color"]) == "white" ? White : ColorNum;

	// 千日手の評価値
	SetDrawValue(options["Draw_Value_Black"], options["Draw_Value_White"]);
	SetEvalCoef(options["Eval_Coef"]);
	const auto book_draw_value_black = (float)options["Book_Draw_Value_Black"] / 1000.0f;
	const auto book_draw_value_white = (float)options["Book_Draw_Value_White"] / 1000.0f;
	draw_score_black = Score(-logf(1.0f / book_draw_value_black - 1.0f) * eval_coef);
	draw_score_white = Score(-logf(1.0f / book_draw_value_white - 1.0f) * eval_coef);

	// αβ探索で特定局面の評価値を置き換える
	init_book_key_eval_map(options["Book_Key_Eval_Map"]);

	// αβ探索の代わりにMCTSを使う
	const bool book_use_mcts = options["Book_Use_Mcts"];
	book_mcts_playouts = options["Book_Mcts_Playouts"];
	book_mcts_threads = options["Book_Mcts_Threads"];
	book_mcts_temperature = options["Book_Mcts_Temperature"] / 100.0f;
	book_mcts_debug = options["Book_Mcts_Debug"];

	// 定跡読み込み
	bookMap.clear();
	read_book(bookFileName, bookMap);

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	std::istringstream ssPosCmd(posCmd);
	setPosition(pos, ssPosCmd);

	// 定跡をmin-max探索
	std::unordered_map<Key, MinMaxBookEntry> bookMapMinMax;
	make_minmax_book(pos, bookMapMinMax, make_book_color, book_use_mcts ? parallel_uct_search : select_best_book_entry);

	// 出力
	saveBookMapMinMax(outFileName, bookMapMinMax);
	std::cout << "minmaxBook.size:" << bookMapMinMax.size() << std::endl;

	// PV出力
	const auto pv = getBookPV(pos, outFileName);
	std::cout << "pv " << pv << std::endl;
}

void MySearcher::makeMctsBook(std::istringstream& ssCmd, const std::string& posCmd) {
	HuffmanCodedPos::init();

	std::string bookFileName;
	std::string outFileName;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;

	// 千日手の評価値
	SetDrawValue(options["Draw_Value_Black"], options["Draw_Value_White"]);
	SetEvalCoef(options["Eval_Coef"]);
	const auto book_draw_value_black = (float)options["Book_Draw_Value_Black"] / 1000.0f;
	const auto book_draw_value_white = (float)options["Book_Draw_Value_White"] / 1000.0f;
	draw_score_black = Score(-logf(1.0f / book_draw_value_black - 1.0f) * eval_coef);
	draw_score_white = Score(-logf(1.0f / book_draw_value_white - 1.0f) * eval_coef);

	book_mcts_playouts = options["Book_Mcts_Playouts"];
	book_mcts_threads = options["Book_Mcts_Threads"];
	book_mcts_temperature = options["Book_Mcts_Temperature"] / 100.0f;
	book_mcts_debug = options["Book_Mcts_Debug"];

	// 定跡読み込み
	bookMap.clear();
	read_book(bookFileName, bookMap);

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	std::istringstream ssPosCmd(posCmd);
	setPosition(pos, ssPosCmd);

	// 定跡をMCTSで探索
	std::map<Key, std::vector<BookEntry> > outMap;
	make_mcts_book(pos, bookMap, outMap);

	// 出力
	{
		size_t count = 0;
		std::ofstream ofs(outFileName.c_str(), std::ios::binary);
		for (auto& elem : outMap) {
			for (auto& elel : elem.second) {
				ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
				count++;
			}
		}
		std::cout << "outMap.size:" << outMap.size() << " count: " << count << std::endl;
	}

	// PV出力
	const auto pv = getBookPV(pos, outFileName);
	std::cout << "pv " << pv << std::endl;
}

void MySearcher::mergeBook(std::istringstream& ssCmd, const std::string& posCmd) {
	std::string bookFileName;
	std::string mergeFileName;
	std::string outFileName;

	ssCmd >> bookFileName;
	ssCmd >> mergeFileName;
	ssCmd >> outFileName;

	// 定跡読み込み
	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(bookFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "input size: " << outMap.size() << std::endl;
		}
	}

	// 定跡マージ
	const auto merged = merge_book(outMap, mergeFileName, false);

	// 出力
	saveOutmap(outFileName, outMap);

	std::cout << "output size: " << outMap.size() << std::endl;
}

// 局面の定跡作成
void MySearcher::makeBookPosition(std::istringstream& ssCmd, const std::string& posCmd) {
	// isreadyを先に実行しておくこと。

	std::string outFileName;
	int playoutNum;

	ssCmd >> outFileName;
	ssCmd >> playoutNum;

	// プレイアウト数固定
	LimitsType limits;
	limits.nodes = playoutNum;

	// 探索打ち切りを使用する
	use_interruption = options["Use_Interruption"];

	// 評価値の閾値
	book_eval_threshold = options["Book_Eval_Threshold"];

	// 訪問回数の閾値(1000分率)
	book_visit_threshold = options["Book_Visit_Threshold"] / 1000.0;

	book_cutoff = options["Book_Cutoff"] / 1000.0f;

	// 訪問回数に応じてランダムに選択する際の温度パラメータ
	book_reciprocal_temperature = 1000.0 / options["Book_Temperature"];

	// 相手定跡の最善手を選択する
	book_best_move = options["Book_Best_Move"];

	// 相手定跡の評価値閾値
	book_eval_diff = (Score)(int)options["Book_Eval_Diff"];

	// 先手、後手どちらの定跡を作成するか("black":先手、"white":後手、それ以外:両方)
	const Color make_book_color = std::string(options["Make_Book_Color"]) == "black" ? Black : std::string(options["Make_Book_Color"]) == "white" ? White : ColorNum;

	// 定期的にマージする定跡ファイル
	const std::string merge_file = options["Book_Merge_File"];

	// マージ時にロックする
	const bool use_book_lock = options["Use_Book_Lock"];

	// MinMaxで選ぶ確率
	book_minmax_prob = options["Book_MinMax_Prob"] / 1000.0;
	book_minmax_prob_opp = options["Book_MinMax_Prob_Opp"] / 1000.0;

	// MinMaxのために相手定跡の手番でも探索する
	make_book_for_minmax = options["Make_Book_For_MinMax"];

	// 千日手の評価値
	const auto book_draw_value_black = (float)options["Book_Draw_Value_Black"] / 1000.0f;
	const auto book_draw_value_white = (float)options["Book_Draw_Value_White"] / 1000.0f;
	draw_score_black = Score(-logf(1.0f / book_draw_value_black - 1.0f) * eval_coef);
	draw_score_white = Score(-logf(1.0f / book_draw_value_white - 1.0f) * eval_coef);

	// 相手定跡から外れた場合USIエンジンを使う
	init_usi_book_engine(options["USI_Book_Engine"], options["USI_Book_Engine_Options"], options["USI_Book_Engine_Nodes"], options["USI_Book_Engine_Prob"] / 1000.0, options["USI_Book_Engine_Nodes_Own"], options["USI_Book_Engine_Prob_Own"] / 1000.0);

	// αβ探索で特定局面の評価値を置き換える
	init_book_key_eval_map(options["Book_Key_Eval_Map"]);

	SetReuseSubtree(options["ReuseSubtree"]);

	// outFileが存在するときは追加する
	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	book_pos_cmd = "position";
	int count = 0;
	{
		std::istringstream ssPosCmd(posCmd);
		std::string token;
		std::string sfen;

		ssPosCmd >> token;
		book_pos_cmd += " " + token;

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

		book_pos_cmd += " moves";
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.doMove(move, pos.searcher()->states->emplace_back());
			book_pos_cmd += " " + token;

			// 定跡にない場合、追加
			{
				const Key key = Book::bookKey(pos);
				const auto itr = outMap.find(key);
				if (outMap.find(key) == outMap.end()) {
					book_starting_pos_key = pos.getKey();
					std::vector<Move> moves;
					Position pos_copy(pos);
					make_book_alpha_beta(pos_copy, limits, bookMap, outMap, count, 0, true, moves);
				}
			}

		}
	}

	// 保存
	saveOutmap(outFileName, outMap);

	// 結果表示
	std::cout << "input\t" << input_num << std::endl;
	std::cout << "output\t" << count << std::endl;
	std::cout << "entries\t" << outMap.size() << std::endl;
}

// 局面の定跡作成(ファイル入力)
void MySearcher::makeBookPositions(std::istringstream& ssCmd) {
	// isreadyを先に実行しておくこと。

	std::string inFileName;
	std::string outFileName;
	int playoutNum;

	ssCmd >> inFileName;
	ssCmd >> outFileName;
	ssCmd >> playoutNum;

	// プレイアウト数固定
	LimitsType limits;
	limits.nodes = playoutNum;

	// 保存間隔
	const int save_book_interval = options["Save_Book_Interval"];

	// 探索打ち切りを使用する
	use_interruption = options["Use_Interruption"];

	// 評価値の閾値
	book_eval_threshold = options["Book_Eval_Threshold"];

	// 訪問回数の閾値(1000分率)
	book_visit_threshold = options["Book_Visit_Threshold"] / 1000.0;

	book_cutoff = options["Book_Cutoff"] / 1000.0f;

	// 訪問回数に応じてランダムに選択する際の温度パラメータ
	book_reciprocal_temperature = 1000.0 / options["Book_Temperature"];

	// 相手定跡の最善手を選択する
	book_best_move = options["Book_Best_Move"];

	// 相手定跡の評価値閾値
	book_eval_diff = (Score)(int)options["Book_Eval_Diff"];

	// 先手、後手どちらの定跡を作成するか("black":先手、"white":後手、それ以外:両方)
	const Color make_book_color = std::string(options["Make_Book_Color"]) == "black" ? Black : std::string(options["Make_Book_Color"]) == "white" ? White : ColorNum;

	// 定期的にマージする定跡ファイル
	const std::string merge_file = options["Book_Merge_File"];

	// マージ時にロックする
	const bool use_book_lock = options["Use_Book_Lock"];

	// MinMaxで選ぶ確率
	book_minmax_prob = options["Book_MinMax_Prob"] / 1000.0;
	book_minmax_prob_opp = options["Book_MinMax_Prob_Opp"] / 1000.0;

	// MinMaxのために相手定跡の手番でも探索する
	make_book_for_minmax = options["Make_Book_For_MinMax"];

	// 千日手の評価値
	const auto book_draw_value_black = (float)options["Book_Draw_Value_Black"] / 1000.0f;
	const auto book_draw_value_white = (float)options["Book_Draw_Value_White"] / 1000.0f;
	draw_score_black = Score(-logf(1.0f / book_draw_value_black - 1.0f) * eval_coef);
	draw_score_white = Score(-logf(1.0f / book_draw_value_white - 1.0f) * eval_coef);

	// 相手定跡から外れた場合USIエンジンを使う
	init_usi_book_engine(options["USI_Book_Engine"], options["USI_Book_Engine_Options"], options["USI_Book_Engine_Nodes"], options["USI_Book_Engine_Prob"] / 1000.0, options["USI_Book_Engine_Nodes_Own"], options["USI_Book_Engine_Prob_Own"] / 1000.0);

	// αβ探索で特定局面の評価値を置き換える
	init_book_key_eval_map(options["Book_Key_Eval_Map"]);

	SetReuseSubtree(options["ReuseSubtree"]);

	// outFileが存在するときは追加する
	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	int count = 0;
	int positions = 0;
	// 局面ファイル読み込み
	std::ifstream ifsInFile(inFileName.c_str());
	std::string line;
	while (std::getline(ifsInFile, line)) {
		if (line == "") continue;

		book_pos_cmd = "position";
		std::istringstream ssPosCmd(line);
		std::string token;
		std::string sfen;

		ssPosCmd >> token;
		book_pos_cmd += " " + token;

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

		book_pos_cmd += " moves";
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.doMove(move, pos.searcher()->states->emplace_back());
			book_pos_cmd += " " + token;

			// 定跡にない場合、追加
			{
				const Key key = Book::bookKey(pos);
				const auto itr = outMap.find(key);
				if (outMap.find(key) == outMap.end()) {
					book_starting_pos_key = pos.getKey();
					std::vector<Move> moves;
					Position pos_copy(pos);
					make_book_alpha_beta(pos_copy, limits, bookMap, outMap, count, 0, true, moves);

					// Save_Book_Intervalごとに途中経過を保存
					if (++positions % save_book_interval == 0)
					{
						// 保存
						saveOutmap(outFileName, outMap);
					}
				}
			}
		}
	}

	// 保存
	saveOutmap(outFileName, outMap);

	// 結果表示
	std::cout << "input\t" << input_num << std::endl;
	std::cout << "output\t" << count << std::endl;
	std::cout << "entries\t" << outMap.size() << std::endl;
	std::cout << "positions\t" << positions << std::endl;
}

// 定跡が未接続の局面を出力する
void MySearcher::outpuNoneConnectPositions(std::istringstream& ssCmd) {
	std::string bookFileName;
	std::string outFileName;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;

	// 定跡読み込み
	bookMap.clear();
	read_book(bookFileName, bookMap);

	Position pos(DefaultStartPositionSFEN, thisptr);
	std::ofstream ofs(outFileName);
	std::unordered_set<Key> exists;
	int count = 0;
	output_none_connect_positions(pos, bookMap, exists, ofs, count);
	std::cout << "exists: " << exists.size() << " count: " << count << std::endl;
}

// USIエンジンで局面を評価する
void MySearcher::evalPositionsWithUsiEngine(std::istringstream& ssCmd, const std::string& posCmd) {
	HuffmanCodedPos::init();

	std::string bookFileName;
	std::string outFileName;
	int engine_num;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;
	ssCmd >> engine_num;

	// 定跡読み込み
	bookMap.clear();
	read_book(bookFileName, bookMap);

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	std::istringstream ssPosCmd(posCmd);
	setPosition(pos, ssPosCmd);

	// 出力定跡読み込み
	std::map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	//  USIエンジンで局面を評価する
	eval_positions_with_usi_engine(pos, bookMap, outMap, options["USI_Book_Engine"], options["USI_Book_Engine_Options"], options["USI_Book_Engine_Nodes"], engine_num);

	// 出力
	{
		size_t count = 0;
		std::ofstream ofs(outFileName.c_str(), std::ios::binary);
		for (auto& elem : outMap) {
			for (auto& elel : elem.second) {
				ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
				count++;
			}
		}
		std::cout << "outMap.size:" << outMap.size() << " count: " << count << std::endl;
	}
}

// 評価値が割れる局面を延長する
void MySearcher::diffEval(std::istringstream& ssCmd, const std::string& posCmd) {
	// isreadyを先に実行しておくこと。

	std::string bookFileName;
	std::string outFileName;
	int playoutNum;
	int diff;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;
	ssCmd >> playoutNum;
	ssCmd >> diff;

	// プレイアウト数固定
	LimitsType limits;
	limits.nodes = playoutNum;

	// 訪問回数の閾値(1000分率)
	book_visit_threshold = options["Book_Visit_Threshold"] / 1000.0;

	book_cutoff = options["Book_Cutoff"] / 1000.0f;

	// 探索打ち切りを使用する
	use_interruption = options["Use_Interruption"];

	SetReuseSubtree(options["ReuseSubtree"]);

	// outFileが存在するときは追加する
	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	// 開始局面設定
	Position pos(DefaultStartPositionSFEN, thisptr);
	book_pos_cmd = "position " + posCmd;
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

		if (token != "moves")
			book_pos_cmd += " moves";
		while (ssPosCmd >> token) {
			const Move move = usiToMove(pos, token);
			if (!move) break;
			pos.doMove(move, pos.searcher()->states->emplace_back());
		}
	}
	book_starting_pos_key = pos.getKey();

	// 定跡読み込み
	std::unordered_map<Key, std::vector<BookEntry> > bookMap;
	read_book(bookFileName, bookMap);

	diff_eval(pos, bookMap, outMap, limits, (Score)diff, outFileName);

	// 結果表示
	std::cout << "outMap.size:" << outMap.size() << std::endl;
}

// 評価値が最大値を超える局面を削除
void MySearcher::deleteOverMaxEval(std::istringstream& ssCmd) {
	std::string outFileName;

	ssCmd >> outFileName;

	int input_num = 0;
	std::unordered_map<Key, std::vector<BookEntry> > outMap;
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].emplace_back(entry);
				input_num++;
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	std::vector<Key> delete_keys;
	for (auto kv : outMap) {
		const auto& entries = kv.second;
		for (const auto& entry : entries) {
			if (entry.score < -ScoreMaxEvaluate || entry.score > ScoreMaxEvaluate) {
				delete_keys.emplace_back(kv.first);
				break;
			}
		}
	}

	for (const Key key : delete_keys) {
		outMap.erase(key);
	}

	// 保存
	saveOutmap(outFileName, outMap);

	std::cout << "outMap.size: " << outMap.size() << std::endl;
}
#endif
