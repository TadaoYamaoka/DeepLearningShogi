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
void go_uct(Position& pos, std::istringstream& ssCmd, const std::string& prevPosCmd);
bool nyugyoku(const Position& pos);
void make_book(std::istringstream& ssCmd);
void mate_test(Position& pos, std::istringstream& ssCmd);
void test(Position& pos, std::istringstream& ssCmd);

ns_dfpn::DfPn dfpn;

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

	TerminateUctSearch();
}

void MySearcher::doUSICommandLoop(int argc, char* argv[]) {
	bool evalTableIsRead = false;
	Position pos(DefaultStartPositionSFEN, thisptr);

	std::string cmd;
	std::string token;
	std::thread th;
	std::string prevPosCmd;

	for (int i = 1; i < argc; ++i)
		cmd += std::string(argv[i]) + " ";

	do {
		if (argc == 1 && !std::getline(std::cin, cmd))
			cmd = "quit";

		//std::cout << "info string " << cmd << std::endl;
		std::istringstream ssCmd(cmd);

		ssCmd >> std::skipws >> token;

		if (token == "quit" || token == "gameover") {
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
			if (token == "ponderhit") {
				StopUctSearch();
				if (th.joinable())
					th.join();
				// 確率的なPonderの場合
				if (pos.searcher()->options["Stochastic_Ponder"]) {
					// 局面を戻す
					std::istringstream ssTmpCmd(prevPosCmd);
					setPosition(pos, ssTmpCmd);
				}
			}
			else {
				if (th.joinable())
					th.join();
			}
			th = std::thread([&pos, tmpCmd = cmd.substr((size_t)ssCmd.tellg() + 1), &prevPosCmd] {
				std::istringstream ssCmd(tmpCmd);
				go_uct(pos, ssCmd, prevPosCmd);
			});
		}
		else if (token == "position") {
			prevPosCmd = cmd.substr((size_t)ssCmd.tellg() + 1);
			setPosition(pos, ssCmd);
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
				InitializeUctSearch(options["UCT_Hash"]);
				const std::string model_paths[max_gpu] = { options["DNN_Model"], options["DNN_Model2"], options["DNN_Model3"], options["DNN_Model4"] };
				SetModelPath(model_paths);
				const int new_thread[max_gpu] = { options["UCT_Threads"], options["UCT_Threads2"], options["UCT_Threads3"], options["UCT_Threads4"] };
				const int new_policy_value_batch_maxsize[max_gpu] = { options["DNN_Batch_Size"], options["DNN_Batch_Size2"], options["DNN_Batch_Size3"], options["DNN_Batch_Size4"] };
				SetThread(new_thread, new_policy_value_batch_maxsize);

				if (options["Mate_Root_Search"] > 0) {
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
			c_init = options["C_init"] / 100.0f;
			c_base = options["C_base"];

			// 初回探索をキャッシュ
			SEARCH_MODE search_mode = GetMode();
			Position pos_tmp(DefaultStartPositionSFEN, thisptr);
			SetMode(CONST_PLAYOUT_MODE);
			SetPlayout(1);
			InitializeSearchSetting();
			Move ponder;
			UctSearchGenmove(&pos_tmp, ponder);
			SetPlayout(CONST_PLAYOUT); // 元に戻す

			SetMode(search_mode); // 元に戻す
			if (search_mode == TIME_SETTING_MODE || search_mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
				// プレイアウト速度測定
				SetConstTime(10);
				InitializeSearchSetting();
				UctSearchGenmove(&pos_tmp, ponder);
			}
			else {
				InitializeSearchSetting();
			}

			// PonderingMode
			SetPonderingMode(options["USI_Ponder"]);

			// DebugMessageMode
			SetDebugMessageMode(options["DebugMessage"]);

			std::cout << "readyok" << std::endl;
		}
		else if (token == "setoption") setOption(ssCmd);
		else if (token == "make_book") make_book(ssCmd);
		else if (token == "mate_test") mate_test(pos, ssCmd);
		else if (token == "test") test(pos, ssCmd);
	} while (token != "quit" && argc == 1);

	if (th.joinable())
		th.join();
}

void go_uct(Position& pos, std::istringstream& ssCmd, const std::string& prevPosCmd) {
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

	// 確率的なPonder
	if (limits.ponder && pos.searcher()->options["Stochastic_Ponder"]) {
		// 相手局面から探索
		std::istringstream ssTmpCmd(prevPosCmd.substr(0, prevPosCmd.rfind(" ")));
		setPosition(pos, ssTmpCmd);
	}

	// Book使用
	static Book book;
	if (!limits.ponder && pos.searcher()->options["OwnBook"]) {
		const std::tuple<Move, Score> bookMoveScore = book.probe(pos, pos.searcher()->options["Book_File"], pos.searcher()->options["Best_Book_Move"]);
		if (std::get<0>(bookMoveScore)) {
			std::cout << "info"
				<< " score cp " << std::get<1>(bookMoveScore)
				<< " pv " << std::get<0>(bookMoveScore).toUSI()
				<< std::endl;

			std::cout << "bestmove " << std::get<0>(bookMoveScore).toUSI() << std::endl;
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
	bool mate = false;
	if (!limits.ponder && pos.searcher()->options["Mate_Root_Search"] > 0) {
		dfpn.set_remaining_time((double)limits.time[pos.turn()] * 0.9f);
		t.reset(new std::thread([&pos, &mate]() {
			if (!pos.inCheck()) {
				Position pos_copy(pos);
				mate = dfpn.dfpn(pos_copy);
				if (mate)
					StopUctSearch();
			}
		}));
	}

	// UCTによる探索
	Move ponderMove = Move::moveNone();
	Move move = UctSearchGenmove(&pos, ponderMove, limits.ponder);

	// Ponderの場合、結果を返さない
	if (limits.ponder)
		return;

	// 詰み探索待ち
	if (pos.searcher()->options["Mate_Root_Search"] > 0) {
		dfpn.dfpn_stop();
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
	if (ponderMove != Move::moveNone())
		std::cout << " ponder " << ponderMove.toUSI();
	std::cout << std::endl;
}

struct child_node_t_copy {
	Move move;  // 着手する座標
	int move_count;  // 探索回数
	float win;         // 勝った回数

	child_node_t_copy(const child_node_t& child) {
		this->move = child.move;
		this->move_count = child.move_count;
		this->win = child.win;
	}
};

void make_book_entry_with_uct(Position& pos, const Key& key, std::map<Key, std::vector<BookEntry> > &outMap, int& count, std::vector<Move> &moves) {
	std::cout << "position startpos moves ";
	for (Move move : moves) {
		std::cout << move.toUSI() << " ";
	}
	std::cout << std::endl;

	// UCT探索を使用
	UctSearchGenmoveNoPonder(&pos);

	if (uct_node[current_root].child_num == 0) {
		std::cerr << "Error: child_num" << std::endl;
		exit(EXIT_FAILURE);
	}

	// 探索回数で降順ソート
	std::vector<child_node_t_copy> movelist;
	int num = 0;
	const child_node_t *uct_child = uct_node[current_root].child;
	for (int i = 0; i < uct_node[current_root].child_num; i++) {
		movelist.emplace_back(uct_child[i]);
		if (double(uct_child[i].move_count) / uct_node[current_root].move_count > 0.1) { // 閾値
			num++;
		}
	}
	if (num == 0) {
		num = (uct_node[current_root].child_num + 2) / 3;
	}

	std::cout << "movelist.size: " << num << std::endl;

	std::sort(movelist.begin(), movelist.end(), [](auto left, auto right) {
		return left.move_count > right.move_count;
	});

	for (int i = 0; i < num; i++) {
		auto &child = movelist[i];
		// 定跡追加
		BookEntry be;
		float wintrate = child.win / child.move_count;
		be.score = Score(int(-logf(1.0f / wintrate - 1.0f) * 754.3f));
		be.key = key;
		be.fromToPro = static_cast<u16>(child.move.proFromAndTo());
		be.count = (u16)((double)child.move_count / (double)uct_node[current_root].move_count * 1000.0);
		outMap[key].emplace_back(be);

		count++;
	}
}

// 定跡作成(再帰処理)
void make_book_inner(Position& pos, std::map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> > &outMap, int& count, int depth, const bool isBlack, std::vector<Move> &moves) {
	pos.setStartPosPly(depth + 1);
	const Key key = Book::bookKey(pos);
	if ((depth % 2 == 0) == isBlack) {

		if (outMap.find(key) == outMap.end()) {
			// 先端ノード
			// UCT探索で定跡作成
			make_book_entry_with_uct(pos, key, outMap, count, moves);
		}
		else {
			// 探索済みの場合
			{
				// 最上位の手を選択
				// (定跡の幅を広げたい場合は確率的に選択するように変更する)
				auto entry = outMap[key][0];

				Move move = move16toMove(Move(entry.fromToPro), pos);

				StateInfo state;
				pos.doMove(move, state);
				// 繰り返しになる場合、別の手を選ぶ
				if (outMap[key].size() >= 2 && pos.isDraw() == RepetitionDraw) {
					pos.undoMove(move);
					entry = outMap[key][1];
					move = move16toMove(Move(entry.fromToPro), pos);
					pos.doMove(move, state);
				}

				moves.emplace_back(move);

				// 次の手を探索
				make_book_inner(pos, bookMap, outMap, count, depth + 1, isBlack, moves);

				pos.undoMove(move);
			}
		}
	}
	else {
		// 定跡を使用
		std::vector<BookEntry> *entries;

		// 局面が定跡にあるか確認
		auto itr = bookMap.find(key);
		if (itr != bookMap.end()) {
			entries = &itr->second;
		}
		else {
			// 定跡にない場合、探索結果を使う
			itr = outMap.find(Book::bookKey(pos));

			if (itr == outMap.end()) {
				// 定跡になく未探索の局面の場合
				// UCT探索で定跡作成
				make_book_entry_with_uct(pos, key, outMap, count, moves);
			}

			entries = &outMap[key];
		}

		// 確率的に手を選択
		std::vector<double> probabilities;
		for (auto& entry : *entries) {
			probabilities.emplace_back(entry.count);
		}
		std::discrete_distribution<std::size_t> dist(probabilities.begin(), probabilities.end());
		size_t selected = dist(g_randomTimeSeed);

		Move move = move16toMove(Move(entries->at(selected).fromToPro), pos);

		StateInfo state;
		pos.doMove(move, state);
		moves.emplace_back(move);

		// 次の手を探索
		make_book_inner(pos, bookMap, outMap, count, depth + 1, isBlack, moves);

		pos.undoMove(move);
	}
}

// 定跡読み込み
void read_book(const std::string& bookFileName, std::map<Key, std::vector<BookEntry> >& bookMap) {
	std::ifstream ifs(bookFileName.c_str(), std::ifstream::in | std::ifstream::binary);
	if (!ifs) {
		std::cerr << "Error: cannot open " << bookFileName << std::endl;
		exit(EXIT_FAILURE);
	}
	BookEntry entry;
	size_t count = 0;
	while (ifs.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
		count++;
		auto itr = bookMap.find(entry.key);
		if (itr != bookMap.end()) {
			// すでにある場合、追加
			itr->second.emplace_back(entry);
		}
		else {
			bookMap[entry.key].emplace_back(entry);
		}
	}
	std::cout << "bookEntries.size:" << bookMap.size() << " count:" << count << std::endl;
}

// 定跡作成
void make_book(std::istringstream& ssCmd) {
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
	SetMode(CONST_PLAYOUT_MODE);
	SetPlayout(playoutNum);
	InitializeSearchSetting();
	SetReuseSubtree(false);

	// outFileが存在するときは追加する
	int input_num = 0;
	std::map<Key, std::vector<BookEntry> > outMap;
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

	Searcher s;
	s.init();
	Position pos(DefaultStartPositionSFEN, &s);

	// 定跡読み込み
	std::map<Key, std::vector<BookEntry> > bookMap;
	read_book(bookFileName, bookMap);

	// シグナル設定
	signal(SIGINT, sigint_handler);

	int black_num = 0;
	int white_num = 0;
	std::vector<Move> moves;
	for (int trial = 0; trial < limitTrialNum; trial += 2) {
		// 先手番
		int count = 0;
		moves.clear();
		// 探索
		pos.set(DefaultStartPositionSFEN);
		make_book_inner(pos, bookMap, outMap, count, 0, true, moves);
		black_num += count;

		// 後手番
		count = 0;
		moves.clear();
		// 探索
		pos.set(DefaultStartPositionSFEN);
		make_book_inner(pos, bookMap, outMap, count, 0, false, moves);
		white_num += count;

		if (stopflg)
			break;
	}

	// 保存
	std::ofstream ofs(outFileName.c_str(), std::ios::binary);
	for (auto& elem : outMap) {
		for (auto& elel : elem.second)
			ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
	}

	// 結果表示
	std::cout << "input\t" << input_num << std::endl;
	std::cout << "black\t" << black_num << std::endl;
	std::cout << "white\t" << white_num << std::endl;
	std::cout << "sum\t" << black_num + white_num << std::endl;
	std::cout << "entries\t" << outMap.size() << std::endl;
}

void mate_test(Position& pos, std::istringstream& ssCmd) {
	/*auto start = std::chrono::system_clock::now();
	bool isCheck;
	int depth;
	ssCmd >> depth;

	if (!pos.inCheck()) {
		//isCheck = mateMoveIn3Ply(pos);
		depth += (depth + 1) % 2;
		isCheck = mateMoveInOddPly(pos, depth);
	}
	else {
		//isCheck = mateMoveIn2Ply(pos);
		depth += depth % 2;
		isCheck = mateMoveInEvenPly(pos, depth);
	}
	auto end = std::chrono::system_clock::now();
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "mateMoveIn" << depth << "Ply : " << isCheck << std::endl;
	std::cout << msec << " msec" << std::endl;*/
}

void test(Position& pos, std::istringstream& ssCmd) {
}
