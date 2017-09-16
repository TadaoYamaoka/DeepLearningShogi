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
#include "book.hpp"

#include "cppshogi.h"
#include "UctSearch.h"

struct MySearcher : Searcher {
	STATIC void doUSICommandLoop(int argc, char* argv[]);
};
void go_uct(Position& pos, std::istringstream& ssCmd);
void make_book(std::istringstream& ssCmd);

int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	//HuffmanCodedPos::init();
	auto s = std::unique_ptr<MySearcher>(new MySearcher);

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

		//std::cout << "info string " << cmd << std::endl;
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
			// 詰み探索用
			if (options["Search_Mate_Depth"] > 0) {
				tt.clear();
				threads.main()->previousScore = ScoreInfinite;
				if (!evalTableIsRead) {
					// 一時オブジェクトを生成して Evaluator::init() を呼んだ直後にオブジェクトを破棄する。
					// 評価関数の次元下げをしたデータを格納する分のメモリが無駄な為、
					std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
					evalTableIsRead = true;
				}
			}

			// 各種初期化
			set_softmax_tempature(options["Softmax_Tempature"] / 100.0);
			SetThread(options["UCT_Threads"]);
			SetModelPath(std::string(options["DNN_Model"]).c_str());
			InitializeUctSearch();
			InitializeUctHash();

			// 初回探索をキャッシュ
			SEARCH_MODE search_mode = GetMode();
			Position pos_tmp(DefaultStartPositionSFEN, threads.main(), thisptr);
			SetMode(CONST_PLAYOUT_MODE);
			SetPlayout(1);
			InitializeSearchSetting();
			UctSearchGenmove(&pos_tmp);
			SetPlayout(CONST_PLAYOUT); // 元に戻す

			SetMode(search_mode); // 元に戻す
			if (search_mode == TIME_SETTING_MODE || search_mode == TIME_SETTING_WITH_BYOYOMI_MODE) {
				// プレイアウト速度測定
				SetTime(1);
				InitializeSearchSetting();
				UctSearchGenmove(&pos_tmp);
			}
			else {
				InitializeSearchSetting();
			}

			std::cout << "readyok" << std::endl;
		}
		else if (token == "setoption") setOption(ssCmd);
		else if (token == "make_book") make_book(ssCmd);
	} while (token != "quit" && argc == 1);

	if (options["Search_Mate_Depth"] > 0)
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
	if (limits.moveTime != 0) {
		limits.moveTime -= pos.searcher()->options["Byoyomi_Margin"];
		SetConstTime(limits.moveTime / 1000.0);
	}
	else if (pos.searcher()->options["Time_Margin"] != 0)
		limits.time[pos.turn()] -= pos.searcher()->options["Time_Margin"];

	// 持ち時間設定
	SetRemainingTime(limits.time[pos.turn()] / 1000.0, pos.turn());
	SetIncTime(limits.inc[pos.turn()] / 1000.0, pos.turn());

	// Book使用
	static Book book;
	if (pos.searcher()->options["OwnBook"]) {
		const std::tuple<Move, Score> bookMoveScore = book.probe(pos, pos.searcher()->options["Book_File"], false);
		if (std::get<0>(bookMoveScore)) {
			std::cout << "info"
				<< " score cp " << std::get<1>(bookMoveScore)
				<< " pv " << std::get<0>(bookMoveScore).toUSI()
				<< std::endl;

			std::cout << "bestmove " << std::get<0>(bookMoveScore).toUSI() << std::endl;
			return;
		}
	}

	// 詰みの探索用
	if (pos.searcher()->options["Search_Mate_Depth"] > 0) {
		limits.depth = static_cast<Depth>((int)pos.searcher()->options["Search_Mate_Depth"]);
		pos.searcher()->alpha = -ScoreMaxEvaluate;
		pos.searcher()->beta = ScoreMaxEvaluate;
		pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);
	}

	// UCTによる探索
	Move move = UctSearchGenmove(&pos);
	if (move == Move::moveNone()) {
		std::cout << "bestmove resign" << std::endl;
		return;
	}

	// 探索待ち
	if (pos.searcher()->options["Search_Mate"]) {
		pos.searcher()->threads.main()->waitForSearchFinished();

		Score score = pos.searcher()->threads.main()->rootMoves[0].score;

		if (!pos.searcher()->threads.main()->rootMoves[0].pv[0]) {
			SYNCCOUT << "bestmove resign" << SYNCENDL;
			return;
		}
		else if (score > ScoreMaxEvaluate) {
			Move move2 = pos.searcher()->threads.main()->rootMoves[0].pv[0];
			std::cout << "info score mate " << ScoreMate0Ply - score << " pv " << move2.toUSI() << std::endl;
			std::cout << "bestmove " << move2.toUSI() << std::endl;
			return;
		}
	}

	std::cout << "bestmove " << move.toUSI() << std::endl;
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

// 定跡作成(再帰処理)
void make_book_inner(Position& pos, std::set<Key>& bookKeys, std::map<Key, std::vector<BookEntry> > &outMap, int& count, int depth, const bool isBlack, const int limitDepth) {
	pos.setStartPosPly(depth + 1);
	std::cout << pos.toSFEN() << std::endl;
	if ((depth % 2 == 0) == isBlack) {
		const Key key = Book::bookKey(pos);

		if (outMap.find(key) == outMap.end()) {
			// UCT探索を使用
			UctSearchGenmove(&pos);

			// 探索回数で降順ソート
			std::vector<child_node_t_copy> movelist;
			child_node_t *uct_child = uct_node[current_root].child;
			for (int i = 0; i < uct_node[current_root].child_num; i++) {
				if (double(uct_child[i].move_count) / uct_node[current_root].move_count > 0.1) { // 閾値
					movelist.emplace_back(uct_child[i]);
				}
			}
			if (movelist.size() == 0) {
				std::cerr << "Error: move_count" << std::endl;
				exit(EXIT_FAILURE);
			}
			std::cout << "movelist.size: " << movelist.size() << std::endl;

			std::sort(movelist.begin(), movelist.end(), [](auto left, auto right) {
				return left.move_count > right.move_count;
			});

			for (auto child : movelist) {
				// 定跡追加
				BookEntry be;
				float wintrate = child.win / child.move_count;
				be.score = Score(int(-logf(1.0f / wintrate - 1.0f) * 754.3f));
				be.key = key;
				be.fromToPro = static_cast<u16>(child.move.proFromAndTo());
				be.count = child.move_count;
				outMap[key].push_back(be);

				count++;

				StateInfo state;
				pos.doMove(child.move, state);

				// 次の手を探索
				if (depth + 1 < limitDepth)
					make_book_inner(pos, bookKeys, outMap, count, depth + 1, isBlack, limitDepth);

				pos.undoMove(child.move);
			}
		}
		else {
			// 探索済みの場合
			if (depth + 1 >= limitDepth)
				return;

			for (auto entry : outMap[key]) {
				Move move = Move::moveNone();
				const Move tmp = Move(entry.fromToPro);
				const Square to = tmp.to();
				if (tmp.isDrop()) {
					const PieceType ptDropped = tmp.pieceTypeDropped();
					move = makeDropMove(ptDropped, to);
				}
				else {
					const Square from = tmp.from();
					const PieceType ptFrom = pieceToPieceType(pos.piece(from));
					const bool promo = tmp.isPromotion();
					if (promo)
						move = makeCapturePromoteMove(ptFrom, from, to, pos);
					else
						move = makeCaptureMove(ptFrom, from, to, pos);
				}

				StateInfo state;
				pos.doMove(move, state);

				// 次の手を探索
				make_book_inner(pos, bookKeys, outMap, count, depth + 1, isBlack, limitDepth);

				pos.undoMove(move);
			}
		}
	}
	else {
		// 定跡を使用
		if (depth + 1 >= limitDepth)
			return;

		// 合法手一覧
		for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
			StateInfo state;
			pos.doMove(ml.move(), state);

			// 局面が定跡にあるか確認
			auto itr = bookKeys.find(Book::bookKey(pos));
			if (itr != bookKeys.end()) {
				// 同じ局面を探索しないようにキーを削除
				bookKeys.erase(itr);

				// 次の手を探索
				make_book_inner(pos, bookKeys, outMap, count, depth + 1, isBlack, limitDepth);
			}

			pos.undoMove(ml.move());
		}
	}
}

// 定跡作成
void make_book(std::istringstream& ssCmd) {
	// isreadyを先に実行しておくこと。

	std::string bookFileName;
	std::string outFileName;
	int limitDepth;
	int playoutNum;
	std::map<Key, std::vector<BookEntry> > outMap;

	ssCmd >> bookFileName;
	ssCmd >> outFileName;
	ssCmd >> limitDepth;
	ssCmd >> playoutNum;

	// プレイアウト数固定
	SetMode(CONST_PLAYOUT_MODE);
	SetPlayout(playoutNum);
	InitializeSearchSetting();
	SetReuseSubtree(false);

	// 定跡読み込み
	std::set<Key> bookKeys;
	std::ifstream ifs(bookFileName.c_str(), std::ifstream::in | std::ifstream::binary);
	if (!ifs) {
		std::cerr << "Error: cannot open " << bookFileName << std::endl;
		exit(EXIT_FAILURE);
	}
	BookEntry entry;
	while (ifs.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
		bookKeys.insert(entry.key);
	}
	std::cout << "bookKeys.size: " << bookKeys.size() << std::endl;

	// outFileが存在するときは追加する
	{
		std::ifstream ifsOutFile(outFileName.c_str(), std::ios::binary);
		if (ifsOutFile) {
			BookEntry entry;
			while (ifsOutFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
				outMap[entry.key].push_back(entry);
			}
			std::cout << "outMap.size: " << outMap.size() << std::endl;
		}
	}

	// 初期局面
	Searcher s;
	s.init();
	const std::string options[] = { "name Threads value 1",
		"name MultiPV value 1",
		"name USI_Hash value 256",
		"name OwnBook value false",
		"name Max_Random_Score_Diff value 0" };
	for (auto& str : options) {
		std::istringstream is(str);
		s.setOption(is);
	}
	Position pos(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);

	// 探索
	int count = 0;
	make_book_inner(pos, bookKeys, outMap, count, 0, true, limitDepth);
	int black_num = outMap.size();

	pos.set(DefaultStartPositionSFEN, s.threads.main());
	make_book_inner(pos, bookKeys, outMap, count, 0, false, limitDepth);
	int white_num = outMap.size() - black_num;

	std::cout << "black\t" << black_num << std::endl;
	std::cout << "white\t" << white_num << std::endl;
	std::cout << "sum\t" << black_num + white_num << std::endl;

	// 保存
	std::ofstream ofs(outFileName.c_str(), std::ios::binary);
	for (auto& elem : outMap) {
		for (auto& elel : elem.second)
			ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
	}
}

#endif