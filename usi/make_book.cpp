#ifdef MAKE_BOOK
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
#include "make_book.h"
#include "USIBookEngine.h"

#include <filesystem>
#include <regex>
#include <omp.h>

struct child_node_t_copy {
	Move move;       // 着手する座標
	int move_count;  // 探索回数
	WinType win;       // 勝った回数

	child_node_t_copy(const child_node_t& child) {
		this->move = child.move;
		this->move_count = child.move_count;
		this->win = child.win;
	}
	bool IsWin() const { return move.value() & VALUE_WIN; }
	bool IsLose() const { return move.value() & VALUE_LOSE; }
};

std::unordered_map<Key, std::vector<BookEntry> > bookMap;
extern std::unique_ptr<NodeTree> tree;
int make_book_sleep = 0;
bool use_interruption = true;
int book_eval_threshold = INT_MAX;
double book_visit_threshold = 0.005;
double book_cutoff = 0.015;
double book_reciprocal_temperature = 1.0;
bool book_best_move = false;
Score book_eval_diff = Score(10000);
// MinMaxで選ぶ確率
double book_minmax_prob = 1.0;
double book_minmax_prob_opp = 0.1;
std::uniform_real_distribution<double> dist_minmax(0, 1);
// MinMaxのために相手定跡の手番でも探索する
bool make_book_for_minmax = false;
// 一定の確率でPriorityBookから確率的に選ぶ
double book_priority_prob = 0;
double book_priority_prob_temperature = 0.01;
int book_priority_minmax_depth = 1;
double book_priority_prob_opp = 0;
double book_priority_prob_temperature_opp = 5.0;
// 千日手の評価値
extern float draw_value_black;
extern float draw_value_white;
extern float eval_coef;
Score draw_score_black;
Score draw_score_white;
// 相手定跡から外れた場合USIエンジンを使う
std::deque<std::unique_ptr<USIBookEngine>> usi_book_engines;
int usi_book_engine_nodes;
double usi_book_engine_prob = 1.0;
// 自分の手番でも一定確率でUSIエンジンを使う
int usi_book_engine_nodes_own;
double usi_book_engine_prob_own = 0.0;
// αβ探索で特定局面の評価値を置き換える
std::unordered_map<Key, Score> book_key_eval_map;

// 定跡用mutex
std::mutex gpu_mutex;
std::mutex usi_mutex;
std::condition_variable usi_cond;


// MinMaxの探索順に使用する定跡読み込み
bool read_minmax_priority_book(const std::string& minmax_priority_book, std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest, const bool force_read) {
	if (minmax_priority_book == "")
		return false;

	if (!force_read) {
		// ファイル更新がある場合のみ処理する
		static std::filesystem::file_time_type prev_time{};
		std::error_code ec;
		const std::filesystem::file_time_type file_time = std::filesystem::last_write_time(minmax_priority_book, ec);
		if (file_time == prev_time)
			return false;
		prev_time = file_time;
	}

	bookMapBest.clear();
	std::cout << "read minmax_priority_book" << std::endl;
	read_book(minmax_priority_book, bookMapBest);
	return true;
}

void copy_minmax_priority_book(const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBestSrc, std::unordered_map<Key, std::vector<BookEntry> >& bookMapBestDst) {
	bookMapBestDst.clear();
	for (auto itr = bookMapBestSrc.begin(); itr != bookMapBestSrc.end(); itr++) {
		auto& entries = bookMapBestDst[itr->first];
		for (const auto& entry : itr->second)
			entries.emplace_back(entry);
	}
}

std::unique_ptr<USIBookEngine> get_usi_book_engine() {
	std::unique_lock<std::mutex> lock(usi_mutex);
	usi_cond.wait(lock, [] { return !usi_book_engines.empty(); });
	std::unique_ptr<USIBookEngine> front = std::move(usi_book_engines.front());
	usi_book_engines.pop_front();
	return std::move(front);
}

void reuse_usi_book_engine(std::unique_ptr<USIBookEngine> usi_engine) {
	std::lock_guard<std::mutex> lock(usi_mutex);
	usi_book_engines.emplace_back(std::move(usi_engine));
	usi_cond.notify_one();
}

inline Move UctSearchGenmoveNoPonder(Position* pos, const std::vector<Move>& moves, const Key& book_starting_pos_key) {
	Move move;
	return UctSearchGenmove(pos, book_starting_pos_key, moves, move);
}

bool make_book_entry_with_uct(Position& pos, LimitsType& limits, const Key& key, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const std::vector<Move>& moves, const std::string& book_pos_cmd, const Key& book_starting_pos_key) {
	std::unique_lock<std::mutex> gpu_lock(gpu_mutex);
	std::cout << omp_get_thread_num() << "# " << book_pos_cmd;
	for (Move move : moves) {
		std::cout << " " << move.toUSI();
	}
	std::cout << std::endl;

	// UCT探索を使用
	limits.startTime.restart();
	SetLimits(limits);
	UctSearchGenmoveNoPonder(&pos, moves, book_starting_pos_key);

	const uct_node_t* current_root = tree->GetCurrentHead();
	if (current_root->child_num == 0) {
		// 詰みの局面
		return false;
	}

	// 探索回数で降順ソート
	std::vector<child_node_t_copy> movelist;
	int num = 0;
	const child_node_t* uct_child = current_root->child.get();
	for (int i = 0; i < current_root->child_num; i++) {
		movelist.emplace_back(uct_child[i]);
	}

	std::sort(movelist.begin(), movelist.end(), [](auto left, auto right) {
		return left.move_count > right.move_count;
		});

	const auto cutoff_threshold = movelist[0].win / movelist[0].move_count - book_cutoff;
	for (const auto& child : movelist) {
		if (double(child.move_count) / current_root->move_count <= book_visit_threshold) // 訪問回数閾値
			break;
		if (child.win / child.move_count < cutoff_threshold) // 勝率閾値
			break;
		num++;
	}
	if (num == 0) {
		num = (current_root->child_num + 2) / 3;
	}

	std::cout << "movelist.size: " << num << std::endl;
	gpu_lock.unlock();

	auto& entries = outMap[key];
	for (int i = 0; i < num; i++) {
		const auto& child = movelist[i];
		// 定跡追加
		BookEntry be;
		const auto wintrate = child.win / child.move_count;
		if (child.IsWin() || wintrate == 0.0) {
			be.score = -ScoreMaxEvaluate;
		}
		else if (child.IsLose() || wintrate == 1.0) {
			be.score = ScoreMaxEvaluate;
		}
		else {
			be.score = Score(int(-log(1.0 / wintrate - 1.0) * 754.3));
		}
		be.key = key;
		be.fromToPro = static_cast<u16>(child.move.proFromAndTo());
		be.count = (u16)((double)child.move_count / (double)current_root->move_count * 1000.0);
		entries.emplace_back(be);

		count++;
	}

	if (make_book_sleep > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(make_book_sleep));

	return true;
}

// min-max(αβ)で選択
/*std::vector<Move> debug_moves;
void print_debug_moves(Score score= ScoreNotEvaluated) {
	std::cout << "position startpos moves";
	for (int i = 0; i < debug_moves.size(); i++) {
		std::cout << " " << debug_moves[i].toUSI();
	}
	std::cout << "\t" << score << std::endl;
}*/
struct Searched {
	int depth;
	Score score;
	Score beta;
};
Score book_search(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, Score alpha, const Score beta, const Score score, std::map<Key, Searched>& searched, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	const Key key = Book::bookKey(pos);

	// 特定局面の評価値を置き換える
	if (book_key_eval_map.size() > 0 && book_key_eval_map.find(key) != book_key_eval_map.end()) {
		//std::cout << pos.toSFEN() << std::endl;
		return -book_key_eval_map[key];
	}

	// 探索済みチェック
	const auto itr_searched = searched.find(key);
	// 深さが同じか浅い場合のみ再利用
	/*if (std::abs(itr_searched->second.score) == 71 && std::abs(beta) == 171) {
		std::cout << key << "\t" << itr_searched->second.depth << "\t" << pos.gamePly() << "\t" << itr_searched->second.beta << "\t" << beta << "\t" << itr_searched->second.score << std::endl;
		__debugbreak();
	}*/
	if (itr_searched != searched.end() && itr_searched->second.depth <= pos.gamePly() && itr_searched->second.beta >= beta) {
		/*if (key == 12668901208309554908UL)
			std::cout << itr_searched->second.depth << "\t" << pos.gamePly() << "\t" << itr_searched->second.beta << "\t" << beta << std::endl;*/
		return -itr_searched->second.score;
	}
	
	const auto itr = outMap.find(key);
	if (itr == outMap.end()) {
		// エントリがない場合、自身の評価値を返す
		return score;
	}
	const auto& entries = itr->second;
	Score value = -ScoreInfinite;

	// MinMaxの探索順に使用する定跡
	auto itr_best = bookMapBest.end();
	if (bookMapBest.size() > 0) {
		itr_best = bookMapBest.find(key);
		if (itr_best != bookMapBest.end()) {
			for (const auto& entry : itr_best->second) {
				Move move = move16toMove(Move(entry.fromToPro), pos);
				StateInfo state;
				pos.doMove(move, state);
				//debug_moves.emplace_back(move);
				switch (pos.isDraw()) {
				case RepetitionDraw:
					// 繰り返しになる場合、千日手の評価値
					value = pos.turn() == Black ? draw_score_white : draw_score_black;
					break;
				case RepetitionWin:
					value = -ScoreInfinite;
					break;
				case RepetitionLose:
					value = ScoreMaxEvaluate;
					break;
				default:
					value = book_search(pos, outMap, -beta, -alpha, entry.score, searched, bookMapBest);
				}
				pos.undoMove(move);
				//debug_moves.pop_back();
				//std::cout << move.toUSI() << "\t" << entry.score << "\t" << value << std::endl;

				alpha = std::max(alpha, value);
				if (alpha >= beta) {
					searched[key] = { pos.gamePly(), value, beta };
					return -value;
				}
			}
		}
	}

	Score trusted_score = entries[0].score;
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		if (itr_best != bookMapBest.end()) {
			const auto& entries_best = itr_best->second;
			if (std::find_if(entries_best.begin(), entries_best.end(),
				[&entry](const BookEntry& entry_best) { return entry_best.fromToPro == entry.fromToPro; }) != entries_best.end()) {
				// 探索済み
				continue;
			}
		}
		// 訪問回数が少ない評価値は信頼しない
		if (entry.score < trusted_score)
			trusted_score = entry.score;
		//std::cout << pos.turn() << "\t" << move.toUSI() << std::endl;
		/*if (key == 7172278114399076909UL && debug_moves[0].toUSI() == "4b5b" && move.toUSI() == "4a5b") {
			print_debug_moves(entry.score);
			__debugbreak();
		}*/
		StateInfo state;
		pos.doMove(move, state);
		//debug_moves.emplace_back(move);
		switch (pos.isDraw()) {
		case RepetitionDraw:
			// 繰り返しになる場合、千日手の評価値
			value = std::max(value, pos.turn() == Black ? draw_score_white : draw_score_black);
			break;
		case RepetitionWin:
			// 相手の勝ち(自分の負け)のためvalueを更新しない
			break;
		case RepetitionLose:
			// 相手の負け(自分の勝ち)
			value = ScoreMaxEvaluate;
			break;
		default:
			value = std::max(value, book_search(pos, outMap, -beta, -alpha, trusted_score, searched, bookMapBest));
		}
		pos.undoMove(move);
		//debug_moves.pop_back();

		/*if (key == 10806437342126107775UL && debug_moves.size() > 8 && debug_moves[8].toUSI() == "6c5d")
			std::cout << "***\t" << move.toUSI() << "\t" << value << "\t" << alpha << "\t" << beta << std::endl;*/

		alpha = std::max(alpha, value);
		if (alpha >= beta) {
			//if (debug_moves.size() > 14 && debug_moves[14] == Move(10437) && value == 127) print_debug_moves(value);
			//if (itr_searched == searched.end() || itr_searched->second.depth >= pos.gamePly() && itr_searched->second.beta <= beta) {
			searched[key] = { pos.gamePly(), value, beta };
			//}
			return -value;
		}
	}
	for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
		const Move& move = ml.move();
		const u16 move16 = (u16)(move.value());
		if (std::any_of(entries.begin(), entries.end(), [move16](const BookEntry& entry) { return entry.fromToPro == move16; }))
			continue;
		if (outMap.find(Book::bookKeyAfter(pos, key, move)) == outMap.end())
			continue;
		if (itr_best != bookMapBest.end()) {
			const auto& entries_best = itr_best->second;
			if (std::find_if(entries_best.begin(), entries_best.end(),
				[&move16](const BookEntry& entry_best) { return entry_best.fromToPro == move16; }) != entries_best.end()) {
				// 探索済み
				continue;
			}
		}
		StateInfo state;
		pos.doMove(move, state);
		//debug_moves.emplace_back(move);
		switch (pos.isDraw()) {
		case RepetitionDraw:
			// 繰り返しになる場合、千日手の評価値
			value = std::max(value, pos.turn() == Black ? draw_score_white : draw_score_black);
			break;
		case RepetitionWin:
			// 相手の勝ち(自分の負け)のためvalueを更新しない
			break;
		case RepetitionLose:
			// 相手の負け(自分の勝ち)
			value = ScoreMaxEvaluate;
			break;
		default:
			const auto ret = book_search(pos, outMap, -beta, -alpha, ScoreNotEvaluated, searched, bookMapBest);
			value = std::max(value, ret);
		}
		pos.undoMove(move);
		//debug_moves.pop_back();

		/*if (key == 10806437342126107775UL)
			std::cout << "***\t" << move.toUSI() << "\t" << value << std::endl;*/

		alpha = std::max(alpha, value);
		if (alpha >= beta) {
			//if (debug_moves.size() > 14 && debug_moves[14] == Move(10437) && value == 127) print_debug_moves(value);
			//if (itr_searched == searched.end() || itr_searched->second.depth >= pos.gamePly() && itr_searched->second.beta <= beta) {
			searched[key] = { pos.gamePly(), value, beta };
			//}
			return -value;
		}
	}
	//if (std::abs(value) == 71) print_debug_moves(value);

	// βカットされなかった場合はsearchedに追加しない
	return -alpha;
}

std::tuple<int, Move, Score> select_best_book_entry(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	const Key key = Book::bookKey(pos);

	Score alpha = -ScoreInfinite;
	Move bestMove = Move::moveNone();
	int bestIndex = -1;
	std::map<Key, Searched> searched;

	// MinMaxの探索順に使用する定跡
	Move topMove = Move::moveNone();
	auto itr_best = bookMapBest.end();
	if (bookMapBest.size() > 0) {
		itr_best = bookMapBest.find(key);
		if (itr_best != bookMapBest.end()) {
			for (const auto& entry : itr_best->second) {
				Move move = move16toMove(Move(entry.fromToPro), pos);
				StateInfo state;
				pos.doMove(move, state);
				//debug_moves.emplace_back(move);
				Score value;
				switch (pos.isDraw()) {
				case RepetitionDraw:
					// 繰り返しになる場合、千日手の評価値
					value = pos.turn() == Black ? draw_score_white : draw_score_black;
					break;
				case RepetitionWin:
					value = -ScoreInfinite;
					break;
				case RepetitionLose:
					value = ScoreMaxEvaluate;
					break;
				default:
					value = book_search(pos, outMap, -ScoreInfinite, -alpha, entry.score, searched, bookMapBest);
				}
				pos.undoMove(move);
				//debug_moves.pop_back();
				//std::cout << move.toUSI() << "\t" << entry.score << "\t" << value << std::endl;

				if (value > alpha) {
					bestMove = move;
					alpha = value;
				}
			}
			topMove = bestMove;
		}
	}

	Score trusted_score = entries[0].score;
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		if (itr_best != bookMapBest.end()) {
			const auto& entries_best = itr_best->second;
			if (std::find_if(entries_best.begin(), entries_best.end(),
				[&entry](const BookEntry& entry_best) { return entry_best.fromToPro == entry.fromToPro; }) != entries_best.end()) {
				// 探索済み
				if (move == topMove && move == bestMove)
					bestIndex = (int)(&entry - &entries[0]);
				continue;
			}
		}
		// 訪問回数が少ない評価値は信頼しない
		if (entry.score < trusted_score)
			trusted_score = entry.score;
		//std::cout << pos.turn() << "\t" << move.toUSI() << std::endl;
		StateInfo state;
		pos.doMove(move, state);
		//debug_moves.emplace_back(move);
		Score value;
		switch (pos.isDraw()) {
		case RepetitionDraw:
			// 繰り返しになる場合、千日手の評価値
			value = pos.turn() == Black ? draw_score_white : draw_score_black;
			break;
		case RepetitionWin:
			value = -ScoreInfinite;
			break;
		case RepetitionLose:
			value = ScoreMaxEvaluate;
			break;
		default:
			value = book_search(pos, outMap, -ScoreInfinite, -alpha, trusted_score, searched, bookMapBest);
		}
		pos.undoMove(move);
		//debug_moves.pop_back();
		//std::cout << move.toUSI() << "\t" << entry.score << "\t" << value << std::endl;

		if (value > alpha) {
			bestIndex = (int)(&entry - &entries[0]);
			bestMove = move;
			alpha = value;
		}
	}
	for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
		const Move& move = ml.move();
		const u16 move16 = (u16)(move.value());
		if (std::any_of(entries.begin(), entries.end(), [move16](const BookEntry& entry) { return entry.fromToPro == move16; }))
			continue;
		if (outMap.find(Book::bookKeyAfter(pos, key, move)) == outMap.end())
			continue;
		if (move == topMove)
			continue;
		StateInfo state;
		pos.doMove(move, state);
		//debug_moves.emplace_back(move);
		Score value;
		switch (pos.isDraw()) {
		case RepetitionDraw:
			// 繰り返しになる場合、千日手の評価値
			value = pos.turn() == Black ? draw_score_white : draw_score_black;
			break;
		case RepetitionWin:
			value = -ScoreInfinite;
			break;
		case RepetitionLose:
			value = ScoreMaxEvaluate;
			break;
		default:
			const auto ret = book_search(pos, outMap, -ScoreInfinite, -alpha, ScoreNotEvaluated, searched, bookMapBest);
			value = ret;
		}
		pos.undoMove(move);
		//debug_moves.pop_back();
		//std::cout << move.toUSI() << "\t" << ScoreNotEvaluated << "\t" << value << std::endl;

		if (value > alpha) {
			bestIndex = -1;
			bestMove = move;
			alpha = value;
		}
	}
	return { bestIndex, bestMove, alpha };
}

Score minmax_priority_book_score(Position& pos, const Key key, const Move move, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest, Score alpha, Score beta, int depth) {
	const auto itr_best = bookMapBest.find(key);

	if (itr_best == bookMapBest.end())
		return ScoreNone;

	StateInfo st;
	pos.doMove(move, st);

	Score trustedScore = ScoreInfinite;
	const auto& entries = itr_best->second;
	for (auto& entry : entries) {
		const Move move16 = Move(entry.fromToPro);
		Score score = entry.score;

		// 回数が少ない評価値は信頼しない
		if (score < trustedScore)
			trustedScore = score;
		score = trustedScore;

		switch (pos.moveIsDraw(move16)) {
		case RepetitionDraw:
		{
			// 千日手の評価で上書き
			score = pos.turn() == Black ? draw_score_black : draw_score_white;
			break;
		}
		case RepetitionWin:
			// 相手の勝ち(自分の負け)
			score = -ScoreInfinite;
			break;
		case RepetitionLose:
			// 相手の負け(自分の勝ち)
			score = ScoreMaxEvaluate;
			break;
		default:
			if (depth > 0) {
				const Move move2 = move16toMove(move16, pos);
				const Score ret = minmax_priority_book_score(pos, Book::bookKeyAfter(pos, key, move2), move2, bookMapBest, -beta, -alpha, depth - 1);
				if (ret != ScoreNone)
					score = -ret;
			}
			break;
		}

		alpha = std::max(alpha, score);
		if (alpha >= beta) {
			break;
		}
	}
	pos.undoMove(move);
	return alpha;
}

// PriorityBookから確率的に選ぶ
std::tuple<Move, Score> select_priority_book_entry(Position& pos, const Key key, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest, double temperature) {
	std::vector<Move> moves_priority;
	std::vector<double> probabilities;
	const auto itr_best = bookMapBest.find(key);
	const auto max_score = itr_best->second[0].score;
	const auto& entries = itr_best->second;
	Score trustedScore = ScoreInfinite;
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		Score score = entry.score;

		// 回数が少ない評価値は信頼しない
		if (score < trustedScore)
			trustedScore = score;
		score = trustedScore;

		switch (pos.moveIsDraw(move)) {
		case RepetitionDraw:
		{
			// 千日手の評価で上書き
			score = pos.turn() == Black ? draw_score_black : draw_score_white;
			break;
		}
		case RepetitionWin:
			// 相手の勝ち(自分の負け)
			score = -ScoreInfinite;
			break;
		case RepetitionLose:
			// 相手の負け(自分の勝ち)
			score = ScoreMaxEvaluate;
			break;
		default:
			const Score ret = minmax_priority_book_score(pos, Book::bookKeyAfter(pos, key, move), move, bookMapBest, -ScoreInfinite, ScoreInfinite, book_priority_minmax_depth - 1);
			if (ret != ScoreNone)
				score = -ret;
			break;
		}

		moves_priority.emplace_back(move);
		probabilities.emplace_back(std::exp((int)(score - max_score) / temperature));
	}
	std::discrete_distribution<std::size_t> dist(probabilities.begin(), probabilities.end());
	const auto selected_index = dist(g_randomTimeSeed);
	return { moves_priority[selected_index], entries[selected_index].score };
}

// 定跡作成(再帰処理)
void make_book_inner(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest, const std::string& book_pos_cmd, const Key& book_starting_pos_key) {
	const Key key = Book::bookKey(pos);
	if ((depth % 2 == 0) == isBlack) {

		const auto itr = outMap.find(key);
		if (itr == outMap.end()) {
			// 先端ノード
			// UCT探索で定跡作成
			make_book_entry_with_uct(pos, limits, key, outMap, count, moves, book_pos_cmd, book_starting_pos_key);
		}
		else {
			// 探索済みの場合
			{
				Move move;
				if (dist_minmax(g_randomTimeSeed) < usi_book_engine_prob_own) {
					// 自分の手番でも一定確率でUSIエンジンを使う
					auto usi_book_engine = get_usi_book_engine();
					const auto usi_result = usi_book_engine->Go(book_pos_cmd, moves, usi_book_engine_nodes_own);
					reuse_usi_book_engine(std::move(usi_book_engine));
					std::cout << omp_get_thread_num() << "# usi move : " << depth << " " << usi_result.info << std::endl;
					if (usi_result.bestMove == "resign" || usi_result.bestMove == "win")
						return;
					move = usiToMove(pos, usi_result.bestMove);
				}
				else {
					const auto& entries = itr->second;
					int index = 0;
					Score score;
					if (bookMapBest.size() > 0 && dist_minmax(g_randomTimeSeed) < book_priority_prob && bookMapBest.find(key) != bookMapBest.end()) {
						// 一定の確率でPriorityBookから確率的に選ぶ
						std::tie(move, score) = select_priority_book_entry(pos, key, bookMapBest, book_priority_prob_temperature);
					}
					else {
						// 一定の確率でmin-maxで選ぶ
						std::tie(index, move, score) = (dist_minmax(g_randomTimeSeed) < book_minmax_prob) ? select_best_book_entry(pos, outMap, entries, moves, bookMapBest) : std::make_tuple(0, move16toMove(Move(entries[0].fromToPro), pos), entries[0].score);
					}

					// 評価値が閾値を超えた場合、探索終了
					if (std::abs(score) > book_eval_threshold) {
						std::cout << omp_get_thread_num() << "# " << book_pos_cmd;
						for (Move move : moves) {
							std::cout << " " << move.toUSI();
						}
						std::cout << "\nentry.score: " << score << std::endl;
						return;
					}

					if (index != 0)
						std::cout << omp_get_thread_num() << "# best move : " << depth << " " << index << " " << move.toUSI() << std::endl;
				}

				StateInfo state;
				pos.doMove(move, state);
				// 繰り返しになる場合、探索終了
				switch (pos.isDraw()) {
				case RepetitionDraw:
				case RepetitionWin:
				case RepetitionLose:
					pos.undoMove(move);
					return;
				}

				moves.emplace_back(move);

				// 次の手を探索
				make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves, select_best_book_entry, bookMapBest, book_pos_cmd, book_starting_pos_key);

				pos.undoMove(move);
			}
		}
	}
	else {
		// MinMaxのために相手定跡の手番でも探索する
		if (make_book_for_minmax) {
			const auto itr_out = outMap.find(key);
			if (itr_out == outMap.end()) {
				// 未探索の局面の場合
				// UCT探索で定跡作成
				if (!make_book_entry_with_uct(pos, limits, key, outMap, count, moves, book_pos_cmd, book_starting_pos_key))
				{
					// 詰みの局面の場合何もしない
					return;
				}
			}
		}

		Move move;
		const auto itr = bookMap.find(key);
		if (itr == bookMap.end() && usi_book_engine_nodes > 0 && dist_minmax(g_randomTimeSeed) < usi_book_engine_prob) {
			// 相手定跡から外れた場合USIエンジンを使う
			auto usi_book_engine = get_usi_book_engine();
			const auto usi_result = usi_book_engine->Go(book_pos_cmd, moves, usi_book_engine_nodes);
			reuse_usi_book_engine(std::move(usi_book_engine));
			std::cout << omp_get_thread_num() << "# usi move : " << depth << " " << usi_result.info << std::endl;
			if (usi_result.bestMove == "resign" || usi_result.bestMove == "win")
				return;
			move = usiToMove(pos, usi_result.bestMove);
		}
		else {
			// 定跡を使用
			const std::vector<BookEntry>* entries;

			// 局面が定跡にあるか確認
			if (itr != bookMap.end()) {
				entries = &itr->second;
			}
			else {
				// 定跡にない場合、探索結果を使う
				const auto itr_out = outMap.find(key);

				if (itr_out == outMap.end()) {
					// 定跡になく未探索の局面の場合
					// UCT探索で定跡作成
					if (!make_book_entry_with_uct(pos, limits, key, outMap, count, moves, book_pos_cmd, book_starting_pos_key))
					{
						// 詰みの局面の場合何もしない
						return;
					}
				}

				entries = &outMap[key];
			}

			if (itr == bookMap.end() && bookMapBest.size() > 0 && dist_minmax(g_randomTimeSeed) < book_priority_prob_opp && bookMapBest.find(key) != bookMapBest.end()) {
				// 一定の確率でPriorityBookから確率的に選ぶ
				Score score;
				std::tie(move, score) = select_priority_book_entry(pos, key, bookMapBest, book_priority_prob_temperature_opp);
			}
			else if (itr == bookMap.end() && dist_minmax(g_randomTimeSeed) < book_minmax_prob_opp) {
				// 一定の確率でmin-maxで選ぶ
				const auto& entry = select_best_book_entry(pos, outMap, *entries, moves, bookMapBest);
				move = std::get<Move>(entry);
			}
			else if (itr != bookMap.end() && book_best_move) {
				// 相手定跡の最善手を選択する
				size_t selected_index = 0;
				u16 max_count = entries->at(0).count;
				for (size_t i = 1; i < entries->size(); ++i) {
					const auto& entry = entries->at(i);
					if (entry.count > max_count) {
						selected_index = i;
						max_count = entry.count;
					}
				}
				move = move16toMove(Move(entries->at(selected_index).fromToPro), pos);
			}
			else {
				// 確率的に手を選択
				std::vector<double> probabilities;
				const Score score_th = entries->at(0).score - book_eval_diff;
				for (const auto& entry : *entries) {
					// 相手定跡の評価値閾値
					if (itr != bookMap.end() && entry.score < score_th)
						continue;
					const auto probability = std::pow((double)entry.count, book_reciprocal_temperature);
					probabilities.emplace_back(probability);
				}
				std::discrete_distribution<std::size_t> dist(probabilities.begin(), probabilities.end());
				const auto selected_index = dist(g_randomTimeSeed);
				move = move16toMove(Move(entries->at(selected_index).fromToPro), pos);
			}
		}

		StateInfo state;
		pos.doMove(move, state);
		moves.emplace_back(move);

		// 次の手を探索
		make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves, select_best_book_entry, bookMapBest, book_pos_cmd, book_starting_pos_key);

		pos.undoMove(move);
	}
}

void make_book_alpha_beta(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest, const std::string& book_pos_cmd, const Key& book_starting_pos_key) {
	std::vector<Move> moves;
	make_book_inner(pos, limits, bookMap, outMap, count, depth, isBlack, moves, select_best_book_entry, bookMapBest, book_pos_cmd, book_starting_pos_key);
}

// 定跡読み込み
void read_book(const std::string& bookFileName, std::unordered_map<Key, std::vector<BookEntry> >& bookMap) {
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

// 定跡マージ
int merge_book(std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::string& merge_file, const bool check_file_time) {
	if (check_file_time) {
		// ファイル更新がある場合のみ処理する
		static std::filesystem::file_time_type prev_time{};
		std::error_code ec;
		const std::filesystem::file_time_type file_time = std::filesystem::last_write_time(merge_file, ec);
		if (file_time == prev_time)
			return 0;
		prev_time = file_time;
	}

	std::ifstream ifsMerge(merge_file.c_str(), std::ios::binary);
	int merged = 0;
	if (ifsMerge) {
		BookEntry entry;
		Key prev_key = 0;
		while (ifsMerge.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
			if (entry.key == 0) {
				std::cerr << "book file is corrupted: " << merge_file << std::endl;
				throw std::runtime_error("book file is corrupted");
			}
			if (entry.key == prev_key || outMap.find(entry.key) == outMap.end()) {
				if (entry.key != prev_key)
					merged++;
				outMap[entry.key].emplace_back(entry);
				prev_key = entry.key;
			}
		}
	}
	std::cout << "merged: " << merged << std::endl;
	return merged;
}

void merge_book_map(std::unordered_map<Key, std::vector<BookEntry> >& dstMap, const std::unordered_map<Key, std::vector<BookEntry> >& srctMap) {
	for (const auto& src : srctMap) {
		auto itr_dst = dstMap.find(src.first);
		if (itr_dst == dstMap.end()) {
			auto& entries = dstMap[src.first];
			entries.reserve(src.second.size());
			std::copy(src.second.begin(), src.second.end(), std::back_inserter(entries));
		}
	}
}

void minmax_book_white(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest);

void minmax_book_black(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	// αβで探索
	const Key key = Book::bookKey(pos);

	// 探索済みの場合、深さが同じか浅い場合、打ち切る
	const auto itr_minmax = bookMapMinMax.find(key);
	if (itr_minmax != bookMapMinMax.end() && itr_minmax->second.depth <= pos.gamePly()) {
		return;
	}

	const auto itr = bookMap.find(key);
	if (itr == bookMap.end()) {
		// エントリがない
		return;
	}

	const std::vector<BookEntry>& entries = itr->second;

	int index;
	Move bestMove;
	Score score;
	std::tie(index, bestMove, score) = select_best_book_entry(pos, bookMap, entries, moves, bookMapBest);

	// 最善手を登録
	auto& minMaxBookEntry = bookMapMinMax[key];
	minMaxBookEntry.move16 = (u16)bestMove.value();
	minMaxBookEntry.score = score;
	minMaxBookEntry.depth = pos.gamePly();

	// 最善手を指す
	StateInfo state;
	pos.doMove(bestMove, state);
	moves.emplace_back(bestMove);
	minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
	moves.pop_back();
	pos.undoMove(bestMove);
}

void minmax_book_white(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	// すべての手を試す
	const Key key = Book::bookKey(pos);

	const auto itr = bookMap.find(key);
	if (itr == bookMap.end()) {
		// エントリがない
		return;
	}

	const std::vector<BookEntry>& entries = itr->second;

	std::vector<Move> candidates;
	candidates.reserve(entries.size());
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		candidates.emplace_back(move);
	}
	for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
		const Move& move = ml.move();
		if (std::find(candidates.begin(), candidates.end(), move) != candidates.end())
			continue;
		if (bookMap.find(Book::bookKeyAfter(pos, key, move)) == bookMap.end())
			continue;
		candidates.emplace_back(move);
	}

	for (size_t i = 0; i < candidates.size(); ++i) {
		const Move move = candidates[i];
		StateInfo state;
		pos.doMove(move, state);
		switch (pos.isDraw()) {
		case RepetitionDraw:
		case RepetitionWin:
		case RepetitionLose:
			// 繰り返しになる場合
			pos.undoMove(move);
			continue;
		default:
			moves.emplace_back(move);
			minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
			moves.pop_back();
		}
		pos.undoMove(move);
	}
}

void make_minmax_book(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, const Color make_book_color, select_best_book_entry_t select_best_book_entry, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	std::vector<Move> moves;
	if (make_book_color == Black)
		minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
	else if (make_book_color == White)
		minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
	else {
		minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
		minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry, bookMapBest);
	}
}

void saveOutmap(const std::string& outFileName, const std::unordered_map<Key, std::vector<BookEntry> >& outMap) {
	// キーをソート
	std::set<Key> keySet;
	for (auto& elem : outMap) {
		keySet.emplace(elem.first);
	}

	std::ofstream ofs(outFileName.c_str(), std::ios::binary);
	for (const Key key : keySet) {
		const auto itr = outMap.find(key);
		const auto& elem = *itr;
		for (auto& elel : elem.second)
			ofs.write(reinterpret_cast<const char*>(&(elel)), sizeof(BookEntry));
	}
}

void saveOutmap(const std::string& outFileName, const std::map<Key, std::vector<BookEntry> >& outMap) {
	std::ofstream ofs(outFileName.c_str(), std::ios::binary);
	for (auto& elem : outMap) {
		for (auto& elel : elem.second) {
			ofs.write(reinterpret_cast<const char*>(&(elel)), sizeof(BookEntry));
		}
	}
}

std::string get_book_pv_inner(Position& pos, const std::string& fileName, Book& book) {
	const auto bookMoveScore = book.probeConsideringDraw(pos, fileName);
	const Move move = std::get<0>(bookMoveScore);
	if (move) {
		StateInfo state;
		pos.doMove(move, state);
		switch (pos.isDraw()) {
		case RepetitionDraw:
		case RepetitionWin:
		case RepetitionLose:
			pos.undoMove(move);
			return move.toUSI();
		}
		const auto moves = get_book_pv_inner(pos, fileName, book);
		pos.undoMove(move);
		return move.toUSI() + " " + moves;
	}
	return "";
}

std::string getBookPV(Position& pos, const std::string& fileName) {
	Book book;
	return get_book_pv_inner(pos, fileName, book);
};

std::unique_ptr<USIBookEngine> create_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const bool random_nodes) {
	std::vector<std::pair<std::string, std::string>> usi_engine_options;
	std::istringstream ss(engine_options);
	std::string field;
	while (std::getline(ss, field, ',')) {
		const auto p = field.find_first_of(":");
		usi_engine_options.emplace_back(field.substr(0, p), field.substr(p + 1));
	}
	return std::make_unique<USIBookEngine>(engine_path, usi_engine_options, random_nodes);
}

void init_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const int nodes, const double prob, const int nodes_own, const double prob_own, const int num_engines, const bool random_nodes) {
	if (engine_path == "")
		return;
	for (int i = 0; i < num_engines; ++i) {
		usi_book_engines.emplace_back(create_usi_book_engine(engine_path, engine_options, random_nodes));
	}
	usi_book_engine_nodes = nodes;
	usi_book_engine_prob = prob;
	usi_book_engine_nodes_own = nodes_own;
	usi_book_engine_prob_own = prob_own;
}

void init_book_key_eval_map(const std::string& str) {
	book_key_eval_map.clear();
	if (str == "")
		return;

	std::istringstream ss(str);
	std::string field;
	while (std::getline(ss, field, ',')) {
		const auto p = field.find_first_of(":");
		const Key key = std::stoull(field.substr(0, p));
		const Score score = (Score)std::stoi(field.substr(p + 1));
		book_key_eval_map.emplace(key, score);
	}
}

// USIエンジンで局面を評価する
void eval_positions_with_usi_engine(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, const std::string& engine_path, const std::string& engine_options, const int nodes, const int engine_num, const std::string& outFileName) {
	// 全局面を列挙
	std::vector<std::pair<HuffmanCodedPos, const std::vector<BookEntry>*>> positions;
	{
		std::unordered_set<Key> exists;

		enumerate_positions(pos, bookMap, positions, exists, outMap);
	}

	std::cout << "positions: " << positions.size() << std::endl;

	const auto engine_num_ = std::min(engine_num, (int)positions.size());

	// USIエンジン初期化
	std::vector<std::unique_ptr<USIBookEngine>> usi_book_engines(engine_num_);
	#pragma omp parallel for num_threads(engine_num_)
	for (int i = 0; i < engine_num_; ++i) {
		usi_book_engines[omp_get_thread_num()] = create_usi_book_engine(engine_path, engine_options, false);
	}
	usi_book_engine_nodes = nodes;

	// 並列で評価
	const int positions_size = (int)positions.size();
	const std::vector<Move> moves = {};
	std::regex re(R"*(score +(cp|mate) +([+\-]?\d*))*");
	int count = 0;
	#pragma omp parallel for num_threads(engine_num_) schedule(dynamic)
	for (int i = 0; i < positions_size; ++i) {
		Position pos;
		pos.set(positions[i].first);
		const auto usi_result = usi_book_engines[omp_get_thread_num()]->Go("position " + pos.toSFEN(), moves, usi_book_engine_nodes);
		if (usi_result.bestMove == "resign" || usi_result.bestMove == "win")
			continue;
		const Key key = Book::bookKey(pos);
		BookEntry be;
		be.key = key;
		be.fromToPro = (u16)usiToMove(pos, usi_result.bestMove).value();
		be.count = 1;
		std::smatch m;
		if (std::regex_search(usi_result.info, m, re)) {
			if (m[1].str() == "cp") {
				be.score = (Score)std::stoi(m[2].str());
			}
			else { // mate
				if (m[2].str()[0] == '-')
					be.score = -ScoreMaxEvaluate;
				else
					be.score = ScoreMaxEvaluate;
			}
			#pragma omp critical
			{
				++count;
				outMap[key].emplace_back(be);
				if (count % 10000 == 0) {
					std::cout << "progress: " << count * 100 / positions_size << "%" << std::endl;
					saveOutmap(outFileName, outMap);
				}
			}
		}
	}
}

struct PositionWithMove {
	Key key;
	Move move;
	int depth;
	const PositionWithMove* parent;
};

// 局面を列挙する
void enumerate_positions_with_move(const Position& pos_root, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::vector<PositionWithMove>& positions) {
	// 最短経路をBFSで探索する
	std::unordered_set<Key> exists;

	std::vector<std::pair<HuffmanCodedPos, const PositionWithMove*>> current_positions;
	std::vector<std::pair<HuffmanCodedPos, const PositionWithMove*>> next_positions;

	const Key key_root = Book::bookKey(pos_root);
	PositionWithMove& potision_root = positions.emplace_back(PositionWithMove{ key_root, Move::moveNone(), 0, nullptr });
	current_positions.push_back({ pos_root.toHuffmanCodedPos(), &potision_root });
	exists.emplace(key_root);

	int depth = 1;
	while (current_positions.size() > 0) {
		for (const auto& position : current_positions) {
			const auto& hcp = position.first;
			const PositionWithMove* parent = position.second;

			Position pos;
			pos.set(hcp);

			// 定跡の指し手の順を優先する
			MoveList<LegalAll> ml(pos);
			std::vector<Move> moves;
			moves.reserve(ml.size());
			const auto itr_curr = bookMap.find(Book::bookKey(pos));
			const auto& entries = itr_curr->second;
			std::vector<Move> book_moves;
			book_moves.reserve(entries.size());
			for (const auto& entry : entries) {
				Move move = move16toMove(Move(entry.fromToPro), pos);
				moves.emplace_back(move);
				book_moves.emplace_back(move);
			}
			for (; !ml.end(); ++ml) {
				const auto move = ml.move();
				if (std::find(book_moves.cbegin(), book_moves.cend(), move) == book_moves.cend())
					moves.emplace_back(ml.move());
			}

			for (const auto& move : moves) {
				StateInfo state;
				pos.doMove(move, state);
				const Key key = Book::bookKey(pos);
				if (!exists.emplace(key).second) {
					// 合流
					pos.undoMove(move);
					continue;
				}
				auto itr = bookMap.find(key);
				if (itr != bookMap.end()) {
					// 追加
					PositionWithMove& potision_next = positions.emplace_back(PositionWithMove{ key, move, depth, parent });
					next_positions.push_back({ pos.toHuffmanCodedPos(), &potision_next });
				}
				pos.undoMove(move);
			}
		}

		current_positions = std::move(next_positions);
		++depth;
	}
}

// 評価値が割れる局面を延長する
void diff_eval(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, LimitsType& limits, const Score diff, const std::string& outFileName, const std::string& book_pos_cmd, const Key& book_starting_pos_key) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size() + 1); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	if (positions.size() > bookMap.size() + 1)
		throw std::runtime_error("positions.size() > bookMap.size()");

	// 評価値が割れる局面を延長する
	for (const auto& position : positions) {
		const Key key = position.key;

		// 評価値差分
		const auto itr_book = bookMap.find(key);
		if (itr_book != bookMap.end()) {
			const auto& itr = outMap.find(key);
			if (itr == outMap.end())
				continue;
			const auto& entry = itr->second[0];
			const Score score = entry.score;
			const auto& opp_entry = itr_book->second[0];
			const auto opp_score = std::min(std::max(opp_entry.score, -ScoreMaxEvaluate), ScoreMaxEvaluate);
			// 相手が詰みを見つけているか
			const bool opp_mate = std::abs(opp_score) >= 30000 && std::abs(score) < 30000;
			if ((score + 150) * opp_score < 0 || (score - 150) * opp_score < 0 || opp_mate) {
				// 評価値の符号が異なり、差がdiff以上、もしくは詰み
				if (std::abs(opp_score - score) >= diff || opp_mate) {
					Position pos_copy(pos);
					const PositionWithMove* position_ptr = &position;
					std::vector<Move> moves(position_ptr->depth);
					for (int j = position_ptr->depth - 1; j >= 0; --j) {
						moves[j] = position_ptr->move;
						position_ptr = position_ptr->parent;
					}
					assert(position_ptr->parent == nullptr);

					// move
					auto states = StateListPtr(new std::deque<StateInfo>(1));
					for (const Move move : moves) {
						states->emplace_back(StateInfo());
						pos_copy.doMove(move, states->back());
					}

					const Move move = (score < opp_score) ?
						// 悲観している局面では、相手の指し手を選ぶ
						move16toMove(Move(opp_entry.fromToPro), pos_copy) :
						// 楽観している局面では、自分の指し手を選ぶ
						move16toMove(Move(entry.fromToPro), pos_copy);
					Key key_after = Book::bookKeyAfter(pos_copy, key, move);
					if (outMap.find(key_after) == outMap.end()) {
						// 最善手が定跡にない場合
						std::cout << "diff: " << score << ", " << opp_entry.score << std::endl;
						// 最善手を指して、定跡を延長
						StateInfo state;
						int count = 0;
						pos_copy.doMove(move, state);
						moves.emplace_back(move);
						make_book_entry_with_uct(pos_copy, limits, key_after, outMap, count, moves, book_pos_cmd, book_starting_pos_key);
						// 保存
						saveOutmap(outFileName, outMap);
					}
				}
			}
		}
	}
}

// 全ての局面についてαβで定跡を作る
void make_all_minmax_book(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, const Color make_book_color, const int threads, const std::unordered_map<Key, std::vector<BookEntry> >& bookMapBest) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size() + 1); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	assert(positions.size() <= bookMap.size());

	std::vector<int> indexes;
	for (int i = 0; i < (int)positions.size(); ++i) {
		if (make_book_color == Black && positions[i].depth % 2 == 0 || make_book_color == White && positions[i].depth % 2 == 1 || make_book_color == ColorNum) {
			if (outMap.find(positions[i].key) == outMap.end())
				indexes.emplace_back(i);
		}
	}

	// 並列でminmax定跡作成
	const int indexes_size = (int)indexes.size();
	const int initial_size = (int)outMap.size();
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
	for (int i = 0; i < indexes_size; ++i) {
		Position pos_copy(pos);

		const PositionWithMove& position = positions[indexes[i]];
		const PositionWithMove* position_ptr = &position;
		std::vector<Move> moves(position_ptr->depth);
		for (int j = position_ptr->depth - 1; j >= 0; --j) {
			moves[j] = position_ptr->move;
			position_ptr = position_ptr->parent;
		}
		assert(position_ptr->parent == nullptr);

		// move
		auto states = StateListPtr(new std::deque<StateInfo>(1));
		for (const Move move : moves) {
			states->emplace_back(StateInfo());
			pos_copy.doMove(move, states->back());
		}

		const Key key = position.key;
		assert(Book::bookKey(pos_copy) == key);
		const auto itr = bookMap.find(key);
		assert(itr != bookMap.end());
		const auto& entries = itr->second;
		int index;
		Move move;
		Score score;
		std::tie(index, move, score) = select_best_book_entry(pos_copy, bookMap, entries, moves, bookMapBest);
		#pragma omp critical
		{
			auto& out_entries = outMap[key];
			assert(out_entries.size() == 0);
			out_entries.emplace_back(BookEntry{ key, (u16)move.value(), 1, score});
			if ((outMap.size() - initial_size) % 10000 == 0)
				std::cout << "progress: " << (outMap.size() - initial_size) * 100 / indexes_size << "%" << std::endl;
		}
	}
}

namespace make_book_ra {
	constexpr size_t MAX_PLY = 256;
	constexpr u16 BOOK_DEPTH_INF = 999;
	struct BookNode;
	struct ChildEdge {
		Move move;
		BookNode* node;
		Score score;
		u16 depth;
		bool is_check;
		Score score_for_update;
		u16 depth_for_update;
	};
	struct ParentEdge {
		BookNode* node;
		ChildEdge* edge;
	};
	struct BookNode {
		Key key;
		int out_count;
		Color color;
		bool in_check;
		std::vector<std::unique_ptr<ChildEdge>> childs;
		std::vector< std::unique_ptr<ParentEdge>> parents;
	};
	void enumerate_book_nodes(Position& pos, const std::unordered_map<Key, std::vector<BookEntry>>& bookMap, std::vector<BookNode>& nodes, BookNode* node, std::unordered_map<Key, BookNode*>& searched) {
		const Key key = node->key;
		const auto itr = bookMap.find(key);

		std::vector<std::pair<Move, Score>> moves;
		{
			const auto itr_curr = bookMap.find(key);
			const auto& entries = itr_curr->second;
			moves.reserve(entries.size());
			Score trusted_score = entries[0].score;
			for (const auto& entry : entries) {
				Move move = move16toMove(Move(entry.fromToPro), pos);
				if (entry.score < trusted_score)
					trusted_score = entry.score;
				moves.emplace_back(move, trusted_score);
			}
			for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
				const auto move = ml.move();
				if (std::find_if(entries.begin(), entries.end(), [&move](const auto& entry) { return entry.fromToPro == (u16)move.value(); }) == entries.end())
					moves.emplace_back(move, ScoreNone);
			}
		}

		for (const auto& m : moves) {
			const auto& move = m.first;
			const Key key_after = Book::bookKeyAfter(pos, key, move);
			const auto itr_after = bookMap.find(key_after);
			if (itr_after == bookMap.end()) {
				// 定跡の指し手にあるか
				if (m.second != ScoreNone) {
					node->childs.emplace_back(new ChildEdge{ move, nullptr, m.second, 0, false, ScoreNotEvaluated, 0 });
				}
				continue;
			}
			BookNode* next_node;
			const bool not_found = searched.find(key_after) == searched.end();
			if (not_found) {
				next_node = &nodes.emplace_back(BookNode{ key_after, 0, oppositeColor(pos.turn()), pos.moveGivesCheck(move), {}, {}});
				searched[key_after] = next_node;
			}
			else {
				next_node = searched[key_after];
			}
			node->out_count++;
			auto& edge = node->childs.emplace_back(new ChildEdge{ move, next_node, ScoreNone, BOOK_DEPTH_INF, next_node->in_check, ScoreNotEvaluated, 0 });
			next_node->parents.emplace_back(new ParentEdge{ node, edge.get()});
			if (not_found) {
				StateInfo st;
				pos.doMove(move, st);
				enumerate_book_nodes(pos, bookMap, nodes, next_node, searched);
				pos.undoMove(move);
			}
		}
	}

	// バッグトラッキングして、連続王手になる手順があるかチェックする
	bool repetition_check(BookNode* node, const Key key, bool is_evasion, std::unordered_set<Key>& searched) {
		bool repetition_any = false;
		for (auto& parent : node->parents) {
			if (parent->edge->depth == BOOK_DEPTH_INF && parent->node->in_check == is_evasion) {
				// evasionでなければ王手であること
				if (!is_evasion && !parent->edge->is_check)
					continue;
				bool repetition = false;
				if (parent->node->key == key) {
					repetition = true;
				}
				else if (searched.find(parent->node->key) != searched.end()) {
					// ループしているためこれ以上探索しない
				}
				else {
					searched.emplace(parent->node->key);
					repetition = repetition_check(parent->node, key, !is_evasion, searched);
				}
				if (repetition) {
					return repetition;
				}
			}
		}
		return false;
	}
}

// 後退解析(Retrograde analysis)
void make_all_minmax_book_ra(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, const double beta) {
	using namespace make_book_ra;
	std::vector<BookNode> nodes;
	nodes.reserve(bookMap.size());
	const Key root_key = Book::bookKey(pos);
	auto& root_node = nodes.emplace_back(BookNode{ Book::bookKey(pos), 0, pos.turn(), pos.inCheck(), {}, {}});
	{
		std::unordered_map<Key, BookNode*> searched;
		searched.emplace(root_key, &root_node);
		enumerate_book_nodes(pos, bookMap, nodes, &root_node, searched);
	}
	std::cout << "nodes: " << nodes.size() << std::endl;
	assert(positions.size() <= bookMap.size());


	// 出次数0のnodeをqueueに追加
	std::vector<BookNode*> queue;
	for (auto& node : nodes)
	{
		if (node.out_count == 0)
			queue.emplace_back(&node);
	}
	int terminal_count = queue.size();

	// 出次数が0のノードがなくなるまで取り除く
	while (queue.size() > 0) {
		auto node = queue.back();
		queue.pop_back();

		Score max_score = -ScoreInfinite;
		ChildEdge* best;
		for (auto& edge : node->childs) {
			//if (edge->node != nullptr) __debugbreak();
			if (edge->score > max_score) {
				max_score = edge->score;
				best = edge.get();
			}
		}
		//if (max_score == -ScoreInfinite) __debugbreak();
		//if (best->depth == BOOK_DEPTH_INF) __debugbreak();
		// 親のedgeを更新
		for (auto& parent : node->parents) {
			parent->edge->node = nullptr;
			parent->edge->score = -max_score;
			parent->edge->depth = best->depth + 1;

			// 出次数を減らす
			parent->node->out_count--;

			// 出次数が0になった場合
			if (parent->node->out_count == 0) {
				queue.emplace_back(parent->node);
				terminal_count++;
			}
		}
		// もう親を辿ることはないため、parentsをクリア
		node->parents.clear();
	}
	std::cout << "terminal_count: " << terminal_count << " rest: " << nodes.size() - terminal_count << std::endl;

	// 連続王手の千日手の初期化
	int repetition_check_count = 0;
	for (auto& node : nodes) {
		for (auto& edge : node.childs) {
			if (edge->depth == BOOK_DEPTH_INF && (edge->is_check || node.in_check)) {
				// バッグトラッキングして、連続王手になる手順があるかチェックする
				// 連続王手になる場合、その評価値を初期化する
				//if (edge->node == nullptr) __debugbreak();
				std::unordered_set<Key> searched;
				bool repetition = repetition_check(&node, edge->node->key, !node.in_check, searched);
				if (repetition) {
					// 評価値を初期化
					if (node.in_check)
						edge->score = ScoreMaxEvaluate;
					else
						edge->score = -ScoreMaxEvaluate;

					repetition_check_count++;
				}
				/*if (edge->node->key == 833690442675171497UL) {
					std::cout << "repetition: " << repetition << " key: " << node.key << " score: " << edge->score << std::endl;
					__debugbreak();
				}*/
			}
		}
	}
	std::cout << "repetition_check_count: " << repetition_check_count << std::endl;
	/*for (auto& edge : root_node.childs) {
		std::cout << edge->move.toUSI() << " score: " << edge->score << " depth: " << edge->depth << std::endl;
	}*/

	// MAX_PLY回だけ評価値を伝播させる
	for (size_t loop = 0; loop < MAX_PLY; ++loop) {
		int update_count = 0;
		for (auto& node : nodes) {
			if (node.parents.size() == 0)
				continue;

			// 最善手を選択
			Score max_score = -ScoreInfinite;
			ChildEdge* best;
			for (auto& edge : node.childs) {
				Score score;
				// 千日手の判定
				const bool draw = edge->depth == BOOK_DEPTH_INF && edge->score == ScoreNone;
				if (draw) {
					score = (node.color == Black) ? draw_score_black : draw_score_white;
				}
				else {
					score = edge->score;
				}
				if (score > max_score) {
					max_score = score;
					best = edge.get();
				}
				else if (score == max_score) {
					if (draw) {
						// 先手は千日手を回避、後手は千日手を選択
						if (node.color == White) {
							best = edge.get();
						}
					}
					else {
						// 手順が長い方(定跡が長く続く方)を選択
						if (edge->depth > best->depth) {
							best = edge.get();
						}
					}
				}
			}
			const auto parent_score = best->score == ScoreNone ? best->score : -best->score;
			const auto parent_depth = best->depth == BOOK_DEPTH_INF ? best->depth : best->depth + 1;
			// 親に伝播
			for (auto& parent : node.parents) {
				if (parent->edge->score != parent_score) {
					update_count++;
				}
				parent->edge->score_for_update = parent_score;
				parent->edge->depth_for_update = parent_depth;
			}
		}

		// update
		for (auto& node : nodes) {
			if (node.parents.size() == 0)
				continue;
			for (auto& edge : node.childs) {
				if (edge->score_for_update != ScoreNotEvaluated) {
					edge->score = edge->score_for_update;
					edge->depth = edge->depth_for_update;
					edge->score_for_update = ScoreNotEvaluated;
				}
			}
		}
		std::cout << "loop: " << loop << " update_count: " << update_count << std::endl;
		if (update_count == 0) break;
	}
	// 出力
	for (auto& node : nodes) {
		for (auto& edge : node.childs) {
			if (edge->score == ScoreNone) {
				edge->score = (node.color == Black) ? draw_score_black : draw_score_white;
			}
		}
		std::stable_sort(node.childs.begin(), node.childs.end(), [&node](const auto& l, const auto& r) {
			if (l->score == r->score) {
				if (l->depth != BOOK_DEPTH_INF && r->depth == BOOK_DEPTH_INF && node.color == Black)
					return true;
				if (l->depth == BOOK_DEPTH_INF && r->depth != BOOK_DEPTH_INF && node.color == White)
					return true;
				return l->depth > r->depth;
			}
			else {
				return l->score > r->score;
			}
			});
		auto& entries = outMap[node.key];
		for (const auto& edge : node.childs) {
			entries.emplace_back(BookEntry{ node.key, (u16)edge->move.value(), 0, edge->score });
		}

		// score to prob
		const Score max_score = entries[0].score;
		std::vector<double> prob;

		// softmax temperature with normalize
		double sum = 0;
		for (const auto& entry : entries) {
			double x = (double)(entry.score - max_score) * beta / 754.3;
			x = exp(x);
			prob.emplace_back(x);
			sum += x;
		}
		// normalize
		for (auto& x : prob) {
			x /= sum;
		}

		for (size_t i = 0; i < entries.size(); ++i) {
			entries[i].count = (u16)(prob[i] * USHRT_MAX);
			if (i > 0 && entries[i].count == entries[0].count) {
				// 最善手のcountと同じにしない
				entries[i].count--;
			}
		}
	}

	for (auto& edge : root_node.childs) {
		std::cout << edge->move.toUSI() << " score: " << edge->score << " depth: " << edge->depth << std::endl;
	}
}

// 評価値が30000以上の局面を再評価
void fix_eval(Position& pos, std::unordered_map<Key, std::vector<BookEntry> >& bookMap, LimitsType& limits, const std::string& book_pos_cmd, const Key& book_starting_pos_key) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size() + 1); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	assert(positions.size() <= bookMap.size());

	// 評価値が30000以上の局面を再評価
	for (const auto& position : positions) {
		const Key key = position.key;

		const auto itr_book = bookMap.find(key);
		if (itr_book != bookMap.end()) {
			const auto& entry = itr_book->second[0];
			if (std::abs(entry.score) >= 30000) {
				Position pos_copy(pos);
				const PositionWithMove* position_ptr = &position;
				std::vector<Move> moves(position_ptr->depth);
				for (int j = position_ptr->depth - 1; j >= 0; --j) {
					moves[j] = position_ptr->move;
					position_ptr = position_ptr->parent;
				}
				assert(position_ptr->parent == nullptr);

				// move
				auto states = StateListPtr(new std::deque<StateInfo>(1));
				for (const Move move : moves) {
					states->emplace_back(StateInfo());
					pos_copy.doMove(move, states->back());
				}

				const auto prev_score = entry.score;
				int count = 0;
				bookMap.erase(key);
				make_book_entry_with_uct(pos_copy, limits, key, bookMap, count, moves, book_pos_cmd, book_starting_pos_key);

				const auto after_score = bookMap[key][0].score;

				std::cout << prev_score << ", " << after_score << std::endl;

			}
		}
	}
}

Score make_hcpe3_cache(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, const std::unordered_map<Key, std::vector<BookEntry> >& minmaxBookMap, std::vector<TrainingData>& trainingData, std::unordered_set<Key>& exists, const double beta) {
	const Key key = Book::bookKey(pos);
	const auto itr = minmaxBookMap.find(key);
	if (itr == minmaxBookMap.end())
		return ScoreNone;

	if (!exists.emplace(key).second)
		return -itr->second[0].score;

	std::unordered_map<u16, Score> candidate_score_map;

	Score trusted_score = itr->second[0].score;

	// Stack overflowを避けるためヒープに確保する
	for (auto ml = std::make_unique<MoveList<LegalAll>>(pos); !ml->end(); ++(*ml)) {
		const Move move = ml->move();
		auto state = std::make_unique<StateInfo>();
		pos.doMove(move, *state);
		Score score = make_hcpe3_cache(pos, bookMap, minmaxBookMap, trainingData, exists, beta);
		pos.undoMove(move);

		if (score == ScoreNone) {
			if ((u16)move.value() != itr->second[0].fromToPro)
				continue;

			score = itr->second[0].score;
		}

		candidate_score_map.emplace((u16)move.value(), score > trusted_score ? trusted_score : score);
	}

	const auto itr_book = bookMap.find(key);
	for (const BookEntry& entry : itr_book->second) {
		if (trusted_score > entry.score) {
			trusted_score = entry.score;
		}
		candidate_score_map.emplace(entry.fromToPro, trusted_score);
	}

	std::unordered_map<u16, float> candidates;

	// score to prob
	if (candidate_score_map.size() == 1) {
		candidates.emplace(candidate_score_map.begin()->first, 1.0f);
	}
	else {
		const Score max_score = itr->second[0].score;

		// softmax temperature with normalize
		double sum = 0;
		for (const auto& entry : candidate_score_map) {
			double x = (double)(entry.second - max_score) * beta / 754.3;
			x = exp(x);
			candidates.emplace(entry.first, (float)x);
			sum += x;
		}
		// normalize
		for (auto& entry : candidates) {
			entry.second = (float)(entry.second / sum);
		}
	}

	const auto value = score_to_value(itr->second[0].score);
	auto& data = trainingData.emplace_back(pos.toHuffmanCodedPos(), value, value);
	data.candidates = std::move(candidates);

	return -itr->second[0].score;
}

// make_all_minmax_bookで作成した定跡からhcpe3キャッシュを作成
void minmax_book_to_cache(Position& pos, std::unordered_map<Key, std::vector<BookEntry> >& bookMap, const std::unordered_map<Key, std::vector<BookEntry> >& minmaxBookMap, const std::string& filepath, const double beta) {
	std::vector<TrainingData> trainingData;
	std::unordered_set<Key> exists;
	make_hcpe3_cache(pos, bookMap, minmaxBookMap, trainingData, exists, beta);

	std::ofstream ofs(filepath, std::ios::binary);

	// インデックス部
	// 局面数
	const size_t num = trainingData.size();
	ofs.write((const char*)&num, sizeof(num));
	// 各局面の開始位置
	size_t start_pos = sizeof(num) + sizeof(start_pos) * num;
	for (const auto& hcpe3 : trainingData) {
		ofs.write((const char*)&start_pos, sizeof(start_pos));
		start_pos += sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * hcpe3.candidates.size();
	}

	// ボディ部
	for (const auto& hcpe3 : trainingData) {
		Hcpe3CacheBody body{
			hcpe3.hcp,
			hcpe3.value,
			hcpe3.result,
			hcpe3.count
		};
		ofs.write((const char*)&body, sizeof(body));

		for (const auto kv : hcpe3.candidates) {
			Hcpe3CacheCandidate candidate{
				kv.first,
				kv.second
			};
			ofs.write((const char*)&candidate, sizeof(candidate));
		}
	}
}

void overwrite_hcpe3_cache(const std::string& original_filepath, const std::string& filepath, const std::string& out_filepath, const double weight) {
	std::ifstream cache(filepath, std::ios::binary);
	size_t num_cache;
	cache.read((char*)&num_cache, sizeof(num_cache));
	std::vector<size_t> cache_pos;
	cache_pos.resize(num_cache + 1);
	cache.read((char*)cache_pos.data(), sizeof(size_t) * num_cache);
	cache.seekg(0, std::ios_base::end);
	cache_pos[num_cache] = cache.tellg();

	std::unordered_map<HuffmanCodedPos, std::pair<Hcpe3CacheBody, std::vector<Hcpe3CacheCandidate>>> cache_map;
	for (size_t i = 0; i < num_cache; ++i) {
		cache.seekg(cache_pos[i], std::ios_base::beg);
		Hcpe3CacheBody body;
		cache.read((char*)&body, sizeof(body));
		auto& data = cache_map[body.hcp];
		data.first = body;

		auto& candidates = data.second;
		const auto num_candidates = (cache_pos[i + 1] - cache_pos[i] - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
		candidates.resize(num_candidates);
		cache.read((char*)candidates.data(), sizeof(Hcpe3CacheCandidate) * num_candidates);
	}

	std::ifstream original(original_filepath, std::ios::binary);
	size_t num_original;
	original.read((char*)&num_original, sizeof(num_original));
	std::vector<size_t> original_pos;
	original_pos.resize(num_original + 1);
	original.read((char*)original_pos.data(), sizeof(size_t) * num_original);
	original.seekg(0, std::ios_base::end);
	original_pos[num_original] = original.tellg();

	std::vector<TrainingData> trainingData;
	trainingData.reserve(num_original + num_cache);

	for (size_t i = 0; i < num_original; ++i) {
		original.seekg(original_pos[i], std::ios_base::beg);
		Hcpe3CacheBody body;
		original.read((char*)&body, sizeof(body));

		const auto itr = cache_map.find(body.hcp);
		if (itr == cache_map.end()) {
			const auto num_candidates = (original_pos[i + 1] - original_pos[i] - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
			std::vector<Hcpe3CacheCandidate> candidates(num_candidates);
			original.read((char*)candidates.data(), sizeof(Hcpe3CacheCandidate) * num_candidates);
			trainingData.emplace_back(body, candidates.data(), num_candidates);
		}
		else if (weight == 1) {
			// 上書き
			auto& data = itr->second;
			trainingData.emplace_back(data.first, data.second.data(), data.second.size());
			cache_map.erase(itr);
		}
		else {
			// 加重平均
			const auto num_candidates = (original_pos[i + 1] - original_pos[i] - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
			std::vector<Hcpe3CacheCandidate> candidates(num_candidates);
			original.read((char*)candidates.data(), sizeof(Hcpe3CacheCandidate) * num_candidates);

			auto& data = itr->second;

			const auto count = body.count;
			body.count = 1;
			body.value = (float)((1 - weight) * body.value / count + weight * data.first.value / data.first.count);
			body.result = (float)((1 - weight) * body.result / count + weight * data.first.result / data.first.count);

			std::map<u16, double> candidate_map;
			for (const auto& candidate : candidates) {
				candidate_map[candidate.move16] = (1 - weight) * candidate.prob / count;
			}
			for (const auto& candidate : data.second) {
				candidate_map[candidate.move16] += weight * candidate.prob / data.first.count;
			}
			std::vector<Hcpe3CacheCandidate> new_candidates;
			new_candidates.reserve(candidate_map.size());
			for (const auto& kv : candidate_map) {
				new_candidates.emplace_back(Hcpe3CacheCandidate{ kv.first, (float)kv.second });
			}

			trainingData.emplace_back(body, new_candidates.data(), new_candidates.size());
			cache_map.erase(itr);
		}
	}

	// 残りを追加
	for (auto itr = cache_map.cbegin(); itr != cache_map.cend(); ++itr) {
		const auto& data = itr->second;
		trainingData.emplace_back(data.first, data.second.data(), data.second.size());
	}

	// 出力
	std::ofstream ofs(out_filepath, std::ios::binary);

	// インデックス部
	// 局面数
	const size_t num = trainingData.size();
	ofs.write((const char*)&num, sizeof(num));
	// 各局面の開始位置
	size_t start_pos = sizeof(num) + sizeof(start_pos) * num;
	for (const auto& hcpe3 : trainingData) {
		ofs.write((const char*)&start_pos, sizeof(start_pos));
		start_pos += sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * hcpe3.candidates.size();
	}

	// ボディ部
	for (const auto& hcpe3 : trainingData) {
		Hcpe3CacheBody body{
			hcpe3.hcp,
			hcpe3.value,
			hcpe3.result,
			hcpe3.count
		};
		ofs.write((const char*)&body, sizeof(body));

		for (const auto kv : hcpe3.candidates) {
			Hcpe3CacheCandidate candidate{
				kv.first,
				kv.second
			};
			ofs.write((const char*)&candidate, sizeof(candidate));
		}
	}
}

void book_to_hcp(Position& pos, const std::string& bookFileName, const std::string& outFileName) {
	std::unordered_map<Key, std::vector<BookEntry> > bookMap;
	read_book(bookFileName, bookMap);

	// 全局面を列挙
	std::vector<std::pair<HuffmanCodedPos, const std::vector<BookEntry>*>> positions;
	{
		std::unordered_set<Key> exists;

		enumerate_positions(pos, bookMap, positions, exists);
	}

	std::cout << "positions: " << positions.size() << std::endl;

	std::ofstream ofs(outFileName, std::ios::binary);
	for (const auto position : positions) {
		ofs.write((const char*)position.first.data, sizeof(HuffmanCodedPos));
	}
}

void make_policy_book_inner(Position& pos,
	const std::unordered_map<Key, std::vector<BookEntry> >& bookMap,
	const std::unordered_map<Key, std::vector<BookEntry> >& minmaxBookMap,
	std::unordered_map<Key, std::vector<BookEntry> > &outBookMap,
	std::unordered_set<Key>& exists, const double beta) {
	const Key key = Book::bookKey(pos);
	const auto itr = bookMap.find(key);
	if (itr == bookMap.end())
		return;

	const auto minmaxItr = minmaxBookMap.find(key);
	if (minmaxItr == minmaxBookMap.end())
		return;

	if (!exists.emplace(key).second)
		return;

	const auto& minmaxEntry = minmaxItr->second[0];

	auto& outEntries = outBookMap[key];
	outEntries.emplace_back(minmaxEntry);
	Score trustedScore = minmaxEntry.score;

	// Stack overflowを避けるためヒープに確保する
	for (auto ml = std::make_unique<MoveList<LegalAll>>(pos); !ml->end(); ++(*ml)) {
		const Move move = ml->move();
		auto state = std::make_unique<StateInfo>();
		pos.doMove(move, *state);
		make_policy_book_inner(pos, bookMap, minmaxBookMap, outBookMap, exists, beta);

		if (minmaxEntry.fromToPro != (u16)move.value()) {
			Key keyAfter = Book::bookKey(pos);
			const auto itrAfter = minmaxBookMap.find(keyAfter);
			if (itrAfter != minmaxBookMap.end()) {
				const auto& entryAfter = itrAfter->second[0];
				auto& outEntry = outEntries.emplace_back();
				outEntry.key = key;
				outEntry.fromToPro = (u16)move.value();
				outEntry.count = 0;
				outEntry.score = std::min(-entryAfter.score, trustedScore);
			}
		}

		pos.undoMove(move);
	}

	for (const auto& entry : itr->second) {
		const auto exist = std::find_if(outEntries.cbegin(), outEntries.cend(), [&entry](const BookEntry& outEntry) { return outEntry.fromToPro == entry.fromToPro; });
		if (exist != outEntries.cend()) {
			trustedScore = exist->score;
			continue;
		}
		if (entry.score < trustedScore) {
			trustedScore = entry.score;
		}
		auto outEntry = entry;
		outEntry.score = trustedScore;
		outEntries.emplace_back(outEntry);
	}

	// score to prob
	if (outEntries.size() > 0) {
		const Score max_score = itr->second[0].score;
		std::vector<double> prob;

		// softmax temperature with normalize
		double sum = 0;
		for (const auto& entry : outEntries) {
			double x = (double)(entry.score - max_score) * beta / 754.3;
			x = exp(x);
			prob.emplace_back(x);
			sum += x;
		}
		// normalize
		for (auto& x : prob) {
			x /= sum;
		}

		for (size_t i = 0; i < outEntries.size(); ++i) {
			outEntries[i].count = (u16)(prob[i] * USHRT_MAX);
		}

		std::stable_sort(outEntries.begin(), outEntries.end(), [](const BookEntry& l, const BookEntry& r) { return l.count > r.count; });
	}
}

void make_policy_book(Position& pos, const std::string& bookFileName, const std::string& minmaxBookFileName, const std::string& outFileName, const double beta) {
	std::unordered_map<Key, std::vector<BookEntry> > bookMap;
	read_book(bookFileName, bookMap);

	std::unordered_map<Key, std::vector<BookEntry> > minmaxBookMap;
	read_book(minmaxBookFileName, minmaxBookMap);

	std::unordered_map<Key, std::vector<BookEntry> > outBookMap;

	// 全局面を列挙
	{
		std::unordered_set<Key> exists;

		make_policy_book_inner(pos, bookMap, minmaxBookMap, outBookMap, exists, beta);
	}

	std::cout << "outBookMap.size: " << outBookMap.size() << std::endl;

	saveOutmap(outFileName, outBookMap);
}

void complement_book(Position& pos, const std::string& bookFileName, std::string& outFileName, const std::string& book_pos_cmd) {
	std::unordered_map<Key, std::vector<BookEntry> > bookMap;
	read_book(bookFileName, bookMap);

	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size() + 1); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	assert(positions.size() <= bookMap.size());

	std::ofstream ofs(outFileName);

	std::set<Key> exists;
	int n = 0;

	// 最善手を指した局面が未登録で、最善手以外の手を指した局面が登録されている場合、
	// 最善手を指した局面を追加
	for (const auto& position : positions) {
		const Key key = position.key;
		const auto itr_book = bookMap.find(key);

		Position pos_copy(pos);
		const PositionWithMove* position_ptr = &position;
		std::vector<Move> moves(position_ptr->depth);
		for (int j = position_ptr->depth - 1; j >= 0; --j) {
			moves[j] = position_ptr->move;
			position_ptr = position_ptr->parent;
		}
		assert(position_ptr->parent == nullptr);

		// move
		auto states = StateListPtr(new std::deque<StateInfo>(1));
		for (const Move move : moves) {
			states->emplace_back(StateInfo());
			pos_copy.doMove(move, states->back());
		}
		assert(Book::bookKey(pos_copy) == key);

		const auto& entries = itr_book->second;

		// 最善手を指した局面が未登録で、最善手以外の手を指した局面が登録されている場合
		int found_index = -1;
		for (int i = 0; i < (int)entries.size(); ++i) {
			const auto& entry = entries[i];
			const auto key_after = Book::bookKeyAfter(pos_copy, key, move16toMove(Move(entry.fromToPro), pos_copy));
			if (bookMap.find(key_after) != bookMap.end()) {
				found_index = i;
				break;
			}
		}

		// 最善手を指した局面を追加
		if (found_index > 0) {
			const auto& entry = entries[0];
			const auto key_after = Book::bookKeyAfter(pos_copy, key, move16toMove(Move(entry.fromToPro), pos_copy));
			if (exists.emplace(key_after).second) {
				ofs << book_pos_cmd.substr(9);
				for (const Move move : moves) {
					ofs << " " << move.toUSI();
				}
				ofs << " " << Move(entry.fromToPro).toUSI() << std::endl;
				n++;
			}
		}
	}

	std::cout << "num: " << n << std::endl;
}
#endif
