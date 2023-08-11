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
Key book_starting_pos_key;
std::string book_pos_cmd;
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
// MinMaxの探索順に使用する定跡
std::unordered_map<Key, std::vector<BookEntry> > bookMapBest;
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
std::map<Key, Score> book_key_eval_map;

// 定跡用mutex
std::mutex gpu_mutex;
std::mutex usi_mutex;
std::condition_variable usi_cond;


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

inline Move UctSearchGenmoveNoPonder(Position* pos, const std::vector<Move>& moves) {
	Move move;
	return UctSearchGenmove(pos, book_starting_pos_key, moves, move);
}

bool make_book_entry_with_uct(Position& pos, LimitsType& limits, const Key& key, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const std::vector<Move>& moves) {
	std::unique_lock<std::mutex> gpu_lock(gpu_mutex);
	std::cout << omp_get_thread_num() << "# " << book_pos_cmd;
	for (Move move : moves) {
		std::cout << " " << move.toUSI();
	}
	std::cout << std::endl;

	// UCT探索を使用
	limits.startTime.restart();
	SetLimits(limits);
	UctSearchGenmoveNoPonder(&pos, moves);

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
Score book_search(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, Score alpha, const Score beta, const Score score, std::map<Key, Searched>& searched) {
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

	using candidate_t = std::tuple<Move, Score>;
	std::vector<candidate_t> candidates;
	{
		candidates.reserve(entries.size() + 4);
		// MinMaxの探索順に使用する定跡
		Move moveBest = Move::moveNone();
		if (bookMapBest.size() > 0) {
			const auto itr_best = bookMapBest.find(key);
			if (itr_best != bookMapBest.end()) {
				moveBest = move16toMove(Move(itr_best->second[0].fromToPro), pos);
				candidates.emplace_back(moveBest, ScoreNotEvaluated);
			}
		}
		Score trusted_score = entries[0].score;
		for (const auto& entry : entries) {
			// 訪問回数が少ない評価値は信頼しない
			if (entry.score < trusted_score)
				trusted_score = entry.score;
			const Move move = move16toMove(Move(entry.fromToPro), pos);
			if (move == moveBest) {
				std::get<Score>(candidates[0]) = trusted_score;
			}
			else {
				candidates.emplace_back(move, trusted_score);
			}
		}
		const auto size = candidates.size();
		for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
			const Move& move = ml.move();
			if (std::any_of(candidates.cbegin(), candidates.cbegin() + size, [move](const auto& candidate) { return std::get<Move>(candidate) == move; }))
				continue;
			if (outMap.find(Book::bookKeyAfter(pos, key, move)) == outMap.end())
				continue;
			candidates.emplace_back(move, ScoreNotEvaluated);
		}
	}

	Score value = -ScoreInfinite;
	for (const auto& candidate : candidates) {
		const Move& move = std::get<Move>(candidate);
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
			value = std::max(value, book_search(pos, outMap, -beta, -alpha, std::get<Score>(candidate), searched));
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
	//if (std::abs(value) == 71) print_debug_moves(value);

	// βカットされなかった場合はsearchedに追加しない
	return -value;
}

std::tuple<int, Move, Score> select_best_book_entry(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves) {
	const Key key = Book::bookKey(pos);

	using candidate_t = std::tuple<Move, Score, const BookEntry*>;
	std::vector<candidate_t> candidates;
	{
		candidates.reserve(entries.size() + 4);
		// MinMaxの探索順に使用する定跡
		Move moveBest = Move::moveNone();
		if (bookMapBest.size() > 0) {
			const auto itr_best = bookMapBest.find(key);
			if (itr_best != bookMapBest.end()) {
				moveBest = move16toMove(Move(itr_best->second[0].fromToPro), pos);
				candidates.emplace_back(moveBest, ScoreNotEvaluated, nullptr);
			}
		}
		Score trusted_score = entries[0].score;
		for (const auto& entry : entries) {
			// 訪問回数が少ない評価値は信頼しない
			if (entry.score < trusted_score)
				trusted_score = entry.score;
			const Move move = move16toMove(Move(entry.fromToPro), pos);
			if (move == moveBest) {
				std::get<Score>(candidates[0]) = trusted_score;
				std::get<const BookEntry*>(candidates[0]) = &entry;
			}
			else {
				candidates.emplace_back(move, trusted_score, &entry);
			}
		}
		const auto size = candidates.size();
		for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
			const Move& move = ml.move();
			if (std::any_of(candidates.cbegin(), candidates.cbegin() + size, [move](const auto& candidate) { return std::get<Move>(candidate) == move; }))
				continue;
			if (outMap.find(Book::bookKeyAfter(pos, key, move)) == outMap.end())
				continue;
			candidates.emplace_back(move, ScoreNotEvaluated, nullptr);
		}
	}

	Score alpha = -ScoreInfinite;
	const candidate_t* best = nullptr;
	std::map<Key, Searched> searched;

	for (const auto& candidate : candidates) {
		const Move& move = std::get<Move>(candidate);
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
			value = book_search(pos, outMap, -ScoreInfinite, -alpha, std::get<Score>(candidate), searched);
		}
		pos.undoMove(move);
		//debug_moves.pop_back();
		//std::cout << move.toUSI() << "\t" << entry.score << "\t" << value << std::endl;

		if (value > alpha) {
			best = &candidate;
			alpha = value;
		}
	}
	const auto& best_entry = std::get<const BookEntry*>(*best);
	return { best_entry != nullptr ? (int)(best_entry - &entries[0]) : -1, std::get<Move>(*best), alpha };
}

// 定跡作成(再帰処理)
void make_book_inner(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry) {
	const Key key = Book::bookKey(pos);
	if ((depth % 2 == 0) == isBlack) {

		const auto itr = outMap.find(key);
		if (itr == outMap.end()) {
			// 先端ノード
			// UCT探索で定跡作成
			make_book_entry_with_uct(pos, limits, key, outMap, count, moves);
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
					// 一定の確率でmin-maxで選ぶ
					int index;
					Move move;
					Score score;
					std::tie(index, move, score) = (dist_minmax(g_randomTimeSeed) < book_minmax_prob) ? select_best_book_entry(pos, outMap, entries, moves) : std::make_tuple(0, move16toMove(Move(entries[0].fromToPro), pos), entries[0].score);

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
				make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves, select_best_book_entry);

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
				if (!make_book_entry_with_uct(pos, limits, key, outMap, count, moves))
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
					if (!make_book_entry_with_uct(pos, limits, key, outMap, count, moves))
					{
						// 詰みの局面の場合何もしない
						return;
					}
				}

				entries = &outMap[key];
			}

			if (itr == bookMap.end() && dist_minmax(g_randomTimeSeed) < book_minmax_prob_opp) {
				// 一定の確率でmin-maxで選ぶ
				const auto& entry = select_best_book_entry(pos, outMap, *entries, moves);
				move = std::get<Move>(entry);
			}
			if (itr != bookMap.end() && book_best_move) {
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
		make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves, select_best_book_entry);

		pos.undoMove(move);
	}
}

void make_book_alpha_beta(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack) {
	std::vector<Move> moves;
	make_book_inner(pos, limits, bookMap, outMap, count, depth, isBlack, moves, select_best_book_entry);
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

void minmax_book_white(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry);

void minmax_book_black(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry) {
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
	std::tie(index, bestMove, score) = select_best_book_entry(pos, bookMap, entries, moves);

	// 最善手を登録
	auto& minMaxBookEntry = bookMapMinMax[key];
	minMaxBookEntry.move16 = (u16)bestMove.value();
	minMaxBookEntry.score = score;
	minMaxBookEntry.depth = pos.gamePly();

	// 最善手を指す
	StateInfo state;
	pos.doMove(bestMove, state);
	moves.emplace_back(bestMove);
	minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry);
	moves.pop_back();
	pos.undoMove(bestMove);
}

void minmax_book_white(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry) {
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
			minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry);
			moves.pop_back();
		}
		pos.undoMove(move);
	}
}

void make_minmax_book(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, const Color make_book_color, select_best_book_entry_t select_best_book_entry) {
	std::vector<Move> moves;
	if (make_book_color == Black)
		minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry);
	else if (make_book_color == White)
		minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry);
	else {
		minmax_book_black(pos, bookMapMinMax, moves, select_best_book_entry);
		minmax_book_white(pos, bookMapMinMax, moves, select_best_book_entry);
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

std::unique_ptr<USIBookEngine> create_usi_book_engine(const std::string& engine_path, const std::string& engine_options) {
	std::vector<std::pair<std::string, std::string>> usi_engine_options;
	std::istringstream ss(engine_options);
	std::string field;
	while (std::getline(ss, field, ',')) {
		const auto p = field.find_first_of(":");
		usi_engine_options.emplace_back(field.substr(0, p), field.substr(p + 1));
	}
	return std::make_unique<USIBookEngine>(engine_path, usi_engine_options);
}

void init_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const int nodes, const double prob, const int nodes_own, const double prob_own, const int num_engines) {
	if (engine_path == "")
		return;
	for (int i = 0; i < num_engines; ++i) {
		usi_book_engines.emplace_back(create_usi_book_engine(engine_path, engine_options));
	}
	usi_book_engine_nodes = nodes;
	usi_book_engine_prob = prob;
	usi_book_engine_nodes_own = nodes_own;
	usi_book_engine_prob_own = prob_own;
}

void init_book_key_eval_map(const std::string& str) {
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
void eval_positions_with_usi_engine(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, const std::string& engine_path, const std::string& engine_options, const int nodes, const int engine_num) {
	// 全局面を列挙
	std::vector<std::pair<HuffmanCodedPos, const std::vector<BookEntry>*>> positions;
	{
		std::unordered_set<Key> exists;

		enumerate_positions(pos, bookMap, positions, exists, outMap);
	}

	std::cout << "positions: " << positions.size() << std::endl;

	// USIエンジン初期化
	std::vector<std::unique_ptr<USIBookEngine>> usi_book_engines(engine_num);
	#pragma omp parallel for num_threads(engine_num)
	for (int i = 0; i < engine_num; ++i) {
		usi_book_engines[omp_get_thread_num()] = create_usi_book_engine(engine_path, engine_options);
	}
	usi_book_engine_nodes = nodes;

	// 並列で評価
	const int positions_size = (int)positions.size();
	const std::vector<Move> moves = {};
	std::regex re(R"*(score +(cp|mate) +([+\-]?\d*))*");
	int count = 0;
	#pragma omp parallel for num_threads(engine_num)
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
				if (count % 10000 == 0)
					std::cout << "progress: " << count * 100 / positions_size << "%" << std::endl;
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

	PositionWithMove& potision_root = positions.emplace_back(PositionWithMove{ Book::bookKey(pos_root), Move::moveNone(), 0, nullptr });
	current_positions.push_back({ pos_root.toHuffmanCodedPos(), &potision_root });

	int depth = 1;
	while (current_positions.size() > 0) {
		for (const auto& position : current_positions) {
			const auto& hcp = position.first;
			const PositionWithMove* parent = position.second;

			Position pos;
			pos.set(hcp);

			for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
				const auto move = ml.move();
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
void diff_eval(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, LimitsType& limits, const Score diff, const std::string& outFileName) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size()); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	assert(positions.size() <= bookMap.size());

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
			if ((score + 100) * opp_score < 0 || (score - 100) * opp_score < 0 || opp_mate) {
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
						make_book_entry_with_uct(pos_copy, limits, key_after, outMap, count, moves);
						// 保存
						saveOutmap(outFileName, outMap);
					}
				}
			}
		}
	}
}

// 全ての局面についてαβで定跡を作る
void make_all_minmax_book(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, const Color make_book_color, const int threads) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size()); // 追加でparentのポインターが無効にならないようにする
	enumerate_positions_with_move(pos, bookMap, positions);
	std::cout << "positions: " << positions.size() << std::endl;
	assert(positions.size() <= bookMap.size());

	std::vector<int> indexes;
	for (int i = 0; i < (int)positions.size(); ++i) {
		if (make_book_color == Black && positions[i].depth % 2 == 0 || make_book_color == White && positions[i].depth % 2 == 1 || make_book_color == ColorNum) {
			indexes.emplace_back(i);
		}
	}

	// 並列でminmax定跡作成
	const int indexes_size = (int)indexes.size();
	#pragma omp parallel for num_threads(threads)
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
		std::tie(index, move, score) = select_best_book_entry(pos_copy, bookMap, entries, moves);
		#pragma omp critical
		{
			auto& out_entries = outMap[key];
			assert(out_entries.size() == 0);
			out_entries.emplace_back(BookEntry{ key, (u16)move.value(), 1, score});
			if (outMap.size() % 10000 == 0)
				std::cout << "progress: " << outMap.size() * 100 / positions.size() << "%" << std::endl;
		}
	}
}

// 評価値が30000以上の局面を再評価
void fix_eval(Position& pos, std::unordered_map<Key, std::vector<BookEntry> >& bookMap, LimitsType& limits) {
	// 局面を列挙する
	std::vector<PositionWithMove> positions;
	positions.reserve(bookMap.size()); // 追加でparentのポインターが無効にならないようにする
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
				make_book_entry_with_uct(pos_copy, limits, key, bookMap, count, moves);

				const auto after_score = bookMap[key][0].score;

				std::cout << prev_score << ", " << after_score << std::endl;

			}
		}
	}
}
#endif
