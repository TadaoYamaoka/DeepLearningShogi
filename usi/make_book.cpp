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

#include <sys/stat.h>

struct child_node_t_copy {
	Move move;       // 着手する座標
	int move_count;  // 探索回数
	WinType win;       // 勝った回数

	child_node_t_copy(const child_node_t& child) {
		this->move = child.move;
		this->move_count = child.move_count;
		this->win = child.win;
	}
};

std::map<Key, std::vector<BookEntry> > bookMap;
Key book_starting_pos_key;
std::string book_pos_cmd;
extern std::unique_ptr<NodeTree> tree;
int make_book_sleep = 0;
bool use_book_policy = true;
bool use_interruption = true;
int book_eval_threshold = INT_MAX;
double book_visit_threshold = 0.005;
double book_cutoff = 0.015;
double book_reciprocal_temperature = 1.0;
// MinMaxで選ぶ確率
double book_minmax_prob = 1.0;
double book_minmax_prob_opp = 0.1;
std::uniform_real_distribution<double> dist_minmax(0, 1);
// 千日手の評価値
extern float draw_value_black;
extern float draw_value_white;
extern float eval_coef;
Score draw_score_black;
Score draw_score_white;

inline Move UctSearchGenmoveNoPonder(Position* pos, const std::vector<Move>& moves) {
	Move move;
	return UctSearchGenmove(pos, book_starting_pos_key, moves, move);
}

bool make_book_entry_with_uct(Position& pos, LimitsType& limits, const Key& key, std::map<Key, std::vector<BookEntry> >& outMap, int& count, const std::vector<Move>& moves) {
	std::cout << book_pos_cmd;
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

	for (int i = 0; i < num; i++) {
		const auto& child = movelist[i];
		// 定跡追加
		BookEntry be;
		const auto wintrate = child.win / child.move_count;
		be.score = Score(int(-log(1.0 / wintrate - 1.0) * 754.3));
		be.key = key;
		be.fromToPro = static_cast<u16>(child.move.proFromAndTo());
		be.count = (u16)((double)child.move_count / (double)current_root->move_count * 1000.0);
		outMap[key].emplace_back(be);

		count++;
	}

	if (make_book_sleep > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(make_book_sleep));

	return true;
}

// min-max(αβ)で選択
Score book_search(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, Score alpha, const Score beta, const Score score) {
	const Key key = Book::bookKey(pos);
	const auto itr = outMap.find(key);
	if (itr == outMap.end()) {
		// エントリがない場合、自身の評価値を返す
		return score;
	}
	const auto& entries = itr->second;
	Score value = -ScoreInfinite;
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		//std::cout << pos.turn() << "\t" << move.toUSI() << std::endl;
		StateInfo state;
		pos.doMove(move, state);
		if (pos.isDraw() == RepetitionDraw) {
			// 繰り返しになる場合、千日手の評価値
			value = std::max(value, pos.turn() == Black ? draw_score_white : draw_score_black);
		}
		else {
			value = std::max(value, -book_search(pos, outMap, -beta, -alpha, -entry.score));
		}
		pos.undoMove(move);

		alpha = std::max(alpha, value);
		if (alpha >= beta)
			break;
	}
	return value;
}

const BookEntry& select_best_book_entry(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries) {
	if (entries.size() == 1)
		return entries[0];

	Score alpha = -ScoreInfinite;
	const BookEntry* best = nullptr;
	for (const auto& entry : entries) {
		const Move move = move16toMove(Move(entry.fromToPro), pos);
		//std::cout << pos.turn() << "\t" << move.toUSI() << std::endl;
		StateInfo state;
		pos.doMove(move, state);
		Score value;
		if (pos.isDraw() == RepetitionDraw) {
			// 繰り返しになる場合、千日手の評価値
			value = pos.turn() == Black ? draw_score_white : draw_score_black;
		}
		else {
			value = -book_search(pos, outMap, -ScoreInfinite, -alpha, -entry.score);
		}
		pos.undoMove(move);
		//std::cout << value << std::endl;

		if (value > alpha) {
			best = &entry;
			alpha = value;
		}
	}
	return *best;
}

// 定跡作成(再帰処理)
void make_book_inner(Position& pos, LimitsType& limits, std::map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves) {
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
				const auto& entries = itr->second;
				// 一定の確率でmin-maxで選ぶ
				const auto& entry = (dist_minmax(g_randomTimeSeed) < book_minmax_prob) ? select_best_book_entry(pos, outMap, entries) : entries[0];

				// 評価値が閾値を超えた場合、探索終了
				if (std::abs(entry.score) > book_eval_threshold) {
					std::cout << book_pos_cmd;
					for (Move move : moves) {
						std::cout << " " << move.toUSI();
					}
					std::cout << "\nentry.score: " << entry.score << std::endl;
					return;
				}

				const Move move = move16toMove(Move(entry.fromToPro), pos);
				if (&entry != &entries[0])
					std::cout << "best move : " << depth << " " << & entry - &entries[0] << " " << move.toUSI() << std::endl;

				StateInfo state;
				pos.doMove(move, state);
				// 繰り返しになる場合、探索終了
				if (pos.isDraw() == RepetitionDraw) {
					pos.undoMove(move);
					return;
				}

				moves.emplace_back(move);

				// 次の手を探索
				make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves);

				pos.undoMove(move);
			}
		}
	}
	else {
		// 定跡を使用
		std::vector<BookEntry>* entries;

		// 局面が定跡にあるか確認
		auto itr = bookMap.find(key);
		if (itr != bookMap.end()) {
			entries = &itr->second;
		}
		else {
			// 定跡にない場合、探索結果を使う
			itr = outMap.find(key);

			if (itr == outMap.end()) {
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

		size_t selected = 0;
		if (dist_minmax(g_randomTimeSeed) < book_minmax_prob_opp) {
			// 一定の確率でmin-maxで選ぶ
			const auto& entry = select_best_book_entry(pos, outMap, *entries);
			selected = &entry - &(*entries)[0];
		}
		else {
			// 確率的に手を選択
			std::vector<double> probabilities;
			for (const auto& entry : *entries) {
				const auto probability = std::pow((double)entry.count, book_reciprocal_temperature);
				probabilities.emplace_back(probability);
			}
			std::discrete_distribution<std::size_t> dist(probabilities.begin(), probabilities.end());
			selected = dist(g_randomTimeSeed);
		}
		const Move move = move16toMove(Move(entries->at(selected).fromToPro), pos);

		StateInfo state;
		pos.doMove(move, state);
		moves.emplace_back(move);

		// 次の手を探索
		make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves);

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

// 定跡マージ
int merge_book(std::map<Key, std::vector<BookEntry> >& outMap, const std::string& merge_file) {
	// ファイル更新がある場合のみ処理する
	static time_t prev_time = 0;
	struct stat st;
	if (stat(merge_file.c_str(), &st) != 0)
		return 0;
	if (st.st_ctime == prev_time)
		return 0;
	prev_time = st.st_ctime;

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
#endif
