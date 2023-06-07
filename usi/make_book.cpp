﻿#ifdef MAKE_BOOK
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
// 千日手の評価値
extern float draw_value_black;
extern float draw_value_white;
extern float eval_coef;
Score draw_score_black;
Score draw_score_white;
// 相手定跡から外れた場合USIエンジンを使う
std::unique_ptr<USIBookEngine> usi_book_engine;
int usi_book_engine_nodes;
double usi_book_engine_prob = 1.0;
// 自分の手番でも一定確率でUSIエンジンを使う
int usi_book_engine_nodes_own;
double usi_book_engine_prob_own = 0.0;
// αβ探索で特定局面の評価値を置き換える
std::map<Key, Score> book_key_eval_map;

inline Move UctSearchGenmoveNoPonder(Position* pos, const std::vector<Move>& moves) {
	Move move;
	return UctSearchGenmove(pos, book_starting_pos_key, moves, move);
}

bool make_book_entry_with_uct(Position& pos, LimitsType& limits, const Key& key, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const std::vector<Move>& moves) {
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
	if (itr_searched != searched.end()/* && itr_searched->second.depth <= pos.gamePly()*/ && itr_searched->second.beta >= beta) {
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
	Score trusted_score = entries[0].score;
	for (const auto& entry : entries) {
		// 訪問回数が少ない評価値は信頼しない
		if (entry.score < trusted_score)
			trusted_score = entry.score;
		const Move move = move16toMove(Move(entry.fromToPro), pos);
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
			value = std::max(value, book_search(pos, outMap, -beta, -alpha, trusted_score, searched));
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
			const auto ret = book_search(pos, outMap, -beta, -alpha, ScoreNotEvaluated, searched);
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
	return -value;
}

std::tuple<int, u16, Score> select_best_book_entry(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves) {
	const Key key = Book::bookKey(pos);

	Score alpha = -ScoreInfinite;
	const BookEntry* best = nullptr;
	BookEntry tmp; // entriesにない要素を返す場合
	std::map<Key, Searched> searched;
	Score trusted_score = entries[0].score;
	for (const auto& entry : entries) {
		// 訪問回数が少ない評価値は信頼しない
		if (entry.score < trusted_score)
			trusted_score = entry.score;
		const Move move = move16toMove(Move(entry.fromToPro), pos);
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
			value = book_search(pos, outMap, -ScoreInfinite, -alpha, trusted_score, searched);
		}
		pos.undoMove(move);
		//debug_moves.pop_back();
		//std::cout << move.toUSI() << "\t" << entry.score << "\t" << value << std::endl;

		if (value > alpha) {
			best = &entry;
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
			const auto ret = book_search(pos, outMap, -ScoreInfinite, -alpha, ScoreNotEvaluated, searched);
			value = ret;
		}
		pos.undoMove(move);
		//debug_moves.pop_back();
		//std::cout << move.toUSI() << "\t" << ScoreNotEvaluated << "\t" << value << std::endl;

		if (value > alpha) {
			tmp.fromToPro = move16;
			best = &tmp;
			alpha = value;
		}
	}
	return { (int)(best - &entries[0]), best->fromToPro, alpha };
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
					const auto usi_result = usi_book_engine->Go(book_pos_cmd, moves, usi_book_engine_nodes_own);
					std::cout << "usi move : " << depth << " " << usi_result.info << std::endl;
					if (usi_result.bestMove == "resign" || usi_result.bestMove == "win")
						return;
					move = usiToMove(pos, usi_result.bestMove);
				}
				else {
					const auto& entries = itr->second;
					// 一定の確率でmin-maxで選ぶ
					int index;
					u16 move16;
					Score score;
					std::tie(index, move16, score) = (dist_minmax(g_randomTimeSeed) < book_minmax_prob) ? select_best_book_entry(pos, outMap, entries, moves) : std::make_tuple(0, entries[0].fromToPro, entries[0].score);

					// 評価値が閾値を超えた場合、探索終了
					if (std::abs(score) > book_eval_threshold) {
						std::cout << book_pos_cmd;
						for (Move move : moves) {
							std::cout << " " << move.toUSI();
						}
						std::cout << "\nentry.score: " << score << std::endl;
						return;
					}

					move = move16toMove(Move(move16), pos);
					if (index != 0)
						std::cout << "best move : " << depth << " " << index << " " << move.toUSI() << std::endl;
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
			if (outMap.find(key) == outMap.end()) {
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
		if (itr == bookMap.end() && usi_book_engine && dist_minmax(g_randomTimeSeed) < usi_book_engine_prob) {
			// 相手定跡から外れた場合USIエンジンを使う
			const auto usi_result = usi_book_engine->Go(book_pos_cmd, moves, usi_book_engine_nodes);
			std::cout << "usi move : " << depth << " " << usi_result.info << std::endl;
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

			u16 selected_move16;
			if (itr == bookMap.end() && dist_minmax(g_randomTimeSeed) < book_minmax_prob_opp) {
				// 一定の確率でmin-maxで選ぶ
				const auto& entry = select_best_book_entry(pos, outMap, *entries, moves);
				selected_move16 = std::get<u16>(entry);
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
				selected_move16 = entries->at(selected_index).fromToPro;
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
				selected_move16 = entries->at(selected_index).fromToPro;
			}
			move = move16toMove(Move(selected_move16), pos);
		}

		StateInfo state;
		pos.doMove(move, state);
		moves.emplace_back(move);

		// 次の手を探索
		make_book_inner(pos, limits, bookMap, outMap, count, depth + 1, isBlack, moves, select_best_book_entry);

		pos.undoMove(move);
	}
}

void make_book_alpha_beta(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves) {
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
	u16 move16;
	Score score;
	std::tie(index, move16, score) = select_best_book_entry(pos, bookMap, entries, moves);

	// 最善手を登録
	auto& minMaxBookEntry = bookMapMinMax[key];
	minMaxBookEntry.move16 = move16;
	minMaxBookEntry.score = score;
	minMaxBookEntry.depth = pos.gamePly();

	// 最善手を指す
	const Move bestMove = move16toMove(Move(move16), pos);
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

void init_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const int nodes, const double prob, const int nodes_own, const double prob_own) {
	if (engine_path == "")
		return;
	usi_book_engine = create_usi_book_engine(engine_path, engine_options);
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

		// 出力定跡に含まれる局面は除外する
		for (const auto& kv : outMap) {
			exists.emplace(kv.first);
		}

		enumerate_positions(pos, bookMap, positions, exists);
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
					outMap[key].emplace_back(be);
					if (outMap.size() % 10000 == 0)
						std::cout << "progress: " << outMap.size() * 100 / positions.size() << "%" << std::endl;
			}
		}
	}
}

// 合流局面を列挙する
struct Junction {
	Ply depth;
	bool searched;
};
void enumerate_junctions(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, Junction>& junctions, std::unordered_map<Key, Ply>& exists) {
	const Key key = Book::bookKey(pos);
	auto itr = bookMap.find(key);
	if (itr == bookMap.end())
		return;

	// 合流
	{
		const auto itr_exists = exists.emplace(key, pos.gamePly());
		if (!itr_exists.second) {
			Junction junction = { std::min(pos.gamePly(), itr_exists.first->second), false };
			auto itr_junction = junctions.emplace(key, junction);
			if (!itr_junction.second) {
				auto& junction2 = itr_junction.first->second;
				if (pos.gamePly() < junction2.depth)
					junction2.depth = pos.gamePly();
			}
			return;
		}
	}

	// Stack overflowを避けるためヒープに確保する
	for (auto ml = std::make_unique<MoveList<LegalAll>>(pos); !ml->end(); ++(*ml)) {
		const Move move = ml->move();
		auto state = std::make_unique<StateInfo>();
		pos.doMove(move, *state);
		enumerate_junctions(pos, bookMap, junctions, exists);
		pos.undoMove(move);
	}
}

// 評価値が割れる局面を延長する
void diff_eval_inner(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, std::unordered_map<Key, Junction>& junctions, LimitsType& limits, const Score diff, std::vector<Move> moves, const std::string& outFileName) {
	const Key key = Book::bookKey(pos);
	auto itr = outMap.find(key);
	if (itr == outMap.end())
		return;

	// 合流
	{
		const auto& itr_junction = junctions.find(key);
		if (itr_junction != junctions.end()) {
			if (!itr_junction->second.searched && pos.gamePly() == itr_junction->second.depth)
				itr_junction->second.searched = true;
			else
				return;
		}
	}

	// 評価値差分
	{
		const auto itr_book = bookMap.find(key);
		if (itr_book != bookMap.end()) {
			const auto& best_entry = itr_book->second[0];
			const Score score = itr->second[0].score;
			if (score * best_entry.score < 0 && std::abs(score - best_entry.score) >= diff) {
				// 評価値の符号が異なり、差がdiff以上
				const Move move = move16toMove(Move(best_entry.fromToPro), pos);
				Key key_after = Book::bookKeyAfter(pos, key, move);
				if (outMap.find(key_after) == outMap.end()) {
					// 最善手が定跡にない場合
					std::cout << "diff: " << score << ", " << best_entry.score << std::endl;
					// 最善手を指して、定跡を延長
					StateInfo state;
					int count = 0;
					pos.doMove(move, state);
					moves.emplace_back(move);
					make_book_entry_with_uct(pos, limits, key_after, outMap, count, moves);
					moves.pop_back();
					pos.undoMove(move);
					// 保存
					saveOutmap(outFileName, outMap);
				}
			}
		}
	}

	// Stack overflowを避けるためヒープに確保する
	for (auto ml = std::make_unique<MoveList<LegalAll>>(pos); !ml->end(); ++(*ml)) {
		const Move move = ml->move();
		auto state = std::make_unique<StateInfo>();
		pos.doMove(move, *state);
		moves.emplace_back(move);
		diff_eval_inner(pos, bookMap, outMap, junctions, limits, diff, moves, outFileName);
		moves.pop_back();
		pos.undoMove(move);
	}
}

// 評価値が割れる局面を延長する
void diff_eval(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, LimitsType& limits, const Score diff, const std::string& outFileName) {
	// 合流局面を列挙する
	std::unordered_map<Key, Junction> junctions;
	{
		std::unordered_map<Key, Ply> exists;
		enumerate_junctions(pos, outMap, junctions, exists);

		std::cout << "positions: " << exists.size() << std::endl;
	}

	// 評価値が割れる局面を延長する
	std::vector<Move> moves;
	diff_eval_inner(pos, bookMap, outMap, junctions, limits, diff, moves, outFileName);
}
#endif
