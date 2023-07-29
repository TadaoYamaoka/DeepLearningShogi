#pragma once
#ifdef MAKE_BOOK
#include "book.hpp"

extern std::unordered_map<Key, std::vector<BookEntry> > bookMap;
extern Key book_starting_pos_key;
extern std::string book_pos_cmd;
extern int make_book_sleep;
extern bool use_interruption;
extern int book_eval_threshold;
extern double book_visit_threshold;
extern double book_cutoff;
extern double book_reciprocal_temperature;
extern bool book_best_move;
extern Score book_eval_diff;
// MinMaxで選ぶ確率
extern double book_minmax_prob;
extern double book_minmax_prob_opp;
// MinMaxのために相手定跡の手番でも探索する
extern bool make_book_for_minmax;
// 千日手の評価値
extern float draw_value_black;
extern float draw_value_white;
extern float eval_coef;
extern Score draw_score_black;
extern Score draw_score_white;

extern int book_mcts_playouts;
extern int book_mcts_threads;
extern float book_mcts_temperature;
extern bool book_mcts_debug;


struct MinMaxBookEntry {
	u16 move16;
	Score score;
	int depth;
	//std::vector<Move> moves;
};

void read_book(const std::string& bookFileName, std::unordered_map<Key, std::vector<BookEntry> >& bookMap);
int merge_book(std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::string& merge_file, const bool check_file_time=true);
typedef std::tuple<int, u16, Score>(*select_best_book_entry_t)(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves);
void make_book_inner(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves, select_best_book_entry_t select_best_book_entry);
void make_book_alpha_beta(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves);
void make_book_iddfs(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves);
std::tuple<int, u16, Score> book_search_iddfs(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves);
void make_book_mcts(Position& pos, LimitsType& limits, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves);
std::tuple<int, u16, Score> select_best_book_entry(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves);
std::tuple<int, u16, Score> parallel_uct_search(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& outMap, const std::vector<BookEntry>& entries, const std::vector<Move>& moves);
void make_minmax_book(Position& pos, std::unordered_map<Key, MinMaxBookEntry>& bookMapMinMax, const Color make_book_color, select_best_book_entry_t select_best_book_entry);
void make_mcts_book(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap);
void saveOutmap(const std::string& outFileName, const std::unordered_map<Key, std::vector<BookEntry> >& outMap);
void saveOutmap(const std::string& outFileName, const std::map<Key, std::vector<BookEntry> >& outMap);
std::string getBookPV(Position& pos, const std::string& fileName);
void init_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const int nodes, const double prob, const int nodes_own, const double prob_own);
void init_book_key_eval_map(const std::string& str);
void output_none_connect_positions(Position& pos, std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_set<Key>& exists, std::ofstream& ofs, int& count);
void enumerate_positions(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::vector<std::pair<HuffmanCodedPos, const std::vector<BookEntry>*>>& positions, std::unordered_set<Key>& exists, const std::map<Key, std::vector<BookEntry> >& skip = {});
void eval_positions_with_usi_engine(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, const std::string& engine_path, const std::string& engine_options, const int nodes, const int engine_num);
void diff_eval(Position& pos, const std::unordered_map<Key, std::vector<BookEntry> >& bookMap, std::unordered_map<Key, std::vector<BookEntry> >& outMap, LimitsType& limits, const Score diff, const std::string& outFileName);
void make_all_minmax_book(Position& pos, std::map<Key, std::vector<BookEntry> >& outMap, const Color make_book_color, const int threads);
#endif