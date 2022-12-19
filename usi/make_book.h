#pragma once
#ifdef MAKE_BOOK
#include "book.hpp"

extern std::map<Key, std::vector<BookEntry> > bookMap;
extern Key book_starting_pos_key;
extern std::string book_pos_cmd;
extern int make_book_sleep;
extern bool use_book_policy;
extern bool use_interruption;
extern int book_eval_threshold;
extern double book_visit_threshold;
extern double book_cutoff;
extern double book_reciprocal_temperature;
// MinMaxで選ぶ確率
extern double make_book_minmax_prob;
extern double make_book_minmax_prob_opp;
// 千日手の評価値
extern float draw_value_black;
extern float draw_value_white;
extern float eval_coef;
extern Score draw_score_black;
extern Score draw_score_white;

void read_book(const std::string& bookFileName, std::map<Key, std::vector<BookEntry> >& bookMap);
int merge_book(std::map<Key, std::vector<BookEntry> >& outMap, const std::string& merge_file);
void make_book_inner(Position& pos, LimitsType& limits, std::map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves);
#endif