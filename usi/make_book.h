﻿#pragma once
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

struct MinMaxBookEntry {
	u16 move16;
	Score score;
	int depth;
	//std::vector<Move> moves;
};

void read_book(const std::string& bookFileName, std::map<Key, std::vector<BookEntry> >& bookMap);
int merge_book(std::map<Key, std::vector<BookEntry> >& outMap, const std::string& merge_file);
void make_book_inner(Position& pos, LimitsType& limits, std::map<Key, std::vector<BookEntry> >& bookMap, std::map<Key, std::vector<BookEntry> >& outMap, int& count, const int depth, const bool isBlack, std::vector<Move>& moves);
void minmax_book(Position& pos, std::map<Key, MinMaxBookEntry>& bookMapMinMax, const Color make_book_color);
std::string getBookPV(Position& pos, const std::string& fileName);
void init_usi_book_engine(const std::string& engine_path, const std::string& engine_options, const int nodes, const double prob, const int nodes_own, const double prob_own);
void init_book_key_eval_map(const std::string& str);
void extract_considering_draw_positions(Position& pos, const std::map<Key, std::vector<BookEntry> >& bookMap, std::vector<Move> histories, std::ofstream& ofs, std::set<Key>& exists);
#endif