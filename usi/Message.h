#pragma once

#include <string>
#include "position.hpp"
#include "UctSearch.h"


//  エラーメッセージの出力の設定
void SetDebugMessageMode(const bool flag);
bool GetDebugMessageMode();

//  探索の情報の表示
void PrintPlayoutInformation(const uct_node_t *root, const po_info_t *po_info, const double finish_time, const int pre_simulated);

//  探索時間の出力
void PrintPlayoutLimits(const double time_limit, const int playout_limit);

//  再利用した探索回数の出力
void PrintReuseCount(const int count);

extern bool debug_message;