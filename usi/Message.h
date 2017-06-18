#pragma once

#include <string>
#include "position.hpp"
#include "UctSearch.h"


//  エラーメッセージの出力の設定
void SetDebugMessageMode(const bool flag);
bool GetDebugMessageMode();

//  盤面の表示
void PrintBoard(const Position *pos);
void PrintRate(const Position *pos);

//  連の情報の表示              
//    呼吸点の数, 座標          
//    連を構成する石の数, 座標  
//    隣接する敵の連のID
void PrintString(const Position *pos);

//  各座標の連IDの表示  
void PrintStringID(const Position *pos);

//  連リストの繋がりを表示(Debug用)
void PrintStringNext(const Position *pos);

//  合法手である候補手を表示 
void PrintLegal(const Position *pos, const int color);

//  オーナーの表示
void PrintOwner(const uct_node_t *root, const int color, double *own);

//  最善応手列の表示
void PrintBestSequence(const Position *pos, const uct_node_t *uct_node, const int root, const int start_color);
void PrintMoveStat(std::ostream& out, const Position *pos, const uct_node_t *uct_node, int current_root);

//  探索の情報の表示
void PrintPlayoutInformation(const uct_node_t *root, const po_info_t *po_info, const double finish_time, const int pre_simulated);

//  座標の出力
void PrintPoint(const int pos);

//  コミの値の出力
void PrintKomiValue(void);

//  Ponderingのプレイアウト回数の出力
void PrintPonderingCount(const int count);

//  探索時間の出力
void PrintPlayoutLimits(const double time_limit, const int playout_limit);

//  再利用した探索回数の出力
void PrintReuseCount(const int count);
