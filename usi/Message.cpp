#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "Message.h"
#include "UctSearch.h"

using namespace std;


bool debug_message = true;


////////////////////////////////////
//  エラーメッセージの出力の設定  //
////////////////////////////////////
void
SetDebugMessageMode(const bool flag)
{
	debug_message = flag;
}

bool
GetDebugMessageMode()
{
	return debug_message;
}

///////////////////////
//  探索の情報の表示  //
///////////////////////
void
PrintPlayoutInformation(const uct_node_t *root, const po_info_t *po_info, const double finish_time, const int pre_simulated)
{
	double winning_percentage = (double)root->win / root->move_count;

	if (!debug_message) return;

	cout << "info string " << "All Playouts       :  " << setw(7) << root->move_count << endl;
	cout << "info string " << "Pre Simulated      :  " << setw(7) << pre_simulated << endl;
	cout << "info string " << "Thinking Time      :  " << setw(7) << finish_time << " sec" << endl;
	cout << "info string " << "Winning Percentage :  " << setw(7) << (winning_percentage * 100) << "%" << endl;
	if (finish_time != 0.0) {
		cout << "info string " << "Playout Speed      :  " << setw(7) << (int)(po_info->count / finish_time) << " PO/sec " << endl;
	}
}

//////////////////////
//  探索時間の出力  //
/////////////////////
void
PrintPlayoutLimits(const double time_limit, const int playout_limit)
{
	if (!debug_message) return;

	cout << "info string " << "Time Limit    : " << time_limit << " Sec" << endl;
	cout << "info string " << "Playout Limit : " << playout_limit << " PO" << endl;
}

////////////////////////////////////////
//  再利用した探索回数の出力          //
////////////////////////////////////////
void
PrintReuseCount(const int count)
{
	if (!debug_message) return;

	cout << "info string " << "Reuse : " << count << " Playouts" << endl;
}
