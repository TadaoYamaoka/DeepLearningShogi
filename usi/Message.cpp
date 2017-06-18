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

//////////////////////
//  探索時間の出力  //
/////////////////////
void
PrintPlayoutLimits(const double time_limit, const int playout_limit)
{
	if (!debug_message) return;

	cerr << "Time Limit    : " << time_limit << " Sec" << endl;
	cerr << "Playout Limit : " << playout_limit << " PO" << endl;
}

////////////////////////////////////////
//  再利用した探索回数の出力          //
////////////////////////////////////////
void
PrintReuseCount(const int count)
{
	if (!debug_message) return;

	cerr << "Reuse : " << count << " Playouts" << endl;
}
