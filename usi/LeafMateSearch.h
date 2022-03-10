#pragma once

#include "UctSearch.h"

void InitLeafMateSearch(const int threads);

void RunLeafMateSearch();

void JoinLeafMateSearch();

void StopLeafMateSearch();

void QueuingLeafMateRequest(std::unique_ptr<Position>& pos, StateListPtr& states, child_node_t* uct_child);
