#pragma once

void dfpn_init();
bool dfpn(Position& r);
void dfpn_stop();
void dfpn_set_maxdepth(uint32_t depth);
Move dfpn_move(Position& pos);
