#include "position.hpp"
#include "move.hpp"
#include "init.hpp"
#include "dfpn.h"

#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_limit(DfPn& dfpn, const char* sfen, int draw_ply, int max_depth, bool expected) {
    Position pos;
    pos.set(sfen);
    const auto before = pos.toSFEN();
    check(!pos.inCheck() && pos.mateMoveIn1Ply(), "fixture must have a mate in one");
    DfPn::set_draw_ply(draw_ply);
    dfpn.set_maxdepth(max_depth);
    check(dfpn.dfpn(pos) == expected, "unexpected mate result at ply limit");
    check(pos.toSFEN() == before, "search changed the position");
    check(dfpn.searchedNode == 0, "mate in one must not expand search nodes");
    if (expected) {
        const auto [pv, depth, move] = dfpn.get_pv(pos);
        check(depth == 1 && move == pos.mateMoveIn1Ply(), "mate PV changed");
    }
}
}

int main() {
    try {
        initTable();
        Position::initZobrist();
        DfPn::set_hashsize(1);
        DfPn dfpn;
        dfpn.init();

        const char* white = "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262";
        check_limit(dfpn, white, 261, 31, false);
        check_limit(dfpn, white, 262, 31, true);
        check_limit(dfpn, white, 263, 31, true);

        const char* black = "lns3kn1/1r4g2/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G2Pl4p 53";
        check_limit(dfpn, black, 52, 31, false);
        check_limit(dfpn, black, 53, 31, true);
        check_limit(dfpn, black, INT_MAX - 1, 31, true);
        check_limit(dfpn, black, 52, 2, false);
        check_limit(dfpn, black, 53, 2, true);

        std::cout << "dfpn mate-in-one ply limit: OK\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
