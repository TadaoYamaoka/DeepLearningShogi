#include "gtest/gtest.h"

#include "init.hpp"
#include "usi.hpp"
#include "sf_position.h"

TEST(StockfishTest, see_ge) {
    initTable();

    Stockfish::Position pos;
    pos.set("ln1g1gsnl/rs3k1b1/pppppppp1/8p/7N1/2P3P1P/PP1PPP1P1/1B1G3R1/LNS1KGS1L b - 13");

    const Move m = usiToMove(pos, "2e3c+");
    pos.see_ge(m, -83);

}

TEST(StockfishTest, legal) {
    initTable();

    Stockfish::Position pos;
    pos.set("lnsg1gsnl/1r1k5/ppppppppb/8p/9/P7P/1PPPPPPP1/1B1K3R1/LNSG1GSNL b - 7");

    const Move m(70953); // 5g5f
    EXPECT_FALSE(pos.legal(m));

}
