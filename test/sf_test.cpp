#include "gtest/gtest.h"

#include "init.hpp"
#include "usi.hpp"
#include "sf_position.h"
#include "sf_movepick.h"
#include "sf_history.h"

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

TEST(StockfishTest, probcut_move_picker) {
    initTable();

    Stockfish::Position pos;
    pos.set("lr5nl/3g1kg2/2nspps1b/p1pp2pPp/1p7/P1PPSPP1P/1PS1P1N2/2GK1G3/LN5RL b BP 67");

    Stockfish::CapturePieceToHistory captureHistory;
    Stockfish::MovePicker mp(pos, Stockfish::Move::none(), -49, &captureHistory);

    Stockfish::Move move;
    while ((move = mp.next_move()) != Stockfish::Move::none())
    {
        assert(move.is_ok());

        if (!pos.legal(move))
            continue;

        std::cerr << move.toUSI();
    }
}
