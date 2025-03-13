#include "gtest/gtest.h"

#include "init.hpp"
#include "usi.hpp"
#include "sf_position.h"
#include "sf_movepick.h"
#include "sf_history.h"
#include "sf_evaluate.h"

#include "cppshogi.h"

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

TEST(StockfishTest, embedding_layers) {
    using namespace Stockfish;

    initTable();

    constexpr int NUM_ITERATIONS = 10000;

    constexpr int NUM_EMBEDDINGS1 = PieceTypeNum * 2;
    constexpr int NUM_EMBEDDINGS2 = MAX_PIECES_IN_HAND_SUM * 2 + 1;

    // 32バイトアライメントでメモリ確保
    float* embedding_table1 = static_cast<float*>(_mm_malloc(NUM_EMBEDDINGS1 * Eval::EMBEDDING_DIM * sizeof(float), 32));
    float* embedding_table2 = static_cast<float*>(_mm_malloc(NUM_EMBEDDINGS2 * Eval::EMBEDDING_DIM * sizeof(float), 32));
    float* output = static_cast<float*>(_mm_malloc(Eval::EMBEDDING_DIM * (int)SquareNum * sizeof(float), 32));

    if (!embedding_table1 || !embedding_table2 || !output) {
        std::cerr << "Error: Failed to allocate aligned memory!" << std::endl;
        return;
    }

    // 乱数生成器の初期化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 埋め込みテーブルを乱数で初期化
    for (int i = 0; i < NUM_EMBEDDINGS1 * Eval::EMBEDDING_DIM; ++i) {
        embedding_table1[i] = dist(gen);
    }
    for (int i = 0; i < NUM_EMBEDDINGS2 * Eval::EMBEDDING_DIM; ++i) {
        embedding_table2[i] = dist(gen);
    }

    Stockfish::Position pos;
    pos.set("+P4g1nl/4g1k2/p3pp1p1/2p2Lp1p/2n4P1/3p1N2P/P1Ps+rPP2/LpK3BS1/R4G2L b B2SNPgp 103");


    // ベンチマーク実行
    std::cout << "\nRunning benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        Eval::embedding_layers(pos, embedding_table1, embedding_table2, output);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // 結果出力
    std::cout << "\nBenchmark results:" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << elapsed.count() << " ms" << std::endl;
    std::cout << "Average time per position: " << std::fixed << std::setprecision(4)
        << elapsed.count() / NUM_ITERATIONS << " ms" << std::endl;
    std::cout << "Positions per second: " << std::fixed << std::setprecision(0)
        << (NUM_ITERATIONS * 1000) / elapsed.count() << std::endl;

    // メモリ解放
    _mm_free(embedding_table1);
    _mm_free(embedding_table2);
    _mm_free(output);
}

TEST(StockfishTest, read_parameters) {
    using namespace Stockfish;

    auto result = Eval::read_parameters(R"(..\..\dlshogi\model-008.bin)");
    EXPECT_TRUE(result);
}

TEST(StockfishTest, evaluate) {
    using namespace Stockfish;

    initTable();

    auto result = Eval::read_parameters(R"(..\..\dlshogi\model-008.bin)");
    EXPECT_TRUE(result);


    Stockfish::Position pos;
    pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
    auto v = Eval::evaluate(pos);

    EXPECT_GT(v, -200);
    EXPECT_LT(v, 200);

    pos.set("l3l2+Nl/1r1s5/p3kg1pp/2P1n4/3B1pP2/1P1S2R2/PG1PPP2P/4K4/LNS5+b w GSN5Pgp 1");
    v = Eval::evaluate(pos);

    EXPECT_GT(v, -300);
    EXPECT_LT(v, -100);

    pos.set("4+r3l/3g1ksB1/3s1s1p1/BLp2pP2/L2gp2Pp/4PN3/3P1+n2P/1p2KG3/6R1L b GNPsn6p 1");
    v = Eval::evaluate(pos);

    EXPECT_GT(v, 0);
    EXPECT_LT(v, 100);
}

TEST(StockfishTest, evaluate_perf) {
    using namespace Stockfish;

    initTable();

    auto result = Eval::read_parameters(R"(..\..\dlshogi\model-008.bin)");
    EXPECT_TRUE(result);

    Stockfish::Position pos[100];
    pos[0].set("l2gk3l/3p3g1/p5n1p/9/1P+B3b2/2P2r3/P1N1P1N1P/1p1SG3s/L1GK4L w SN6Prs3p 1");
    pos[1].set("8l/+P1g1k1gs1/2nsppnp1/1rp5p/3P5/1SP2PP2/1PN1P1N1P/1G1K1GS2/+l6RL w B3Pbl2p 1");
    pos[2].set("ln1g4l/3k2g2/2ppppr1p/p8/3s3R1/P1s2N3/1PSPPP2P/2G6/LNK2G1NL b B2Pbs4p 1");
    pos[3].set("l8/9/3Sp4/p1p2pkgp/1L3sgp1/P1Pp4P/1P+B1P+s3/1RG1+n4/LK5NL b RGN3Pbsn3p 1");
    pos[4].set("ln1gk2nl/1rs3gs1/p1pppp1pp/9/8b/2P4R1/PP1PPPP1P/1BGK2S2/LNS2G1NL b 2Pp 1");
    pos[5].set("ln1g4l/5kg2/4p2sp/pr1pspPp1/2P6/1P1PBP3/P1S1P1+n1P/4KSGR1/LN1G4L b 4Pbn 1");
    pos[6].set("1nsk2snl/Ppp1gb1p1/3p1g2p/2P1p1l2/2G+R5/6p2/2BPP1N1P/3SK1S2/1N3G2L b L3Pr3p 1");
    pos[7].set("lnsgk2nl/1r3sg2/p1ppp1bpp/5pp2/1p5P1/2P6/PPBPPPP1P/1SG2S1R1/LN2KG1NL b - 1");
    pos[8].set("l2g2snl/2s2kgb1/2nppp1p1/pr6p/2P6/P5PRP/1P1PPP3/1BG1K1S2/LNS2G1NL w 3Pp 1");
    pos[9].set("lnsg1g1nl/1k2rb3/1ppp4p/p4p1S1/4p4/P1s1P4/1P1P1P2P/1BK1G2R1/LNSG3NL b 2P3p 1");
    pos[10].set("ln1g2knl/1r4gp1/p2ppp2p/2psbs1P1/1P7/1SP4R1/P1NPPPP1P/2G1K1S2/L4G1NL w B2P 1");
    pos[11].set("l6nl/6gk1/2+P+P1gspp/1SK3p2/PRG6/p3P4/1PN+sb3P/2P6/L3+pb2L w RSN5Pgnp 1");
    pos[12].set("lr5nl/2g1k1gs1/p2spp1p1/3p2P2/Pp1n3N1/2PBPS+b1p/1P1P1P3/1SKGG4/LN5RL w 3Pp 1");
    pos[13].set("1+R3B2l/l4pn1s/3ppggkp/2P1n1p2/P5+r2/2SBNP3/3PP4/1PK3+l2/LN1S5 b 5P2gs2p 1");
    pos[14].set("1n5nl/4k2g1/p2gpp1sp/2pP3p1/5N3/2s1B1G2/P1N1PP+B1P/3K1S3/+r1S3R1L w L3Pgl4p 1");
    pos[15].set("ln4Rnl/4s1s2/p1p2p1kp/3p2pp1/4n4/PPP6/2N1PPL2/1K2G2G1/L6P1 b RB3Pb2g2s2p 1");
    pos[16].set("lS5nl/5kg2/3gpp1p1/6s2/Pn4PPp/2nS1Bp2/1P1PP3P/4K1S2/L6RL w RG4Pbgn2p 1");
    pos[17].set("l6nl/2r1g4/4bgks1/p3p1Npp/2s2pS2/PPGPP2NP/3GBP3/1K5R1/L2s3NL b 2P5p 1");
    pos[18].set("l1r4nl/4gk3/3p1sgp1/pnP5p/2sS3P1/PP1PSPp2/2G1P3P/4G2R1/L1K4NL b BN2Pb3p 1");
    pos[19].set("ln5nl/1ks1G4/1pb4pp/4r1p2/P6P1/pPPP1pP1P/1K5R1/2S1G2B1/L2G3NL b G4P2snp 1");
    pos[20].set("l5knl/r3g2g1/4ps3/p4spp1/3pNNP1P/P1p2+BS2/1P1P5/2G2K2L/L6R1 w BN4Pgs3p 1");
    pos[21].set("lnsg1s3/4k1s1+R/p1pp1p2p/4p1p2/5n3/2PbP4/P1NP1PP1P/2G1K1S2/+rP1G1G1NL b LPbl2p 1");
    pos[22].set("l2s1k1nl/1P7/n2pgp1gp/1p7/P1p6/3P2P1P/4PP3/3S1K3/4S2GL w R2BNPrgsnl5p 1");
    pos[23].set("2+R3gnl/3g1skb1/p2p1p1p1/b1P1p1p2/4Pl3/4S3p/P4PSP1/2+r1G2KP/L4G1NL b S2NP4p 1");
    pos[24].set("ln1g1k1nl/1r1s1sgb1/p1pp1pp1p/4p2p1/1p7/2P6/PPBPPPP1P/2GS1S1R1/LN2KG1NL b p 1");
    pos[25].set("ln1gk2nl/1r1s1sgb1/p1pppp1pp/1p4p2/9/2PP3P1/PP1SPPP1P/1B5R1/LN1GKGSNL b - 1");
    pos[26].set("lr5nl/2s1k1g2/p3pps1p/2p1b1pp1/9/1P3BP1P/P1SPPPN2/1+p1GK1SR1/+p4G2L b G2nl2p 1");
    pos[27].set("lr5nl/2g1k1g2/2ns1psp1/p1ppp3p/7P1/P1PPPS3/1PSG1P2P/1KG3pR1/LN3b1NL b B2P 1");
    pos[28].set("ln1g2sn1/5kgbP/2spppppl/prp6/8p/P1P4R1/1P1PPPP2/1BG1K1S2/LNS2G1NL b 2p 1");
    pos[29].set("4gk1nl/5bn2/1GnpPSs2/r1p4pp/1p2pp3/2PP5/PPSG1P2P/2K1GS3/+lN5RL b bl4p 1");
    pos[30].set("ls1g2g1+R/4ksg2/p1Ppppn1p/2p6/1N7/P2B1P3/1+p1PP1P1P/1S2K1GP1/Lr4SNL w BNLP2p 1");
    pos[31].set("l2g3nl/1ks4r1/2ns2bpp/ppppp1pP1/5g3/PPPSP1P2/1SBP4P/L1G4R1/KNG4NL w Pp 1");
    pos[32].set("l4k1nl/1r3sgb1/p1ngsp1p1/2pp2p1p/1p4P2/P1PP1S3/1PSG1P2P/1BG4R1/LN1K3NL w 2Pp 1");
    pos[33].set("l1s2R2l/1kgsp4/3g2npp/pbp1n4/PNPp3P1/1S2P1P2/3PS3P/1KGG1N3/L4P2+b w R3Plp 1");
    pos[34].set("l4R2l/2g3+R2/ps2ppn1p/5k3/1p1bsb3/S1pN2P2/PP1P1NN1P/2G3S2/L4+p1KL b 2g6p 1");
    pos[35].set("l7l/1r2gk3/pPnpss1gp/2pb1p1p1/4p1P2/1nPB1S3/PG1P1P2P/3S3R1/LNK1G3L w N3Pp 1");
    pos[36].set("3g1k3/2s2l3/2np5/1r1+B3g1/l1p2p2p/pPPBP3P/3P1PP2/PSG1KSG2/LN5P+r w SNL3Pnp 1");
    pos[37].set("ln5nl/1r4ks1/2sp1ggp1/ppp1ppp2/7Pp/P1PP1SP2/1PS1PPN1P/1KG4R1/LN3G2L w Bb 1");
    pos[38].set("l4k1+Pl/5g3/p1npspn2/1r2bsG1p/2p6/1PPPpP2P/P2S2PpN/2GB1G3/LN1K3RL b 2Psp 1");
    pos[39].set("ln3k1nl/2r2sg2/2sp1gppp/p2+bppP2/1p4SP1/4PP3/PPSP4P/2GG3R1/LNK4NL b B2p 1");
    pos[40].set("lr5nl/3gk1gb1/p1ns1psp1/2p1p1p2/Pp1p4p/2P2PP2/1P1PPSN1P/1BGS1G3/LN1K3RL b P 1");
    pos[41].set("l1+r6/3sk3+R/pP2pp2p/2p1sg3/1n3n1P1/2PP2S2/P1SpPP2P/4K1G2/L1G4NL w GL3P2bnp 1");
    pos[42].set("l4B1nl/1r4g1k/p2p2sP1/4Npppp/P1S3P2/3PP2RP/nP+n1K4/2s1G4/L7L b B2G4Psp 1");
    pos[43].set("ln1g2s1l/2sk2g2/p1pppp2p/6bp1/7nB/2P2Pp2/PP1PP2PP/1S3KSr1/LNG2G1NL b RPp 1");
    pos[44].set("ln1g1gsnl/1r1s1k3/p1ppppbpp/1p4p2/9/2PP3P1/PP1SPPP1P/1B3S1R1/LN1GKG1NL w - 1");
    pos[45].set("l8/3sgs3/p1n1pk3/2r1lpp2/1p4+R2/4bBP2/PPSP1P2+p/2G3S2/LN1K1G3 b GNPnl6p 1");
    pos[46].set("ln1g2snl/1rs2kP2/p2ppp1+Pp/6p2/2P6/1S5g1/PPNPPP2P/3K2S2/+b2G1G2L b BL2Prnp 1");
    pos[47].set("7nl/1r3+P3/4S2pp/2p2ppk1/PP1n2Pg1/2Ppp2+n1/5L3/2G1K1G2/L1S4L1 b RBGSN2Pbs4p 1");
    pos[48].set("r6nl/2g3gk1/p3ps1p1/3pnpP1p/1pp3sP1/2lP2+p2/PPSBPP2P/K1GG5/LN5RL b BSnp 1");
    pos[49].set("ln1g5/1psk2g2/2ppps1pp/pP7/4+B1P2/2P6/2NPPBS+rP/1SGK5/L4G2L b NL3Prn2p 1");
    pos[50].set("lr3n2l/2g2kg2/2sp1p2p/pnN1psPp1/1S4S2/PP1GB1R2/1+n2PP2P/4K4/L1+p5L w BG4Pp 1");
    pos[51].set("l2g4l/2s1k1gs1/2npppn1p/pr1b2p2/6R2/2P3P2/PPNPPP2P/1SG1K1GS1/L6NL b B4P 1");
    pos[52].set("l6nl/4k1g2/p1g1pp2p/2ps2Pp1/6r2/2P6/PP1SPP2P/1S2K2RB/LN5NL b GSNPbg4p 1");
    pos[53].set("l6rl/3gk1g2/2nsbsnp1/p1ppppp1p/1p5P1/P1PPSPP1P/1P1SP1N2/1KGG5/LN2B2RL w - 1");
    pos[54].set("l7l/3gk3s/p1ns5/1r2p1gPp/2P1P2N1/1ppP1B3/P4P2P/1PG1G1B1+r/LN1K1S3 b SN4Plp 1");
    pos[55].set("l2gk2nl/3s2g2/p3p1spp/2pp1pp2/P2n3P1/2r3P2/3PPPS1P/1SGK3R1/LN3G1NL b BPb2p 1");
    pos[56].set("ls1nB3l/1ks3gs1/p1pppp2p/2P2b1P1/N1R6/P1S6/1P1PKPP1P/2G2N1r1/L3G2NL b 2Pg2p 1");
    pos[57].set("ln5nl/1r2gkgs1/p2pp2pp/2Bsb4/5p3/1PPP1S3/P1S1PP2P/2G4R1/LN2KG1NL b 4Pp 1");
    pos[58].set("l6rl/b3k1g2/p1gppp3/1P1s1ns1p/1pP4P1/P6RP/3PPPP1N/2G1KSG2/L1S5L w BN3Pn 1");
    pos[59].set("ln6l/3r1k3/p2gppg2/2p4Pp/9/P1PPP4/BPN3G+pP/1SG1K4/L7L b B2SN4Prsnp 1");
    pos[60].set("ln5nl/prgsk1gs1/3ppb1pp/P1P2pP2/1p1PP3P/2S4R1/LPN2P3/2GK2SB1/5G1NL w 2Pp 1");
    pos[61].set("l6rl/2k2g3/2n2p3/png1ps1Pp/1ppp5/P2P+BP2P/1PS1PSp2/2G2G1+b1/LNK2R2L b SPn2p 1");
    pos[62].set("ln1gk2nl/1r1s1sg2/p1ppppbpp/6p2/1p5P1/2PP5/PPB1PPP1P/2S1GS1R1/LN1GK2NL w - 1");
    pos[63].set("+P+R1g1k2l/2+b1r1gp1/p1p1ppn1p/3p2p2/9/2P6/P2PPPP1P/2G1K1Ss1/LNS2G1NL w SNLb2p 1");
    pos[64].set("lG1B3nl/5bgk1/p2p1s1p1/4pps2/3Pn2Pp/1SP3p2/PPNK3R1/1p7/L7L b RGSN5Pg 1");
    pos[65].set("l7l/1r3k1g1/p1n1p1npp/2b2pp2/6b2/Rp1S5/PP2PPP1P/1G2G4/LN2K2NL b G2S2Ps3p 1");
    pos[66].set("ln3+P1nl/6gk1/p1ppPp+bs1/7p1/3N2p1p/2Ps1P1P1/P5PSP/5+p1K1/LR5NL b RB2G2Pgs 1");
    pos[67].set("4kg2l/2s3+r2/ppp1Pps1p/9/5+bp2/1B7/PPS2P2P/5KSL1/LNPG3NL b R2N3P2g3p 1");
    pos[68].set("l5s1l/1sg1k1g2/3pppn1p/ppp2b1R1/2rn4P/P1P6/2NPPPP2/1SG1KS3/L3G2NL w B2P2p 1");
    pos[69].set("ln3gsn1/1r3b1kl/p1pp1g3/3sppppp/1pP6/3P2PPP/PP2PP1S1/2R1G1SK1/LN2BG1NL b - 1");
    pos[70].set("ln1gkn2l/1r5g1/p1pBp4/1p1p1sPpp/5l3/2P4R1/PPNPP3P/1SGK2S2/L5G1+b w SPn3p 1");
    pos[71].set("lrs5l/4+N2pg/1k+b1p+P2p/p1p6/1p7/3p5/PPSPP3P/K1GN2+r2/LNG5L w SNbgs5p 1");
    pos[72].set("l6nl/1r1k2gs1/2nspg1pp/p1pp1pP2/1p3bSP1/P1PP5/1PGGPP2P/2KS2R2/LN5NL w BP 1");
    pos[73].set("ln5nl/2r2kgb1/pp1gpps+Pp/1s7/2Pp5/4P1R2/PPBG1PP1P/2S2S3/LN1GK2NL w 3Pp 1");
    pos[74].set("l3g4/4g2kl/p3pgnp1/2p1s4/3+b2p2/P2p4p/4S1PP1/7KP/B4+RNNL b RNL3Pg2s4p 1");
    pos[75].set("+R5g1l/7k1/p4pnp1/3p1s1Pp/4ps3/1K1P2p2/P1P2P2P/2G1G2+b1/LN5NL w G2SNL2Prb3p 1");
    pos[76].set("ln1g1ksnl/1rs3gb1/p1p1pp1pp/1p1p2p2/9/2PP3P1/PP2PPP1P/1BGS1S1R1/LN2KG1NL b - 1");
    pos[77].set("lnsgkg1nl/1r4sb1/ppppp2pp/5pp2/9/P1P6/1P1PPPPPP/1BG2S1R1/LNS1KG1NL w - 1");
    pos[78].set("l5knl/4g1g2/p1np1pbs1/2rs3pp/1p4P2/3SPS1R1/PP1P1PN1P/2GBG4/LNK5L w 4Pp 1");
    pos[79].set("l1s3pnl/1+R1+Bk4/3s5/p1p1g3p/1P1ns4/P1Pp2P1P/5P3/1KG2G3/LN6L w GSN2Prb5p 1");
    pos[80].set("l7+P/2g1sg3/3+P4p/p3pp1p1/1PpN2P2/P1N5k/2B1B4/L1SG5/K5RP1 w RN2Pg2sn2l4p 1");
    pos[81].set("1+N1pl2nl/2+R2bk2/5ggp1/S1p1npp1p/3P3s1/1PPSp4/3K1P2P/9/2B4RL b 2GN3Psl3p 1");
    pos[82].set("ln1g3nl/1r3kgs1/p2ppp2p/2ps2Pp1/1p4S2/1PPB5/P1SPPP2P/2G4R1/LN1K1G1NL w Pbp 1");
    pos[83].set("4Rnk1l/+P8/3G1pgpp/2p6/3p2pP1/l1P1NPP1B/1P+nPKS2P/4G4/b1+p4+lL b RNPg3s2p 1");
    pos[84].set("ln1g2+R2/1slkl2p1/3p1p1gp/p1pnp1p2/3B1+b1+r1/2PP5/PGSNPP2P/2KS2G2/6P2 w NLs3p 1");
    pos[85].set("l3r3l/1kg3g2/1p1s1pnp1/p1ppps2p/6pP1/P1PS1P2P/1P2G4/1KS5B/L2R3NL w BN2Pgnp 1");
    pos[86].set("1r5nl/2kg5/l6gp/1sPppbs2/1n3pP2/pS1PP2S1/1P1GB2pP/2G4R1/LN1K3NL b P5p 1");
    pos[87].set("l2g4l/2s1k1gs1/2npppn2/p1p5p/6r2/P1P4RP/1P1PPPP1N/1G1SKGS2/LN6L b B2Pb2p 1");
    pos[88].set("ln3gsnl/1r2gk1b1/p2ppppp1/2ps5/1p2P3p/2PS5/PPBP1PPPP/4R1K2/LN1G1GSNL b - 1");
    pos[89].set("ln1g1ksnl/3r2gb1/ppp1pp1pp/3ps1p2/9/2PPP4/PP3PP1P/1BG1GS1R1/LNS1K2NL w P 1");
    pos[90].set("l6+R1/3skg3/p2p1g1Pp/4l1p2/2p2pP2/1P2L4/PG1PS1NpP/2s5R/+bN2K3L w G2N4Pbsp 1");
    pos[91].set("l+N5nl/5kg2/p1bg1p1s1/7pp/1Pp1P1p2/6P2/P+b1PpSN1P/5GK2/L6RL b RSNgs5p 1");
    pos[92].set("7nl/5gsk1/3s3pp/2p1ppg2/6r2/S1PPPP1L1/2SK2P1P/P2G3+B1/2+p4NL b BG4Pr2nlp 1");
    pos[93].set("l6rl/3k1p3/2ns4p/2pp1s1p1/pp2ps3/2PP4B/PPSG2+p1P/1KG6/LN6L w RGNPbgn3p 1");
    pos[94].set("l4B1nl/1P3g1k1/3+B3g1/p1r1psp2/1sp4Pp/P2PPSP2/1K1S2NpP/2G2G3/LNNr4L b 2P3p 1");
    pos[95].set("l2g2s1l/3k2g2/1sppp2pp/p3l4/5pN+B1/5nRNP/1P1PP1p2/+p1GKG4/6R1+l b BSNPs4p 1");
    pos[96].set("l2g2Bnl/1ks1r1s2/ppn1g2pp/2ppp4/P4p1P1/2P1P1R2/1PSP1P2P/2K1G4/LN1G3N+b b Pslp 1");
    pos[97].set("+Bng4nl/1ps2kgs1/lbpppp1pp/p1r6/1P4p2/2P4R1/PG1PPPP1P/4K1S2/LNS2G1NL b p 1");
    pos[98].set("1n5nl/1r4gk1/3+B1psp1/2g3p1p/Lp5P1/1nPP2P1P/1PS1P4/1K2G4/1N5RL w GSL5Pbsp 1");
    pos[99].set("ln1g4l/7r1/1k1pg3p/ppPs3b1/3S3p1/P2PB4/1PG5P/1KG3+p2/LNR5L w SNPsn6p 1");

    constexpr int num_iterations = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations / 100; ++iter) {
        for (int i = 0; i < 100; ++i) {
            Eval::evaluate(pos[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Performance: " << num_iterations << " iterations in "
        << elapsed.count() << " ms\n";
    std::cout << "Average time per iteration: " << elapsed.count() / num_iterations << " ms\n";
}
