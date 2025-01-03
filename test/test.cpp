#include "gtest/gtest.h"

#include <iostream>
#include <chrono>
#include <fstream>

#include "cppshogi.h"
#include "python_module.h"
#include "usi.hpp"

using namespace std;

TEST(HcpeTest, make_hcpe) {
	// hcpe作成
	initTable();
	const int num = 2;
	Position pos[num];
	pos[0].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
	pos[1].set("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1");

	vector<HuffmanCodedPosAndEval> hcpevec;
	std::ofstream ofs("test.hcpe", std::ios::binary);
	for (int i = 0; i < num; i++) {
		hcpevec.emplace_back(HuffmanCodedPosAndEval());
		HuffmanCodedPosAndEval& hcpe = hcpevec.back();
		hcpe.hcp = pos[i].toHuffmanCodedPos();
		MoveList<Legal> ml(pos[i]);
		hcpe.bestMove16 = static_cast<u16>(ml.move().value());
		hcpe.gameResult = BlackWin;
		hcpe.eval = 0;
	}
	ofs.write(reinterpret_cast<char*>(hcpevec.data()), sizeof(HuffmanCodedPosAndEval) * hcpevec.size());
}

TEST(Hcpe3Test, merge_cache) {
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();

	// hcpe3作成
	Position pos[3];
	pos[0].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
	pos[1].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL w - 2");
	pos[2].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2");

	u16 move_2g2f = (u16)usiToMove(pos[0], "2g2f").value();
	u16 move_9g9f = (u16)usiToMove(pos[0], "9g9f").value();
	u16 move_7g7f = (u16)usiToMove(pos[0], "7g7f").value();
	u16 move_8c8d = (u16)usiToMove(pos[1], "8c8d").value();
	u16 move_4a3b = (u16)usiToMove(pos[1], "4a3b").value();
	u16 move_8b3b = (u16)usiToMove(pos[2], "8b3b").value();

	std::vector<MoveVisits> candidates1_1;
	candidates1_1.emplace_back(MoveVisits{ move_2g2f, 3 });
	candidates1_1.emplace_back(MoveVisits{ move_9g9f, 1 });

	std::vector<MoveVisits> candidates1_2;
	candidates1_2.emplace_back(MoveVisits{ move_2g2f, 5 });
	candidates1_2.emplace_back(MoveVisits{ move_7g7f, 7 });

	std::vector<MoveVisits> candidates2;
	candidates2.emplace_back(MoveVisits{ move_8c8d, 4 });
	candidates2.emplace_back(MoveVisits{ move_4a3b, 1 });

	std::vector<MoveVisits> candidates3;
	candidates3.emplace_back(MoveVisits{ move_8b3b, 11 });

	std::ofstream ofs_hcpe3_1("test1.hcpe3", std::ios::binary);
	std::ofstream ofs_hcpe3_2("test2.hcpe3", std::ios::binary);

	HuffmanCodedPosAndEval3 hcpe3_1{ pos[0].toHuffmanCodedPos(), 2, BlackWin, 0 };
	ofs_hcpe3_1.write(reinterpret_cast<char*>(&hcpe3_1), sizeof(HuffmanCodedPosAndEval3));
	MoveInfo move_info1_1{ move_2g2f, 200, 2 };
	ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info1_1), sizeof(MoveInfo));
	ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates1_1.data()), sizeof(MoveVisits) * candidates1_1.size());
	MoveInfo move_info2{ move_8c8d, 210, 2 };
	ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info2), sizeof(MoveInfo));
	ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates2.data()), sizeof(MoveVisits) * candidates2.size());

	HuffmanCodedPosAndEval3 hcpe3_2{ pos[0].toHuffmanCodedPos(), 2, WhiteWin, 0 };
	ofs_hcpe3_2.write(reinterpret_cast<char*>(&hcpe3_2), sizeof(HuffmanCodedPosAndEval3));
	MoveInfo move_info1_2{ move_7g7f, 220, 2 };
	ofs_hcpe3_2.write(reinterpret_cast<char*>(&move_info1_2), sizeof(MoveInfo));
	ofs_hcpe3_2.write(reinterpret_cast<char*>(candidates1_2.data()), sizeof(MoveVisits) * candidates1_2.size());
	MoveInfo move_info3{ move_8b3b, 230, 1 };
	ofs_hcpe3_2.write(reinterpret_cast<char*>(&move_info3), sizeof(MoveInfo));
	ofs_hcpe3_2.write(reinterpret_cast<char*>(candidates3.data()), sizeof(MoveVisits) * candidates3.size());

	ofs_hcpe3_1.close();
	ofs_hcpe3_2.close();

	// cache作成
	extern size_t __load_hcpe3(const std::string& filepath, bool use_average, double a, double temperature, size_t &len);
	extern void __hcpe3_create_cache(const std::string& filepath);
	size_t len = 0;
	__load_hcpe3("test1.hcpe3", false, 600, 1, len);
	EXPECT_EQ(2, len);
	__hcpe3_create_cache("test1.cache");

	// __hcpe3_create_cacheはtrainingDataをクリアする
	len = 0;
	__load_hcpe3("test2.hcpe3", false, 600, 1, len);
	EXPECT_EQ(2, len);
	__hcpe3_create_cache("test2.cache");

	// merge_cache
	extern void __hcpe3_merge_cache(const std::string& file1, const std::string& file2, const std::string& out);
	__hcpe3_merge_cache("test1.cache", "test2.cache", "out.cache");

	// マージしたcache読み込み
	// インデックス読み込み
	std::ifstream cache("out.cache", std::ios::binary);
	size_t num;
	cache.read((char*)&num, sizeof(num));
	std::vector<size_t> cache_pos(num + 1);
	cache.read((char*)cache_pos.data(), sizeof(size_t) * num);
	cache.seekg(0, std::ios_base::end);
	cache_pos[num] = cache.tellg();
	EXPECT_EQ(3, num);

	struct Hcpe3CacheBuf {
		Hcpe3CacheBody body;
		Hcpe3CacheCandidate candidates[MaxLegalMoves];
	} buf;

	// pos1
	auto pos1 = cache_pos[0];
	const size_t candidate_num1 = ((cache_pos[1] - pos1) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
	cache.seekg(pos1, std::ios_base::beg);
	cache.read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num1);
	EXPECT_EQ(pos[0].toHuffmanCodedPos(), buf.body.hcp);
	EXPECT_EQ(1.17314f, buf.body.value);
	EXPECT_EQ(1.0f, buf.body.result);
	EXPECT_EQ(2, buf.body.count);
	EXPECT_EQ(3, candidate_num1);
	EXPECT_EQ(move_2g2f, buf.candidates[0].move16);
	EXPECT_EQ(3.0f / (3.0f + 1.0f) + 5.0f / (5.0f + 7.0f), buf.candidates[0].prob);
	EXPECT_EQ(move_9g9f, buf.candidates[1].move16);
	EXPECT_EQ(1.0f / (3.0f + 1.0f), buf.candidates[1].prob);
	EXPECT_EQ(move_7g7f, buf.candidates[2].move16);
	EXPECT_EQ(7.0f / (5.0f + 7.0f), buf.candidates[2].prob);

	// pos2
	auto pos2 = cache_pos[1];
	const size_t candidate_num2 = ((cache_pos[2] - pos2) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
	cache.seekg(pos2, std::ios_base::beg);
	cache.read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num2);
	EXPECT_EQ(pos[1].toHuffmanCodedPos(), buf.body.hcp);
	EXPECT_EQ(0.586415410f, buf.body.value);
	EXPECT_EQ(0.0f, buf.body.result);
	EXPECT_EQ(1, buf.body.count);
	EXPECT_EQ(2, candidate_num2);
	EXPECT_EQ(move_8c8d, buf.candidates[0].move16);
	EXPECT_EQ(4.0f / (4.0f + 1.0f), buf.candidates[0].prob);
	EXPECT_EQ(move_4a3b, buf.candidates[1].move16);
	EXPECT_EQ(1.0f / (4.0f + 1.0f), buf.candidates[1].prob);

	// pos3
	auto pos3 = cache_pos[2];
	const size_t candidate_num3 = ((cache_pos[3] - pos3) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
	cache.seekg(pos3, std::ios_base::beg);
	cache.read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num3);
	EXPECT_EQ(pos[2].toHuffmanCodedPos(), buf.body.hcp);
	EXPECT_EQ(0.594411194f, buf.body.value);
	EXPECT_EQ(1.0f, buf.body.result);
	EXPECT_EQ(1, buf.body.count);
	EXPECT_EQ(1, candidate_num3);
	EXPECT_EQ(move_8b3b, buf.candidates[0].move16);
	EXPECT_EQ(1.0f, buf.candidates[0].prob);
}

TEST(FeaturesTest, make_input_features2) {
    // features2
    initTable();
    Position pos;

    // case 1
    {
        pos.set("8l/+S8/1P4+Sp1/K4p2p/PNG1g2G1/2P6/6+n1k/9/3+lP4 b BG2SNL9P2rbnl2p 1");


        features1_t features1{};
        features2_t features2{};
        make_input_features(pos, features1, features2);

        float data2[MAX_FEATURES2_NUM];
        for (size_t i = 0; i < MAX_FEATURES2_NUM; ++i) data2[i] = *((float*)features2 + (size_t)SquareNum * i);

        // 先手持ち駒
        float* begin = data2, * end = data2 + MAX_HPAWN_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HLANCE_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HKNIGHT_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HSILVER_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HGOLD_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HBISHOP_NUM;
        EXPECT_EQ((std::vector<float>{1, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HROOK_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        // 後手持ち駒
        begin = end, end = begin + MAX_HPAWN_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HLANCE_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HKNIGHT_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HSILVER_NUM;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HGOLD_NUM;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HBISHOP_NUM;
        EXPECT_EQ((std::vector<float>{1, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HROOK_NUM;
        EXPECT_EQ((std::vector<float>{1, 1}), (std::vector<float>{begin, end}));
        // 王手
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{0, }), (std::vector<float>{begin, end}));
    }

    // case 2
    {
        pos.set("6+B1+P/1K+B3+L2/1+L1G5/5+R2P/7S1/P4P1n1/p1P1p1p1p/6sks/2+r3gP1 w 2GNL6Ps2nl2p 1");
        features1_t features1{};
        features2_t features2{};
        make_input_features(pos, features1, features2);

        float data2[MAX_FEATURES2_NUM];
        for (size_t i = 0; i < MAX_FEATURES2_NUM; ++i) data2[i] = *((float*)features2 + (size_t)SquareNum * i);

        // 後手持ち駒
        float* begin = data2, * end = data2 + MAX_HPAWN_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HLANCE_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HKNIGHT_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HSILVER_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HGOLD_NUM;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HBISHOP_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HROOK_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        // 先手持ち駒
        begin = end, end = begin + MAX_HPAWN_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 1, 1, 1, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HLANCE_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HKNIGHT_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HSILVER_NUM;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HGOLD_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HBISHOP_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HROOK_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        // 王手
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{1, }), (std::vector<float>{begin, end}));
    }
}

TEST(PythonModuleTest, hcpe3_clean) {
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    __hcpe3_clean(R"(F:\hcpe3\selfplay_pre55-012_book_policy_po5000-01_broken.hcpe3)", R"(R:\cleaned.hcpe3)");
}

#if 0
int main() {
	initTable();
	Position pos;
	//pos.set("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1");
	//pos.set("lnsgkg1nl/1r7/p1pppp1sp/6pP1/1p6B/2P6/PP1PPPP1P/7R1/LNSGKGSNL b Pb 1"); // dcBB
	pos.set("lnsgkg1nl/1r5s1/pppppp1pp/6p2/b8/2P6/PPNPPPPPP/7R1/L1SGKGSNL b B 1"); // pinned

	Bitboard occupied = pos.occupiedBB();
	occupied.printBoard();

	pos.bbOf(Black).printBoard();
	pos.bbOf(White).printBoard();

	// 駒の利き
	/*for (Color c = Black; c < ColorNum; ++c) {
		for (Square sq = SQ11; sq < SquareNum; sq++) {
			Bitboard bb = pos.attackersTo(Black, sq, occupied);
			std::cout << sq << ":" << bb.popCount() << std::endl;
			bb.printBoard();
		}
	}*/

	// 駒の利き(駒種でマージ)
	/*Bitboard attacks[ColorNum][PieceTypeNum] = {
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
		{ { 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 },{ 0, 0 } },
	};

	for (Square sq = SQ11; sq < SquareNum; sq++) {
		Piece p = pos.piece(sq);
		if (p != Empty) {
			Color pc = pieceToColor(p);
			PieceType pt = pieceToPieceType(p);
			Bitboard bb = pos.attacksFrom(pt, pc, sq, occupied);
			attacks[pc][pt] |= bb;
		}
	}

	for (Color c = Black; c < ColorNum; ++c) {
		for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
			std::cout << c << ":" << pt << std::endl;
			attacks[c][pt].printBoard();
		}
	}*/

	// 王手情報
	std::cout << pos.inCheck() << std::endl;
	CheckInfo checkInfo(pos);
	std::cout << "dcBB" << std::endl;
	checkInfo.dcBB.printBoard();
	std::cout << "pinned" << std::endl;
	checkInfo.pinned.printBoard();
	
}
#endif

// 王手生成テスト
TEST(GenerateMovesTest, Check) {
	initTable();
	Position pos;

    struct TestData {
        std::string sfen;
        std::vector<std::string> moves;
    };

	std::vector<TestData> sfens = {
        // 直接王手 歩成
        { "lnsgkgsnl/1r5b1/ppppPpppp/9/9/9/PPPP1PPPP/1B5R1/LNSGKGSNL b p 1", { "5c5b+" } },
        // 直接王手 香車成
        { "k1+B5l/l1gL2g2/2ng1pn2/2p1p1ppp/1pn2P3/LS1PP1S1P/1P1p5/1KG6/1N1s4B b 2RSP4p 1", { "1i7c+", "7a8a", "7a8b", "9f9b+", "S*8b", "R*8a" } },
        // 直接王手 桂馬成
		{ "lnsgkgsnl/1r5b1/pppp1pppp/4N4/9/9/PPPPPPPPP/1B5R1/LNSGKGS1L b p 1", { "5d4b+", "5d6b+" } },
        // 直接王手 角成
		{ "lnsgkgsnl/1r1p3b1/pp2ppppp/2p6/B8/9/PPPPPPPPP/7R1/LNSGKGSNL b - 1", { "9e6b+" } },
        // 直接王手 角成(対角線外)
		{ "lnskggsnl/1r5b1/pp1pppppp/2p6/B8/9/PPPPPPPPP/7R1/LNSGKGSNL b - 1", { "9e5a+", "9e6b+" } },
        // 直接王手 飛車成
		{ "lnsgkgsnl/4p2b1/pppp1Rppp/5p3/9/4R4/PPPPPPPPP/1B7/LNSGKGSNL b - 1", { "4c4a+", "4c4b+", "5f5b+" } },
        // 間接王手 銀
		{ "lnsgkgsnl/1r5b1/ppppSpppp/9/9/4L4/PPPPPPPPP/1B5R1/LNSGKG1N1 b p 1", { "5c4b+", "5c4b", "5c4d+", "5c4d", "5c5b", "5c5b+", "5c6b+", "5c6b", "5c6d+", "5c6d" } },
        // 間接王手 桂馬
		{ "lnsgkgsnl/1r5b1/pppp1pppp/9/9/4N4/PPPPLPPPP/1B5R1/LNSGKGS2 b 2p 1", { "5f4d", "5f6d" } },
        // 間接王手 香車
		{ "lnsg1gsnl/1r2k2b1/p1p1ppppp/1p1p5/1L7/B8/PPPPpPPPP/7R1/LNSGKGSN1 b - 1", { "8e8d" } },
        // 間接王手 香車成
		{ "lnsgkgsnl/1r5b1/ppLpppppp/2p6/B8/9/PPPPpPPPP/7R1/LNSGKGSN1 b - 1", { "7c7a+", "7c7b+" } },
        // 間接王手 香車成(後手)
		{ "lnsgkgsn1/1r7/ppppppppp/9/6P1b/9/PPPPPPlPP/1B5R1/LNSGKGSNL w - 1", { "3g3h+", "3g3i+" } },
        // 間接王手 香車成(直接あり)
		{ "lnsg1gsnl/1r2k2b1/p1pLppppp/1p1p5/9/B8/PPPPpPPPP/7R1/LNSGKGSN1 b - 1", { "6c6a+", "6c6b+" } },
        // 歩が成って王手
		{ "lnsgkgsnl/1r1P3b1/ppppPPppp/4pp3/9/9/PPP3PPP/1B5R1/LNSGKGSNL b - 1", { "4c4b+", "5c5b+", "6b6a+" } },
        // 歩が成って王手
		{ "lnsg1gsnl/1r1P3b1/ppppk1ppp/5P3/4Pp3/4p4/PPP3PPP/1B5R1/LNSGKGSNL b - 1", { "4d4c+", "5e5d" } },
        // 間接王手 歩成
		{ "lnsgkgsnl/1r5b1/ppPpppppp/2p6/B8/9/PP1PpPPPP/7R1/LNSGKGSNL b - 1", { "7c7b+" } },
        // 間接王手 歩成(直接あり)
		{ "lnsgkgsnl/1r1P3b1/pp1pppppp/2p6/B8/9/PP1PpPPPP/7R1/LNSGKGSNL b - 1", { "6b6a+" } },
        // 間接王手 歩成(直線上)
		{ "lnsgkgsnl/1r5b1/ppppPpppp/9/9/4L4/PPPP1PPPP/1B5R1/LNSGKGSN1 b p 1", { "5c5b+" } },
        // 間接王手 桂馬成
		{ "lnsg1gsnl/1r2k2b1/ppp2pppp/3p5/1N7/B8/PPPPLPPPP/7R1/LNSGKGS2 b 2p 1", { "8e7c+", "8e7c", "8e9c+", "8e9c" } },
        // 間接王手 桂馬成(直接あり)
		{ "lnsgkgsnl/1r5b1/pppp1pppp/4N4/9/9/PPPPLPPPP/1B5R1/LNSGKGS2 b 2p 1", { "5d4b+", "5d6b+" } },
        // 間接王手 金
		{ "lnsgkgsnl/1r5b1/pp1pppppp/1Gp6/B8/9/PPPPpPPPP/7R1/LNSGK1SNL b - 1", { "8d7d", "8d8c", "8d8e", "8d9c", "8d9d" } },
        // 間接王手 金(直接あり)
		{ "lnsgkgsnl/1r5b1/ppGpppppp/2p6/B8/9/PPPPpPPPP/7R1/LNSGK1SNL b - 1", { "7c6b", "7c6c", "7c7b", "7c7d", "7c8b", "7c8c" } },
        // 間接王手 成銀
		{ "lnsgkgsnl/1r5b1/pppp1pppp/4+S4/9/4L4/PPPPpPPPP/1B5R1/LNSGKG1N1 b p 1", { "5d4c", "5d4d", "5d6c", "5d6d" } },
        // 間接王手 成銀(直接あり)
		{ "lnsgkgsnl/1r5b1/pppp+Spppp/9/9/4L4/PPPPpPPPP/1B5R1/LNSGKG1N1 b p 1", { "5c4b", "5c4c", "5c5b", "5c6b", "5c6c" } },
        // 間接王手 角成
		{ "lnsgkgsnl/1r5b1/ppp1B1ppp/3p1p3/9/4R4/PPPPpPPPP/9/LNSGKGSNL b p 1", { "5c3a+", "5c4b+", "5c4d+", "5c6b+", "5c6d+", "5c7a+" } },
        // 間接王手 飛車成
		{ "lnsgkgsnl/1r5b1/ppRpppppp/2p6/B8/9/PPPPpPPPP/9/LNSGKGSNL b - 1", { "7c6c+", "7c7a+", "7c7b+", "7c7d+", "7c8c+" } },
        // 合法手の数：65
		{ "9/R1S1k1S1R/2+P3G2/2G3G2/9/B1NL1LN1B/9/4K4/4L4 b G2S2NL17P 1", { "3b2a+", "3b2a", "3b2c+", "3b2c", "3b3a+", "3b3a", "3b4a+", "3b4a", "3b4c+", "3b4c", "3d2c", "3d2d", "3d3e", "3d4c", "3d4d", "5h4g", "5h4h", "5h4i", "5h6g", "5h6h", "5h6i", "7b6a+", "7b6a", "7b6c+", "7b6c", "7b7a+", "7b7a", "7b8a+", "7b8a", "7b8c+", "7b8c", "7d6c", "7d6d", "7d7e", "7d8c", "7d8d", "3c4b", "3c4c", "3f4d", "4f4b+", "4f4c+", "6f6b+", "6f6c+", "7c6b", "7c6c", "7f6d", "P*5c", "L*5c", "L*5d", "L*5e", "L*5f", "L*5g", "N*4d", "N*6d", "S*4a", "S*4c", "S*5c", "S*6a", "S*6c", "G*4b", "G*4c", "G*5a", "G*5c", "G*6b", "G*6c" } },
        // 合法手の数：67
        { "5S1S1/RS5k1/5G3/9/5NL1L/9/9/1K7/B8 b RB3GS3N2L18P 1", { "8b7a+", "8b7a", "8b7c+", "8b7c", "8b8a+", "8b8a", "8b9a+", "8b9a", "8b9c+", "8b9c", "8h7h", "8h7i", "8h8g", "8h8i", "8h9g", "8h9h", "1e1b+", "1e1c+", "2a1b+", "2a3b+", "3e3b+", "3e3c+", "4a3b+", "4c3b", "4c3c", "4e3c+", "P*2c", "L*2c", "L*2d", "L*2e", "L*2f", "L*2g", "L*2h", "L*2i", "N*1d", "N*3d", "S*1a", "S*1c", "S*2c", "S*3a", "S*3c", "G*1b", "G*1c", "G*2c", "G*3b", "G*3c", "B*1a", "B*1c", "B*3a", "B*3c", "B*4d", "B*5e", "B*6f", "B*7g", "R*1b", "R*2c", "R*2d", "R*2e", "R*2f", "R*2g", "R*2h", "R*2i", "R*3b", "R*4b", "R*5b", "R*6b", "R*7b" } },
        // 合法手の数：91
		{ "+B7+B/7R1/2R6/9/3Sk1G2/6G2/3+PS1+P2/9/4L1N1K b GSNLPgs2n2l15p 1", { "2b1b+", "2b2a+", "2b2c+", "2b2d+", "2b2e+", "2b2f+", "2b2g+", "2b2h+", "2b2i+", "2b3b+", "2b4b+", "2b5b+", "2b6b+", "2b7b+", "2b8b+", "2b9b+", "5g4f", "5g4h", "5g5f", "5g6f", "5g6h", "7c1c+", "7c2c+", "7c3c+", "7c4c+", "7c5c+", "7c6c+", "7c7a+", "7c7b+", "7c7d+", "7c7e+", "7c7f+", "7c7g+", "7c7h+", "7c7i+", "7c8c+", "7c9c+", "3e4e", "3f4e", "3f4f", "3g4f", "3i4g", "6e5f", "6e6d", "6g5f", "6g6f", "P*5f", "L*5f", "N*4g", "S*4d", "S*4f", "S*5f", "S*6d", "S*6f", "G*4e", "G*4f", "G*5d", "G*5f", "G*6f" } },
	};

	// 王手生成
	for (const auto data : sfens) {
		pos.set(data.sfen);
		MoveList<Check> ml(pos);
        std::vector<std::string> moves;
        moves.reserve(ml.size());
		for (; !ml.end(); ++ml) {
            moves.emplace_back(ml.move().toUSI());
		}
        EXPECT_EQ(data.moves, moves) << data.sfen;
	}
}

#if 0
// 王手生成テスト(ファイルからsfen読み込み)
int main(int argc, char* argv[]) {
	initTable();
	Position pos;

	std::ifstream ifs(argv[1]);

	string sfen;
	while (ifs) {
		std::getline(ifs, sfen);
		if (sfen.size() == 0) break;
		pos.set(sfen);
		if (pos.inCheck()) continue;

		MoveList<CheckAll> ml(pos);

		std::cout << sfen << "\t" << ml.size();
		std::vector<std::string> movelist;
		movelist.reserve(ml.size());
		for (; !ml.end(); ++ml) {
			movelist.emplace_back(ml.move().toUSI());
		}
		std::sort(movelist.begin(), movelist.end());
		for (const auto& move : movelist) {
			std::cout << "\t" << move;
		}
		std::cout << std::endl;
	}

	return 0;
}
#endif

#if 0
#include "mate.h"
// 詰み探索計測
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	vector<string> sfens = {
		// 詰み
		"lnsgk1snl/1r4g2/p1pppp1pp/6pP1/1p7/2P6/PPGPPPP1P/6SR1/LN+b1KG1NL w bs 11",
		"l3S1kpl/3r1gs2/1p2p2P1/p1p2P1+Bp/3s2Ps1/2P2p+b1P/PP2K4/7R1/LN1g4L w GNPg2n3p 5",
		"l1r2k1nl/1+S4gs1/3p1g1pp/4p1p2/p2N1p1P1/1P2N1P2/P3P1N1P/2G1G1SR1/+b1K5L w bsl5p 7",
		"l1r2k1nl/3+S2gs1/3p1g1pp/4p1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w Pbsl5p 9",
		"l1r2k1nl/6gs1/3p1g1pp/3Sp1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w Pbsl5p 10",
		"7+P1/3pksg1l/4pp1pp/3G1gn2/1+R4pR1/4PPn1K/2P+l2P1P/1P7/L+b4+b1L w 2SN4Pgsnp 8",
		"l1r2k1nl/3S2gs1/3p1g1pp/4p1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w Pbsl5p 10",
		"l1r2k1nl/3+S2gs1/3p1g1pp/4p1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w Pbsl5p 10",
		"lk6+P/9/2s2L3/6gP1/1p1p5/p1P1P+s3/1P1Pg1P1N/1gS2P3/LNBKG1S1L w N7P2rbn 11",
		"l3rg3/b3p1k2/p5np1/6+R2/1p1P5/P1P1g1G2/1PN4P+l/3S+p1P2/5GK2 b B2SN2Psn2l5p 9",
		"l1r2k1nl/6gs1/3p1g1pp/1+S2p1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w Pbsl5p 10",
		"lk6+P/9/2s2L3/6gP1/1p1p5/p1P1P+s3/1P1Pg1P1N/1gS2P3/LNBKG1S1L w N7P2rbn 12",
		"l1r2k1nl/3+S1ggs1/3p3pp/4pPp2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w bsl5p 11",
		"l1r2k1nl/3+S1ggs1/3p3pp/4pPp2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w bsl5p 12",
		"l1r2k1nl/1+S3ggs1/3p3pp/4pPp2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w bsl5p 12",
		"l3rg3/b3p1k2/p5np1/6+R2/1p1P5/P1P1g1G2/1PN2+p1P+l/3SSpP2/5GK2 b B2SN2Pn2l4p 12",
		"l3rg3/b3p1k2/p5np1/6+R2/1p1P5/P1P1g1Gn1/1PN2+p1P+l/3SS1P2/5GK2 b B2SN2P2l5p 12",
		"l1r2k1nl/1+S3g1s1/3p1g1pp/4p1p2/p2N+bN1P1/1P2N1P2/P3P3P/2G1G1SR1/2K5L w bsl6p 13",
		"ln2k4/2sg2+P1l/p3p2+R1/2Pp4p/1bGP2S2/2p2Pp2/P+r2P3P/2+p1G4/L3K2NL b BGSNPsn3p 16",
		"ln2k4/2sg2+P1l/p3p2+R1/2Pp4p/2GP2S2/2p2Pp2/P+r2P3P/2+p1G1s2/L3K2NL b BGSNPbn3p 16",
		"lnkg2+R1l/3g5/1s1p4p/2p1N1pP1/Pp7/n1PPPPP2/1P1SS3P/2G3+r2/K7L w BGS4Pbnl 19",
		"1n1g1k3/5s2+B/1p1p1LnPp/5ppp1/2ps4P/1K1PrPP2/1P2+b1N2/2G3S2/+lN5RL w GL2Pgs3p 20",
		"1n1g3B1/5k2l/1p1p2nPp/5ppp1/2ps1N2P/1K1PrPP2/1P2+b4/2G3S2/+lN5RL w GS2Pgsl3p 21",
		"1n1g5/5k2l/1p1p2nPp/1l3ppp1/2p+B4P/3PrPP2/1PK1+b1N2/2G3S2/+lN5RL w G2S2Pgs3p 21",
		"1n3g1nl/5ksb1/1p2pr1pp/2sPP1P2/5N3/1P2G2P1/1K2+nS2P/7R1/2S5L w G8Pbg2l 18",
		// 不詰み
		"1+B5n1/5g1k1/4pp1p1/l5p2/2PR1P2P/1P1pP1P2/1S2s2PL/1K1+b5/g6N1 b RG2SN2L4Pgn2p 2",
		"1+B5n1/5g1+N1/4ppkp1/l5p2/2PR1P2P/1P1pP1P2/1S2s2PL/1K1+b5/g6N1 b RG2SN2L4Pg2p 2",
		"1+B5n1/5g1k1/4pp1p1/l5p2/2PR1P2P/1P1pP1P2/1S2s2PL/1K1+b5/g6N1 b RG2SN2L4Pgn2p 3",
		"1+B5n1/5g1+N1/4ppkp1/l5p2/2PR1P2P/1P1pP1P2/1S2s2PL/1K1+b5/g6N1 b RG2SN2L4Pg2p 3",
		"l1r2k1nl/3S2gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 3",
		"l1r2k1nl/3S2gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 4",
		"1+B4g1R/3r1sknS/p4+b2l/1p1p2ppp/2pP1N3/4G1P1P/PP2S2P1/L2+n1+s1K1/8L w 2P2gnl3p 3",
		"l1r2k1nl/1S4gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 4",
		"l1r2k1nl/3S2gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 5",
		"l1r2k1nl/1S4gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 5",
		"l2R3n1/5sgkl/2n+P1g1p1/p2P2p2/2P1bPPNP/PP5P1/5S1K1/5G3/LN1+b4L w 2S6Prg 2",
		"l1r2k1nl/3S2gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 6",
		"l1r2k1nl/1S4gs1/3p1g1pp/4p1p2/p2N1p1P1/1P1bN1P2/P3P1N1P/2G1G1SR1/L1K5L w bs5p 6",
		"l2R3n1/3S1sgkl/2n+P1g1p1/p5p2/2P1bPPNP/PP5P1/5S1K1/5G3/LN1+b4L w S7Prg 2",
		"1+B4g1R/3r1sknS/p7l/1p1p1+bppp/2pP1NP2/4GS2P/PP2p2P1/L2+n1+s1K1/8L w 2gnl4p 5",
		"l2R3n1/4Psgkl/2n+P1g1p1/p5p2/2P1bPPNP/PP5P1/5S1K1/5G3/LN1+b4L w 2S6Prg 2",
		"l3S1kpl/3r1gs2/1p2p2P1/p1p2P1+Bp/5+bPs1/2Ps1p2P/PP7/7R1/LN1K4L w 2GNPg2n3p 9",
		"l3S1kpl/3r1gs2/1p2p2P1/p1p2P1+Bp/5+bPs1/2Ps1p2P/PP7/7R1/LN1K4L w 2GNPg2n3p 10",
		"1+R5+P1/3pksg1l/4pp1pp/3G1gn2/6pR1/4PPn2/2P+l2PKP/1P7/L+b4+b1L w 2SN4Pgsnp 7",
		"7+P1/3pksg1l/4pp1pp/3G1gn2/1+R4pR1/4PPn2/2P+l2PKP/1P7/L+bN3+b1L w 2S4Pgsnp 8",
		"1+R5+P1/3pksg1l/4pp1pp/3G1gn2/6pR1/4PPn2/2P+l2PKP/1P7/L+b4+b1L w 2SN4Pgsnp 8",
		"7+P1/3pksg1l/4pp1pp/3G1gn2/3+R2pR1/4PPn2/2P+l2PKP/1P7/L+b4+b1L w 2SN4Pgsnp 8",
		"lns4n1/2r1k1s2/ppggpp1pl/2pp2p1p/5P+bP1/P2SP4/1PPP1GN1P/1B3S3/LN1GK4 w rlp 8",
		"7+P1/4ksg1l/3ppp1pp/3G1gn2/1+R1N2pR1/4PPn2/2P+l2PKP/1P7/L+b4+b1L w S4Pg2snp 10",
	};

	auto start0 = std::chrono::system_clock::now();
	auto total = start0 - start0;
	for (string sfen : sfens) {
		pos.set(sfen);
		auto start = std::chrono::system_clock::now();
		bool ret = mateMoveInOddPly<7>(pos);
		auto end = std::chrono::system_clock::now();

		auto time = end - start;
		total += time;

		auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();

		cout << ret << "\t";
		cout << time_ns / 1000000.0 << endl;
	}
	auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
	cout << total_ns / 1000000.0 << endl;

	return 0;
}
#endif

#if 0
#include "mate.h"
// 詰み探索計測(ファイルからsfen読み込み)
int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	Position pos;

	std::ifstream ifs(argv[1]);

	std::chrono::high_resolution_clock::duration total{};
	string sfen;
	while (ifs) {
		std::getline(ifs, sfen);
		if (sfen.size() == 0) break;
		pos.set(sfen);

		auto start = std::chrono::high_resolution_clock::now();

		auto ret = mateMoveInOddPly<5>(pos);

		total += std::chrono::high_resolution_clock::now() - start;

		cout << (bool)ret << "\t" << sfen << std::endl;
	}
	cout << std::chrono::duration_cast<std::chrono::nanoseconds>(total).count() / 1000000.0 << endl;

	return 0;
}
#endif

#if 0
#include "mate.h"
// 奇数手詰めの判定間違い
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	string sfen = "lnG3Gnl/4k2b1/pppp2ppp/2N1GGN2/9/9/PPPP1PPPP/1B5R1/L1S1K1S1L b 3Pr2s 1";
	vector<string> moves{ "P*5c", "5b5a" };

	pos.set(sfen);
	pos.searcher()->states = StateListPtr(new std::deque<StateInfo>(1));
	for (auto token : moves) {
		const Move move = usiToMove(pos, token);
		pos.searcher()->states->push_back(StateInfo());
		pos.doMove(move, pos.searcher()->states->back());
	}

	bool ret = mateMoveInOddPly<5>(pos);

	std::cout << ret << std::endl;

	return 0;
}
#endif

#if 0
#include "mate.h"
// 最大手数チェック
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	// WCSCのルールでは、最大手数で詰ました場合は勝ちになる
	vector<pair<string, int>> sfens = {
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 264 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 264 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 264 }, // 262手目で詰み → 詰み
	};

	for (auto& sfen_draw : sfens) {
		pos.set(sfen_draw.first);
		bool ret = mateMoveInOddPly<5>(pos, sfen_draw.second);
		cout << ret << endl;
	}

	return 0;
}
#endif

#if 0
#include "dfpn.h"
// DfPnテスト
int main()
{
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(400000);
	dfpn.set_maxdepth(33);

	Position pos;

	vector<string> sfens = {
		// 詰み
		"9/9/+N8/p1p4p1/6p1p/1P7/3k3PP/2+p5L/6+rGK w R2B2G3Sgs3n3l9p 1",
		"1n1g3+Pl/k1p1s4/1ng5p/pSP1p1pp1/1n3p3/P1K3P1P/1P7/9/L1G5L b 2R2BG2SL5Pn 161", // mate 15
		"ln6K/9/1sp2+P3/pp4G1p/6P2/+rl+B+R5/k8/+b8/9 b 2G2SNL2Pgs2nl10p 1", // mate 15
		"ln1s+R3K/2s6/p1pp1p3/kp+r4pp/N3p4/1Sg6/P2B2P1P/5g3/LL3g1NL b BGS2Pn5p 1", // mate 17
		"n1gg2+R2/l2l5/p1k1p3K/1ppp5/3PBp3/9/P4P2P/g8/8L b RG2S2NL6Pb2sn2p 1", // mate 33
		"1+Rp4nl/3+b1+Rss1/7k1/p3p1p1p/1PBp1p1PP/4P1n1K/3S1PS2/4G2g1/5G1NL b NL4Pgl2p 1", // mate 37
		"l1+R5K/4p2+P1/1pp6/p2+b4p/2k2G1p1/PL2+n3P/N5+s2/2+nP5/L4+R2L b B2G2S3Pgsn5p 1", // mate 13
		"ln1gl1R2/ks1gn1R2/pp2pp2K/1PB3+P2/5P3/9/P7P/8p/+n4S2L b B2G2SNL3P5p 1", // mate 13
		"7n1/+R5glK/n4s1p1/p4pkl1/2s3n2/P3PP2P/1P+pG5/2P6/9 b R2B2G2S2L8Pn 1", // mate 13
		"ln6K/1ksPg1sG1/5s2p/ppp3p2/9/P1P3P2/4+p3P/1+r7/LN3L2L w RBSb2g2n7p 1", // mate 19
		"l6n1/b+R4glK/n4s1p1/p5kl1/2sp2n2/P1p1P3P/1P3P3/2P1G4/2+b6 b R2G2SNL7Pp 1", // mate 15
		"7+L1/1+B1nkg2+R/3p2s1K/1pp1g1p2/1n1P5/lSP4P1/1P7/1G7/1+l7 b RBG2S2NL9Pp 1", // mate 11
		"ln6K/9/1sp2+P3/pp4G1p/6P2/+r8/9/9/k+l1+R5 b B2G2SNLPbgs2nl11p 1", // mate 13
		"lnS1r4/GGs5K/2k5p/pppp5/9/PLPP5/1P+n1PS2P/4G4/g1+r3+p2 b BS2N2L6Pb 1", // mate 15
		"7s1/k2g5/n1p1p1P+LK/s2P2n2/p5pP1/2P1P4/5+p3/8L/L3+b1sN1 w NL4P2rb3gs4p 1", // mate 13
		"l7K/9/p6sp/1g1ppRpbk/7n1/1P2SP2P/P3P1P2/2+n1s2+p1/LN2+r2NL b B3GSL6P 1", // mate 9
		"1n3G1nK/5+r1R1/p2L+P3p/2p2Gp2/9/3P2B2/P1P5+n/5S1p1/L1S1L1Ggk b 2SNL6Pb3p 1", // mate 7
		"ln3+P1+PK/1r1k3+B1/3p1+L1+S1/p1p2p1+B1/3+r3s1/7s1/4p1+n+pp/+p3+n2p+p/1+p3+p+p+p+p b 2GN2L2gsp 1", // mate 15
		"lR6K/4p2+P1/1p7/p7p/2k1+b2p1/Pg1n+n3P/N5+s2/1L1P5/L4+R2L b B2G2S3Pgsn6p 1", // mate 15
		"lnp1+RS2K/1k5+P1/1pgps3p/4p4/6+Rp1/3+n5/+pP2n1P2/2+b1P4/8+p b B2G2SN2L2Pgl4p 1", // mate 23
		"lns3kn1/1r4g2/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G2Pl4p 53", // mate 1
		"lns3kn1/1r3g3/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G3Pl3p 51", // mate 3
		"1n+S1l3n/2s6/1pp3pg1/3p2s2/1kP4PK/4p1n1P/+l1G2+B3/1+l1G1+R3/5P3 b RL6Pbgsn3p 1", // mate 3
		"knS5+P/1g7/Pgp+B4+P/l1n1pp1+B1/7L1/pp2PPPPK/3P2+pR1/5g3/LR5S1 b GS2Ps2nl2p 1", // mate 15
		"+B+R5n1/5gk2/p1pps1gp1/4ppnsK/6pP1/1PPSP3L/PR1P1PP2/6S2/L2G1G3 w B2N2LP2p 1", // mate 3(循環)
		"lnl5l/2b6/ppk6/3p1p2p/Ps2p1bP1/1NP3g1K/LP6P/9/1N6+p b R3G2SN2Prs4p 1", // mate 25
		"knS5K/llGp3G1/pp2+R2S1/2n6/9/6S2/P3+n3P/2P2P2L/6GNb b RBGSL11P 1", // mate 5
		"ln5+LK/1rk1+B2S1/p1sp5/4p1pp1/1PPP1P3/2S1P3+l/P1B2S3/1R2G2+p1/LN3G3 b 2GN5Pnp", // mate 13
		"1p+Bnn+R2K/3+R5/4ps1pp/3p2p2/1NG1s4/6kPP/P2PP4/3G1+s1G1/L8 b BSN3L6Pgp 1", // mate 11
		"l2g2p1+P/1k2n4/ppns5/2pb2g1+L/4PP1pK/PP5S1/3+b1+s2P/7P1/8+r w 4Pr2gs2n2l2p 1", // mate 11
		"+S5knl/R2g3g1/4sp1p1/2P1pNp2/Bp3P1Pp/p1pp2P1P/GP2P1N2/K1+n2G3/L7L b B2SLPrp 101", // mate 33(nodes 368571が必要)
		"lr1s4l/3k1+P3/p2p4p/1NL2P3/1PpS5/P6pP/K1+b6/NR1P5/7NL w BSN3P4gs4p 166", // mate 33(depath 35, nodes 3000000が必要)
		// 不詰み
		"lns3kn1/1r7/4pgs+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b BG3Pl3p 49", // 不詰み
		"lns4n1/1r3k3/4pgsg1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG2R1/LN1K3N+l b B3Pl3p 47", // 不詰み
		"7nl/5Psk1/1+P1+P1p1pp/K3g4/6p1B/1SP4P1/PsS3P1P/1N7/+r6NL w GLrb2gnl6p 1", // 不詰み
		"ln3+P1+PK/1rk4+B1/3p1+L1+S1/p1p2p1+B1/3+r3s1/7s1/4p1+n+pp/+p3+n2p+p/1+p3+p+p+p+p b 2GN2L2gsp 1", // 不詰み
		"l2+S1p2K/1B4G2/p4+N1p1/3+B3sk/5P1s1/P1G3p1p/2P1Pr1+n1/9/LNS5L b R2GL8Pnp 1", // 不詰み
		"+B2B1n2K/7+R1/p2p1p1ps/3g2+r1k/1p3n3/4n1P+s1/PP7/1S6p/L7L b 3GS7Pn2l2p 1", // 不詰み
		"l6GK/2p2+R1P1/p1nsp2+Sp/1p1p2s2/2+R2bk2/3P4P/P4+p1g1/2s6/L7L b B2GNL2n7p 1", // 不詰み
		"1n3G1nK/2+r2P3/p3+P1n1p/2p2Gp2/5l3/3P5/P1P3S2/6+Bpg/L1S1L3k b R2SNL5Pbg3p 1", // 不詰み
		"+B2B1n2K/7+R1/p2p1p1ps/3g2+r1k/1p3n3/4n1P+s1/PP7/1S7/L8 b 3GSL7Pn2l3p 1", // 不詰み
		"ln2g3l/2+Rskg3/p2sppL2/2pp1sP1p/2P2n3/B2P1N1p1/P1NKPP2P/1G1S1+p1P1/7+rL b B2Pg 98", // 不詰み
	};
	
	auto start0 = std::chrono::system_clock::now();
	auto total = start0 - start0;
	for (string sfen : sfens) {
		pos.set(sfen);
		auto start = std::chrono::system_clock::now();
		bool ret = dfpn.dfpn(pos);
		auto end = std::chrono::system_clock::now();

		auto time = end - start;
		total += time;

		auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();

		cout << ret << "\t" << dfpn.searchedNode << "\t";
		cout << time_ns / 1000000.0 << endl;
	}
	auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
	cout << total_ns / 1000000.0 << endl;
}
#endif

#if 0
#include "dfpn.h"
// 最大手数チェック
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	DfPn dfpn;
	dfpn.init();

	// WCSCのルールでは、最大手数で詰ました場合は勝ちになる
	vector<pair<string, int>> sfens = {
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 261 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 262 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 261 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 262 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 261 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 262 }, // 262手目で詰み → 詰み
		{ "+P2+Rb1gnl/7k1/n1p1+Bp1pp/p5p2/1p1pP2P1/2+s6/PsNGSPP1P/3KG4/L5RNL b SL3Pg 83", 83 + 9 }, // 11手で詰み → 持将棋
		{ "+P2+Rb1gnl/7k1/n1p1+Bp1pp/p5p2/1p1pP2P1/2+s6/PsNGSPP1P/3KG4/L5RNL b SL3Pg 83", 83 + 10 }, // 11手で詰み → 詰み
		{ "l7l/4g4/4s3p/p3b1p2/1PS3n2/4P1g2/P5n1P/2+n+rSpK2/LN2k3L w 2GS3Prb7p 254", 256 }, // 256手で詰み → 詰み
	};

	for (auto& sfen_draw : sfens) {
		pos.set(sfen_draw.first);
		DfPn::set_draw_ply(sfen_draw.second);
		bool ret = dfpn.dfpn(pos);
		cout << ret << endl;
	}

	return 0;
}
#endif

#if 0
#include "dfpn.h"
// 最大深さチェック
int main()
{
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();

	Position pos;

	vector<pair<string, int>> sfens = {
		{ "lns3kn1/1r4g2/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G2Pl4p 53", 1 }, // mate 1
		{ "lns3kn1/1r4g2/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G2Pl4p 53", 2 }, // mate 1
		{ "lns3kn1/1r3g3/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G3Pl3p 51", 2 }, // mate 3
		{ "lns3kn1/1r3g3/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G3Pl3p 51", 3 }, // mate 3
		{ "knS5K/llGp3G1/pp2+R2S1/2n6/9/6S2/P3+n3P/2P2P2L/6GNb b RBGSL11P 1", 6 }, // mate 7
		{ "knS5K/llGp3G1/pp2+R2S1/2n6/9/6S2/P3+n3P/2P2P2L/6GNb b RBGSL11P 1", 7 }, // mate 7
	};

	for (auto& sfen_depth : sfens) {
		pos.set(sfen_depth.first);
		dfpn.set_maxdepth(sfen_depth.second);
		bool ret = dfpn.dfpn(pos);
		cout << ret << endl;
	}

	return 0;
}
#endif

#if 0
#include "dfpn.h"
// DfPnのPV表示テスト
int main()
{
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(400000);
	dfpn.set_maxdepth(33);

	Position pos;

	vector<string> sfens = {
		// 詰み
		"9/9/+N8/p1p4p1/6p1p/1P7/3k3PP/2+p5L/6+rGK w R2B2G3Sgs3n3l9p 1",
		"1n1g3+Pl/k1p1s4/1ng5p/pSP1p1pp1/1n3p3/P1K3P1P/1P7/9/L1G5L b 2R2BG2SL5Pn 161", // mate 15
		"ln6K/9/1sp2+P3/pp4G1p/6P2/+rl+B+R5/k8/+b8/9 b 2G2SNL2Pgs2nl10p 1", // mate 15
		"ln1s+R3K/2s6/p1pp1p3/kp+r4pp/N3p4/1Sg6/P2B2P1P/5g3/LL3g1NL b BGS2Pn5p 1", // mate 17
		"n1gg2+R2/l2l5/p1k1p3K/1ppp5/3PBp3/9/P4P2P/g8/8L b RG2S2NL6Pb2sn2p 1", // mate 33
		"1+Rp4nl/3+b1+Rss1/7k1/p3p1p1p/1PBp1p1PP/4P1n1K/3S1PS2/4G2g1/5G1NL b NL4Pgl2p 1", // mate 37
		"l1+R5K/4p2+P1/1pp6/p2+b4p/2k2G1p1/PL2+n3P/N5+s2/2+nP5/L4+R2L b B2G2S3Pgsn5p 1", // mate 13
		"ln1gl1R2/ks1gn1R2/pp2pp2K/1PB3+P2/5P3/9/P7P/8p/+n4S2L b B2G2SNL3P5p 1", // mate 13
		"7n1/+R5glK/n4s1p1/p4pkl1/2s3n2/P3PP2P/1P+pG5/2P6/9 b R2B2G2S2L8Pn 1", // mate 13
		"ln6K/1ksPg1sG1/5s2p/ppp3p2/9/P1P3P2/4+p3P/1+r7/LN3L2L w RBSb2g2n7p 1", // mate 19
		"l6n1/b+R4glK/n4s1p1/p5kl1/2sp2n2/P1p1P3P/1P3P3/2P1G4/2+b6 b R2G2SNL7Pp 1", // mate 15
		"7+L1/1+B1nkg2+R/3p2s1K/1pp1g1p2/1n1P5/lSP4P1/1P7/1G7/1+l7 b RBG2S2NL9Pp 1", // mate 11
		"ln6K/9/1sp2+P3/pp4G1p/6P2/+r8/9/9/k+l1+R5 b B2G2SNLPbgs2nl11p 1", // mate 13
		"lnS1r4/GGs5K/2k5p/pppp5/9/PLPP5/1P+n1PS2P/4G4/g1+r3+p2 b BS2N2L6Pb 1", // mate 15
		"7s1/k2g5/n1p1p1P+LK/s2P2n2/p5pP1/2P1P4/5+p3/8L/L3+b1sN1 w NL4P2rb3gs4p 1", // mate 13
		"l7K/9/p6sp/1g1ppRpbk/7n1/1P2SP2P/P3P1P2/2+n1s2+p1/LN2+r2NL b B3GSL6P 1", // mate 9
		"1n3G1nK/5+r1R1/p2L+P3p/2p2Gp2/9/3P2B2/P1P5+n/5S1p1/L1S1L1Ggk b 2SNL6Pb3p 1", // mate 7
		"ln3+P1+PK/1r1k3+B1/3p1+L1+S1/p1p2p1+B1/3+r3s1/7s1/4p1+n+pp/+p3+n2p+p/1+p3+p+p+p+p b 2GN2L2gsp 1", // mate 15
		"lR6K/4p2+P1/1p7/p7p/2k1+b2p1/Pg1n+n3P/N5+s2/1L1P5/L4+R2L b B2G2S3Pgsn6p 1", // mate 15
		"lnp1+RS2K/1k5+P1/1pgps3p/4p4/6+Rp1/3+n5/+pP2n1P2/2+b1P4/8+p b B2G2SN2L2Pgl4p 1", // mate 23
		"lns3kn1/1r4g2/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G2Pl4p 53", // mate 1
		"lns3kn1/1r3g3/3Bp1s+R1/2pp1p3/pp2P4/2P1SP3/PPSP5/2GBG4/LN1K3N+l b G3Pl3p 51", // mate 3
		"1n+S1l3n/2s6/1pp3pg1/3p2s2/1kP4PK/4p1n1P/+l1G2+B3/1+l1G1+R3/5P3 b RL6Pbgsn3p 1", // mate 3
		"knS5+P/1g7/Pgp+B4+P/l1n1pp1+B1/7L1/pp2PPPPK/3P2+pR1/5g3/LR5S1 b GS2Ps2nl2p 1", // mate 15
		"+B+R5n1/5gk2/p1pps1gp1/4ppnsK/6pP1/1PPSP3L/PR1P1PP2/6S2/L2G1G3 w B2N2LP2p 1", // mate 3(循環)
		"lnl5l/2b6/ppk6/3p1p2p/Ps2p1bP1/1NP3g1K/LP6P/9/1N6+p b R3G2SN2Prs4p 1", // mate 25
		"knS5K/llGp3G1/pp2+R2S1/2n6/9/6S2/P3+n3P/2P2P2L/6GNb b RBGSL11P 1", // mate 5
		"ln5+LK/1rk1+B2S1/p1sp5/4p1pp1/1PPP1P3/2S1P3+l/P1B2S3/1R2G2+p1/LN3G3 b 2GN5Pnp", // mate 13
		"1p+Bnn+R2K/3+R5/4ps1pp/3p2p2/1NG1s4/6kPP/P2PP4/3G1+s1G1/L8 b BSN3L6Pgp 1", // mate 11
		"l2g2p1+P/1k2n4/ppns5/2pb2g1+L/4PP1pK/PP5S1/3+b1+s2P/7P1/8+r w 4Pr2gs2n2l2p 1", // mate 11
		"+S5knl/R2g3g1/4sp1p1/2P1pNp2/Bp3P1Pp/p1pp2P1P/GP2P1N2/K1+n2G3/L7L b B2SLPrp 101", // mate 33(nodes 368571が必要)
		"l1g4ng/1k1s+R+P3/p2p2pP1/2s1B4/2bnp4/PR2P1P2/K1+p2PN2/3L4+l/Ls7 w 2GSPn6p 150", // mate1
		"l3kgsnl/9/p1pS+Bp3/7pp/6PP1/9/PPPPPPn1P/1B1GG2+r1/LNS1K3L w RG3Psnp 54", // mate1
		"l1r3bn1/3k1g1sl/2G1pp3/p2p1Pp1p/4P4/PP1P2PRP/1g1S5/4+b4/LNK3sNL w Pgsn4p 104", // mate3 王手をかけられている局面から
		"l4k1nl/7B1/p2sgpgp1/2pp2p1p/P6P1/G4PPs1/2N3n1P/5R3/L3+rNK1L w BS2Pgs4p 78", // mate5
		"l1l5l/7g1/2GK2+b2/1k1s+R4/p8/2sNP+r3/PP+pP4+p/2G6/LN5+p1 b GS2N3Pbs7p 147", // mate3
		"+R2+S5/2+R5S/pN4g1p/2p1N1ppk/3g3P1/1P1p1NSsP/P8/2K6/L7g b 2BGN3L8P 1", // mate3
		"lnkg5/3bgs1p1/pG1p1p3/4P4/4+r1p2/PPPn1L3/4rPGSP/1S7/LNK4NL w BSP6p 88", // mate7
		"ln2+B1gnl/2R1G1k2/p2p1p+bp1/2p3p1p/9/2P1S3P/P2P1PPP1/4s4/2+r2GKNL b GSL3Psnp 1", // mate27
		"l4k2l/3+R5/p3pg3/1p6p/3n1p1N1/P2SP1PbP/1PPP1P1p1/2G1G4/LNK1+n2rL w GSPb2s3p 1", // mate13
		"2s+R1+N3/1ks5+R/lpggp1+Pp1/p1pp5/2n2P1P1/P1Pn5/1PNPP3P/1K7/L1S5L b BS2Pb2glp 1", // mate15
		"ln3+S2l/6k2/p2+Pp3p/1NP6/2g1Gp3/2pP1Pr1P/PP1K3p1/1S2G+p1s1/LN6+r b 2BGNLPs3p 1", // mate23
		"+LR2R2n1/3g1lg2/3N2kp1/p2pp4/5S1Sp/1PP3SP1/P2PP2B1/1G1S2G2/LN2K4 b N6Pblp 121", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"l4p2l/7g1/6n1b/P2g1kS1p/3p2p2/2S2N1p1/1PKP1S1NP/4g1G2/L1+B1+p1S1L b RN5Pr3p 171", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"lkB1s3+N/4bs3/2RL2pp1/p1pGpp2p/1n1p5/8P/PP1PPPP1N/1G1S2S+R1/LN2KG2L b Pg2p 89", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"1n1g4l/+B1s2+B2k/p1p2sr1p/4gl1SP/3p2p1G/5L3/PP1PPPP2/2G2S3/3R1K1NL b 2N4P2p 1", // mate7(9手になる)
		"ln3+BR2/6n1k/2s1sppr1/p1g1p2P1/1p2B2p1/9/PPSPPPP2/2G2KS2/LN3G1N1 b 2Pg2l3p 81", // mate7(9手になる)
		"9/3S1k3/3pg2+R1/4spP2/1p1PN3L/PP1nPP3/3l1SN2/K1g2G3/L3b+b3 w RGSL7Pn2p 126", // mate7(角成らず打ち歩詰めにならず最短で詰ますことができる)
		"l8/2+P2+N1k1/4G2p1/2p2ps1G/p3s4/1pP1S4/PlN1gP+p2/1bGp5/L3K4 b R4Prbs2nl3p 193", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"l4ksRl/3+S5/p4pngp/2pp1bp2/9/P1PP1PP1P/1PNG1K3/1G4r2/L5bNL w G2SN2P3p 78", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"lR3p2l/2g1k1s2/ps1ppPn2/2pg2p1p/9/2PB1NP2/P1NPP1N1P/2S1KGS2/L2G4L b RPb3p 83", // mate7(飛車成らずにより打ち歩詰めにならず最短で詰ますことができる)
		"5pG+Rl/2S6/1SS1g4/pB2k3P/3p1Lp2/PR3PS2/1P1+p2K2/9/L7L b N5Pb2g3n4p 205", // 手順に劣等局面を含む
		"lnGS5/k4p+R2/1ppp4p/1+b2p1p2/3P5/SP2P3P/K5P2/4gP3/+b1P4NL b RG2SNL2Pgnl2p 1", // mate9
		"4p2pl/1+N7/l3k2P1/8p/3gsn3/1GP3P1P/P2K1p3/9/3g4L b 2BGSNL6P2r2sn3p 1", // mate19
		"2g5l/+S8/3kp1ppp/Npp6/p1sp1rP2/5p2P/2P2P1B1/1K7/9 w B3GSN3L2Prs2n3p 1", // mate9
	};

	auto start0 = std::chrono::system_clock::now();
	auto total = start0 - start0;
	for (string sfen : sfens) {
		pos.set(sfen);
		auto start = std::chrono::system_clock::now();
		bool ret = dfpn.dfpn(pos);
		auto end = std::chrono::system_clock::now();

		auto time = end - start;
		total += time;

		auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();

		cout << ret << "\t" << dfpn.searchedNode << "\t";
		cout << time_ms;

		// pv
		if (ret) {
			std::string pv;
			int depth;
			Move move;
			auto start_pv = std::chrono::system_clock::now();
			std::tie(pv, depth, move) = dfpn.get_pv(pos);
			auto end_pv = std::chrono::system_clock::now();

			cout << "\t" << move.toUSI() << "\t" << pv << "\t" << depth << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_pv - start_pv).count();
		}
		cout << endl;
	}
	auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total).count();
	cout << total_ms << endl;
}
#endif

#if 0
#include "dfpn.h"
// DfPnのPV表示テスト(ファイルからsfen読み込み)
int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	Position pos;

	if (argc < 3) return 1;

	std::ifstream ifs(argv[1]);

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(400000);
	dfpn.set_maxdepth(std::atoi(argv[2]));

	std::chrono::system_clock::duration total{};
	string sfen;
	while (ifs) {
		std::getline(ifs, sfen);
		if (sfen.size() == 0) break;
		pos.set(sfen);

		auto start = std::chrono::system_clock::now();
		bool ret = dfpn.dfpn(pos);
		auto end = std::chrono::system_clock::now();

		auto time = end - start;
		total += time;

		auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();

		cout << ret << "\t" << dfpn.searchedNode << "\t";
		cout << time_ms;

		// pv
		if (ret) {
			std::string pv;
			int depth;
			Move move;
			auto start_pv = std::chrono::system_clock::now();
			std::tie(pv, depth, move) = dfpn.get_pv(pos);
			auto end_pv = std::chrono::system_clock::now();

			cout << "\t" << move.toUSI() << "\t" << pv << "\t" << depth << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_pv - start_pv).count();
		}
		cout << endl;
	}
	auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total).count();
	cout << total_ms << endl;

	return 0;
}
#endif

#if 0
#include "mate.h"
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	// 王手している角の利きには移動してしまうバグの再現局面
	string sfen = "6sn1/2+P3g1l/p1+P1r3p/2bpk2p1/6s1N/3PK1pPP/P1N2P3/2S1G1SR1/L4G1NL b B5Pglp 1";

	pos.set(sfen);
	bool ret = mateMoveInEvenPly<4>(pos);

	return 0;
}
#endif

#if 0
#include "fastmath.h"
int main() {
	for (int i = 0; i < 10; ++i) {
		float a0 = logf(100.0f * (i + 1));
		float a1 = FastLog(100.0f * (i + 1));

		cout << a0 << ", " << a1 << ", " << a0 - a1 << endl;
	}

	float a = 0;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < 100000; ++i) {
		a += logf(i + 1);
	}
	auto end = std::chrono::system_clock::now();
	cout << a << "\t" << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "[ns]" << endl;

	a = 0;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < 100000; ++i) {
		a += FastLog(i + 1);
	}
	end = std::chrono::system_clock::now();
	cout << a << "\t" << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "[ns]" << endl;
}
#endif

#if 0
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	vector<pair<string, string>> sfens = {
		{ "2+B+L4l/3R5/1P1s1pn2/pp1ppsk2/2p6/P7P/3SPPPPS/4G1GG1/LNR2BKN1 b G2Pnl3p 1", "black_win1.hcp" }, // 先手の勝ち
		{ "2+B+L4l/3R5/1P1s1pn2/pp1ppsk2/2p6/P7P/3SPPPPS/4G1GG1/LNR2BKN1 w G2Pnl3p 1", "black_win2.hcp" }, // 先手の勝ち(後手番開始)
		{ "1nkb2rnl/1gg1g4/spppps3/p7p/6P2/2KSPP1PP/2NP1S1p1/5r3/L4+l+b2 b NL3Pg2p 1", "white_win1.hcp" }, // 後手の勝ち
		{ "1nkb2rnl/1gg1g4/spppps3/p7p/6P2/2KSPP1PP/2NP1S1p1/5r3/L4+l+b2 w NL3Pg2p 1", "white_win2.hcp" }, // 後手の勝ち(後手番開始)
		{ "+N1K4G+S/2+PGP4/+N+P+R1+P3g/3S2B2/5p2p/1p7/+p2+lp1+p1P/2+sk+p4/5P2+r b BGS3L3P2n2p 1", "black_nyugyoku1.hcp" }, // 先手の入玉宣言
		{ "+N1K4G+S/2+PGP4/+N+P+R1+P3g/3S2B2/5p2p/1p7/+p2+lp1+p1P/2+sk+p4/5P2+r w BGS3L3P2n2p 1", "black_nyugyoku2.hcp" }, // 先手の入玉宣言(後手番開始)
		{ "+R1+Pp5/1+P2+PK+S2/p3P+L3/7P1/P2P5/2b2s+r2/G3+p2+p+n/4pg+p2/+sg4k1+n b bgs2n3l5p 1", "white_nyugyoku1.hcp" }, // 後手の入玉宣言
		{ "+R2p5/4+PK+S2/p1+P1P+L2+P/7P1/P2P5/2b2s+r2/G3+p2+p+n/4pg+p2/+sg4k1+n w 2N2Pbgs3l3p 1", "white_nyugyoku2.hcp" }, // 後手の入玉宣言(後手番開始)
	};

	// hcp出力
	for (auto sfen : sfens) {
		pos.set(sfen.first);
		HuffmanCodedPos hcp = pos.toHuffmanCodedPos();
		ofstream ofs(sfen.second, ios::binary);
		ofs.write((char*)&hcp, sizeof(hcp));
	}

	// テスト
	// selfplay --threashold 1 --threads 1 --usi_engine E:\game\shogi\apery_wcsc28\bin\apery_wcsc28_bmi2.exe --usi_engine_num 1 --usi_threads 1 --usi_options USI_Ponder:False,Threads:1,Byoyomi_Margin:0 F:\model\model_rl_val_wideresnet10_selfplay_236 R:\hcp\black_win.hcp R:\hcpe 1 800 0 1

	return 0;
}
#endif

#if 0
#include "mate.h"
// 詰み探索（王手生成のバグのあった局面）
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	vector<string> sfens = {
		// 詰み
		"1n6+P/l1B1s1g2/p1+LG3p1/2pspS1kp/1P3P3/P1P3R2/3P+nN1PP/3+n2G2/L3s1K1L b G3Prb2p 1",
		"1+B6+P/l3sg3/p+Ls1k1Ppb/2pspPp2/1P3s2K/P1P2+r3/3P1N1PP/2g+n2G2/L7L w GN3Prnp 1",
		"1n6+P/l3s1g2/p1sk2Ppp/1LpspPp2/1P4N2/P1P2p3/2+nP3PP/2g6/L3+rNK1L w 2BS2Pr2g 1",
		"1n6+P/l1B1s1g2/p1+L1k1spp/2pspP2b/1P4R2/P1P3P1P/4+n2PK/4g1+p1N/L7+r w GS3Pgnlp 1",
	};

	for (string sfen : sfens) {
		pos.set(sfen);
		bool ret = mateMoveInOddPly<5>(pos);
		cout << ret << endl;
	}

	return 0;
}
#endif

#if 0
// 合法手生成
int main() {
	initTable();
	Position::initZobrist();
	Position pos;

	vector<string> sfens = {
		"l5p1K/p8/2nrpp1g1/LLppk1L2/9/9/9/9/9 b r2b3g4s3n12p 1", // 2段目への香の不成(3d3b,8d8b,9d9b)
		"9/9/9/9/9/2l1KPPll/1G1PPRN2/8P/k1P5L w R2B3G4S3N12P 1" // 2段目への香の不成(1f1h,2f2h,7f7h)
	};

	for (string sfen : sfens) {
		std::cout << sfen << "\n";
		pos.set(sfen);
		for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
			std::cout << ml.move().toUSI() << "\n";
		}
		std::cout << std::endl;
	}

	return 0;
}
#endif

#if 0
// 自己対局で異常終了した詰み探索局面の調査
#include <fstream>
#include <regex>
#include "dfpn.h"
int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();

	// Positionをメモリダンプしたテキストを読み込む
	std::ifstream is(argv[1]);
	std::string str;
	char buf[sizeof(Position)] = {};
	auto re = std::regex(R"([0-9a-f]{2})");
	
	for (int i = 0; i < sizeof(Position); ) {
		is >> str;
		if (str.size() != 2) continue;
		if (!std::regex_match(str, re)) continue;

		buf[i++] = (char)std::stoi(str, nullptr, 16);
	}

	Position pos;
	std::memcpy((void*)&pos, buf, sizeof(Position));
	auto sfen = pos.toSFEN().substr(5);
	std::cout << sfen << std::endl;

	Position pos_copy;
	pos_copy.set(sfen);

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(150000);
	dfpn.set_maxdepth(25);

	bool mate;
	if (!pos_copy.inCheck()) {
		mate = dfpn.dfpn(pos_copy);
	}
	else {
		mate = dfpn.dfpn_andnode(pos_copy);
	}
	std::cout << mate << std::endl;

	return 0;
}
#endif

#if 0
#include "dfpn.h"
// DfPnで不正な手を返すバグ
int main()
{
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(150000);
	dfpn.set_maxdepth(27);

	Position pos;

	vector<string> sfens = {
		"3B2p1p/l5g2/6np1/p2RpS2k/1pp1gbsP1/P1n3g1l/LP3+p2L/K4p3/1NR3N2 b GS2Ps4p 137", // 1b1bを返す
	};

	auto start0 = std::chrono::system_clock::now();
	auto total = start0 - start0;
	for (string sfen : sfens) {
		pos.set(sfen);
		auto start = std::chrono::system_clock::now();
		bool ret = dfpn.dfpn(pos);
		auto end = std::chrono::system_clock::now();

		const auto move = dfpn.dfpn_move(pos);

		auto time = end - start;
		total += time;

		auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();

		cout << ret << "\t" << move.value() << "\t" << move.toUSI() << "\t";
		cout << time_ns / 1000000.0 << endl;
	}
	auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
	cout << total_ns / 1000000.0 << endl;
}
#endif

#if 0
int main(int argc, char* argv[]) {
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;
	std::string filepath = argv[1];

	std::ifstream ifs(filepath, std::ifstream::binary);
	if (!ifs) {
		std::cerr << "read error" << std::endl;
		return 1;
	}
	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			std::cout << p << std::endl;
			break;
		}

		// 局面
		Position pos;
		if (!pos.set(hcpe.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}
		if (hcpe.hcp.color() != pos.turn()) {
			std::cerr << "hcpe.hcp.color() != pos.turn()" << std::endl;
			break;
		}
	}

	return 0;
}
#endif

#if 0
int main() {
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	vector<string> sfens = {
		"1n6+P/l1B1s1g2/p1+LG3p1/2pspS1kp/1P3P3/P1P3R2/3P+nN1PP/3+n2G2/L3s1K1L b G3Prb2p 1",
		"1+B6+P/l3sg3/p+Ls1k1Ppb/2pspPp2/1P3s2K/P1P2+r3/3P1N1PP/2g+n2G2/L7L w GN3Prnp 1",
		"1n6+P/l3s1g2/p1sk2Ppp/1LpspPp2/1P4N2/P1P2p3/2+nP3PP/2g6/L3+rNK1L w 2BS2Pr2g 1",
		"1n6+P/l1B1s1g2/p1+LG3p1/2pspS1kp/1P3P3/P1P3R2/3P+nN1PP/3+n2G2/L3s1K1L b G3Prb2p 1",
		"1n6+P/l3s1g2/p1sk2Ppp/1LpspPp2/1P4N2/P1P2p3/2+nP3PP/2g6/L3+rNK1L w 2BS2Pr2g 1",
		"1n6+P/l1B1s1g2/p1+L1k1spp/2pspP2b/1P4R2/P1P3P1P/4+n2PK/4g1+p1N/L7+r w GS3Pgnlp 1",
	};

	std::unordered_map<HuffmanCodedPos, int> map;

	for (string sfen : sfens) {
		pos.set(sfen);
		const auto hcp = pos.toHuffmanCodedPos();
		auto itr = map.find(hcp);
		auto p = map.emplace(hcp, 1);
		if (p.second) {
			std::cout << "insert" << std::endl;
		}
		else {
			std::cout << "exist" << std::endl;
		}
	}

}
#endif

#if 0
#include "dfpn.h"
// 王手がかかっているときDfPnで不正な手を返すバグ修正
int main()
{
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_maxdepth(33);

	Position pos;

	vector<string> sfens = {
		"7n1/l1g4k1/pp5gl/4s1ppp/3b+Bn3/1G1K5/PP1P1P2P/3R5/2+r1SG2L w N6P2snl2p 144", // oute_kaihimore
		"3b3n1/l1g1B2k1/pp2sK1gl/3s2ppp/4sn3/1G7/PP1P1P2P/3R5/2+r1SG2L w N6Pnl2p 152", // 1手
		"7n1/l1g4k1/pp5gl/3ss1ppp/3b+Bn3/1G1K5/PP1P1P2P/3R5/2+r1SG2L w N6Psnl2p 1" // 王手回避&1手詰め
	};

	auto start0 = std::chrono::system_clock::now();
	auto total = start0 - start0;
	for (string sfen : sfens) {
		pos.set(sfen);
		auto start = std::chrono::system_clock::now();
		bool ret = dfpn.dfpn(pos);
		auto end = std::chrono::system_clock::now();

		const auto move = dfpn.dfpn_move(pos);

		auto time = end - start;
		total += time;

		auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time).count();

		cout << ret << "\t" << move.value() << "\t" << move.toUSI() << "\t";
		cout << time_ns / 1000000.0 << endl;
	}
	auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
	cout << total_ns / 1000000.0 << endl;
}
#endif

#if 0
// hcpe3から最後の局面と指し手を抽出
int main(int argc, char* argv[])
{
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	const auto filepath = argv[1];
	std::ifstream ifs(argv[1], std::ios::binary);

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		// 開始局面
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}

		std::stringstream sfen("sfen ");
		sfen << pos.toSFEN() << " moves";

		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (moveInfo.candidateNum > 0) {
				ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

			if (i == hcpe3.moveNum - 1) {
				std::cout << sfen.str() << "\t" << move.toUSI() << "\t" << (int)hcpe3.result << std::endl;
				break;
			}

			sfen << " " << move.toUSI();
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return 0;
}
#endif

#if 0
// hcpe3から最後の局面を抽出
int main(int argc, char* argv[])
{
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	const auto filepath = argv[1];
	std::ifstream ifs(argv[1], std::ios::binary);

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		// 開始局面
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}

		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (moveInfo.candidateNum > 0) {
				ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

			if (i == hcpe3.moveNum - 1 && (hcpe3.result == BlackWin || hcpe3.result == WhiteWin)) {
				std::cout << pos.toSFEN(1).substr(5) << std::endl;
				break;
			}

			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return 0;
}
#endif

#if 0
// hcpe3から最後の局面を検索
int main(int argc, char* argv[])
{
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	if (argc < 3) return 1;

	const auto filepath = argv[1];
	std::ifstream ifs(argv[1], std::ios::binary);

	const auto sfen = argv[2];
	pos.set(sfen);
	const auto key = pos.getKey();

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		// 開始局面
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}

		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (moveInfo.candidateNum > 0) {
				ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

			if (i == hcpe3.moveNum - 1 && (hcpe3.result == BlackWin || hcpe3.result == WhiteWin)) {
				if (pos.getKey() == key) {
					std::cout << pos.toSFEN() << "\n";
					std::cout << move.toUSI() << "\n";
					std::cout << (int)moveInfo.candidateNum << "\n";
					std::cout << (int)moveInfo.eval << "\n";
					std::cout << (int)hcpe3.result << "\n";
					std::cout << (int)hcpe3.opponent << "\n";
				}
				break;
			}

			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return 0;
}
#endif

#if 0
// hcpe3をopponentで分割
int main(int argc, char* argv[])
{
	if (argc < 2)
		return 1;

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	const std::string filepath{ argv[1] };
	std::ifstream ifs(argv[1], std::ios::binary);
	std::ofstream ofs[3];
	MoveVisits moveVisits[593];

	while (ifs) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		if (!ofs[hcpe3.opponent].is_open()) {
			const auto ext_pos = filepath.rfind('.');
			const std::string basepath = (ext_pos == std::string::npos) ? filepath : filepath.substr(0, ext_pos);
			const std::string ext = (ext_pos == std::string::npos) ? "" : filepath.substr(ext_pos);

			ofs[hcpe3.opponent].open(basepath + "_opp" + std::to_string(hcpe3.opponent) + ext, std::ios::binary);
		}

		ofs[hcpe3.opponent].write((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			ofs[hcpe3.opponent].write((char*)&moveInfo, sizeof(MoveInfo));
			if (moveInfo.candidateNum > 0) {
				ifs.read((char*)moveVisits, sizeof(MoveVisits) * moveInfo.candidateNum);
				ofs[hcpe3.opponent].write((char*)moveVisits, sizeof(MoveVisits) * moveInfo.candidateNum);
			}
		}
	}

	return 0;
}
#endif

#if 0
// hcpe3の同一局面の数
int main(int argc, char* argv[])
{
	if (argc < 2)
		return 1;

	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
	Position pos;

	const std::string filepath{ argv[1] };
	std::ifstream ifs(argv[1], std::ios::binary);
	MoveVisits moveVisits[593];

	struct Data {
		int black;
		int white;
		int draw;
		int ply;
		int count;

		Data() : black(0), white(0), draw(0), ply(0), count(0) {}
	};
	std::vector<std::unordered_map<HuffmanCodedPos, Data>> map_counts(3);

	Position position;
	int position_num[3] = {};
	while (ifs) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}
		position.set(hcpe3.hcp);
		if (!pos.set(hcpe3.hcp)) {
			return 1;
		}
		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (moveInfo.candidateNum > 0) {
				position_num[hcpe3.opponent]++;
				auto& data = map_counts[hcpe3.opponent][pos.toHuffmanCodedPos()];
				const auto result = hcpe3.result & 3;
				if (result == BlackWin)
					data.black++;
				else if (result == WhiteWin)
					data.white++;
				else if (result == Draw)
					data.draw++;
				data.ply = i;
				data.count++;
				ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
				
			}
			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	for (int opp = 0; opp < 3; opp++) {
		// サマリ
		std::cout << "opponent\t" << opp << std::endl;
		std::cout << "total position num\t" << position_num[opp] << std::endl;
		std::cout << "unique position num\t" << map_counts[opp].size() << std::endl;

		// ソート
		const size_t num = std::min<size_t>(1000, map_counts[opp].size());
		std::vector<std::pair<HuffmanCodedPos, Data>> counts;
		for (auto v : map_counts[opp]) {
			counts.emplace_back(v.first, v.second);
		}
		std::partial_sort(counts.begin(), counts.begin() + num, counts.end(), [](const auto& lhs, const auto& rhs) {
			if (lhs.second.count == rhs.second.count)
				return lhs.second.ply < rhs.second.ply;
			return lhs.second.count > rhs.second.count;
		});

		for (int i = 0; i < num; i++) {
			position.set(counts[i].first);
			const auto& data = counts[i].second;
			std::cout << position.toSFEN() << "\t" << data.ply << "\t" << data.count << "\t" << data.black << ":" << data.white << ":" << data.draw << std::endl;
		}
	}

	return 0;
}
#endif

#if 0
// hcpe3の同一手順の棋譜を削除
#include <unordered_map>
int main(int argc, char* argv[])
{
	if (argc < 3)
		return 1;

	initTable();
	HuffmanCodedPos::init();

	std::ifstream ifs(argv[1], std::ios::binary);
	std::vector<char> buf;

	std::ofstream ofs(argv[2], std::ios::binary);

	std::unordered_map<std::string, int> map;

	int game_num = 0;
	while (ifs) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		long long size = 0;
		std::stringstream ss;

		ss.write((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (ifs.eof()) {
				std::cerr << "read error" << std::endl;
				goto L_EXIT;
			}
			size += sizeof(MoveInfo);
			ss.write((char*)&moveInfo.selectedMove16, sizeof(moveInfo.selectedMove16));
			if (moveInfo.candidateNum > 0) {
				const size_t move_visits_size = sizeof(MoveVisits) * moveInfo.candidateNum;
				ifs.seekg(move_visits_size, std::ios_base::cur);
				size += move_visits_size;
			}
		}

		const auto ret = map.try_emplace(ss.str(), 1);
		if (ret.second) {
			// 書き出し
			ifs.seekg(-size, std::ios_base::cur);
			buf.resize(size);
			ifs.read(buf.data(), size);
			if (ifs.eof()) {
				std::cerr << "read error" << std::endl;
				break;
			}
			ofs.write((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
			ofs.write(buf.data(), size);
		}
		else {
			ret.first->second++;
		}

		game_num++;
	}
L_EXIT:

	ofs.close();
	ifs.close();

	// サマリ
	std::cout << "total game num\t" << game_num << std::endl;
	std::cout << "unique game num\t" << map.size() << std::endl;

	if (argc >= 4) {
		size_t num = 0;
		try {
			num = std::stoi(argv[3]);
		}
		catch (std::invalid_argument&) {
			return 0;
		}

		std::cout << "sfen\tmove num\tresult\topponent\tcount" << std::endl;

		Position::initZobrist();
		Position pos;

		// ソート
		num = std::min<size_t>(num, map.size());
		std::vector<std::pair<std::string, int>> counts;
		for (auto v : map) {
			counts.emplace_back(v.first, v.second);
		}
		std::partial_sort(counts.begin(), counts.begin() + num, counts.end(), [](const auto& lhs, const auto& rhs) {
			return lhs.second > rhs.second;
			});

		for (int i = 0; i < num; i++) {
			const char* data = counts[i].first.data();
			HuffmanCodedPosAndEval3* phcpe3 = (HuffmanCodedPosAndEval3*)data;
			pos.set(phcpe3->hcp);
			u16* moves = (u16*)(data + sizeof(HuffmanCodedPosAndEval3));
			const auto move_num = (counts[i].first.size() - sizeof(HuffmanCodedPosAndEval3)) / sizeof(u16);
			std::cout << pos.toSFEN() << " moves";
			for (size_t j = 0; j < move_num; j++) {
				const auto move = move16toMove((Move)moves[j], pos);
				std::cout << " " << move.toUSI();
			}
			std::cout << "\t" << (int)phcpe3->moveNum;
			std::cout << "\t" << (int)phcpe3->result;
			std::cout << "\t" << (int)phcpe3->opponent;
			std::cout << "\t" << counts[i].second << std::endl;
		}
	}

	std::_Exit(0);
	return 0;
}
#endif

#if 0
// hcpe3の同一手順の棋譜を平均化
#include <unordered_map>
int main(int argc, char* argv[])
{
	if (argc < 3)
		return 1;

	initTable();
	HuffmanCodedPos::init();

	std::ifstream ifs(argv[1], std::ios::binary);
	std::vector<char> buf;

	std::ofstream ofs(argv[2], std::ios::binary);

	struct MoveInfo2 {
		u16 selectedMove16; // 指し手
		int eval; // 評価値
		int count;
	};
	struct Data {
		int count;
		HuffmanCodedPosAndEval3 hcpe3;
		std::vector<MoveInfo2> vecMoveInfo;
		std::vector<std::map<u16, unsigned int>> vecMoveVisits;
	};
	std::unordered_map<std::string, Data> map;

	int game_num = 0;
	while (ifs) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}

		std::stringstream ss;
		std::vector<MoveInfo> vecMoveInfo;
		std::vector<std::vector<MoveVisits>> vecMoveVisits;

		ss.write((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo& moveInfo = vecMoveInfo.emplace_back();
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			if (ifs.eof()) {
				std::cerr << "read error" << std::endl;
				goto L_EXIT;
			}
			ss.write((char*)&moveInfo.selectedMove16, sizeof(moveInfo.selectedMove16));
			std::vector<MoveVisits> moveVisits(moveInfo.candidateNum);
			if (moveInfo.candidateNum > 0) {
				ifs.read((char*)moveVisits.data(), sizeof(MoveVisits) * moveInfo.candidateNum);
				if (ifs.eof()) {
					std::cerr << "read error" << std::endl;
					goto L_EXIT;
				}
			}
			vecMoveVisits.emplace_back(std::move(moveVisits));
		}

		const auto ret = map.try_emplace(ss.str(), Data());
		if (ret.second) {
			auto& data = ret.first->second;
			data.count = 1;
			data.hcpe3 = hcpe3;
			for (size_t i = 0; i < vecMoveVisits.size(); i++) {
				const auto& moveInfo = vecMoveInfo[i];
				auto& moveInfo2 = data.vecMoveInfo.emplace_back();
				moveInfo2.selectedMove16 = moveInfo.selectedMove16;
				moveInfo2.eval = moveInfo.eval;
				auto& mapMoveVisits = data.vecMoveVisits.emplace_back();
				if (moveInfo.candidateNum > 0) {
					moveInfo2.count = 1;

					const auto& moveVisits = vecMoveVisits[i];
					for (const auto& v : moveVisits) {
						mapMoveVisits.emplace(v.move16, v.visitNum);
					}
				}
				else {
					moveInfo2.count = 0;
				}
			}
		}
		else {
			auto& data = ret.first->second;
			data.count++;
			for (size_t i = 0; i < vecMoveVisits.size(); i++) {
				const auto& moveInfo = vecMoveInfo[i];
				if (moveInfo.candidateNum > 0) {
					data.vecMoveInfo[i].count++;
					data.vecMoveInfo[i].eval += moveInfo.eval;

					const auto& moveVisits = vecMoveVisits[i];
					auto& mapMoveVisits = data.vecMoveVisits[i];
					for (const auto& v : moveVisits) {
						const auto ret2 = mapMoveVisits.try_emplace(v.move16, v.visitNum);
						if (!ret2.second) {
							ret2.first->second += v.visitNum;
						}
					}
				}
			}
		}

		game_num++;
	}
L_EXIT:

	// 書き出し
	for (const auto& v : map) {
		const auto& data = v.second;
		ofs.write((char*)&data.hcpe3, sizeof(HuffmanCodedPosAndEval3));
		for (size_t i = 0; i < data.vecMoveInfo.size(); i++) {
			const auto& moveInfo2 = data.vecMoveInfo[i];
			const auto& mapMoveVisits = data.vecMoveVisits[i];
			MoveInfo moveInfoAvr{ moveInfo2.selectedMove16, moveInfo2.count > 0 ? (s16)(moveInfo2.eval / moveInfo2.count) : moveInfo2.eval, (u16)mapMoveVisits.size() };
			ofs.write((char*)&moveInfoAvr, sizeof(MoveInfo));

			for (const auto& moveVisits : mapMoveVisits) {
				MoveVisits moveVisitsAvr{ moveVisits.first, (u16)(moveVisits.second / moveInfo2.count) };
				ofs.write((char*)&moveVisitsAvr, sizeof(MoveVisits));
			}
		}
	}
	ofs.close();
	ifs.close();

	// サマリ
	std::cout << "total game num\t" << game_num << std::endl;
	std::cout << "unique game num\t" << map.size() << std::endl;

	if (argc >= 4) {
		size_t num = 0;
		try {
			num = std::stoi(argv[3]);
		}
		catch (std::invalid_argument&) {
			return 0;
		}

		std::cout << "sfen\tmove num\tresult\topponent\tcount" << std::endl;

		Position::initZobrist();
		Position pos;

		// ソート
		num = std::min<size_t>(num, map.size());
		std::vector<std::pair<std::string, Data>> counts;
		for (auto v : map) {
			counts.emplace_back(v.first, v.second);
		}
		std::partial_sort(counts.begin(), counts.begin() + num, counts.end(), [](const auto& lhs, const auto& rhs) {
			return lhs.second.count > rhs.second.count;
			});

		for (int i = 0; i < num; i++) {
			const char* data = counts[i].first.data();
			HuffmanCodedPosAndEval3* phcpe3 = (HuffmanCodedPosAndEval3*)data;
			pos.set(phcpe3->hcp);
			u16* moves = (u16*)(data + sizeof(HuffmanCodedPosAndEval3));
			const auto move_num = (counts[i].first.size() - sizeof(HuffmanCodedPosAndEval3)) / sizeof(u16);
			std::cout << pos.toSFEN() << " moves";
			for (size_t j = 0; j < move_num; j++) {
				const auto move = move16toMove((Move)moves[j], pos);
				std::cout << " " << move.toUSI();
			}
			std::cout << "\t" << (int)phcpe3->moveNum;
			std::cout << "\t" << (int)phcpe3->result;
			std::cout << "\t" << (int)phcpe3->opponent;
			std::cout << "\t" << counts[i].second.count << std::endl;
		}
	}

	std::_Exit(0);
	return 0;
}
#endif

#if 0
int main()
{
	initTable();
	Position pos;
	pos.set("lnsgkgsnl/7b1/ppppppppp/9/P2R3P1/6P2/1PPPPPN1P/1B5R1/LNSGKGS1L b - 1");

	const Bitboard occ = pos.occupiedBB();
	occ.printBoard();

	Bitboard bb;

	bb = rookAttack(SQ28, occ);
	bb.printBoard();

	bb = rookAttack(SQ65, occ);
	bb.printBoard();

	bb = lanceAttack(Black, SQ19, occ);
	bb.printBoard();
}
#endif

#if 0
int main()
{
	initTable();
	Position pos;
	//pos.set("l6nl/6gk1/p6pp/4s1s2/1pB1pp1P1/2PP5/PPS1P+b2P/1KG6/LN5NL b RGN3Prgs2p 85");
	pos.set("l1S4nl/3R2gP1/2P1ppbsp/2gpk1Np1/1p3PP2/p5K1P/1PB1P1N2/3G1Sg2/L1SR4L w Pn3p 116");

	features1_t features1{};
	features2_t features2{};
	make_input_features(pos, features1, features2);

	std::cout << "features1\n";
	for (Color c = Black; c < ColorNum; ++c) {
		std::cout << "color:" << c << "\n";
		for (int f1idx = 0; f1idx < MAX_FEATURES1_NUM; ++f1idx) {
			for (Square sq = SQ11; sq < SquareNum; ++sq) {
				const auto v = (float)features1[c][f1idx][sq];
				std::cout << v << ",";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	std::cout << "features2\n";
	for (Color c = Black; c < ColorNum; ++c) {
		std::cout << "color:" << c << "\n";
		for (int f2idx = 0; f2idx < MAX_PIECES_IN_HAND_SUM; ++f2idx) {
			const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx;
			const auto v = (float)features2[idx][0];
			std::cout << v << ",";
		}
		std::cout << "\n";
	}
	{
		const int idx = MAX_FEATURES2_HAND_NUM;
		const auto v = (float)features2[idx][0];
		std::cout << v << "\n";
	}
	std::cout << "\n";


	packed_features1_t packed_features1{};
	packed_features2_t packed_features2{};
	make_input_features(pos, packed_features1, packed_features2);

	std::cout << "packed_features1\n";
	for (Color c = Black; c < ColorNum; ++c) {
		std::cout << "color:" << c << "\n";
		for (int f1idx = 0; f1idx < MAX_FEATURES1_NUM; ++f1idx) {
			for (Square sq = SQ11; sq < SquareNum; ++sq) {
				const int idx = MAX_FEATURES1_NUM * (int)SquareNum * (int)c + (int)SquareNum * f1idx + sq;
				const auto v = ((packed_features1[idx >> 3] >> (idx & 7)) & 1);
				std::cout << v << ",";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

	std::cout << "packed_features2\n";
	for (Color c = Black; c < ColorNum; ++c) {
		std::cout << "color:" << c << "\n";
		for (int f2idx = 0; f2idx < MAX_PIECES_IN_HAND_SUM; ++f2idx) {
			const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx;
			const auto v = ((packed_features2[idx >> 3] >> (idx & 7)) & 1);
			std::cout << v << ",";
		}
		std::cout << "\n";
	}
	{
		const int idx = MAX_FEATURES2_HAND_NUM;
		const auto v = ((packed_features2[idx >> 3] >> (idx & 7)) & 1);
		std::cout << v << "\n";
	}

	return 0;
}
#endif

#if 0
#include "unpack.h"
int main()
{
	initTable();
	Position pos;
	std::vector<std::string> sfens = {
		"l6nl/6gk1/p6pp/4s1s2/1pB1pp1P1/2PP5/PPS1P+b2P/1KG6/LN5NL b RGN3Prgs2p 85",
		"l1S4nl/3R2gP1/2P1ppbsp/2gpk1Np1/1p3PP2/p5K1P/1PB1P1N2/3G1Sg2/L1SR4L w Pn3p 116",
	};
	const int batch_size = sfens.size();

	auto print_features = [](features1_t features1, features2_t features2)
	{
		std::cout << "features1\n";
		for (Color c = Black; c < ColorNum; ++c) {
			std::cout << "color:" << c << "\n";
			for (int f1idx = 0; f1idx < MAX_FEATURES1_NUM; ++f1idx) {
				for (Square sq = SQ11; sq < SquareNum; ++sq) {
					if (sq > SQ11 && sq % 9 == 0) std::cout << " ";
					const auto v = (float)features1[c][f1idx][sq];
					std::cout << v << ",";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}

		std::cout << "features2\n";
		for (Color c = Black; c < ColorNum; ++c) {
			std::cout << "color:" << c << "\n";
			for (int f2idx = 0; f2idx < MAX_PIECES_IN_HAND_SUM; ++f2idx) {
				const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx;
				const auto v = (float)features2[idx][0];
				std::cout << v << ",";
			}
			std::cout << "\n";
		}
		{
			const int idx = MAX_FEATURES2_HAND_NUM;
			const auto v = (float)features2[idx][0];
			std::cout << v << "\n";
		}
		std::cout << "\n";
	};

	auto print_packed_features = [](packed_features1_t packed_features1, packed_features2_t packed_features2)
	{
		std::cout << "packed_features1\n";
		for (Color c = Black; c < ColorNum; ++c) {
			std::cout << "color:" << c << "\n";
			for (int f1idx = 0; f1idx < MAX_FEATURES1_NUM; ++f1idx) {
				for (Square sq = SQ11; sq < SquareNum; ++sq) {
					if (sq > SQ11 && sq % 9 == 0) std::cout << " ";
					const int idx = MAX_FEATURES1_NUM * (int)SquareNum * (int)c + (int)SquareNum * f1idx + sq;
					const auto v = ((packed_features1[idx >> 3] >> (idx & 7)) & 1);
					std::cout << v << ",";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}

		std::cout << "packed_features2\n";
		for (Color c = Black; c < ColorNum; ++c) {
			std::cout << "color:" << c << "\n";
			for (int f2idx = 0; f2idx < MAX_PIECES_IN_HAND_SUM; ++f2idx) {
				const int idx = MAX_PIECES_IN_HAND_SUM * (int)c + f2idx;
				const auto v = ((packed_features2[idx >> 3] >> (idx & 7)) & 1);
				std::cout << v << ",";
			}
			std::cout << "\n";
		}
		{
			const int idx = MAX_FEATURES2_HAND_NUM;
			const auto v = ((packed_features2[idx >> 3] >> (idx & 7)) & 1);
			std::cout << v << "\n";
		}
		std::cout << "\n";
	};


	features1_t* features1 = new features1_t[batch_size];
	features2_t* features2 = new features2_t[batch_size];
	std::fill_n((DType*)features1, sizeof(features1_t) / sizeof(DType) * batch_size, _zero);
	std::fill_n((DType*)features2, sizeof(features2_t) / sizeof(DType) * batch_size, _zero);

	for (int i = 0; i < batch_size; ++i) {
		pos.set(sfens[i]);
		make_input_features(pos, features1[i], features2[i]);
		std::cout << "batch:" << i << "\n";
		print_features(features1[i], features2[i]);
	}

	packed_features1_t* packed_features1 = new packed_features1_t[batch_size];
	packed_features2_t* packed_features2 = new packed_features2_t[batch_size];
	std::fill_n((char*)packed_features1, sizeof(packed_features1_t) * batch_size, 0);
	std::fill_n((char*)packed_features2, sizeof(packed_features2_t) * batch_size, 0);

	for (int i = 0; i < batch_size; ++i) {
		pos.set(sfens[i]);
		make_input_features(pos, packed_features1[i], packed_features2[i]);
		//std::cout << "batch:" << i << "\n";
		//print_packed_features(packed_features1[i], packed_features2[i]);
	}

	packed_features1_t* p1_dev;
	packed_features2_t* p2_dev;
	features1_t* x1_dev;
	features2_t* x2_dev;
	cudaMalloc((void**)&p1_dev, sizeof(packed_features1_t) * batch_size);
	cudaMalloc((void**)&p2_dev, sizeof(packed_features2_t) * batch_size);
	cudaMalloc((void**)&x1_dev, sizeof(features1_t) * batch_size);
	cudaMalloc((void**)&x2_dev, sizeof(features2_t) * batch_size);
	features1_t* x1 = new features1_t[batch_size];
	features2_t* x2 = new features2_t[batch_size];

	cudaMemcpyAsync(p1_dev, packed_features1, sizeof(packed_features1_t) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread);
	cudaMemcpyAsync(p2_dev, packed_features2, sizeof(packed_features2_t) * batch_size, cudaMemcpyHostToDevice, cudaStreamPerThread);
	unpack_features1(batch_size, p1_dev, x1_dev, cudaStreamPerThread);
	unpack_features2(batch_size, p2_dev, x2_dev, cudaStreamPerThread);
	cudaMemcpyAsync(x1, x1_dev, sizeof(features1_t) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread);
	cudaMemcpyAsync(x2, x2_dev, sizeof(features2_t) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread);
	cudaStreamSynchronize(cudaStreamPerThread);

	for (int i = 0; i < batch_size; ++i) {
		std::cout << "batch:" << i << "\n";
		print_features(x1[i], x2[i]);
	}

	return 0;
}
#endif

#if 0
#include <fstream>
#include <regex>
#include "book.hpp"
int main(int argc, char* argv[]) {
	initTable();
	Position pos;

	struct Entry {
		Move move;
		int eval;
		double prob;
	};

	constexpr int evalLimit = -200;
	std::regex re(R"(^(\S+) \S+ (\S+) \d+ \d+$)");

	std::ifstream ifs(argv[1]);
	std::string line;
	std::vector<Entry> entries;
	std::map<Key, std::vector<BookEntry> > outMap;

	auto outputEntries = [&entries, &outMap](Key key) {
		std::vector<BookEntry> bookEntries;
		bookEntries.reserve(entries.size());
		double max = DBL_MIN;
		constexpr double beta = 1.0 / 0.1;
		for (int i = 0; i < entries.size(); i++) {
			double& x = entries[i].prob;
			x *= beta;
			if (x > max) {
				max = x;
			}
		}
		// オーバーフローを防止するため最大値で引く
		double sum = 0.0;
		for (int i = 0; i < entries.size(); i++) {
			double& x = entries[i].prob;
			x = exp(x - max);
			sum += x;
		}
		// normalize
		for (int i = 0; i < entries.size(); i++) {
			double& x = entries[i].prob;
			x /= sum;
		}

		for (auto& entry : entries) {
			u16 count = (u16)(entry.prob * 1000);
			if (count > 0)
				bookEntries.emplace_back(BookEntry{ key, (u16)entry.move.value(), count, (Score)entry.eval });
		}

		if (bookEntries.size() > 0)
			outMap.emplace(key, std::move(bookEntries));
	};

	auto score_to_value = [](const int score) {
		return 1.0 / (1.0 + exp(-(double)score / 100.0));
	};


	while (ifs) {
		std::getline(ifs, line);
		if (line.size() == 0) {
			outputEntries(Book::bookKey(pos));
			break;
		}

		if (line[0] == '#') continue;

		if (line.substr(0, 4) == "sfen") {
			if (entries.size() > 0) {
				outputEntries(Book::bookKey(pos));
			}

			pos.set(line.substr(5));
			entries.clear();
		}
		else {
			std::smatch m;
			if (std::regex_match(line, m, re)) {
				int eval = std::stoi(m[2]);
				if (eval >= evalLimit) {
					entries.emplace_back(Entry{ usiToMove(pos, m[1]), eval, score_to_value(eval) });
				}
			}
		}
	}

	std::ofstream ofs(argv[2], std::ios::binary);
	for (auto& elem : outMap) {
		for (auto& elel : elem.second)
			ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
	}

	std::cout << outMap.size() << std::endl;

	return 0;
}
#endif
