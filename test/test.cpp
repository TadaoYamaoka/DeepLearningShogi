#include "gtest/gtest.h"

#include <iostream>
#include <chrono>
#include <fstream>

#include "cppshogi.h"
#include "python_module.h"
#include "usi.hpp"
#include "dfpn.h"

using namespace std;

TEST(Position, moveIsDraw) {
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    Position pos;
    {
        pos.set(DefaultStartPositionSFEN);
        std::istringstream ssPosCmd("2g2f 4a3b 6i7h 8c8d 7g7f 3a4b 2f2e 8d8e 1g1f 8e8f 8g8f 8b8f 2e2d 2c2d 2h2d P*2c 2d2f 8f8b P*8g 6a5b 3i3h 1c1d 7i6h 9c9d 3g3f 3c3d 8h2b+ 3b2b 8i7g 6c6d 3h3g 4b3c 6g6f 7a6b 6h6g 2b3b 5i6h 8a9c 3g4f 9c8e 7g8e 8b8e 2i3g 6b6c 2f2e 8e8a 4i4h 9d9e 2e2i 5a6b 2i8i N*5d 8g8f 5d4f 4g4f 6c7d 8f8e P*8g 8i8g 7d8e 8g8i P*8h 8i8h 8e8f P*8c 8a8c N*7e 8c8d P*8g 8f7e 7f7e 8d8a N*7f 6d6e 6f6e B*9d S*7g 6b5a P*2b 3c2b 8g8f 5a4b 8f8e N*5d 9g9f P*6f 6g6f 9e9f 6f5e P*6g 6h6g 8a8e 8h8e 9d8e 5e5d 5c5d B*6d 4b3c R*8a S*6i 7h6h P*6f 6g6f 5b6c 8a8e+ 6c6d 6e6d B*4d 6f5f 2b3a B*5a 3c2b P*2d 2c2d P*2c 2b2c P*2e 2c1b 2e2d P*2b 8e8b P*8a 8b5b S*4b 5a7c+ R*7i 7g6f 6i7h+ 6h6g 7i1i+ 6d6c+ 1i6i 4f4e 4d1g+ N*2c L*5e 5f6e 2b2c 2d2c+ 1b2c P*2d 2c2d 7c5e 1g1f 5e4f 2d1e L*1h N*1g G*2e 1f2e 3g2e G*1f 1h1g 1f1g B*6b P*6d 7f6d L*3e 6g7f P*2f 3f3e 6i9i 6e7d 2f2g+ 3e3d 4c4d 6d7b+ 9i8h 4f9a 1e2e P*8d 7h7g 6f7g 8h4h 6b7a+ 4h4e 7b8a 9f9g+ 7d8c 4e3d 7e7d 2a3c 5b5d 1g2h 7d7c+ 2e1f 7a4d 3d2d 8c9b 3c2e 8d8c+ 4b3c 5d5f N*2f 4d7a 3a4b 7g6f 9g8g 5f6e P*9f 5g5f 1d1e 5f5e P*3f 5e5d 9f9g+ 5d5c+ 9g9f 6f7e 8g8f 7e8f 9f8f 7f8f S*4d L*4i 4b5c 6c5c 4d5c 7a5c L*4d 4i4d 3c4d 5c6b P*4f 8a8b P*5e 6e7d L*4c P*4e 4d3c 7d2d 3c2d S*7e 1a1c S*4a G*3a 4a3b+");

        StateListPtr states{ new std::deque<StateInfo>(1) };
        std::string token;
        while (ssPosCmd >> token) {
            const Move move = usiToMove(pos, token);
            pos.doMove(move, states->emplace_back());
        }

        const Move move = usiToMove(pos, "3a3b");
        const auto draw = pos.moveIsDraw(move, 16);
        EXPECT_EQ(NotRepetition, draw);
    }
    {
        pos.set(DefaultStartPositionSFEN);
        std::istringstream ssPosCmd("2g2f 7a7b 2f2e 4a3b 7g7f 8c8d 8h7g 3c3d 7i6h 2b3c 7g3c+ 3b3c 6h7g 5a4b 3g3f 9c9d 2i3g 6a5b 3i3h 3a2b 5i6h 7c7d 1g1f 8d8e 6i7h 9d9e 1f1e 8a7c 4g4f 6c6d 3h4g 7b6c 4i4h 8b8a 4f4e 4b5a 6h5h 5a6b 2h2i 3c3b 2e2d 2c2d 2i2d 4c4d 4e4d 2b3c 2d2i 3d3e P*2b P*2g 2i2g 3c4d 2g2d 3e3f 4g3f P*3e 3f4g 4d3c 2d2f 3c2b 5h6h 2b2c 6g6f 2c3d 7f7e P*2e 2f2i 6d6e 7e7d 6c7d P*7e 7d6c 6f6e 8e8f 8g8f P*8h 7h8h 7c6e 7g7f P*6g 6h5h B*5e 8h8g 6c5d 7e7d P*7b 5h6g 5e9i+ P*6f 9i9h 6g7h L*8b 7f7e 3d4c 4h5h 2a3c 6f6e 5d6e 8g7g 6e7f P*6c 6b5a N*6d P*4f 4g5f 8b8f P*8b 8a8b 6d5b+ 4c5b P*8c 8b8c 6c6b+ 5a6b P*8d 8f8h+ 7h6h 8c8a P*6c 5b6c P*6d 6c5d B*3d P*6g 6h5i 4f4g+ 5f4g P*4c 3g2e P*2d 7g7f 9h7f S*6c 5d6c 6d6c+ 6b6c S*6d 6c5b 2e3c+ 3b3c N*4e G*4b 4e3c+ 4b3c 3d4e 8h7h G*6c 5b4a 6d5c+ 6g6h+ 5h6h 7h6h 5i6h P*6g 6h5h 4a3b P*3d 6g6h+ 5h6h N*5f 4e5f P*6g 6h5i S*5h 5i5h 6g6h+ 5h6h G*6g 5f6g 7f6g 6h6g N*5e 6g6f 5e6g+ 6f6g B*4e 6g7f 4e6g+");

        StateListPtr states{ new std::deque<StateInfo>(1) };
        std::string token;
        while (ssPosCmd >> token) {
            const Move move = usiToMove(pos, token);
            pos.doMove(move, states->emplace_back());
        }

        const Move move = usiToMove(pos, "7f6g");
        const auto draw = pos.moveIsDraw(move, 16);
        EXPECT_EQ(RepetitionInferior, draw);
    }
}

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

TEST(Hcpe3Test, cache_re_eval) {
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
    u16 move_3c3d = (u16)usiToMove(pos[1], "3c3d").value();
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

    HuffmanCodedPosAndEval3 hcpe3_1{ pos[0].toHuffmanCodedPos(), 2, BlackWin, 0 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&hcpe3_1), sizeof(HuffmanCodedPosAndEval3));
    MoveInfo move_info1_1{ move_2g2f, 200, 2 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info1_1), sizeof(MoveInfo));
    ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates1_1.data()), sizeof(MoveVisits) * candidates1_1.size());
    MoveInfo move_info2{ move_8c8d, 210, 2 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info2), sizeof(MoveInfo));
    ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates2.data()), sizeof(MoveVisits) * candidates2.size());

    HuffmanCodedPosAndEval3 hcpe3_2{ pos[0].toHuffmanCodedPos(), 2, WhiteWin, 0 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&hcpe3_2), sizeof(HuffmanCodedPosAndEval3));
    MoveInfo move_info1_2{ move_7g7f, 220, 2 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info1_2), sizeof(MoveInfo));
    ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates1_2.data()), sizeof(MoveVisits) * candidates1_2.size());
    MoveInfo move_info3{ move_8b3b, 230, 1 };
    ofs_hcpe3_1.write(reinterpret_cast<char*>(&move_info3), sizeof(MoveInfo));
    ofs_hcpe3_1.write(reinterpret_cast<char*>(candidates3.data()), sizeof(MoveVisits) * candidates3.size());

    ofs_hcpe3_1.close();

    // cache作成
    size_t actual_len = 0;
    auto len = __load_hcpe3("test1.hcpe3", true, 600, 1, actual_len);
    EXPECT_EQ(4, actual_len);
    EXPECT_EQ(3, len);
    __hcpe3_create_cache("test1.cache");

    // load cache
    len = __hcpe3_load_cache("test1.cache");
    EXPECT_EQ(3, len);

    // cache_re_eval
    {
        unsigned int ndindex[2] = { 0, 1 };
        float ndlogits[2][9 * 9 * MAX_MOVE_LABEL_NUM] = {
            { 2.3149f, 2.1708f, 2.7576f, 2.0594f, 2.5686f, 14.5921f, 2.4196f, 2.6541f, 2.1082f, 2.2258f, 3.1912f, 2.0370f, 3.0874f, 2.1830f, 18.4873f, 2.0236f, 1.7270f, 2.4317f, 2.8381f, 2.4372f, 1.7415f, 1.7039f, 2.3347f, 9.5120f, 2.6061f, 12.1772f, 2.0853f, 2.7911f, 2.4047f, 2.7865f, 2.5838f, 2.9813f, 5.8800f, 2.1315f, 4.6005f, 1.9467f, 3.6836f, 2.4413f, 2.8944f, 2.8033f, 1.7526f, 6.5291f, 2.2490f, 4.2536f, 2.0074f, 2.4073f, 2.6662f, 1.9774f, 2.4424f, 2.4030f, 11.6023f, 1.6402f, 5.3750f, 2.0215f, 3.2166f, 1.9301f, 2.5359f, 2.7069f, 1.8531f, 18.0288f, 2.2421f, 11.5020f, 1.8854f, 1.8093f, 2.6572f, 1.3468f, 1.9366f, 1.4704f, 8.5868f, 2.3736f, 1.9748f, 2.2524f, 1.9160f, 1.9430f, 1.5537f, 2.0791f, 2.8906f, 13.5199f, 2.3970f, 2.9394f, 2.2125f, 1.8093f, 1.9273f, 1.7792f, 1.5569f, 1.4853f, 3.8751f, 1.6263f, 1.5152f, 1.4395f, 1.4492f, 2.2090f, 1.4523f, 1.9127f, 1.7324f, 5.1959f, 1.5063f, 1.5564f, 1.4621f, 2.5528f, 2.5733f, 1.6901f, 1.3260f, 1.6386f, 2.7056f, 1.3545f, 3.2536f, 1.5387f, 2.0957f, 2.1734f, 3.5098f, 2.0593f, 1.8540f, 2.1091f, 1.0546f, 12.1970f, 1.3578f, 2.2154f, 1.8977f, 2.2455f, 3.1499f, 1.2727f, 1.7262f, 1.0097f, 8.1115f, 1.4996f, 1.6731f, 2.2427f, 1.7243f, 2.1621f, 1.8759f, 2.5555f, 1.1008f, 7.3835f, 1.3755f, 1.9078f, 1.5491f, 2.0819f, 1.8509f, 1.5764f, 5.2252f, 1.1581f, 16.0878f, 1.5394f, 0.9806f, 1.9414f, 1.7168f, 2.4597f, 1.4290f, 1.8123f, 2.1107f, 2.6156f, 1.4309f, 1.0985f, 2.1021f, 1.8546f, 1.9117f, 1.4453f, 3.5270f, 1.9612f, 1.5990f, 1.5622f, 1.9025f, 2.4371f, 2.1509f, 1.7685f, 1.5899f, 3.8413f, 1.6169f, 1.8070f, 1.3908f, 2.0044f, 2.5343f, 2.2570f, 2.0095f, 1.4606f, 4.8578f, 1.1723f, 2.1714f, 1.2746f, 2.7396f, 3.0414f, 2.1418f, 1.7659f, 1.7090f, 2.4074f, 1.4945f, 2.9499f, 1.5146f, 2.3456f, 3.2027f, 4.5974f, 2.4853f, 1.6648f, 2.1902f, 1.1118f, 4.6962f, 1.4651f, 2.6334f, 2.7689f, 2.5730f, 3.4984f, 1.1069f, 1.9603f, 1.2009f, 3.6532f, 1.5027f, 1.6665f, 3.1663f, 2.3545f, 2.3789f, 2.0749f, 3.3631f, 1.1016f, 11.8295f, 1.5358f, 2.6163f, 1.2855f, 2.7904f, 1.9103f, 1.3123f, 4.5578f, 2.9346f, 5.4517f, 1.5221f, 1.2303f, 2.1126f, 1.1384f, 2.4327f, 1.1178f, 2.4515f, 1.2409f, 1.1116f, 1.2262f, 1.2787f, 1.7672f, 1.6081f, 1.6705f, 1.4521f, 3.5305f, 1.2552f, 1.4021f, 1.3793f, 1.5318f, 1.7040f, 1.9593f, 1.4082f, 1.4151f, 3.2872f, 1.5026f, 1.2415f, 1.3001f, 1.7154f, 2.4810f, 1.8592f, 1.4913f, 1.5563f, 4.7845f, 1.4416f, 2.1954f, 1.4152f, 2.8374f, 2.5628f, 1.9953f, 1.3709f, 1.5984f, 2.1942f, 1.4538f, 5.7846f, 1.3017f, 2.3944f, 1.9774f, 3.1910f, 2.5280f, 1.7287f, 2.1783f, 1.2885f, 4.7452f, 1.2365f, 2.4429f, 2.4470f, 2.5233f, 2.8147f, 1.4515f, 2.1450f, 0.9572f, 6.3558f, 1.8789f, 2.0845f, 3.0921f, 1.9199f, 2.1935f, 1.3097f, 2.6523f, 1.2192f, 9.1079f, 2.2330f, 2.5168f, 1.9017f, 2.4339f, 1.8159f, 1.9241f, 4.2191f, 1.0666f, 8.0077f, 1.6545f, 1.4831f, 2.3167f, 1.3414f, 1.8238f, 0.8305f, 1.9742f, 1.1338f, 1.3751f, 1.1155f, 1.7146f, 1.8849f, 2.6153f, 1.8916f, 1.4070f, 3.6261f, 1.5910f, 1.6864f, 1.5100f, 1.5866f, 2.2477f, 2.3937f, 1.7125f, 1.1218f, 3.3299f, 1.3635f, 2.5885f, 1.3334f, 2.0535f, 2.5786f, 1.7530f, 1.6230f, 1.1556f, 4.4656f, 1.2503f, 1.9846f, 1.3280f, 3.0519f, 2.2902f, 2.3543f, 1.4634f, 1.1464f, 2.3515f, 1.3274f, 2.8114f, 1.5803f, 2.4679f, 2.7692f, 3.6859f, 2.1843f, 1.4243f, 1.4481f, 1.1277f, 3.4700f, 1.6536f, 2.4558f, 2.5762f, 3.3342f, 3.3931f, 1.5511f, 1.5682f, 1.0034f, 3.0391f, 1.4534f, 1.9205f, 2.1434f, 1.9950f, 2.1999f, 1.1856f, 2.8002f, 1.1055f, 4.1424f, 1.0279f, 2.2879f, 1.4013f, 2.3409f, 2.3396f, 1.8455f, 4.5375f, 1.2825f, 3.9152f, 1.3012f, 1.3649f, 1.9516f, 1.6151f, 2.0231f, 1.0310f, 1.5256f, 1.0056f, 1.1959f, 1.4153f, 1.3284f, 1.7350f, 1.9074f, 1.6737f, 1.3934f, 3.0463f, 1.0988f, 1.5526f, 1.4198f, 1.5400f, 1.9257f, 3.0468f, 1.7869f, 1.1739f, 3.1450f, 1.3207f, 1.2672f, 1.2982f, 1.9131f, 2.1294f, 2.4270f, 1.6900f, 1.5387f, 4.1409f, 1.5255f, 1.5290f, 1.5768f, 2.5590f, 2.1625f, 2.2144f, 1.5270f, 2.0838f, 2.1987f, 1.2150f, 3.2914f, 1.3647f, 2.1259f, 2.8307f, 4.2814f, 2.1982f, 1.2593f, 1.9168f, 0.9268f, 3.5393f, 1.1216f, 2.4458f, 2.2211f, 3.3885f, 3.0245f, 1.0712f, 1.8938f, 1.0482f, 3.2022f, 1.1090f, 2.0652f, 2.2914f, 2.4949f, 2.3316f, 1.0109f, 2.7827f, 0.9599f, 3.3825f, 1.3697f, 2.5052f, 1.4169f, 2.5884f, 2.1920f, 1.5339f, 4.1616f, 0.9561f, 4.8167f, 1.4011f, 1.5590f, 1.9937f, 1.6611f, 3.0900f, 0.9292f, 1.6536f, 1.1130f, 1.1108f, 1.5484f, 1.2465f, 1.7169f, 2.5236f, 1.9611f, 1.2117f, 3.1026f, 1.3262f, 1.4127f, 1.3914f, 1.7180f, 1.6306f, 1.5659f, 1.2386f, 0.9526f, 2.6924f, 1.1988f, 1.0763f, 1.0990f, 1.6501f, 2.3226f, 1.6167f, 1.5147f, 1.1566f, 3.5889f, 1.0498f, 0.9895f, 1.1379f, 2.3572f, 2.3386f, 1.7476f, 1.2733f, 1.3439f, 1.8200f, 1.0426f, 2.5632f, 1.0804f, 1.8824f, 2.2761f, 3.7603f, 1.3791f, 0.9326f, 1.2793f, 0.8520f, 3.9975f, 0.8756f, 2.0774f, 2.0186f, 2.7159f, 3.7025f, 0.9247f, 1.2419f, 0.7865f, 3.1521f, 0.9670f, 1.6760f, 2.2994f, 2.0608f, 1.9337f, 1.0978f, 2.2700f, 0.8912f, 3.4260f, 1.2603f, 2.0781f, 1.6534f, 2.6405f, 1.9960f, 1.5280f, 3.4128f, 0.9365f, 4.2981f, 1.0825f, 1.3638f, 2.2329f, 1.7966f, 2.4750f, 0.9849f, 1.5324f, 0.9031f, 0.7555f, 1.1775f, 1.1810f, 2.1429f, 1.8943f, 2.0244f, 1.1173f, 2.7814f, 1.0714f, 1.0680f, 1.5421f, 1.4468f, 2.0635f, 1.7679f, 1.4982f, 0.9033f, 2.6468f, 1.3391f, 1.1919f, 1.2715f, 1.5883f, 2.8263f, 1.6096f, 1.4142f, 1.0072f, 3.6833f, 1.1809f, 1.3998f, 1.2044f, 2.5441f, 2.3911f, 1.9635f, 1.4458f, 1.2685f, 1.4926f, 1.1767f, 2.8129f, 1.1891f, 1.8342f, 2.9729f, 3.2815f, 1.9164f, 1.2138f, 1.4137f, 0.9746f, 3.6372f, 1.1017f, 2.1185f, 2.4230f, 2.5336f, 3.4092f, 1.2027f, 1.4784f, 1.1540f, 2.7072f, 1.1085f, 1.7095f, 2.7351f, 1.9388f, 2.7850f, 1.2517f, 2.2886f, 0.8827f, 3.8869f, 0.9752f, 2.5411f, 1.5929f, 1.7462f, 2.6383f, 1.3152f, 3.3388f, 1.1496f, 4.4694f, 1.4913f, 1.2024f, 1.9077f, 2.5101f, 2.8668f, 1.1701f, 2.4452f, 1.2572f, 0.9288f, 1.2826f, 1.0350f, 1.7764f, 1.6191f, 1.7113f, 1.3868f, 2.5212f, 1.1119f, 1.2255f, 1.3373f, 1.2820f, 1.2682f, 1.6816f, 1.1915f, 0.8010f, 2.4375f, 1.2697f, 0.9991f, 0.9091f, 1.1234f, 2.1304f, 2.1129f, 1.1139f, 0.9949f, 3.5941f, 1.0516f, 1.1139f, 0.8105f, 2.3512f, 1.3435f, 1.7293f, 1.1737f, 1.2978f, 1.7607f, 1.1106f, 2.2557f, 0.9036f, 1.8121f, 1.8450f, 3.3618f, 1.4934f, 1.1773f, 1.4044f, 0.7728f, 2.7778f, 0.7486f, 2.0049f, 1.4858f, 2.5050f, 1.8721f, 0.9742f, 1.2651f, 0.7909f, 2.3826f, 0.8442f, 1.7676f, 1.8758f, 2.0293f, 1.5151f, 1.0780f, 2.2640f, 0.7657f, 3.1808f, 0.7920f, 2.1190f, 1.3091f, 2.9790f, 1.9165f, 0.9885f, 3.2567f, 0.9083f, 3.4447f, 0.9821f, 1.5045f, 1.9537f, 1.7159f, 1.6740f, 0.9242f, 1.5723f, 1.0634f, 0.8333f, 0.8389f, 0.7774f, 1.4227f, 1.9439f, 1.8365f, 1.1228f, 2.3582f, 1.0764f, 0.9561f, 0.8412f, 1.7888f, 1.0568f, 2.0959f, 1.3285f, 1.2220f, 2.5738f, 0.9528f, 0.9564f, 0.9200f, 1.6513f, 2.2923f, 2.5365f, 1.3848f, 0.8085f, 3.4963f, 0.8834f, 1.1604f, 0.9419f, 2.1606f, 1.4638f, 2.2157f, 1.2523f, 1.0392f, 2.0071f, 0.7832f, 2.3257f, 1.1787f, 1.7108f, 1.9600f, 3.8453f, 1.3972f, 0.9539f, 1.3216f, 0.7339f, 2.9479f, 1.0829f, 1.8904f, 1.9846f, 3.1501f, 2.7497f, 1.0491f, 1.2860f, 0.7686f, 2.0913f, 1.2851f, 1.7232f, 2.0488f, 2.4391f, 1.9777f, 1.0845f, 2.1506f, 0.7515f, 2.8718f, 1.2959f, 2.0579f, 0.8224f, 3.0417f, 2.2483f, 1.1760f, 3.3476f, 0.6066f, 3.9188f, 1.1303f, 1.3261f, 1.9411f, 2.2603f, 2.1959f, 0.8121f, 1.4287f, 0.9183f, 0.8295f, 0.9354f, 0.5679f, 1.5354f, 1.8736f, 1.7570f, 1.0061f, 2.2511f, 0.9958f, 0.9301f, 0.9603f, 1.5717f, 2.3317f, 2.4098f, 1.1883f, 1.0617f, 3.4296f, 1.2487f, 1.2153f, 1.1623f, 1.1774f, 1.7320f, 1.6780f, 1.3816f, 0.5314f, 4.4711f, 0.8444f, 0.8998f, 1.3934f, 1.7929f, 1.7858f, 1.6126f, 0.6376f, 1.0553f, 2.5115f, 1.1246f, 3.3245f, 1.2061f, 1.2317f, 2.4012f, 1.6490f, 0.9598f, 1.0820f, 1.7767f, 0.9466f, 2.3333f, 1.0829f, 1.9623f, 1.8110f, 1.8670f, 1.7272f, 0.7326f, 2.0123f, 1.1142f, 1.9945f, 0.8532f, 1.0345f, 1.8997f, 1.1567f, 0.9012f, 0.9697f, 2.7569f, 0.7890f, 2.9718f, 1.1117f, 1.6770f, 1.5068f, 1.9227f, 1.6541f, 0.8694f, 4.2591f, 0.7901f, 4.1561f, 1.1096f, 0.9083f, 1.7317f, 1.9513f, 1.4184f, 0.9047f, 2.2911f, 1.0581f, 0.9689f, 1.2740f, 1.0466f, 1.6825f, 1.7290f, 1.3808f, 1.4192f, 3.3296f, 1.2522f, 1.4067f, 1.2449f, 1.2545f, 1.7741f, 1.5386f, 0.7613f, 0.5271f, 1.1955f, 0.5501f, 0.4673f, 0.6070f, 0.7138f, 1.5846f, 1.5494f, 0.6549f, 0.5925f, 1.9285f, 0.5081f, 0.5306f, 0.5142f, 1.4578f, 1.7558f, 1.4013f, 0.3742f, 0.5745f, 0.9594f, 0.4030f, 1.2609f, 0.6448f, 0.9558f, 1.9263f, 2.0682f, 0.6491f, 0.6811f, 0.7639f, 0.4333f, 3.3001f, 0.4898f, 0.9383f, 1.5473f, 1.4833f, 1.7931f, 0.3858f, 0.6626f, 0.3550f, 2.5824f, 0.5160f, 0.7961f, 1.5261f, 1.0728f, 0.9045f, 0.5862f, 0.8919f, 0.4127f, 1.5248f, 0.5082f, 0.7521f, 1.0205f, 1.2912f, 1.2042f, 0.5383f, 1.7282f, 0.1868f, 3.9702f, 0.6221f, 0.5999f, 1.4057f, 2.1066f, 1.7944f, 0.5266f, 0.5449f, 0.8296f, 0.8364f, 0.5661f, 0.6539f, 1.5840f, 1.6002f, 1.0701f, 0.7439f, 1.2894f, 0.7718f, 0.5342f, 0.7033f, 1.3182f, 1.9071f, 1.7632f, 0.7992f, 0.4839f, 0.9476f, 0.6208f, 0.7097f, 0.6342f, 0.9952f, 1.6708f, 2.2124f, 0.7692f, 0.5720f, 1.5820f, 0.4981f, 0.6472f, 0.4444f, 1.6899f, 2.0072f, 1.7807f, 0.4885f, 0.5749f, 0.5520f, 0.5830f, 1.1585f, 0.6792f, 1.2172f, 2.5513f, 2.8157f, 0.8540f, 0.6094f, 0.8796f, 0.5514f, 0.4581f, 0.5261f, 1.4493f, 1.9192f, 1.9146f, 1.8704f, 0.3832f, 0.7590f, 0.5237f, 0.7462f, 0.5132f, 0.8420f, 2.2831f, 1.5039f, 1.0193f, 0.6391f, 1.1216f, 0.4044f, 2.9858f, 0.5186f, 1.3346f, 0.9236f, 1.7733f, 1.1447f, 0.5242f, 1.3537f, 1.0385f, 0.4682f, 0.4636f, 0.6862f, 1.5778f, 1.7198f, 1.7756f, 0.3320f, 0.8217f, 0.2911f, 0.2690f, 0.4638f, 0.8003f, 1.3022f, 1.7158f, 0.9562f, 0.7638f, 1.1423f, 0.5902f, 0.5553f, 0.6726f, 0.9074f, 1.5120f, 1.2986f, 0.5779f, 0.3111f, 0.8629f, 0.3952f, 0.3104f, 0.3352f, 0.4371f, 1.1576f, 1.2545f, 0.4432f, 0.3570f, 1.4680f, 0.3567f, 0.5942f, 0.4512f, 1.2040f, 1.2368f, 1.1008f, 0.3385f, 0.4046f, 0.4361f, 0.2795f, 1.0061f, 0.4924f, 0.7679f, 1.7316f, 1.5291f, 0.5130f, 0.3640f, 0.4677f, 0.3103f, 1.5617f, 0.3726f, 0.8194f, 1.3293f, 1.3228f, 1.5077f, 0.2003f, 0.3911f, 0.1906f, 1.6005f, 0.4293f, 0.5019f, 1.3283f, 0.9077f, 0.5643f, 0.2972f, 0.7090f, 0.2831f, 1.4612f, 0.4171f, 0.6088f, 0.8461f, 1.1366f, 0.9613f, 0.3272f, 1.0867f, 0.2024f, 1.6481f, 0.5063f, 0.4691f, 1.1741f, 1.6672f, 1.2359f, 0.2625f, 0.3319f, 0.2810f, 0.4612f, 0.3598f, 0.6389f, 1.2339f, 1.3200f, 0.7935f, 0.5337f, 0.9913f, 0.5285f, 0.3778f, 0.5380f, 0.9611f, 1.4314f, 1.4138f, 0.6049f, 0.3317f, 1.0371f, 0.4198f, 0.5185f, 0.3892f, 0.6339f, 1.1785f, 1.2971f, 0.4543f, 0.3881f, 1.7040f, 0.3763f, 0.5728f, 0.4298f, 1.3626f, 1.1489f, 1.2149f, 0.3269f, 0.3014f, 0.5679f, 0.3242f, 1.1661f, 0.5685f, 0.8250f, 1.7897f, 1.6345f, 0.4532f, 0.3675f, 0.6083f, 0.3391f, 1.5887f, 0.4272f, 0.8829f, 1.4081f, 1.4347f, 1.4289f, 0.2830f, 0.4826f, 0.3335f, 1.2468f, 0.3840f, 0.6908f, 1.2381f, 0.8288f, 0.6752f, 0.2995f, 0.8770f, 0.3375f, 1.6637f, 0.3310f, 0.9470f, 0.5257f, 0.9694f, 0.9681f, 0.3316f, 1.4570f, 0.3566f, 2.2334f, 0.4884f, 0.4515f, 1.0534f, 1.6843f, 1.3229f, 0.2508f, 0.6974f, 0.3296f, 0.4874f, 0.4050f, 0.5842f, 1.1163f, 1.1216f, 0.6373f, 0.5209f, 1.0231f, 0.4794f, 0.4414f, 0.4688f, 1.0473f, 1.5819f, 1.9155f, 0.9428f, 0.4654f, 1.0733f, 0.4863f, 0.4092f, 0.4460f, 0.6757f, 1.2151f, 1.3526f, 0.5173f, 0.2300f, 1.5485f, 0.5490f, 0.5513f, 0.5344f, 1.2555f, 1.3772f, 1.3429f, 0.5230f, 0.6259f, 0.7751f, 0.4279f, 1.1757f, 0.5701f, 0.8394f, 2.0341f, 2.1423f, 0.6444f, 0.5401f, 0.6046f, 0.3826f, 1.4846f, 0.3885f, 0.9932f, 1.5065f, 1.7080f, 1.9263f, 0.3577f, 0.6033f, 0.3810f, 1.3575f, 0.4086f, 0.7985f, 1.5416f, 1.1458f, 1.0628f, 0.4685f, 1.0335f, 0.3850f, 1.7175f, 0.4982f, 0.8558f, 0.7939f, 1.2750f, 1.1690f, 0.5723f, 1.3533f, 0.3216f, 1.5886f, 0.5769f, 0.5302f, 1.1961f, 1.7089f, 1.7221f, 0.3318f, 0.6486f, 0.4093f, 0.4588f, 0.4476f, 0.5766f, 1.1985f, 1.5069f, 0.9401f, 0.6529f, 1.1909f, 0.6391f, 0.4816f, 0.5952f, 1.2735f, 1.7337f, 1.6360f, 0.7468f, 0.4040f, 1.0748f, 0.4315f, 0.4554f, 0.3961f, 0.7519f, 1.4971f, 1.7011f, 0.5816f, 0.3794f, 1.6964f, 0.4391f, 0.4036f, 0.4827f, 1.5617f, 1.7019f, 1.4901f, 0.4582f, 0.4305f, 0.7292f, 0.2859f, 0.9336f, 0.6305f, 0.9757f, 2.2423f, 2.5139f, 0.4819f, 0.3288f, 0.5471f, 0.3298f, 1.9101f, 0.4433f, 1.0360f, 1.7713f, 1.9893f, 1.9217f, 0.2758f, 0.3857f, 0.2477f, 1.4261f, 0.5023f, 1.0995f, 1.9095f, 1.4787f, 0.8681f, 0.3329f, 0.7671f, 0.3157f, 1.2119f, 0.6094f, 1.1355f, 1.2523f, 1.8805f, 1.2923f, 0.4881f, 1.5392f, 0.3585f, 2.3210f, 0.4803f, 0.7400f, 1.6512f, 2.2283f, 1.7919f, 0.3574f, 0.2648f, 0.2378f, 0.3069f, 0.4950f, 0.6772f, 1.6175f, 1.8950f, 1.2691f, 0.5673f, 1.1260f, 0.5109f, 0.3834f, 0.6634f, 1.3456f, 1.8987f, 1.6550f, 0.8905f, 0.4455f, 1.1081f, 0.5428f, 0.6653f, 0.5049f, 0.8182f, 1.9576f, 1.5459f, 0.6173f, 0.4620f, 1.6003f, 0.4942f, 0.4642f, 0.5027f, 1.5806f, 1.6591f, 1.5016f, 0.4770f, 0.4549f, 0.6181f, 0.4782f, 1.1805f, 0.5669f, 1.0010f, 2.3365f, 2.1519f, 0.7502f, 0.5828f, 0.7816f, 0.5449f, 1.6102f, 0.4457f, 1.1223f, 1.7779f, 1.6590f, 1.9624f, 0.5551f, 0.6108f, 0.5476f, 1.0709f, 0.4352f, 0.9530f, 1.8722f, 1.2522f, 1.3266f, 0.5048f, 0.9769f, 0.3874f, 1.1439f, 0.3629f, 1.2692f, 1.1501f, 1.1705f, 1.6492f, 0.5100f, 1.4830f, 0.5109f, 1.9670f, 0.4364f, 0.6852f, 1.4942f, 2.4639f, 1.9160f, 0.4606f, 0.8007f, 0.4101f, 0.3625f, 0.5128f, 0.7765f, 1.5194f, 1.6091f, 1.0554f, 0.7876f, 1.1620f, 0.5958f, 0.6316f, 0.6606f, 0.9646f, 1.6702f, 1.5885f, 0.7322f, 0.3678f, 0.8079f, 0.7042f, 0.5425f, 0.5101f, 0.5656f, 1.4254f, 1.7765f, 0.5731f, 0.6610f, 1.4380f, 0.6043f, 0.5411f, 0.4734f, 1.7643f, 1.3251f, 1.5438f, 0.4139f, 0.5607f, 0.5394f, 0.5060f, 0.8104f, 0.5504f, 0.9740f, 2.2750f, 2.2517f, 0.7952f, 0.5919f, 0.6355f, 0.4344f, 1.1105f, 0.3889f, 1.2582f, 1.5598f, 1.8158f, 1.5535f, 0.2943f, 0.5521f, 0.3474f, 1.3703f, 0.2997f, 0.9505f, 1.7000f, 1.2079f, 0.7924f, 0.5304f, 0.8653f, 0.3959f, 1.6668f, 0.3580f, 1.1745f, 1.2131f, 1.9548f, 1.2420f, 0.4095f, 1.0888f, 0.4881f, 1.6158f, 0.5606f, 0.8914f, 1.4345f, 2.0067f, 1.4034f, 0.3575f, 0.5004f, 0.4126f, 0.4691f, 0.4965f, 0.7035f, 1.2268f, 1.9135f, 1.2013f, 0.7451f, 0.8953f, 0.5989f, 0.5064f, 0.5365f, 1.3603f, 1.4590f, 1.8940f, 0.8168f, 0.6860f, 0.9184f, 0.4297f, 0.5023f, 0.5065f, 0.9841f, 1.5350f, 2.0788f, 0.7684f, 0.5020f, 1.3667f, 0.4509f, 0.5519f, 0.5592f, 1.5872f, 1.3831f, 1.9108f, 0.4398f, 0.3244f, 0.7218f, 0.2733f, 0.8459f, 0.7533f, 0.8627f, 2.3466f, 2.5581f, 0.6963f, 0.4115f, 0.5582f, 0.3999f, 1.1509f, 0.6437f, 1.1491f, 1.9275f, 2.2812f, 2.2240f, 0.3342f, 0.5642f, 0.3311f, 1.0480f, 0.6229f, 0.8801f, 1.7994f, 1.4860f, 1.1431f, 0.5027f, 0.7794f, 0.3822f, 1.3666f, 0.7309f, 1.1100f, 0.7696f, 1.9339f, 1.4782f, 0.5408f, 1.1734f, 0.2089f, 1.8685f, 0.6500f, 0.7176f, 1.3807f, 2.4047f, 1.7827f, 0.2509f, 0.3823f, 0.2756f, 0.4288f, 0.5577f, 0.5245f, 1.2583f, 1.7963f, 1.0819f, 0.6389f, 0.8086f, 0.5249f, 0.4721f, 0.6153f, 3.9592f, 3.0935f, 3.6231f, 3.0040f, 2.7794f, 4.1443f, 2.4605f, 2.7396f, 2.6827f, 3.8077f, 2.8604f, 3.3281f, 3.3884f, 3.7392f, 5.7049f, 2.2561f, 2.1929f, 2.7613f, 4.0116f, 3.6781f, 2.4395f, 3.9305f, 2.4271f, 3.2835f, 2.9307f, 3.9775f, 2.7230f, 3.5422f, 4.2086f, 2.8089f, 3.4265f, 2.6282f, 2.9631f, 2.9818f, 3.6631f, 2.6046f, 4.0819f, 3.4955f, 3.6880f, 3.4505f, 2.4938f, 2.8394f, 2.2991f, 2.8745f, 2.5333f, 3.4270f, 4.3272f, 3.5640f, 3.1153f, 2.0791f, 3.9466f, 2.1566f, 3.4233f, 2.4372f, 4.2249f, 3.8019f, 2.8990f, 3.5058f, 2.4028f, 4.8181f, 2.4709f, 4.7854f, 2.8825f, 3.6842f, 4.4205f, 4.1356f, 3.5896f, 2.4746f, 3.2521f, 3.0079f, 2.4320f, 2.8211f, 2.7790f, 3.6850f, 3.1274f, 3.2359f, 2.7029f, 3.6095f, 2.3825f, 2.4225f, 2.6552f, 1.8945f, 2.5071f, 2.6053f, 1.7515f, 1.5503f, 2.9842f, 1.4014f, 1.4189f, 1.4194f, 2.2605f, 2.9431f, 3.1731f, 2.2814f, 1.7148f, 4.3571f, 2.2627f, 1.9846f, 2.2729f, 2.4499f, 2.5688f, 1.5501f, 1.7571f, 1.8906f, 2.7299f, 1.6261f, 2.8215f, 1.7008f, 2.5612f, 3.0594f, 3.3992f, 1.9087f, 1.3640f, 1.9205f, 1.4065f, 2.7940f, 1.7426f, 2.8710f, 2.3752f, 2.5180f, 1.9071f, 1.5787f, 1.9806f, 1.1961f, 2.4027f, 1.7694f, 1.9854f, 3.4843f, 3.2259f, 2.4037f, 2.1965f, 2.7131f, 1.4383f, 2.4531f, 1.6869f, 2.6349f, 2.8881f, 3.0805f, 2.2308f, 1.3987f, 3.7084f, 1.4995f, 4.1708f, 1.7380f, 2.1200f, 3.8062f, 3.4856f, 2.4210f, 1.2803f, 1.9617f, 1.2410f, 1.2050f, 2.0630f, 1.7392f, 2.6572f, 2.7973f, 2.3506f, 1.6686f, 3.0777f, 1.6857f, 1.9317f, 1.7986f, 1.6259f, 4.7940f, 2.5449f, 2.1933f, 2.0174f, 3.2057f, 1.6037f, 1.9548f, 1.8580f, 2.0317f, 2.3423f, 3.9759f, 2.5223f, 1.6569f, 4.4510f, 1.4797f, 1.7419f, 1.5788f, 2.3147f, 4.0949f, 2.2603f, 2.9921f, 2.9638f, 3.0074f, 1.4596f, 2.7337f, 1.2490f, 2.6500f, 4.6627f, 4.8617f, 1.9731f, 1.8009f, 2.3542f, 1.5418f, 3.0388f, 1.3583f, 2.8774f, 4.2489f, 4.1012f, 2.6525f, 2.4261f, 2.0495f, 1.6440f, 2.4705f, 1.5075f, 2.2553f, 4.3796f, 4.8176f, 2.0678f, 1.6911f, 2.7993f, 1.4590f, 3.0039f, 1.3402f, 2.5966f, 4.2515f, 4.5445f, 3.3698f, 1.5840f, 3.8704f, 1.1177f, 4.0013f, 1.4579f, 2.9555f, 4.0161f, 4.5829f, 3.0993f, 2.9504f, 2.1517f, 1.7611f, 1.1024f, 1.6405f, 1.1758f, 5.7944f, 3.4045f, 4.0618f, 1.7257f, 3.2303f, 1.7155f, 2.0547f, 2.0173f, 2.2502f, 3.6226f, 3.2800f, 1.7596f, 1.9427f, 3.3539f, 1.7456f, 1.7000f, 1.5683f, 1.9899f, 1.0464f, 2.4108f, 1.5863f, 1.9796f, 4.5068f, 1.3096f, 1.6383f, 1.8209f, 2.1554f, 3.3501f, 1.7144f, 2.0864f, 1.5483f, 2.6364f, 1.5706f, 2.8296f, 1.2788f, 1.6379f, 1.2661f, 1.7140f, 1.1118f, 1.5652f, 2.0204f, 1.7189f, 2.9710f, 1.5097f, 1.8955f, 1.1522f, 1.6153f, 1.4204f, 2.4539f, 1.5249f, 1.6437f, 2.0340f, 1.6231f, 1.3270f, 4.0829f, 2.3656f, 1.2137f, 2.2018f, 2.5817f, 1.6287f, 3.1046f, 1.7712f, 2.9837f, 3.8307f, 2.4169f, 2.3330f, 2.3384f, 4.0603f, 1.1418f, 4.3150f, 1.8078f, 2.8448f, 3.7591f, 3.1697f, 3.5043f, 2.3491f, 2.0643f, 1.6152f, 1.0949f, 2.2329f, 2.2626f, 3.4532f, 2.8668f, 3.7976f, 2.1418f, 3.1218f, 1.5022f, 2.0610f, 1.8819f, 2.2819f, 2.7877f, 2.6537f, 1.9286f, 1.7493f, 3.0861f, 1.4118f, 1.5718f, 1.3441f, 1.3692f, 1.6039f, 1.8676f, 1.8739f, 1.1907f, 4.0239f, 1.1464f, 1.6692f, 1.4195f, 1.9424f, 1.8406f, 1.9293f, 1.8908f, 1.9101f, 2.2639f, 1.3403f, 2.5815f, 1.1339f, 1.4243f, 1.5319f, 1.3924f, 0.9761f, 1.3646f, 1.6488f, 1.6294f, 2.9919f, 1.2175f, 1.8088f, 2.3913f, 1.1385f, 1.1455f, 1.7264f, 1.5647f, 1.3756f, 2.5301f, 1.3166f, 1.3066f, 2.5998f, 2.4941f, 2.0513f, 1.6320f, 2.2763f, 1.3623f, 3.1236f, 1.3649f, 1.8944f, 3.3690f, 2.9555f, 2.0851f, 1.4470f, 3.5767f, 1.5183f, 4.5434f, 1.1681f, 2.0619f, 3.2928f, 2.9086f, 2.6983f, 2.8558f, 2.1937f, 1.5989f, 1.4824f, 1.9265f, 1.7031f, 2.2365f, 2.2208f, 2.9095f, 1.6358f, 2.7518f, 1.7186f, 1.5307f, 1.5602f, 1.8707f, 1.6390f, 2.5749f, 1.1261f, 1.1002f, 2.2826f, 1.2196f, 1.2876f, 1.3477f, 1.6917f, 2.1828f, 2.5938f, 1.4000f, 1.5194f, 3.7587f, 1.2031f, 1.2917f, 1.4045f, 1.9630f, 2.2258f, 2.2470f, 1.2131f, 1.3559f, 1.7750f, 1.3969f, 2.3644f, 1.0556f, 1.6244f, 1.9292f, 3.2259f, 1.8874f, 1.4186f, 1.6897f, 1.3196f, 2.7848f, 1.2464f, 2.1509f, 1.4015f, 1.6455f, 2.0927f, 1.4720f, 1.4830f, 1.3270f, 2.2455f, 1.0547f, 1.4610f, 2.1563f, 1.8824f, 2.2640f, 1.7385f, 2.3447f, 1.0340f, 2.8428f, 1.2415f, 2.0555f, 1.9795f, 3.0553f, 2.4413f, 1.5257f, 3.1071f, 0.8589f, 3.6570f, 1.0605f, 1.8126f, 2.4791f, 2.6484f, 2.6551f, 1.6846f, 1.2997f, 1.0616f, 0.8276f, 1.5133f, 1.4948f, 1.8271f, 2.7620f, 2.0034f, 1.3043f, 2.3670f, 1.3608f, 1.4452f, 1.3477f, 2.9876f, 5.2001f, 3.0209f, 1.4931f, 1.7877f, 3.2815f, 1.3110f, 1.7189f, 1.4406f, 2.3077f, 2.8842f, 1.9099f, 1.7554f, 1.3655f, 4.1205f, 1.1597f, 1.2533f, 1.5979f, 2.4836f, 2.4815f, 1.7415f, 1.8598f, 1.7031f, 2.5159f, 1.5169f, 2.9038f, 1.4124f, 1.8179f, 4.5032f, 2.0890f, 1.7986f, 1.1581f, 1.3529f, 1.7833f, 3.2832f, 1.6932f, 2.1575f, 3.9258f, 2.4403f, 1.4355f, 1.5682f, 1.5971f, 1.5812f, 2.2545f, 1.6398f, 1.7216f, 4.3218f, 3.4201f, 1.9335f, 1.3142f, 2.8931f, 1.3206f, 3.0075f, 1.4949f, 2.0469f, 4.2646f, 3.2766f, 2.3671f, 1.2356f, 3.8988f, 1.6652f, 4.7953f, 1.4200f, 1.8200f, 3.8783f, 3.4929f, 3.4849f, 1.7288f, 1.2560f, 1.2826f, 1.0396f, 2.0557f, 2.0697f, 4.8956f, 2.7703f, 2.7120f, 1.3683f, 3.0067f, 1.6143f, 2.1877f, 1.8287f },
            { 2.7642f, 2.0740f, 2.4566f, 2.0529f, 2.6215f, 13.0104f, 2.4415f, 2.6344f, 2.3976f, 2.2849f, 3.1677f, 2.1365f, 3.0956f, 2.8992f, 17.9921f, 2.3092f, 2.1869f, 2.5322f, 3.5648f, 2.1957f, 1.5626f, 1.5520f, 2.5349f, 6.3654f, 2.6318f, 9.3402f, 2.1996f, 2.8703f, 2.0863f, 2.4867f, 2.4508f, 2.9143f, 5.1544f, 2.2862f, 3.7174f, 2.0795f, 3.1193f, 2.3766f, 2.5595f, 2.3305f, 2.0323f, 5.2162f, 2.6120f, 3.5934f, 2.1143f, 2.3939f, 2.0508f, 2.0417f, 2.4347f, 2.1350f, 7.8581f, 2.3770f, 4.5028f, 2.0769f, 3.3130f, 1.6460f, 2.6908f, 2.0138f, 2.5173f, 18.7274f, 2.8866f, 10.3212f, 1.8605f, 2.1483f, 2.6744f, 2.2997f, 2.3617f, 1.9384f, 9.0874f, 1.5507f, 2.7475f, 2.3233f, 2.3595f, 1.9575f, 2.1195f, 2.2234f, 4.2043f, 12.5794f, 2.6568f, 2.9932f, 2.5301f, 2.0805f, 1.7227f, 1.9175f, 1.5183f, 1.3664f, 3.3442f, 1.5666f, 1.5358f, 1.5816f, 1.7708f, 2.3113f, 1.5506f, 1.9251f, 2.0649f, 4.6294f, 1.5006f, 1.8275f, 1.4372f, 2.8140f, 2.3855f, 1.7671f, 1.0399f, 1.5984f, 2.2320f, 1.3178f, 2.4482f, 1.4898f, 2.2573f, 1.7713f, 2.7531f, 2.0071f, 2.0299f, 1.9988f, 1.3415f, 7.9828f, 1.4218f, 1.9401f, 1.5699f, 1.8814f, 2.6815f, 1.5226f, 1.6525f, 1.2030f, 5.5898f, 1.4209f, 1.6554f, 1.8416f, 1.7312f, 2.2948f, 1.6753f, 2.1438f, 1.3749f, 5.0338f, 1.2845f, 1.7894f, 1.5049f, 1.8861f, 2.3012f, 1.4922f, 5.6546f, 1.5882f, 15.6829f, 1.3062f, 1.3915f, 2.1623f, 2.6838f, 2.2556f, 2.2120f, 2.4296f, 2.1222f, 2.7025f, 1.4646f, 1.1302f, 2.2727f, 1.6628f, 2.3304f, 2.1406f, 2.7655f, 2.1534f, 1.9517f, 1.7427f, 2.1194f, 2.0755f, 2.3163f, 1.5298f, 1.4966f, 3.6804f, 1.5463f, 1.6944f, 1.5074f, 2.1302f, 2.2020f, 2.1483f, 1.7861f, 1.6967f, 4.8176f, 1.1820f, 2.3516f, 1.2525f, 2.9979f, 3.0569f, 1.8135f, 1.2132f, 1.4573f, 1.9009f, 1.4836f, 2.3154f, 1.3919f, 2.2006f, 2.5445f, 4.0798f, 2.2020f, 1.6394f, 2.0678f, 1.2967f, 3.5996f, 1.4451f, 2.3473f, 1.9187f, 2.4782f, 3.1852f, 1.4026f, 1.9595f, 1.2341f, 2.7940f, 1.4179f, 1.8460f, 2.2156f, 2.1002f, 2.3265f, 1.9464f, 2.5351f, 1.4474f, 9.9536f, 1.3967f, 2.6009f, 1.5468f, 1.8568f, 2.4570f, 1.2497f, 4.8506f, 3.6930f, 5.6163f, 1.1938f, 1.8070f, 2.0910f, 2.1200f, 2.6424f, 1.6815f, 2.6315f, 1.3875f, 1.1561f, 1.2761f, 1.4108f, 1.7913f, 1.6426f, 1.8418f, 2.1378f, 2.9280f, 1.3017f, 1.6279f, 1.3409f, 1.9309f, 1.6390f, 2.0317f, 1.4770f, 1.3530f, 2.9002f, 1.5278f, 1.2468f, 1.4449f, 1.8759f, 2.6219f, 2.0839f, 1.6563f, 1.8001f, 4.4404f, 1.3890f, 2.3012f, 1.3982f, 3.0471f, 2.4864f, 2.1545f, 1.2380f, 1.5525f, 1.4496f, 1.5275f, 4.1181f, 1.2440f, 2.3715f, 1.7785f, 2.6512f, 2.2517f, 1.7992f, 2.0319f, 1.5332f, 3.5995f, 1.2594f, 2.2984f, 2.1884f, 2.2578f, 2.5842f, 1.5144f, 1.9455f, 1.1634f, 5.1331f, 1.8550f, 2.2777f, 2.5938f, 1.9902f, 2.2538f, 1.4910f, 2.1613f, 1.3314f, 7.4574f, 1.8620f, 2.5404f, 2.0102f, 2.6106f, 1.5956f, 1.5179f, 4.4865f, 1.5037f, 6.7278f, 1.2730f, 2.1402f, 2.4647f, 2.0666f, 2.2743f, 0.9744f, 2.1905f, 1.0559f, 1.6142f, 1.3420f, 1.9731f, 2.1167f, 2.5255f, 2.0454f, 2.0459f, 2.5486f, 1.5867f, 1.6983f, 1.6653f, 1.7386f, 2.0368f, 2.8845f, 1.5076f, 1.3342f, 2.9312f, 1.4091f, 2.4130f, 1.3683f, 2.0832f, 2.4940f, 1.8070f, 1.5835f, 1.4844f, 4.2764f, 1.1172f, 2.1570f, 1.2945f, 3.0929f, 2.7659f, 2.2219f, 1.4213f, 1.0688f, 1.5786f, 1.2507f, 2.3603f, 1.4372f, 2.2780f, 2.8914f, 3.5999f, 2.2295f, 1.2995f, 1.2738f, 1.2770f, 2.5279f, 1.5949f, 2.3574f, 2.4053f, 3.1418f, 3.1139f, 1.5464f, 1.3535f, 1.0728f, 2.4320f, 1.3000f, 1.9427f, 1.7950f, 2.2453f, 2.8879f, 1.3920f, 2.0590f, 1.3350f, 3.6099f, 1.0746f, 2.4402f, 1.4945f, 3.0692f, 2.4924f, 1.2918f, 4.5492f, 1.7021f, 3.9730f, 1.0842f, 1.9592f, 1.7072f, 2.3829f, 2.3062f, 1.2140f, 2.1497f, 0.8963f, 1.4597f, 1.3938f, 1.3627f, 1.6116f, 2.0091f, 1.7638f, 1.9855f, 2.5801f, 1.0328f, 1.4439f, 1.4602f, 1.7864f, 1.8485f, 2.8842f, 1.8794f, 1.0959f, 2.8261f, 1.3358f, 1.2486f, 1.3894f, 1.8892f, 2.1710f, 2.6729f, 1.9947f, 1.7271f, 3.9407f, 1.4209f, 1.6353f, 1.5288f, 2.6247f, 2.1244f, 1.8874f, 1.2262f, 1.9662f, 1.5034f, 1.1307f, 2.6305f, 1.3007f, 1.8950f, 2.2449f, 4.0371f, 1.9794f, 1.4986f, 1.7652f, 0.9885f, 2.6469f, 1.1695f, 2.0964f, 2.1062f, 3.4742f, 3.3503f, 1.2263f, 1.6618f, 1.0795f, 2.5515f, 1.1564f, 2.0607f, 1.6988f, 2.5798f, 2.6378f, 1.1795f, 2.0176f, 1.2114f, 2.8133f, 1.3204f, 2.3283f, 1.3007f, 2.8249f, 2.3706f, 1.7431f, 4.0737f, 1.2785f, 4.8173f, 1.2621f, 2.1185f, 2.0757f, 2.4227f, 3.2186f, 1.4266f, 2.0640f, 0.9705f, 1.5046f, 1.5743f, 1.3341f, 1.7713f, 2.6114f, 2.4594f, 1.8927f, 2.6712f, 1.1969f, 1.4245f, 1.5293f, 2.0040f, 1.5719f, 1.5761f, 1.1958f, 0.8986f, 2.2676f, 1.2718f, 1.1574f, 1.2105f, 1.7156f, 2.2182f, 1.7660f, 1.4811f, 1.3342f, 3.2325f, 0.9395f, 1.0834f, 1.1588f, 2.4928f, 2.1511f, 1.7163f, 0.9843f, 1.2989f, 1.0679f, 1.1075f, 2.0432f, 0.9789f, 1.8808f, 1.7632f, 3.2430f, 1.3593f, 1.0226f, 1.0962f, 1.1254f, 3.0512f, 0.9128f, 1.7755f, 1.6438f, 2.1663f, 3.6625f, 1.1628f, 1.1556f, 0.8958f, 2.3857f, 1.0033f, 1.6113f, 1.5270f, 2.1370f, 2.2073f, 0.8977f, 1.5905f, 1.1666f, 2.8612f, 1.1309f, 2.0125f, 1.5837f, 2.4002f, 1.8937f, 1.2044f, 3.6960f, 1.2921f, 4.2380f, 0.9825f, 1.9192f, 2.3002f, 2.7925f, 2.6397f, 1.3808f, 2.2449f, 0.8193f, 1.0860f, 1.1775f, 1.2217f, 2.1686f, 1.8566f, 3.4405f, 1.7497f, 2.2348f, 0.9937f, 1.3083f, 1.8705f, 1.8119f, 1.9842f, 1.9202f, 1.4630f, 0.9698f, 2.3382f, 1.3574f, 1.2673f, 1.3508f, 1.7544f, 2.6648f, 1.8328f, 1.5457f, 1.4313f, 3.2808f, 1.1473f, 1.5253f, 1.1793f, 2.6760f, 2.3115f, 1.9099f, 1.3163f, 1.2196f, 1.1000f, 1.1124f, 2.0581f, 1.1220f, 1.9189f, 2.5945f, 3.2412f, 2.0704f, 1.2744f, 1.3546f, 1.2478f, 2.5411f, 1.1493f, 1.9561f, 2.0982f, 2.3560f, 3.2071f, 1.3719f, 1.3779f, 1.1725f, 2.0776f, 1.0932f, 1.4819f, 1.8374f, 2.4527f, 2.7294f, 1.3242f, 1.8019f, 1.1641f, 3.4406f, 0.9961f, 2.5363f, 1.6337f, 1.8584f, 2.5289f, 1.3377f, 3.5655f, 1.3428f, 4.3400f, 1.2809f, 1.5810f, 1.7901f, 2.6967f, 2.9035f, 1.2119f, 2.3947f, 1.1531f, 1.2964f, 1.1823f, 1.0767f, 1.8882f, 1.5595f, 1.6684f, 2.0303f, 1.8453f, 1.0888f, 1.3763f, 1.6089f, 1.4793f, 1.0775f, 1.5040f, 1.1319f, 0.8188f, 2.2834f, 1.1053f, 0.9447f, 0.9454f, 1.3940f, 1.9593f, 2.2161f, 1.2012f, 1.3084f, 3.3662f, 0.8997f, 1.2336f, 0.7658f, 2.3280f, 1.1206f, 1.3189f, 1.0202f, 1.2560f, 1.2608f, 0.8348f, 1.6763f, 0.8722f, 1.9083f, 1.3506f, 3.2238f, 1.2196f, 1.1890f, 1.2390f, 0.6142f, 2.0348f, 0.6846f, 1.7910f, 1.0873f, 1.8157f, 1.9977f, 0.9138f, 1.1055f, 0.7890f, 1.9416f, 0.7639f, 1.6596f, 1.1961f, 1.9560f, 1.4552f, 0.9855f, 1.4940f, 0.8246f, 2.7040f, 0.6693f, 1.9750f, 1.0996f, 2.4958f, 1.6759f, 0.9837f, 3.2769f, 1.0021f, 3.5626f, 0.7590f, 2.0379f, 1.7765f, 2.3367f, 1.8444f, 0.7696f, 1.3846f, 1.1300f, 0.9991f, 0.9124f, 0.8679f, 1.3853f, 2.0572f, 2.2756f, 1.5519f, 1.9644f, 1.2163f, 1.0305f, 0.8769f, 1.8181f, 0.9503f, 1.8522f, 1.3357f, 1.0879f, 2.3632f, 0.8962f, 0.9606f, 0.9937f, 1.7611f, 2.0296f, 2.4239f, 1.3404f, 1.0722f, 3.2473f, 0.8439f, 1.3453f, 0.9130f, 2.0910f, 1.3660f, 1.7392f, 0.8272f, 0.9586f, 1.3080f, 0.7058f, 1.7292f, 1.0850f, 1.5581f, 1.3855f, 3.6722f, 1.2473f, 0.9993f, 1.1664f, 0.7308f, 2.0408f, 0.9535f, 1.5838f, 1.3883f, 2.5260f, 2.7706f, 0.9754f, 1.0501f, 0.7782f, 1.4895f, 1.1519f, 1.7279f, 1.1473f, 2.3450f, 1.8979f, 1.2844f, 1.2727f, 0.8832f, 2.4080f, 1.0858f, 2.0976f, 0.8295f, 2.6543f, 2.0322f, 1.1157f, 3.5258f, 0.7814f, 3.8649f, 0.8778f, 2.2647f, 1.5659f, 2.7879f, 2.2641f, 0.8000f, 1.4924f, 0.8463f, 0.9905f, 0.9977f, 0.6710f, 1.2015f, 1.9449f, 2.1792f, 1.4469f, 1.9567f, 1.0122f, 0.9489f, 0.9768f, 1.5541f, 1.9634f, 2.3382f, 1.2948f, 0.9454f, 3.0257f, 1.3728f, 1.2011f, 1.2976f, 1.0990f, 1.5998f, 1.8728f, 1.3855f, 1.1245f, 4.3009f, 0.8864f, 1.1027f, 1.4304f, 1.9583f, 1.4917f, 1.4197f, 0.7418f, 1.1002f, 1.7172f, 1.1922f, 2.5563f, 1.2499f, 1.0501f, 1.7889f, 1.4585f, 0.8743f, 1.0135f, 1.7239f, 0.9532f, 1.5997f, 1.1348f, 1.4551f, 1.3363f, 1.6927f, 1.5694f, 0.8415f, 1.6171f, 1.2622f, 1.4919f, 0.8958f, 0.9429f, 1.3113f, 1.1283f, 1.0046f, 0.7813f, 2.2585f, 1.2639f, 2.2767f, 1.2064f, 1.4738f, 0.9777f, 2.1085f, 1.0683f, 1.2431f, 4.5089f, 1.3593f, 3.8189f, 1.0441f, 1.0609f, 1.2430f, 2.1092f, 0.9892f, 0.7722f, 2.7907f, 0.6795f, 1.3429f, 1.2238f, 1.2232f, 1.0239f, 1.7936f, 1.4860f, 1.8842f, 2.8249f, 1.2817f, 1.3243f, 1.3409f, 1.1339f, 1.4317f, 1.7132f, 0.9021f, 0.4599f, 0.9546f, 0.6253f, 0.5216f, 0.7199f, 0.9011f, 1.3615f, 1.7387f, 0.7245f, 0.6517f, 1.6841f, 0.4362f, 0.6746f, 0.5240f, 1.5656f, 1.5537f, 1.3202f, 0.3704f, 0.5263f, 0.6163f, 0.4276f, 0.9433f, 0.6380f, 0.9127f, 1.4405f, 1.5844f, 0.6908f, 0.7868f, 0.6456f, 0.5700f, 2.2532f, 0.5265f, 0.7133f, 1.1012f, 1.2034f, 1.5167f, 0.4795f, 0.5284f, 0.4362f, 1.8082f, 0.4884f, 0.7464f, 1.1137f, 1.1636f, 0.9508f, 0.6205f, 0.8113f, 0.5473f, 0.9140f, 0.4950f, 0.8110f, 0.8421f, 1.3078f, 0.9837f, 0.5465f, 1.7849f, 0.3330f, 3.6269f, 0.5225f, 0.9180f, 1.0978f, 2.2061f, 1.0179f, 0.5898f, 0.9402f, 0.6700f, 1.3080f, 0.5517f, 0.5630f, 1.0446f, 1.1855f, 1.1516f, 0.9754f, 0.7633f, 0.7248f, 0.6224f, 0.8048f, 1.1716f, 1.5335f, 1.9015f, 0.8555f, 0.5094f, 0.8205f, 0.7167f, 0.7084f, 0.7216f, 1.0628f, 1.2090f, 2.2210f, 0.7668f, 0.6115f, 1.4186f, 0.4784f, 0.8190f, 0.4561f, 1.6248f, 1.9423f, 1.5137f, 0.3520f, 0.4847f, 0.3509f, 0.6171f, 0.8263f, 0.6206f, 0.9042f, 1.8299f, 2.4173f, 0.8278f, 0.5808f, 0.8223f, 0.6253f, 0.4981f, 0.5178f, 1.1751f, 1.1983f, 1.7841f, 1.7818f, 0.5061f, 0.6663f, 0.5803f, 0.6423f, 0.4891f, 0.8542f, 1.4655f, 1.3929f, 0.9853f, 0.5824f, 1.0162f, 0.6359f, 2.5886f, 0.4727f, 1.2816f, 0.9333f, 1.5076f, 1.0646f, 0.4868f, 1.2102f, 1.3483f, 0.3311f, 0.3881f, 1.0907f, 1.1198f, 1.9239f, 1.2347f, 0.3518f, 0.9762f, 0.3501f, 0.5896f, 0.4677f, 0.7389f, 0.8027f, 1.4365f, 1.0419f, 0.9742f, 0.7542f, 0.5153f, 0.5649f, 0.6514f, 0.7447f, 1.1705f, 1.4905f, 0.6847f, 0.2589f, 0.7078f, 0.4981f, 0.3210f, 0.3695f, 0.5316f, 0.8724f, 1.4847f, 0.4657f, 0.4115f, 1.2509f, 0.2654f, 0.7181f, 0.4620f, 1.2811f, 1.1054f, 0.9968f, 0.3172f, 0.3338f, 0.2338f, 0.3087f, 0.7458f, 0.4580f, 0.6500f, 1.2237f, 1.3309f, 0.4730f, 0.3790f, 0.3644f, 0.3853f, 1.0119f, 0.3380f, 0.6575f, 0.8855f, 1.0735f, 1.3216f, 0.2545f, 0.3397f, 0.2500f, 1.2835f, 0.4573f, 0.5190f, 0.7987f, 0.9672f, 0.6184f, 0.3448f, 0.4166f, 0.4200f, 1.2825f, 0.3881f, 0.7019f, 0.6392f, 1.2048f, 0.5561f, 0.2506f, 1.1169f, 0.3515f, 1.6557f, 0.4039f, 0.8453f, 0.7282f, 1.7164f, 0.7245f, 0.1844f, 0.5466f, 0.2890f, 0.8036f, 0.3885f, 0.6385f, 0.7343f, 1.1357f, 0.9384f, 0.6205f, 0.5270f, 0.4488f, 0.3996f, 0.5754f, 0.8844f, 1.1805f, 1.6720f, 0.7079f, 0.3798f, 0.9603f, 0.5593f, 0.5591f, 0.4398f, 0.7543f, 0.8608f, 1.4840f, 0.4608f, 0.5348f, 1.4335f, 0.3527f, 0.7289f, 0.4495f, 1.3950f, 1.2235f, 1.0129f, 0.3518f, 0.2860f, 0.3384f, 0.3542f, 0.7379f, 0.5395f, 0.6879f, 1.3395f, 1.5204f, 0.5040f, 0.3800f, 0.5773f, 0.4206f, 0.9013f, 0.4092f, 0.8043f, 0.9161f, 1.2618f, 1.2991f, 0.3580f, 0.3870f, 0.3911f, 0.9202f, 0.3815f, 0.6979f, 0.7562f, 0.9149f, 0.7925f, 0.5103f, 0.5680f, 0.5203f, 1.4284f, 0.3264f, 0.9145f, 0.4083f, 1.1107f, 0.8148f, 0.3837f, 1.2812f, 0.5353f, 2.1498f, 0.3659f, 0.9164f, 0.5521f, 1.7034f, 0.7448f, 0.3422f, 0.9187f, 0.3585f, 0.7698f, 0.4158f, 0.5924f, 0.5312f, 0.9191f, 0.6641f, 0.6715f, 0.5852f, 0.4205f, 0.4184f, 0.4729f, 0.8759f, 1.2676f, 1.9584f, 1.0612f, 0.4834f, 1.0454f, 0.5906f, 0.4279f, 0.4974f, 0.7591f, 0.9388f, 1.5682f, 0.6361f, 0.4688f, 1.4392f, 0.4357f, 0.6642f, 0.5365f, 1.3951f, 1.2282f, 1.0938f, 0.4444f, 0.6288f, 0.5119f, 0.4170f, 0.9262f, 0.5226f, 0.7183f, 1.4384f, 2.0079f, 0.6159f, 0.5977f, 0.5346f, 0.4099f, 1.0242f, 0.3781f, 0.8258f, 1.0607f, 1.6053f, 1.8711f, 0.3877f, 0.4963f, 0.3859f, 1.1139f, 0.3957f, 0.8102f, 0.9425f, 1.2274f, 1.1315f, 0.6089f, 0.6266f, 0.5695f, 1.3966f, 0.4403f, 0.8916f, 0.6194f, 1.4336f, 0.9556f, 0.6658f, 1.4922f, 0.4315f, 1.5317f, 0.4654f, 0.9669f, 0.8099f, 1.8411f, 1.2368f, 0.4439f, 0.8946f, 0.3762f, 0.7278f, 0.4321f, 0.6034f, 0.6887f, 1.3379f, 1.2251f, 0.8557f, 0.7991f, 0.5320f, 0.4870f, 0.6142f, 1.1609f, 1.4317f, 1.6739f, 0.8744f, 0.3677f, 0.8861f, 0.5848f, 0.5217f, 0.4507f, 0.8907f, 1.2364f, 1.8536f, 0.5626f, 0.4871f, 1.4099f, 0.3455f, 0.5243f, 0.5021f, 1.5719f, 1.4983f, 1.3848f, 0.3670f, 0.4529f, 0.3461f, 0.3794f, 0.6599f, 0.5753f, 0.8644f, 1.5279f, 2.1742f, 0.4896f, 0.4411f, 0.4629f, 0.4421f, 1.2061f, 0.4126f, 0.8938f, 1.1306f, 1.6000f, 1.9038f, 0.3813f, 0.3113f, 0.3027f, 1.0651f, 0.4706f, 0.9950f, 1.1220f, 1.4915f, 0.9460f, 0.3921f, 0.4830f, 0.5274f, 0.8788f, 0.4926f, 1.0880f, 1.0044f, 1.8934f, 0.9151f, 0.4484f, 1.4850f, 0.4697f, 2.3883f, 0.4144f, 1.1752f, 1.2007f, 2.4878f, 1.2944f, 0.4842f, 0.6190f, 0.2751f, 0.6512f, 0.5187f, 0.6340f, 1.0774f, 1.6501f, 2.0208f, 0.7837f, 0.7830f, 0.4261f, 0.4476f, 0.6885f, 1.2001f, 1.6004f, 1.8408f, 1.0138f, 0.4823f, 0.9436f, 0.6526f, 0.7087f, 0.5478f, 0.9989f, 1.6935f, 1.8156f, 0.6711f, 0.6162f, 1.3531f, 0.4587f, 0.5905f, 0.5191f, 1.6580f, 1.5364f, 1.4102f, 0.4872f, 0.4793f, 0.3802f, 0.4629f, 0.8743f, 0.5280f, 0.9726f, 1.8188f, 1.9514f, 0.9039f, 0.7101f, 0.7752f, 0.5955f, 0.9922f, 0.4090f, 1.0352f, 1.3359f, 1.4913f, 1.7905f, 0.6319f, 0.5373f, 0.5555f, 0.8696f, 0.4011f, 0.8656f, 1.1862f, 1.5411f, 1.2633f, 0.6193f, 0.8167f, 0.6205f, 0.9907f, 0.3349f, 1.3856f, 0.9191f, 1.3306f, 1.2604f, 0.6147f, 1.4704f, 0.5800f, 2.0606f, 0.3964f, 1.0194f, 1.0020f, 2.3037f, 1.4683f, 0.4268f, 0.9615f, 0.4052f, 0.7202f, 0.4589f, 0.6920f, 0.9703f, 1.2480f, 0.9696f, 1.0196f, 0.7503f, 0.5153f, 0.6248f, 0.6778f, 1.0532f, 1.3990f, 1.6227f, 0.7997f, 0.3725f, 0.7289f, 0.7502f, 0.5043f, 0.5260f, 0.7333f, 1.1298f, 2.1098f, 0.6645f, 0.7171f, 1.3321f, 0.5076f, 0.6277f, 0.4452f, 1.7478f, 1.2219f, 1.3890f, 0.5791f, 0.6106f, 0.3776f, 0.4568f, 0.5740f, 0.5530f, 0.8993f, 1.6388f, 2.0115f, 0.6934f, 0.5816f, 0.5869f, 0.4224f, 0.8348f, 0.3909f, 1.0338f, 1.0337f, 1.4678f, 1.5131f, 0.3273f, 0.4625f, 0.3959f, 1.1546f, 0.3048f, 0.7492f, 1.0312f, 1.3248f, 0.9357f, 0.4879f, 0.7627f, 0.5144f, 1.3239f, 0.3304f, 1.0239f, 0.8973f, 1.9327f, 0.8800f, 0.4791f, 0.9172f, 0.6609f, 1.5470f, 0.4466f, 1.1100f, 0.9151f, 2.1060f, 1.0788f, 0.3087f, 0.6733f, 0.3981f, 0.7351f, 0.5032f, 0.6928f, 0.8549f, 1.7067f, 1.3470f, 0.9306f, 0.5908f, 0.6159f, 0.5557f, 0.5649f, 1.3118f, 1.2605f, 1.8820f, 0.9382f, 0.5682f, 0.7902f, 0.5619f, 0.5103f, 0.5494f, 1.0163f, 1.1607f, 2.2354f, 0.7555f, 0.5112f, 1.2404f, 0.4428f, 0.6944f, 0.5467f, 1.5292f, 1.3924f, 1.7044f, 0.4028f, 0.3469f, 0.3985f, 0.3853f, 0.6002f, 0.7090f, 0.5850f, 1.6522f, 2.3044f, 0.7020f, 0.4206f, 0.5165f, 0.5103f, 0.7649f, 0.5946f, 0.8520f, 1.2489f, 1.9991f, 2.0969f, 0.3562f, 0.4116f, 0.3851f, 0.7208f, 0.5882f, 0.7750f, 0.9657f, 1.5992f, 1.2720f, 0.6945f, 0.5920f, 0.5583f, 1.0491f, 0.6406f, 1.1098f, 0.6405f, 1.9961f, 1.1393f, 0.5695f, 1.1134f, 0.4523f, 1.6655f, 0.5168f, 1.2557f, 0.6961f, 2.4109f, 1.3650f, 0.3037f, 0.7470f, 0.1444f, 0.6931f, 0.5510f, 0.5260f, 0.6411f, 1.5472f, 1.1867f, 0.8192f, 0.5869f, 0.4398f, 0.4745f, 0.6263f, 4.1163f, 3.4149f, 3.6399f, 3.0789f, 2.8816f, 3.8359f, 2.6338f, 3.0906f, 2.9430f, 3.6785f, 2.4956f, 4.4207f, 3.5577f, 3.9294f, 5.3130f, 2.2928f, 2.2248f, 2.7929f, 3.7797f, 3.5260f, 2.5337f, 3.1498f, 2.6917f, 3.0829f, 2.8251f, 3.5352f, 2.5926f, 3.1927f, 4.2570f, 3.0319f, 3.5744f, 2.6152f, 2.8199f, 2.8517f, 3.0425f, 2.3584f, 3.6341f, 3.4677f, 3.6600f, 3.4711f, 2.7201f, 2.6932f, 2.7418f, 2.8346f, 2.4860f, 2.9934f, 3.5125f, 3.1453f, 2.8009f, 1.9909f, 3.3954f, 2.7442f, 3.5561f, 2.2965f, 3.9759f, 3.2355f, 3.0664f, 3.0618f, 2.1300f, 5.0274f, 2.9988f, 5.2057f, 2.6232f, 3.7508f, 3.0838f, 2.9836f, 3.8086f, 2.5307f, 3.0352f, 2.5959f, 2.3646f, 2.4851f, 3.2988f, 3.6488f, 3.6029f, 3.1622f, 3.6931f, 3.2873f, 2.3522f, 2.5130f, 2.6366f, 1.8889f, 2.3205f, 2.3049f, 1.5549f, 1.5957f, 2.6541f, 1.4683f, 1.3854f, 1.5457f, 2.1481f, 2.3337f, 2.9023f, 2.3990f, 1.8434f, 3.9386f, 1.9490f, 1.7520f, 2.1220f, 2.1836f, 1.9787f, 1.4369f, 1.2088f, 1.7370f, 1.8565f, 1.4729f, 2.2718f, 1.6072f, 2.0483f, 2.4393f, 2.8885f, 1.5965f, 1.4526f, 1.8305f, 1.2561f, 2.0025f, 1.5559f, 2.3576f, 1.7513f, 2.0523f, 1.5012f, 1.3380f, 1.6983f, 1.1848f, 1.9910f, 1.6211f, 1.6522f, 2.2295f, 2.6709f, 2.1873f, 1.7630f, 1.5705f, 1.5266f, 1.9407f, 1.4172f, 2.3774f, 1.8947f, 2.6559f, 1.3813f, 1.2604f, 3.6352f, 1.6963f, 4.0244f, 1.6288f, 1.7621f, 2.5215f, 2.1148f, 2.9330f, 1.3788f, 2.4648f, 1.4284f, 1.6405f, 1.9853f, 1.9921f, 2.1261f, 2.8822f, 2.2990f, 2.0801f, 2.8049f, 1.5323f, 1.6084f, 1.9201f, 1.7054f, 4.3711f, 2.7661f, 1.9543f, 1.8134f, 2.7718f, 1.5833f, 1.9360f, 1.9035f, 2.1517f, 2.0746f, 4.0537f, 2.1252f, 1.8891f, 4.0004f, 1.3003f, 1.5748f, 1.5456f, 2.4066f, 3.3628f, 2.0654f, 2.2510f, 2.1157f, 2.0067f, 1.1791f, 2.1230f, 1.2200f, 2.5024f, 4.1637f, 4.4711f, 1.6685f, 1.9851f, 2.1222f, 1.3974f, 2.1729f, 1.1749f, 2.6872f, 3.6825f, 3.6141f, 2.4820f, 2.2293f, 1.8719f, 1.6585f, 2.1907f, 1.3949f, 2.1333f, 2.9991f, 3.7502f, 1.8530f, 1.4183f, 1.8888f, 1.9465f, 2.4539f, 1.3530f, 2.4432f, 2.8607f, 4.1367f, 2.7547f, 1.3753f, 3.8854f, 1.6189f, 4.0081f, 1.4555f, 3.0384f, 3.4216f, 3.8053f, 3.6960f, 1.7499f, 2.3495f, 1.9556f, 1.3216f, 1.6158f, 1.4180f, 5.0042f, 4.3903f, 4.0977f, 2.6767f, 2.6626f, 1.6163f, 1.9279f, 2.0891f, 2.9371f, 4.5325f, 3.1994f, 1.9513f, 2.0392f, 3.0878f, 1.9147f, 1.8117f, 1.7179f, 2.4114f, 1.4169f, 3.2094f, 1.3823f, 2.2590f, 4.2540f, 1.3999f, 1.6113f, 1.7021f, 2.8216f, 4.0904f, 1.7881f, 2.0097f, 1.8473f, 2.2376f, 1.5941f, 2.2511f, 1.3157f, 1.4576f, 1.6388f, 2.0054f, 1.1204f, 1.5775f, 2.0769f, 1.8367f, 2.0606f, 1.4316f, 1.6839f, 1.4063f, 1.5595f, 1.7004f, 2.2946f, 1.6362f, 1.8285f, 1.6535f, 1.5450f, 1.1817f, 2.6927f, 2.5396f, 2.1609f, 1.6917f, 1.6524f, 2.2150f, 2.8888f, 1.6553f, 2.8161f, 3.7354f, 3.4563f, 2.5902f, 1.9868f, 4.0488f, 2.0751f, 4.3067f, 1.6025f, 2.6328f, 3.6659f, 2.4117f, 3.8901f, 1.7248f, 1.9821f, 1.9712f, 1.6710f, 2.4388f, 2.7641f, 4.1638f, 3.0955f, 3.3278f, 2.7616f, 2.8680f, 1.7034f, 2.0179f, 1.9476f, 2.5671f, 3.4005f, 2.9794f, 2.4872f, 1.6516f, 3.0037f, 1.5486f, 1.6383f, 1.5489f, 1.6938f, 1.8431f, 2.8902f, 1.9529f, 1.6226f, 4.0147f, 1.1009f, 1.6560f, 1.4509f, 2.4043f, 3.1118f, 2.5165f, 2.5726f, 2.1556f, 2.1529f, 1.2688f, 2.2885f, 1.2238f, 1.6697f, 1.9725f, 2.6563f, 1.0561f, 1.8388f, 2.0241f, 1.7297f, 2.4233f, 1.2943f, 1.5185f, 2.2582f, 1.4433f, 0.9635f, 2.0765f, 1.7128f, 1.3798f, 2.3529f, 1.3356f, 1.0250f, 1.6687f, 2.0616f, 2.1154f, 1.9112f, 1.8746f, 1.7301f, 2.7578f, 1.4463f, 1.5974f, 1.9569f, 2.9551f, 1.6139f, 2.5130f, 3.8021f, 2.0135f, 4.6122f, 1.2754f, 1.7153f, 2.3332f, 2.2946f, 3.5646f, 1.9137f, 2.7836f, 1.6694f, 1.7063f, 2.0086f, 1.7953f, 2.0693f, 2.7161f, 2.0185f, 2.9467f, 2.9709f, 1.8886f, 1.9363f, 1.8733f, 1.9740f, 1.6641f, 2.4284f, 1.1708f, 0.9615f, 1.9044f, 1.3111f, 1.3941f, 1.4168f, 1.4610f, 1.7288f, 2.4740f, 1.4485f, 1.5234f, 3.1277f, 0.8887f, 1.2165f, 1.3463f, 1.9311f, 2.0590f, 1.8576f, 1.1733f, 1.2963f, 1.0890f, 1.3918f, 2.1541f, 1.0682f, 1.4803f, 1.7743f, 3.0494f, 1.5979f, 1.5565f, 1.5130f, 1.3166f, 2.4162f, 1.2405f, 1.9122f, 1.1866f, 1.5201f, 1.8645f, 1.3468f, 1.3792f, 1.4132f, 1.9350f, 1.1074f, 1.3898f, 1.5819f, 1.2516f, 2.2036f, 1.7666f, 1.8447f, 1.3778f, 2.4733f, 1.2469f, 1.9194f, 1.5848f, 2.3264f, 1.7765f, 1.4893f, 3.0406f, 1.2875f, 3.4124f, 1.1346f, 1.7251f, 2.0211f, 1.7794f, 2.3304f, 0.9639f, 1.7664f, 1.0843f, 1.0701f, 1.5084f, 1.6116f, 1.2056f, 2.4793f, 1.6197f, 1.5963f, 1.8756f, 1.3527f, 1.5173f, 1.4792f, 3.1890f, 5.4557f, 3.0673f, 1.8776f, 1.7519f, 2.7146f, 1.4883f, 1.8512f, 1.5376f, 2.2715f, 2.6897f, 2.5477f, 1.6170f, 1.8778f, 3.6154f, 0.9819f, 1.1289f, 1.4931f, 2.3705f, 3.7881f, 2.7907f, 1.9005f, 1.9329f, 1.8096f, 1.4820f, 2.4313f, 1.4240f, 1.7815f, 4.3332f, 2.1096f, 2.0945f, 1.7013f, 1.5151f, 1.8774f, 2.4906f, 1.6685f, 1.7317f, 3.0323f, 2.1728f, 1.4093f, 1.9206f, 1.4853f, 1.7133f, 1.9853f, 1.5303f, 1.5165f, 2.2749f, 2.6638f, 1.9758f, 1.7615f, 1.7626f, 1.8294f, 2.5394f, 1.4369f, 1.7965f, 3.0170f, 2.7556f, 2.2753f, 1.1972f, 3.5200f, 1.8568f, 4.7450f, 1.6064f, 2.0431f, 3.0402f, 3.1289f, 3.5977f, 1.5140f, 1.4530f, 1.8212f, 1.3993f, 2.0454f, 2.4185f, 4.8495f, 2.4138f, 2.3109f, 1.8713f, 2.5806f, 1.7578f, 2.1035f, 1.7713f },
        };
        float ndvalue[2]{ 0.5633f, 0.4181f };
        __hcpe3_cache_re_eval(2, (char*)ndindex, (char*)ndlogits, (char*)ndvalue, 0.9f, 0.9f, 0.9f, 0.3f, 10);
    }
    {
        unsigned int ndindex[1] = { 2 };
        float ndlogits[1][9 * 9 * MAX_MOVE_LABEL_NUM] = {
            { 1.9759f, 1.8800f, 2.2340f, 2.1143f, 2.5600f, 15.8510f, 2.3378f, 3.1995f, 2.0966f, 1.5875f, 2.9426f, 1.7042f, 2.4739f, 2.3382f, 17.9522f, 1.9609f, 1.6684f, 2.4498f, 2.7837f, 2.4690f, 2.0082f, 1.8867f, 1.8502f, 8.0496f, 2.3199f, 11.4931f, 2.0139f, 2.4144f, 1.5713f, 1.7978f, 2.3642f, 3.3703f, 8.0576f, 2.1666f, 5.2542f, 1.9576f, 2.6641f, 2.0388f, 2.7946f, 2.6601f, 1.9676f, 7.4080f, 2.4334f, 4.9192f, 2.1328f, 2.7467f, 2.1924f, 2.2468f, 3.1611f, 2.7435f, 8.1440f, 1.9564f, 6.5091f, 2.1378f, 2.9921f, 1.7486f, 2.6991f, 2.3358f, 2.5194f, 18.4509f, 2.6067f, 9.6553f, 2.0268f, 2.0622f, 2.4480f, 1.0413f, 2.1361f, 2.0953f, 9.1869f, 2.3741f, 2.7390f, 2.3814f, 1.5500f, 1.5544f, 1.4742f, 2.1216f, 3.3371f, 14.4371f, 2.5464f, 3.3219f, 2.4335f, 1.4156f, 1.7418f, 1.7546f, 1.3602f, 1.3778f, 4.1921f, 1.4564f, 1.6336f, 1.4040f, 1.2698f, 1.6106f, 1.2173f, 1.4597f, 1.8625f, 4.8275f, 1.4997f, 1.4529f, 1.3066f, 1.7738f, 1.9619f, 2.0750f, 1.3278f, 1.6611f, 2.1320f, 1.3818f, 3.5024f, 1.2813f, 1.7253f, 1.8113f, 2.4375f, 1.7342f, 2.4148f, 2.5610f, 1.2738f, 12.5936f, 1.3238f, 1.5400f, 1.5741f, 2.1362f, 2.7767f, 1.3499f, 2.0895f, 1.1803f, 9.2986f, 1.4313f, 1.7367f, 2.0001f, 2.2820f, 2.7323f, 1.7066f, 2.3582f, 1.2738f, 8.5753f, 1.4201f, 1.4416f, 1.5282f, 2.3313f, 2.0236f, 1.9704f, 5.5837f, 1.4459f, 16.5422f, 1.4887f, 1.1572f, 2.4020f, 2.1949f, 2.6115f, 2.1467f, 2.1607f, 1.9341f, 3.1725f, 1.4651f, 0.7421f, 2.2118f, 1.7391f, 2.2254f, 1.6411f, 3.6031f, 1.6277f, 1.8651f, 1.6623f, 1.4632f, 2.0911f, 2.5164f, 1.3047f, 1.5100f, 4.2241f, 1.5545f, 1.7922f, 1.3217f, 1.5729f, 2.6345f, 1.4137f, 1.5723f, 1.7274f, 4.6114f, 1.2032f, 1.8109f, 1.1871f, 1.9191f, 1.8984f, 2.4647f, 1.4021f, 1.3697f, 1.9795f, 1.4748f, 3.4537f, 1.2482f, 2.0335f, 2.3773f, 3.4934f, 2.0700f, 2.2549f, 2.3718f, 1.1861f, 5.4844f, 1.3526f, 1.9546f, 2.1673f, 2.6005f, 3.1574f, 1.2104f, 2.3464f, 1.2443f, 4.3485f, 1.4339f, 1.8679f, 2.6364f, 3.4014f, 3.0755f, 1.9312f, 2.2299f, 1.3329f, 11.2297f, 1.5514f, 1.9968f, 1.5252f, 2.6227f, 2.3253f, 1.4853f, 4.8393f, 3.1898f, 5.9817f, 1.5609f, 1.2767f, 1.9488f, 1.3672f, 2.5023f, 1.6008f, 2.1616f, 1.4096f, 1.7338f, 1.1443f, 0.8221f, 1.8262f, 1.6176f, 1.8488f, 1.5349f, 3.6506f, 1.2595f, 1.6850f, 1.3410f, 1.2268f, 1.5765f, 1.5327f, 1.2266f, 1.2716f, 3.3679f, 1.4358f, 1.4353f, 1.2401f, 1.3570f, 2.4759f, 1.4234f, 1.0985f, 1.5705f, 4.5792f, 1.3360f, 1.6548f, 1.3532f, 1.9725f, 1.3622f, 2.2231f, 1.6271f, 1.4700f, 1.7949f, 1.4565f, 8.5629f, 1.0781f, 2.0650f, 1.6544f, 2.1937f, 2.3372f, 2.1894f, 2.5082f, 1.2403f, 5.6601f, 1.0846f, 1.8053f, 1.9662f, 2.5209f, 2.7855f, 1.4871f, 2.5465f, 1.1201f, 8.5721f, 1.9344f, 2.1165f, 2.4866f, 1.8613f, 3.4169f, 1.3488f, 2.1073f, 1.3516f, 11.6259f, 2.3279f, 2.0582f, 1.9381f, 2.5823f, 2.1744f, 2.6206f, 4.2878f, 1.0665f, 10.4100f, 1.7432f, 1.4816f, 2.5925f, 1.7055f, 2.1989f, 1.3293f, 2.0183f, 1.4929f, 2.7559f, 1.0919f, 1.2151f, 1.8535f, 2.6151f, 2.0513f, 1.4874f, 3.5388f, 1.5320f, 1.8393f, 1.5748f, 1.4991f, 1.9880f, 2.5504f, 1.1836f, 1.1807f, 3.6837f, 1.2535f, 3.8083f, 1.1922f, 1.5072f, 2.2538f, 1.3671f, 1.2827f, 1.4372f, 4.6325f, 1.0890f, 1.6845f, 1.2387f, 1.7817f, 1.8000f, 2.1733f, 1.3405f, 1.4191f, 1.9142f, 1.2577f, 3.3698f, 1.5825f, 2.0207f, 2.5026f, 2.9079f, 2.4638f, 2.5072f, 1.5196f, 1.1866f, 3.2543f, 1.9164f, 1.6259f, 2.2266f, 3.2188f, 2.6733f, 1.8475f, 1.8133f, 1.1061f, 3.4916f, 1.4073f, 1.9180f, 2.0568f, 2.6833f, 3.1020f, 1.4540f, 2.0603f, 1.3154f, 4.3929f, 0.9990f, 2.0368f, 1.8315f, 2.9544f, 2.9075f, 2.5598f, 4.4369f, 1.3562f, 4.3498f, 1.2446f, 1.0738f, 2.2541f, 1.9053f, 2.3479f, 1.3476f, 1.3870f, 1.1451f, 1.9819f, 1.3372f, 0.7840f, 1.6922f, 2.1497f, 1.7488f, 1.3905f, 3.1022f, 1.0888f, 1.6203f, 1.4645f, 1.1322f, 1.9216f, 2.5900f, 1.5365f, 0.8782f, 3.2043f, 1.2464f, 1.3672f, 1.2403f, 1.3749f, 2.1929f, 2.0732f, 1.4145f, 1.6096f, 4.0977f, 1.3437f, 1.2863f, 1.7045f, 1.6399f, 1.5272f, 2.0873f, 1.4995f, 1.4324f, 1.7316f, 1.0411f, 3.6284f, 1.1673f, 1.7851f, 2.0937f, 3.4256f, 1.9005f, 1.8480f, 2.4691f, 1.0714f, 3.4574f, 1.0894f, 1.6778f, 1.9106f, 3.5929f, 3.4924f, 1.6373f, 1.9841f, 1.1077f, 3.7153f, 1.1879f, 1.8776f, 1.6985f, 2.7988f, 3.5098f, 1.0343f, 2.2076f, 1.0735f, 3.5144f, 1.4567f, 2.0397f, 1.3659f, 3.1230f, 2.8914f, 2.1488f, 4.2222f, 1.0408f, 5.3254f, 1.3897f, 1.3959f, 2.3008f, 1.7964f, 3.4502f, 1.6179f, 2.1908f, 1.1455f, 1.6821f, 1.5434f, 0.8138f, 1.7048f, 2.9313f, 2.5625f, 1.2845f, 3.1901f, 1.1974f, 1.4789f, 1.6067f, 1.2002f, 1.4693f, 1.2998f, 0.9920f, 0.9313f, 2.9718f, 1.1817f, 1.1911f, 1.0824f, 1.2326f, 2.0651f, 1.8609f, 1.2270f, 1.5163f, 3.4432f, 1.0882f, 0.8903f, 1.1151f, 1.5185f, 1.8167f, 2.0622f, 0.7898f, 1.4335f, 1.5349f, 1.1462f, 2.8182f, 0.8817f, 1.5805f, 1.9526f, 2.5085f, 1.2372f, 1.6692f, 1.3600f, 0.9933f, 3.9131f, 0.8620f, 1.4472f, 1.6223f, 2.6168f, 3.5946f, 1.1524f, 1.4532f, 0.9004f, 3.4349f, 1.0267f, 1.6832f, 1.7632f, 2.1931f, 2.4181f, 1.1607f, 1.6962f, 1.2730f, 3.7870f, 1.3269f, 1.6135f, 1.5970f, 2.9394f, 2.2273f, 1.8951f, 3.5393f, 1.0644f, 4.7799f, 1.2426f, 1.4111f, 2.8249f, 2.4536f, 2.9344f, 1.5757f, 1.5719f, 1.1254f, 1.3030f, 1.3979f, 0.6932f, 2.1439f, 2.0983f, 3.4805f, 1.6041f, 2.7225f, 1.0362f, 1.2271f, 1.6536f, 1.2934f, 1.9762f, 1.9613f, 1.3531f, 1.0197f, 3.0059f, 1.2670f, 1.2095f, 1.2570f, 1.2075f, 2.5734f, 1.9053f, 1.4504f, 1.4278f, 3.6389f, 1.1917f, 1.3134f, 1.1785f, 1.2833f, 2.1439f, 1.9432f, 0.9079f, 1.2965f, 1.5688f, 1.2823f, 3.2476f, 1.1012f, 1.3119f, 2.5710f, 2.6405f, 1.9255f, 1.9909f, 1.9322f, 1.1989f, 3.7701f, 1.1695f, 1.3260f, 2.0160f, 2.6841f, 3.1967f, 1.6096f, 1.7028f, 1.2598f, 3.1441f, 1.1476f, 1.4785f, 1.8961f, 2.8580f, 2.8563f, 1.4501f, 1.9943f, 1.2576f, 4.0249f, 1.0719f, 2.2288f, 2.0041f, 1.8388f, 3.5034f, 1.8194f, 3.5866f, 1.2881f, 4.7091f, 1.5479f, 0.8768f, 1.9686f, 2.4957f, 3.1048f, 1.6326f, 2.7045f, 1.4563f, 1.5406f, 1.3358f, 0.6441f, 1.8067f, 1.5959f, 1.9178f, 1.5546f, 2.6163f, 1.1228f, 1.3350f, 1.4535f, 1.1871f, 1.1961f, 1.4828f, 0.9071f, 0.8253f, 2.6629f, 1.1034f, 0.9598f, 0.8330f, 1.1734f, 1.5953f, 1.6204f, 0.9040f, 1.0572f, 3.2888f, 0.9121f, 0.8421f, 0.7376f, 1.5304f, 1.1167f, 1.0442f, 1.0524f, 1.2406f, 1.5249f, 1.0084f, 2.3650f, 0.7231f, 1.6082f, 1.3700f, 2.0049f, 1.2709f, 1.4009f, 1.5833f, 0.7438f, 2.8838f, 0.6575f, 1.6926f, 0.7821f, 2.3047f, 1.6178f, 0.8980f, 1.4199f, 0.7814f, 2.6042f, 0.7768f, 1.4603f, 1.2462f, 1.6566f, 1.9544f, 1.0351f, 1.3899f, 1.0457f, 3.3228f, 0.7966f, 1.8410f, 1.0413f, 2.7243f, 1.9782f, 1.1827f, 3.4859f, 0.7721f, 3.4788f, 1.0288f, 1.6674f, 1.7645f, 1.5571f, 1.7489f, 1.1380f, 1.4485f, 1.1301f, 1.3993f, 0.8409f, 0.6741f, 1.3521f, 2.0344f, 2.3375f, 1.1895f, 2.4727f, 1.0340f, 1.0135f, 0.8415f, 1.4125f, 0.9826f, 1.4234f, 0.8976f, 1.1058f, 2.8001f, 0.8279f, 0.9837f, 0.8411f, 1.2879f, 1.8927f, 1.9448f, 1.0650f, 0.9395f, 3.3179f, 0.7950f, 0.8533f, 0.9075f, 1.2267f, 1.0716f, 1.5674f, 0.9198f, 0.8430f, 1.6171f, 0.8165f, 2.3381f, 0.9924f, 1.2453f, 1.5721f, 2.4352f, 1.4309f, 1.1899f, 1.5714f, 0.5772f, 3.0136f, 1.0133f, 1.4390f, 1.0640f, 3.0665f, 2.3492f, 1.0566f, 1.4553f, 0.7375f, 2.3482f, 1.2062f, 1.5763f, 1.2359f, 2.3134f, 2.4588f, 1.1481f, 1.3179f, 0.8950f, 2.9373f, 1.2159f, 1.9228f, 0.7312f, 2.8725f, 2.3632f, 1.3487f, 3.4575f, 0.5717f, 4.0166f, 1.1164f, 1.4519f, 1.5333f, 1.8382f, 2.2915f, 1.1338f, 1.5974f, 0.9898f, 1.3153f, 0.9856f, 0.4141f, 1.3013f, 1.8670f, 2.1622f, 1.0531f, 2.3756f, 0.9480f, 0.9467f, 0.9450f, 1.1973f, 1.9728f, 2.1416f, 1.4169f, 0.8372f, 3.8157f, 1.3214f, 1.2194f, 1.1795f, 0.8275f, 1.4230f, 1.7168f, 1.1374f, 0.7588f, 4.2373f, 0.7638f, 0.8599f, 1.3751f, 1.4972f, 1.4558f, 0.8065f, 1.0140f, 0.7571f, 2.0913f, 1.0317f, 3.3320f, 1.0563f, 0.8244f, 1.9000f, 1.2030f, 0.8340f, 1.2072f, 2.2180f, 0.9302f, 2.5264f, 1.0294f, 1.2602f, 1.2706f, 1.7179f, 1.6944f, 0.7826f, 2.1892f, 1.1788f, 2.5340f, 0.8556f, 1.0161f, 1.4288f, 1.6776f, 1.4032f, 0.8965f, 2.0337f, 0.9188f, 3.1664f, 1.1920f, 1.3055f, 0.9412f, 2.0652f, 1.5861f, 0.9749f, 4.5502f, 1.0696f, 4.5540f, 1.1323f, 0.8638f, 1.0212f, 1.3749f, 1.5900f, 1.1231f, 2.6592f, 1.0328f, 1.3176f, 1.2546f, 0.6434f, 1.0412f, 1.3905f, 1.5483f, 1.3518f, 3.5224f, 1.2586f, 1.3631f, 1.3466f, 0.8109f, 1.5768f, 1.5931f, 0.9308f, 0.4154f, 1.3641f, 0.5444f, 0.4303f, 0.6097f, 0.6368f, 0.8281f, 1.1708f, 0.5815f, 0.6113f, 1.8609f, 0.4908f, 0.4760f, 0.4228f, 0.6899f, 1.2322f, 0.7806f, 0.4013f, 0.5110f, 0.6750f, 0.4249f, 1.5798f, 0.4684f, 0.6184f, 1.6206f, 1.4128f, 0.7257f, 0.9900f, 0.9467f, 0.5475f, 3.3247f, 0.4548f, 0.5565f, 1.0351f, 1.2624f, 1.4317f, 0.4526f, 0.8588f, 0.3829f, 2.9582f, 0.4395f, 0.6677f, 1.1396f, 1.5551f, 1.2620f, 0.5608f, 0.8356f, 0.5036f, 1.9313f, 0.5414f, 0.5260f, 0.7538f, 1.4549f, 1.2017f, 0.7385f, 1.9405f, 0.3449f, 4.2764f, 0.5849f, 0.5437f, 0.9591f, 1.9102f, 1.7978f, 0.8804f, 0.7501f, 0.7639f, 1.3008f, 0.5738f, 0.3297f, 1.3842f, 1.0832f, 1.3172f, 0.6549f, 1.2375f, 0.6011f, 0.5848f, 0.6809f, 0.8764f, 1.6898f, 1.9162f, 0.7694f, 0.4171f, 1.0756f, 0.6980f, 0.6621f, 0.6089f, 0.8061f, 1.3014f, 1.2802f, 0.6943f, 0.6288f, 1.4014f, 0.4657f, 0.5686f, 0.4100f, 1.1070f, 1.0677f, 1.1063f, 0.4581f, 0.4683f, 0.4541f, 0.6140f, 1.5338f, 0.4860f, 0.8229f, 1.9091f, 2.0425f, 0.8486f, 0.9438f, 0.8681f, 0.5505f, 0.5538f, 0.4611f, 0.8981f, 1.3045f, 1.7191f, 1.7324f, 0.4441f, 0.9325f, 0.5316f, 1.0561f, 0.4503f, 0.7880f, 1.8036f, 2.3033f, 1.4150f, 0.5814f, 0.8141f, 0.5689f, 2.8224f, 0.5475f, 0.9876f, 0.7414f, 1.8334f, 1.2784f, 0.6497f, 1.5044f, 1.1134f, 0.7643f, 0.4596f, 0.5796f, 0.8649f, 1.4049f, 1.7056f, 0.5706f, 0.7375f, 0.4644f, 0.7198f, 0.4467f, 0.4200f, 1.1587f, 1.3768f, 1.2845f, 0.6303f, 1.1232f, 0.5890f, 0.5913f, 0.5948f, 0.4542f, 1.2709f, 1.2578f, 0.6704f, 0.2816f, 0.9686f, 0.4394f, 0.3180f, 0.3045f, 0.3791f, 0.5622f, 1.0398f, 0.3577f, 0.3403f, 1.3478f, 0.3243f, 0.4946f, 0.3639f, 0.5964f, 0.6863f, 0.4461f, 0.3519f, 0.3601f, 0.3194f, 0.2833f, 1.4922f, 0.3219f, 0.4722f, 1.3555f, 1.1172f, 0.3929f, 0.6170f, 0.4552f, 0.2921f, 1.5643f, 0.3121f, 0.5036f, 0.8011f, 1.0812f, 1.1873f, 0.2311f, 0.5153f, 0.2381f, 1.8893f, 0.4052f, 0.3862f, 0.8452f, 1.2391f, 0.9191f, 0.2864f, 0.4013f, 0.4350f, 1.8674f, 0.4518f, 0.4057f, 0.5638f, 1.3649f, 1.0232f, 0.4062f, 1.3901f, 0.3364f, 1.9654f, 0.4633f, 0.3805f, 0.6875f, 1.4031f, 1.3275f, 0.4451f, 0.3655f, 0.4121f, 0.9432f, 0.3729f, 0.3386f, 0.9507f, 0.9718f, 1.1001f, 0.4023f, 0.9283f, 0.4826f, 0.3892f, 0.4999f, 0.6325f, 1.2317f, 1.3921f, 0.6479f, 0.2595f, 1.1910f, 0.4302f, 0.5800f, 0.3545f, 0.4885f, 0.5516f, 0.9170f, 0.4122f, 0.3735f, 1.5464f, 0.3242f, 0.4811f, 0.3753f, 0.6361f, 0.7447f, 0.4589f, 0.3084f, 0.3415f, 0.4271f, 0.3444f, 1.4048f, 0.4198f, 0.4934f, 1.4932f, 1.2396f, 0.3891f, 0.6476f, 0.7651f, 0.3275f, 1.6376f, 0.4131f, 0.5584f, 0.8708f, 1.2613f, 0.8935f, 0.3718f, 0.6248f, 0.3523f, 1.5543f, 0.3389f, 0.5583f, 0.8330f, 1.3438f, 0.9425f, 0.3438f, 0.6021f, 0.4130f, 1.8092f, 0.3356f, 0.7709f, 0.3927f, 1.2402f, 1.0947f, 0.4953f, 1.7222f, 0.4484f, 2.3836f, 0.4183f, 0.3244f, 0.4581f, 1.2701f, 1.3131f, 0.4025f, 0.8653f, 0.4059f, 0.9082f, 0.3557f, 0.2537f, 0.8092f, 0.8162f, 0.8332f, 0.3171f, 1.0596f, 0.4527f, 0.4344f, 0.4171f, 0.5638f, 1.4017f, 1.7848f, 1.0403f, 0.3658f, 1.1972f, 0.5011f, 0.4103f, 0.4382f, 0.5364f, 0.7168f, 1.1953f, 0.5331f, 0.3758f, 1.4545f, 0.5106f, 0.4577f, 0.5054f, 0.6541f, 0.7660f, 0.5387f, 0.5111f, 0.5986f, 0.6406f, 0.3960f, 1.3338f, 0.3937f, 0.6118f, 1.5820f, 1.6683f, 0.6987f, 0.6870f, 0.6484f, 0.4215f, 1.5279f, 0.3578f, 0.6673f, 1.0178f, 1.5878f, 1.8907f, 0.4463f, 0.7030f, 0.3774f, 1.4863f, 0.3809f, 0.7300f, 1.0826f, 1.6066f, 1.5959f, 0.4487f, 0.6790f, 0.4581f, 1.7108f, 0.5198f, 0.7082f, 0.5554f, 1.6508f, 1.4150f, 0.6902f, 1.5304f, 0.3789f, 1.6852f, 0.5279f, 0.5162f, 0.7905f, 1.4247f, 1.8631f, 0.5665f, 0.7112f, 0.4692f, 0.8620f, 0.4077f, 0.2937f, 0.9549f, 1.2741f, 1.3725f, 0.5204f, 1.1143f, 0.5439f, 0.4627f, 0.5982f, 0.7716f, 1.5117f, 1.4028f, 0.8404f, 0.3252f, 1.1978f, 0.4843f, 0.4715f, 0.3479f, 0.6271f, 0.8247f, 1.4812f, 0.5215f, 0.5057f, 1.5529f, 0.3746f, 0.3230f, 0.4310f, 0.9674f, 1.0706f, 0.8632f, 0.3052f, 0.5355f, 0.4467f, 0.3495f, 1.0181f, 0.4357f, 0.6519f, 1.7523f, 1.7002f, 0.5006f, 0.5852f, 0.5504f, 0.3485f, 1.8351f, 0.3886f, 0.7694f, 1.1510f, 1.6956f, 1.6931f, 0.3422f, 0.5282f, 0.2789f, 1.6585f, 0.4631f, 0.9389f, 1.3992f, 1.7524f, 1.2287f, 0.3438f, 0.5221f, 0.4933f, 1.3438f, 0.6357f, 0.9644f, 0.9379f, 2.1465f, 1.3589f, 0.5873f, 1.7420f, 0.4445f, 2.6047f, 0.4781f, 0.7034f, 1.2120f, 2.1162f, 1.9292f, 0.6643f, 0.3771f, 0.3920f, 0.6947f, 0.5536f, 0.2945f, 1.3516f, 1.6605f, 2.2111f, 0.5827f, 1.0707f, 0.4860f, 0.4013f, 0.5858f, 0.9471f, 1.7304f, 1.7252f, 1.0746f, 0.4586f, 1.2057f, 0.5455f, 0.6264f, 0.4787f, 0.7342f, 1.1273f, 1.4435f, 0.6547f, 0.6117f, 1.4652f, 0.4464f, 0.4071f, 0.4515f, 0.7332f, 1.3228f, 0.8412f, 0.4030f, 0.5554f, 0.5831f, 0.5031f, 1.2590f, 0.4184f, 0.5832f, 1.9465f, 1.6485f, 0.9096f, 0.8072f, 0.8740f, 0.5607f, 1.5291f, 0.4103f, 0.6605f, 1.2362f, 1.4992f, 1.6057f, 0.6882f, 0.7093f, 0.5494f, 1.2653f, 0.3755f, 0.7134f, 1.2600f, 1.9188f, 1.4182f, 0.5355f, 0.8839f, 0.5416f, 1.1987f, 0.3674f, 1.1449f, 1.0359f, 1.3348f, 2.0194f, 0.6547f, 1.6783f, 0.5777f, 2.0401f, 0.3719f, 0.4657f, 0.9064f, 2.0459f, 2.0055f, 0.6610f, 1.0020f, 0.5647f, 0.7446f, 0.5148f, 0.4080f, 1.3056f, 1.1625f, 1.2576f, 0.6265f, 1.1356f, 0.5631f, 0.6081f, 0.5942f, 0.7670f, 1.4974f, 1.4898f, 0.8165f, 0.3467f, 0.9100f, 0.7097f, 0.4833f, 0.4418f, 0.6126f, 0.7639f, 1.4320f, 0.5205f, 0.5877f, 1.3373f, 0.4653f, 0.4688f, 0.3973f, 1.0156f, 0.9370f, 0.6775f, 0.5610f, 0.7028f, 0.4989f, 0.4736f, 1.1674f, 0.4155f, 0.7201f, 1.8655f, 1.5301f, 0.8077f, 0.8035f, 0.7402f, 0.5338f, 1.2266f, 0.3549f, 0.8868f, 0.9797f, 1.5824f, 1.3311f, 0.3396f, 0.6417f, 0.3986f, 1.6891f, 0.2641f, 0.6078f, 1.2139f, 1.5231f, 1.1568f, 0.4550f, 0.6847f, 0.5507f, 1.8548f, 0.4323f, 0.8931f, 0.7832f, 2.0786f, 1.2933f, 0.5301f, 1.3421f, 0.4928f, 2.0860f, 0.5800f, 0.7320f, 0.8322f, 1.7008f, 1.4081f, 0.5414f, 0.5312f, 0.4766f, 0.8867f, 0.4346f, 0.4344f, 1.0139f, 1.6163f, 1.6613f, 0.6524f, 0.9664f, 0.6014f, 0.5368f, 0.5358f, 0.9383f, 1.2840f, 1.4212f, 0.7861f, 0.5514f, 1.0247f, 0.4681f, 0.4920f, 0.4303f, 0.6944f, 0.9888f, 1.6485f, 0.6349f, 0.4789f, 1.3592f, 0.3530f, 0.4572f, 0.5142f, 0.7495f, 0.8644f, 1.0696f, 0.4379f, 0.3593f, 0.5616f, 0.3474f, 1.0983f, 0.6145f, 0.3981f, 2.0080f, 1.8190f, 0.9065f, 0.6303f, 0.7157f, 0.3918f, 1.2281f, 0.6281f, 0.6625f, 1.1729f, 2.1428f, 1.8821f, 0.4470f, 0.6624f, 0.3586f, 1.3746f, 0.5734f, 0.6692f, 1.1691f, 2.0232f, 1.5245f, 0.5095f, 0.6105f, 0.4225f, 1.4622f, 0.7341f, 0.9489f, 0.4909f, 2.1219f, 1.5750f, 0.6362f, 1.3271f, 0.2953f, 2.3688f, 0.6174f, 0.5163f, 0.5824f, 1.8732f, 1.7988f, 0.5071f, 0.6417f, 0.3378f, 0.7637f, 0.5295f, 0.2190f, 0.9082f, 1.4160f, 1.4393f, 0.5185f, 0.8915f, 0.5259f, 0.4640f, 0.5962f, 3.5598f, 3.0360f, 3.8687f, 3.6549f, 3.0726f, 4.1627f, 2.6764f, 2.9509f, 3.1988f, 3.1987f, 2.7339f, 3.4437f, 3.2767f, 3.2892f, 5.0510f, 2.0442f, 2.1268f, 2.6724f, 3.5524f, 4.0393f, 3.3949f, 3.5234f, 2.7683f, 3.1113f, 2.9385f, 4.2045f, 2.7402f, 2.9360f, 3.4631f, 3.3765f, 3.0728f, 2.7657f, 3.0930f, 3.0596f, 3.8868f, 2.6316f, 3.4266f, 2.6715f, 3.6169f, 3.9546f, 2.6529f, 3.2033f, 2.6137f, 3.3784f, 2.4956f, 3.2140f, 3.4041f, 3.1518f, 3.2380f, 2.6299f, 3.5970f, 2.0260f, 3.5930f, 2.5043f, 3.7847f, 2.9875f, 2.6531f, 3.5038f, 2.3403f, 4.8719f, 2.4978f, 4.9424f, 2.9194f, 3.4506f, 3.3364f, 3.2369f, 3.8384f, 2.7682f, 3.6036f, 2.7278f, 2.7564f, 2.7057f, 2.7938f, 2.9211f, 3.1411f, 3.6404f, 3.2030f, 3.8890f, 2.5397f, 2.7910f, 2.6398f, 1.7693f, 2.5442f, 2.7227f, 2.1520f, 1.6242f, 3.0506f, 1.6368f, 1.4555f, 1.5760f, 1.5585f, 2.2842f, 3.3045f, 1.8335f, 1.3421f, 3.8902f, 1.7613f, 1.5595f, 2.0307f, 1.8946f, 1.7325f, 1.5849f, 1.4334f, 1.8630f, 2.6481f, 1.8305f, 3.0342f, 1.4388f, 1.9798f, 1.3396f, 1.9537f, 1.3073f, 1.5380f, 2.3897f, 1.4415f, 2.9751f, 1.6592f, 2.3570f, 1.2034f, 1.7078f, 1.7985f, 1.4076f, 2.1710f, 1.2353f, 2.6689f, 1.5494f, 1.7279f, 1.6250f, 1.7827f, 2.5370f, 2.0320f, 1.7460f, 1.2584f, 2.6085f, 1.6254f, 2.5172f, 1.6406f, 2.1419f, 2.0450f, 1.3355f, 3.4215f, 1.1644f, 4.3451f, 1.7627f, 1.8257f, 2.6988f, 2.3289f, 2.5848f, 1.7287f, 2.1582f, 1.5648f, 1.7765f, 1.9906f, 1.5971f, 1.9539f, 2.3931f, 2.9004f, 1.9493f, 3.2272f, 1.8969f, 2.1337f, 1.9123f, 1.8341f, 4.9110f, 3.4868f, 2.0997f, 1.8634f, 3.2591f, 1.5882f, 1.8891f, 2.0783f, 2.1773f, 2.1150f, 4.8907f, 2.5837f, 1.8193f, 3.8253f, 1.3392f, 1.4751f, 1.6446f, 2.5087f, 4.6773f, 2.3937f, 2.3041f, 1.9102f, 2.6234f, 1.4115f, 2.8909f, 1.2634f, 2.4891f, 3.3783f, 4.1474f, 2.0224f, 2.1223f, 2.6214f, 1.5463f, 2.9183f, 1.2934f, 2.7260f, 3.4066f, 3.6811f, 2.4777f, 2.3924f, 2.5058f, 1.8364f, 2.9044f, 1.5549f, 2.0234f, 2.9513f, 4.2067f, 2.2036f, 1.8533f, 2.4436f, 1.6976f, 3.3803f, 1.3705f, 2.4697f, 3.5420f, 4.2228f, 3.4200f, 1.8109f, 3.9955f, 1.2519f, 4.2156f, 1.7442f, 2.9023f, 3.9072f, 3.8903f, 3.1994f, 3.2952f, 2.4225f, 2.1790f, 1.7285f, 1.8526f, 1.2402f, 5.3494f, 3.1148f, 4.4442f, 2.0037f, 3.3782f, 1.7144f, 2.1139f, 2.0936f, 2.3397f, 4.6526f, 3.3039f, 2.8535f, 2.1156f, 3.5508f, 2.0554f, 1.8883f, 1.8183f, 1.8425f, 1.4930f, 3.4418f, 1.8357f, 2.1881f, 4.2560f, 1.6247f, 1.5353f, 1.8766f, 1.6459f, 4.0411f, 2.4652f, 2.3909f, 2.2401f, 2.7958f, 1.7113f, 3.0879f, 1.5718f, 1.3646f, 1.6901f, 2.6012f, 1.7727f, 1.8591f, 2.6086f, 2.0417f, 3.3304f, 1.5825f, 1.4848f, 2.2010f, 2.2472f, 1.9354f, 2.3577f, 2.1450f, 1.9019f, 2.5887f, 1.8676f, 1.3605f, 3.5246f, 2.9288f, 2.0961f, 2.1272f, 3.0529f, 2.0171f, 3.2619f, 2.0367f, 2.9900f, 4.3475f, 3.2545f, 2.8705f, 2.5228f, 4.5746f, 1.8035f, 4.9093f, 2.1103f, 2.7988f, 3.6753f, 3.9717f, 4.0953f, 3.6038f, 2.9722f, 2.4621f, 2.1792f, 2.5807f, 1.7915f, 3.9835f, 3.1847f, 3.8661f, 2.4278f, 3.6584f, 1.7128f, 2.4167f, 2.3689f, 2.3789f, 3.6565f, 3.0548f, 3.2938f, 2.0129f, 3.2057f, 1.2805f, 1.4806f, 1.7074f, 1.7313f, 2.3912f, 3.8346f, 2.5965f, 1.5068f, 3.6709f, 1.4149f, 1.4719f, 1.5027f, 1.8734f, 3.4108f, 2.9012f, 1.9678f, 1.9337f, 2.4546f, 1.4204f, 2.7231f, 1.1964f, 1.5172f, 1.9083f, 1.4657f, 1.0573f, 1.1516f, 2.3472f, 1.8638f, 3.3513f, 1.3884f, 1.8503f, 1.7981f, 1.3028f, 0.8321f, 1.7545f, 1.8918f, 1.4953f, 2.9446f, 1.4403f, 1.0889f, 2.0158f, 2.2073f, 1.8100f, 1.5719f, 2.1633f, 1.5958f, 3.4400f, 1.6324f, 1.9054f, 2.8218f, 2.8666f, 2.2426f, 1.7311f, 3.7165f, 1.2305f, 5.3293f, 1.5130f, 2.1107f, 3.7202f, 3.0941f, 2.8460f, 3.9992f, 2.3866f, 1.9281f, 2.1645f, 2.0300f, 1.4469f, 2.7205f, 2.7377f, 3.6538f, 1.8601f, 3.2112f, 1.7247f, 1.9059f, 1.9076f, 1.6533f, 1.7174f, 2.7163f, 1.5872f, 1.2246f, 2.4303f, 1.1985f, 1.1280f, 1.3561f, 1.1974f, 1.7390f, 1.9428f, 1.3761f, 1.4106f, 3.3377f, 1.2258f, 1.1152f, 1.2653f, 1.6517f, 1.9180f, 1.0227f, 1.3161f, 1.4985f, 1.7401f, 1.3392f, 2.5706f, 0.9396f, 1.5586f, 0.9926f, 1.7170f, 1.0179f, 1.4701f, 1.7669f, 1.2991f, 2.8045f, 1.2176f, 1.8059f, 0.8351f, 1.1689f, 1.9912f, 1.3213f, 1.7367f, 1.3753f, 2.4738f, 0.9424f, 1.2329f, 1.2344f, 1.1225f, 3.1183f, 1.6667f, 2.1469f, 1.0371f, 2.9429f, 1.3021f, 2.0369f, 1.1899f, 1.8197f, 2.7864f, 1.7392f, 3.1561f, 0.9817f, 4.0353f, 1.1896f, 1.6461f, 1.8316f, 1.3525f, 2.6936f, 2.2300f, 1.5073f, 1.2165f, 1.1946f, 1.3620f, 1.0266f, 1.1241f, 2.3340f, 2.6638f, 1.2619f, 2.4981f, 1.3772f, 1.4104f, 1.3221f, 2.4555f, 5.3835f, 3.3432f, 2.4037f, 1.8892f, 3.1932f, 1.2783f, 1.8450f, 1.5663f, 1.5194f, 2.1085f, 2.7786f, 2.4623f, 1.4853f, 3.6255f, 0.8892f, 0.9107f, 1.4328f, 1.8372f, 2.7286f, 4.1707f, 1.5766f, 1.8650f, 2.2752f, 1.2382f, 3.2062f, 1.4676f, 1.7362f, 3.7518f, 1.5755f, 1.8534f, 1.1834f, 1.7910f, 1.7630f, 3.7171f, 1.7517f, 1.7642f, 3.0404f, 1.7850f, 1.1809f, 1.4631f, 1.7381f, 1.7001f, 2.7033f, 1.7268f, 1.4402f, 2.8247f, 2.5993f, 1.8116f, 1.8670f, 2.5664f, 1.4797f, 3.3744f, 1.9056f, 2.0429f, 3.2195f, 2.1462f, 3.1802f, 1.9428f, 4.1058f, 1.7647f, 5.3017f, 1.6167f, 1.3842f, 3.6240f, 3.6517f, 3.9470f, 2.1839f, 1.2701f, 1.6064f, 1.2547f, 2.1960f, 1.4060f, 4.4680f, 2.1896f, 3.2750f, 1.3883f, 3.2716f, 1.7178f, 2.3181f, 1.8003f },
        };
        float ndvalue[1]{ 0.4142f };
        __hcpe3_cache_re_eval(1, (char*)ndindex, (char*)ndlogits, (char*)ndvalue, 0.9f, 0.9f, 0.9f, 0.99f, 3);
    }

    // create cache
    __hcpe3_create_cache("out.cache");

    // 再評価したcache読み込み
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
    EXPECT_EQ(1.17314f / 2 * 0.1f + 0.5633f * 0.9f, buf.body.value);
    EXPECT_EQ(0.5f, buf.body.result);
    EXPECT_EQ(1, buf.body.count);
    EXPECT_EQ(3, candidate_num1);
    EXPECT_EQ(move_2g2f, buf.candidates[0].move16);
    EXPECT_NEAR(0.609726f, buf.candidates[0].prob, 0.0001f);
    EXPECT_EQ(move_9g9f, buf.candidates[1].move16);
    EXPECT_NEAR(0.0125f, buf.candidates[1].prob, 0.0001f);
    EXPECT_EQ(move_7g7f, buf.candidates[2].move16);
    EXPECT_NEAR(0.377774f, buf.candidates[2].prob, 0.0001f);

    // pos2
    auto pos2 = cache_pos[1];
    const size_t candidate_num2 = ((cache_pos[2] - pos2) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
    cache.seekg(pos2, std::ios_base::beg);
    cache.read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num2);
    EXPECT_EQ(pos[1].toHuffmanCodedPos(), buf.body.hcp);
    EXPECT_EQ(0.586415410f / 1 * 0.1f + 0.4181f * 0.9f, buf.body.value);
    EXPECT_EQ(0.0f, buf.body.result);
    EXPECT_EQ(1, buf.body.count);
    EXPECT_EQ(3, candidate_num2);
    EXPECT_EQ(move_8c8d, buf.candidates[0].move16);
    EXPECT_NEAR(0.08f, buf.candidates[0].prob, 0.0001f);
    EXPECT_EQ(move_4a3b, buf.candidates[1].move16);
    EXPECT_NEAR(0.02f, buf.candidates[1].prob, 0.0001f);
    EXPECT_EQ(move_3c3d, buf.candidates[2].move16);
    EXPECT_NEAR(0.9f, buf.candidates[2].prob, 0.0001f);

    // pos3
    auto pos3 = cache_pos[2];
    const size_t candidate_num3 = ((cache_pos[3] - pos3) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
    cache.seekg(pos3, std::ios_base::beg);
    cache.read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num3);
    EXPECT_EQ(pos[2].toHuffmanCodedPos(), buf.body.hcp);
    EXPECT_EQ(0.594411194f / 1 * 0.1f + 0.4142f * 0.9f, buf.body.value);
    EXPECT_EQ(1.0f, buf.body.result);
    EXPECT_EQ(1, buf.body.count);
    EXPECT_EQ(4, candidate_num3);
    EXPECT_EQ(move_3c3d, buf.candidates[0].move16);
    EXPECT_NEAR(0.512647212f, buf.candidates[0].prob, 0.0001f);
    EXPECT_EQ(move_8b3b, buf.candidates[1].move16);
    EXPECT_NEAR(0.1f, buf.candidates[1].prob, 0.0001f);
    EXPECT_EQ(move_8c8d, buf.candidates[2].move16);
    EXPECT_NEAR(0.311340958f, buf.candidates[2].prob, 0.0001f);
    EXPECT_EQ(move_4a3b, buf.candidates[3].move16);
    EXPECT_NEAR(0.0760118142f, buf.candidates[3].prob, 0.0001f);
}

TEST(Hcpe3Test, stat_hecpe3_cache) {
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    __hcpe3_load_cache(R"(F:\hcpe3\test_re_eval.cache)");
    __hcpe3_stat_cache();
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

#ifdef NYUGYOKU_FEATURES
TEST(NyugyokuFeaturesTest, make_input_features) {
    // 入玉特徴量
    initTable();
    Position pos;

    // case 1
    {
        pos.set("8l/+S8/1P4+Sp1/K4p+B1p/PNG1g2G1/2P2P2P/6+n+rk/3r5/L2+lP4 b G2SN7Pbnl2p 1");


        features1_t features1{};
        features2_t features2{};
        make_input_features(pos, features1, features2);

        float data2[MAX_FEATURES2_NUM];
        for (size_t i = 0; i < MAX_FEATURES2_NUM; ++i) data2[i] = *((float*)features2 + (size_t)SquareNum * i);

        // 先手持ち駒
        float *begin = data2, *end = data2 + MAX_HPAWN_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 1, 1, 1, 1, 1, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HLANCE_NUM;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HKNIGHT_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HSILVER_NUM;
        EXPECT_EQ((std::vector<float>{1, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HGOLD_NUM;
        EXPECT_EQ((std::vector<float>{1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_HBISHOP_NUM;
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
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
        EXPECT_EQ((std::vector<float>{0, 0}), (std::vector<float>{begin, end}));
        // 王手
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{0, }), (std::vector<float>{begin, end}));
        // 先手入玉特徴量
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{0, }), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_OPP_FIELD;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_SCORE;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
        // 後手入玉特徴量
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{1, }), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_OPP_FIELD;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_SCORE;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}), (std::vector<float>{begin, end}));
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
        // 後手入玉特徴量
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{1, }), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_OPP_FIELD;
        EXPECT_EQ((std::vector<float>{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_SCORE;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}), (std::vector<float>{begin, end}));
        // 先手入玉特徴量
        begin = end, end = begin + 1;
        EXPECT_EQ((std::vector<float>{1, }), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_OPP_FIELD;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
        begin = end, end = begin + MAX_NYUGYOKU_SCORE;
        EXPECT_EQ((std::vector<float>{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}), (std::vector<float>{begin, end}));
    }
}
#endif

TEST(PythonModuleTest, hcpe3_clean) {
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    __hcpe3_clean(R"(F:\hcpe3\selfplay_pre55-012_book_policy_po5000-01_broken.hcpe3)", R"(R:\cleaned.hcpe3)");
}

TEST(DfPn, dfpn_bug_neighborcheck) {
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    Position pos;
    DfPn dfpn;
    dfpn.init();

    // 探索中の1手詰めの局面で開き王手を近接王手と誤って判定するため、証明駒が正しく設定されず、不詰みが詰みになる
    // moveGivesNeighborCheckを修正
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R1Lk2+l b NL4Pgl3p 1");
        dfpn.set_max_search_node(1746);
        bool mate = dfpn.dfpn(pos);
        EXPECT_EQ(false, mate);
    }
}

TEST(DfPn, dfpn_neighbor_check) {
    extern bool moveGivesNeighborCheck(const Position & pos, const Move & move);

    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();

    Position pos;

    // 直接王手
    // 歩
    {
        pos.set("2+P5+N/1k3P1B1/p2sg2p+P/1Pp2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R5+l b GN2L3Pl3p 1");
        const Move move = usiToMove(pos, "8d8c");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 香車
    {
        pos.set("2+P5+N/1k3P1B1/pp1sg2p+P/1Lp2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R5+l b GNL4Pl2p 1");
        const Move move = usiToMove(pos, "8d8c");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 桂馬
    {
        pos.set("2+P5+N/1k3P1B1/p2sg2p+P/1pp2s1P1/8R/1N3KP1s/P3+BNN1g/1S4+p1g/+p1R5+l b G2L4Pl2p 1");
        const Move move = usiToMove(pos, "8f7d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 銀
    {
        pos.set("2+P5+N/4sP1B1/pp1Sg2p+P/5s1P1/1k6R/5KP1s/P3+BNN1g/6+p1g/+p1R5+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "6c7d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 金
    {
        pos.set("2+P5+N/1p2sP1B1/pG1Sg2p+P/5s1P1/1k6R/5KP1s/P3+BNN1g/6+p1g/+p1R5+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "8c8d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 角
    {
        pos.set("2+P5+N/1p2sP3/pG1Sg2p+P/5s1P1/1k1B4R/5KP1s/P3+BNN1g/6+p1g/+p1R5+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "6e7d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 飛車
    {
        pos.set("2+P5+N/4sP3/pG1Sg2p+P/5s1P1/1k1B4R/1p3KP1s/P3+BNN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "8i8f");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // と金
    {
        pos.set("8+N/4sP3/pG1Sg2p+P/5s1P1/1k1B4R/5KP1s/Pp+P1+BNN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "7g8f");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 馬
    {
        pos.set("8+N/4sP3/pG1Sg2p+P/3+B1s1P1/1k1B4R/5KP1s/Pp+P2NN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "6d7d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 竜
    {
        pos.set("2G5+N/1k1SsP3/pp2g2p+P/3+B1s1P1/3B4R/5KP1s/P1+P2NN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "8i8c+");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }

    // 遠隔王手
    // 香車
    {
        pos.set("2+P5+N/1k3P1B1/p2sg2p+P/1pp2s1P1/1L6R/5KP1s/P3+BNN1g/1S4+p1g/+p1R5+l b GNL4Pl2p 1");
        const Move move = usiToMove(pos, "8e8d");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
        EXPECT_EQ(false, check);
    }
    // 角
    {
        pos.set("2+P5+N/4sP3/pG1Sg2p+P/5s1P1/1k1p4R/4BKP1s/P3+BNN1g/6+p1g/+p1R5+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "5f6g");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 飛車
    {
        pos.set("2+P5+N/4sP3/pG1Sg2p+P/5s1P1/1k1B4R/5KP1s/Pp2+BNN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "8i8g");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 馬
    {
        pos.set("8+N/3SsP3/pG2g2p+P/3+B1s1P1/1k1B4R/5KP1s/Pp+P2NN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "6d6c");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 竜
    {
        pos.set("1k1G4+N/3SsP3/pp2g2p+P/3+B1s1P1/3B4R/5KP1s/P1+P2NN1g/6+p1g/+pR6+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "8i8c+");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }

    // 開き王手
    // 歩
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R1Pk2+l b GN2L3Pl3p 1");
        const Move move = usiToMove(pos, "5i5h");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 香車
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R1Lk2+l b GNL4Pl3p 7");
        const Move move = usiToMove(pos, "5i5h");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 桂馬
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/1S4+p1g/+p1R1Nk2+l b G2L4Pl3p 1");
        const Move move = usiToMove(pos, "5i6g");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 銀
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/6+p1g/+p1R1Sk2+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "5i4h");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 金
    {
        pos.set("2+P5+N/5P1B1/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/6+p1g/+pSRG1k2+l b N2L4Pl3p 1");
        const Move move = usiToMove(pos, "6i5h");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 角
    {
        pos.set("2+P5+N/5P3/p2sg2p+P/2p2s1P1/8R/5KP1s/P3+BNN1g/6+p1g/+pSR1Bk2+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "5i4h");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 飛車
    {
        pos.set("2+P5+N/5P3/p2sg1Bp+P/2p2s1P1/8R/3+B1KP1s/P3RNN1g/5k+p1g/+pS6+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "5g5i");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 馬
    {
        pos.set("2+P5+N/5P3/p2sg1Bp+P/2p2s1P1/8R/5KP1s/P4NN1g/2R+B1k+p1g/+pS6+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "6h6i");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 竜
    {
        pos.set("2+P5+N/5P3/p2sg1Bp+P/2p2s1P1/2+B5R/3+R1KP1s/P4NN1g/5k+p1g/+pS6+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "6f6g");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
    // 玉
    {
        pos.set("2+P5+N/5P3/p2sg1Bp+P/2p+B1s1P1/8R/6P1s/P4NN1g/2+RK1k+p1g/+pS6+l b GN2L4Pl3p 1");
        const Move move = usiToMove(pos, "6h6g");
        bool check = moveGivesNeighborCheck(pos, move);
        EXPECT_EQ(false, check);
        EXPECT_EQ(true, pos.moveGivesCheck(move));
    }
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

// DfPnテスト
TEST(DfPn, dfpn) {
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(400000);
	dfpn.set_maxdepth(33);

	Position pos;

    vector<string> sfens_mate = {
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
        //"lr1s4l/3k1+P3/p2p4p/1NL2P3/1PpS5/P6pP/K1+b6/NR1P5/7NL w BSN3P4gs4p 166", // mate 33(depath 35, nodes 3000000が必要)
    };
    vector<string> sfens_nomate = {
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
    for (const auto& [sfens, mate] : {
            std::pair<const vector<string>&, bool>{sfens_mate, true},
            std::pair<const vector<string>&, bool>{sfens_nomate, false}
        })
    {
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

            EXPECT_EQ(mate, ret) << sfen;
        }
    }
	auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(total).count();
	cout << total_ns / 1000000.0 << endl;
}

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

// DfPnのPV表示テスト
TEST(DfPn, dfpn_get_pv) {
    initTable();
    Position::initZobrist();

    DfPn dfpn;
    dfpn.init();
    dfpn.set_max_search_node(400000);
    dfpn.set_maxdepth(33);

    Position pos;

    vector<pair<string, string>> sfens = {
        {"l5p1k/3+R1p3/3p1g+N2/p1S1p2S1/1p5Kp/5PP2/PP2P1sPP/2P2GG2/L5+r1L b L2P2bgs3np 1", "L*1c 1a2a 1c1b+ 2a1b 2d2c+ 1b1a 2c2b"},
    };

    auto start0 = std::chrono::system_clock::now();
    auto total = start0 - start0;
    for (const auto& [sfen, pv_truth] : sfens) {
        pos.set(sfen);
        bool ret = dfpn.dfpn(pos);

        EXPECT_EQ(true, ret);

        // pv
        if (ret) {
            std::string pv;
            int depth;
            Move move;
            std::tie(pv, depth, move) = dfpn.get_pv(pos);

            EXPECT_EQ(pv_truth, pv);
        }
    }
}

// DfPnのPV表示テスト
TEST(DfPn, dfpn_get_pv_time) {
	initTable();
	Position::initZobrist();

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(800000);
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

        EXPECT_EQ(true, ret) << sfen;

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


// DfPnのPV表示テスト(ファイルからsfen読み込み)
TEST(DfPn, dfpn_fromfile) {
	initTable();
	Position::initZobrist();
	Position pos;

	std::ifstream ifs(R"(H:\home\2020\12_20\shogi\mate\mate7.sfen)");
    std::ofstream ofs(R"(R:\mate7.txt)");

	DfPn dfpn;
	dfpn.init();
	dfpn.set_max_search_node(400000);
	dfpn.set_maxdepth(7);

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

        ofs << ret << "\t" << dfpn.searchedNode << "\t";
        ofs << time_ms;

		// pv
		if (ret) {
			std::string pv;
			int depth;
			Move move;
			auto start_pv = std::chrono::system_clock::now();
			std::tie(pv, depth, move) = dfpn.get_pv(pos);
			auto end_pv = std::chrono::system_clock::now();

            ofs << "\t" << move.toUSI() << "\t" << pv << "\t" << depth << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_pv - start_pv).count();
		}
        ofs << endl;
	}
	auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total).count();
    ofs << total_ms << endl;
}

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
