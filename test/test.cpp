#include <iostream>
#include <chrono>

#include "cppshogi.h"

using namespace std;

#if 0
#include "nn.h"
int main() {
	initTable();
	// 入力データ作成
	const int batchsize = 2;
	features1_t features1[batchsize] = {};
	features2_t features2[batchsize] = {};

	Position pos[batchsize];
	pos[0].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", nullptr);
	pos[1].set("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1", nullptr);

	make_input_features(pos[0], features1, features2);
	make_input_features(pos[1], features1 + 1, features2 + 1);

	NN nn(batchsize);

	nn.load_model(R"(H:\src\DeepLearningShogi\dlshogi\model_rl_val_wideresnet10_110_1)");

	float y1[batchsize][MAX_MOVE_LABEL_NUM * SquareNum];
	float y2[batchsize];
	nn.foward(batchsize, features1, features2, (float*)y1, y2);

	for (int i = 0; i < batchsize; i++) {
		// policyの結果
		for (int j = 0; j < MAX_MOVE_LABEL_NUM * SquareNum; j++) {
			cout << y1[i][j] << endl;
		}
		// valueの結果
		cout << y2[i] << endl;

		// 合法手一覧
		std::vector<Move> legal_moves;
		std::vector<float> legal_move_probabilities;
		for (MoveList<Legal> ml(pos[i]); !ml.end(); ++ml) {
			const Move move = ml.move();
			const int move_label = make_move_label((u16)move.proFromAndTo(), pos[i].turn());
			legal_moves.emplace_back(move);
			legal_move_probabilities.emplace_back(y1[i][move_label]);
		}

		// Boltzmann distribution
		softmax_tempature_with_normalize(legal_move_probabilities);

		// print result
		for (int j = 0; j < legal_moves.size(); j++) {
			const Move& move = legal_moves[j];
			const int move_label = make_move_label((u16)move.proFromAndTo(), pos[i].turn());
			cout << move.toUSI() << " logit:" << y1[i][move_label] << " rate:" << legal_move_probabilities[j] << endl;
		}

	}

	return 0;
}
#endif

#if 0
// hcpe作成
#include <fstream>
int main() {
	initTable();
	const int num = 2;
	Position pos[num];
	pos[0].set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", nullptr);
	pos[1].set("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1", nullptr);

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
#endif

#if 0
int main() {
	initTable();
	Position pos;
	//pos.set("lnsgkgsnl/1r7/ppppppbpp/6pP1/9/9/PPPPPPP1P/1B5R1/LNSGKGSNL w - 1", nullptr);
	//pos.set("lnsgkg1nl/1r7/p1pppp1sp/6pP1/1p6B/2P6/PP1PPPP1P/7R1/LNSGKGSNL b Pb 1", nullptr); // dcBB
	pos.set("lnsgkg1nl/1r5s1/pppppp1pp/6p2/b8/2P6/PPNPPPPPP/7R1/L1SGKGSNL b B 1", nullptr); // pinned

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

#if 0
// 王手生成テスト
int main() {
	initTable();
	Position pos;
	//pos.set("lnsgkgsnl/1r5b1/ppppSpppp/9/9/4L4/PPPPPPPPP/1B5R1/LNSGKG1N1 b p 1", nullptr); // 間接王手 銀
	//pos.set("lnsgkgsnl/1r5b1/pppp1pppp/9/9/4N4/PPPPLPPPP/1B5R1/LNSGKGS2 b 2p 1", nullptr); // 間接王手 桂馬
	//pos.set("lnsgkgsnl/1r5b1/ppLpppppp/2p6/B8/9/PPPPpPPPP/7R1/LNSGKGSN1 b - 1", nullptr); // 間接王手 香車
	//pos.set("lnsgkgsnl/1r1P3b1/ppppPPppp/4pp3/9/9/PPP3PPP/1B5R1/LNSGKGSNL b - 1", nullptr); // 歩が成って王手
	pos.set("lnsg1gsnl/1r1P3b1/ppppk1ppp/5P3/4Pp3/4p4/PPP3PPP/1B5R1/LNSGKGSNL b - 1", nullptr); // 歩が成って王手

	// 王手生成
	for (MoveList<Check> ml(pos); !ml.end(); ++ml) {
		std::cout << ml.move().toUSI() << std::endl;
	}

	return 0;
}
#endif

#if 1
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
		pos.set(sfen, nullptr);
		auto start = std::chrono::system_clock::now();
		bool ret = mateMoveInOddPly(pos, 7);
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
// cuDNN推論テスト
#include "nn.h"

// 検証用
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB
#include <boost/python/numpy.hpp>
namespace py = boost::python;
namespace np = boost::python::numpy;
py::object dlshogi_predict;

int main() {
	// Boost.PythonとBoost.Numpyの初期化
	Py_Initialize();
	np::initialize();
	// Pythonモジュール読み込み
	py::object dlshogi_ns = py::import("dlshogi.predict").attr("__dict__");
	// modelロード
	py::object dlshogi_load_model = dlshogi_ns["load_model"];
	dlshogi_load_model(R"(H:\src\DeepLearningShogi\dlshogi\model_rl_val_wideresnet10_110_1)");
	// 予測関数取得
	dlshogi_predict = dlshogi_ns["predict"];

	initTable();
	// 入力データ作成
	const int batchsize = 20;
	features1_t* features1 = new features1_t[batchsize];
	features2_t* features2 = new features2_t[batchsize];
	// set all zero
	std::fill_n((float*)features1, batchsize * (int)ColorNum * MAX_FEATURES1_NUM * (int)SquareNum, 0.0f);
	std::fill_n((float*)features2, batchsize * MAX_FEATURES2_NUM * (int)SquareNum, 0.0f);

	vector<Position> pos(batchsize);
	pos[0].set("l5snl/4ggkb1/1rn2p2p/p1ps2pp1/1p1pp3P/P1PP1PPP1/1PBSP1NS1/3RG1GK1/LN6L b - 1 : 372 : 1 : 6f6e", nullptr);
	pos[1].set("l4pk2/4+B1s1g/3+N3p1/p+B7/2P2PL2/3P1GPK1/PS2+p2P1/1p1+n5/L5+sN1 w R2GSN7Prl 1 : -5828 : -1 : L*2d", nullptr);
	pos[2].set("1+Bs3+P+R1/l2kg2p1/2+B1p1p1p/p1pp1p3/9/2P1P1P2/PP1PGP2P/9/LNSGK1SNL w RGS2NLP 1 : -9493 : -1 : 6b7c", nullptr);
	pos[3].set("l3p2+R1/1Ss1gr2l/1pnp2p1p/kss2P3/8P/pG3+B2L/1PBP1GP1N/L1K1G4/1N7 w 6Pnp 1 : -5163 : -1 : N*9e", nullptr);
	pos[4].set("ln1l1R3/2k6/1p5p+P/p1pp+S1p2/6b2/2P1P4/PP1P1PSPL/1+bSGKR3/LN1GG2N1 b GS4Pn 1 : 6923 : 1 : 3g4f", nullptr);
	pos[5].set("ln4b1l/r2n5/3p1S3/ppp1pP2p/3SPkpl1/P1P5P/1PSP5/1KGB2+n2/LNG5+r b 2Gs4p 1 : 3503 : 1 : G*4f", nullptr);
	pos[6].set("lns3g2/2kgrs1bl/p1pppp1pp/1p7/5n3/2P3P2/PPSPPP1PP/L2G2R2/1N1K1GSNL b BP 1 : 920 : 1 : 3i2h", nullptr);
	pos[7].set("ln1gp+P2+R/2kss4/1pp2pp1l/p2pP3p/2P4+b1/P1S3P2/1PGP+p1G2/1K2+p4/LN5N+b w RGSLPn 1 : -1307 : -1 : 1i2i", nullptr);
	pos[8].set("l4kbnl/1r1g2s2/2Pp3pp/p2snpp2/2p1p1PP1/1P4N2/PS1GP1S1P/1B3R2L/L1K4N1 w 2P2gp 1 : 1715 : 1 : P*8e", nullptr);
	pos[9].set("2g+bk1+Bn1/1s2g3l/5P1p1/3Pp1p1p/1+R3p1P1/2P1N4/pKS1P1P1P/5S1R1/LN5NL w 2GS4Plp 1 : -6691 : -1 : L*4a", nullptr);
	pos[10].set("ln1gk1gn1/8l/2p1s2+Rp/pr1+bp1+P2/3p2p2/1l6P/N1PPPS2N/2GKS+bP2/L1S2G3 w 5Pp 1 : -986 : -1 : 4h4g", nullptr);
	pos[11].set("+Bn4gnl/2g1ssk2/2+N2p1p1/prSPpb2p/1p7/2p1P3P/PP2GPPP1/3+p1SK2/LR1G3NL b 2Pl 1 : -585 : -1 : 6d6c+", nullptr);
	pos[12].set("lnB3knl/1rs1g1gs1/p3p2pp/2pp1pp2/1p7/2PP1P1P1/PPS1P1P1P/2GKS2R1/LN3G1NL w b 1 : 724 : 1 : 8b8c", nullptr);
	pos[13].set("1n1rg2n1/1Bs2k3/lpp1psgpl/p2p2p1p/1P3R3/2P1S4/P2PP1PPP/2GS5/LN1K2GNL b Pbp 1 : 485 : 1 : 4e4i", nullptr);
	pos[14].set("ln7/3+R5/1ppSpgS1p/k8/b2P1Pp2/2S1P1P2/B1PKG+p2P/7S1/LN1G3NL b G4Prnl2p 1 : 4563 : 1 : G*8e", nullptr);
	pos[15].set("ln7/1r3g2G/2p1pksp1/5pp1P/2Pp2lP1/p1n1P3+b/1l2B4/2+r6/1N2K2N1 b 2g3sl7p 1 : -9389 : -1 : 1d1c+", nullptr);
	pos[16].set("1n5+P1/lks1r1p2/1p2gn3/1l1pg2+B1/2Pspp3/p1G3P2/1PNS1PG2/L1KS4+p/L3R4 b BN4P2p 1 : 3016 : 1 : 7g6e", nullptr);
	pos[17].set("1ns1k1s+B1/lrg3g2/pppppp1p1/6p2/9/2P5p/PPNPPPPPP/2SK2SRL/+b2GG2N1 w NLl 1 : -739 : -1 : L*7d", nullptr);
	pos[18].set("6pn1/4g3l/3kpp1+Bp/pN1p1bP2/1sP1P3P/2+n4P1/1P1P1PN1L/4GSS2/L+p+r1GK3 w Grsl3p 1 : 2989 : 1 : L*5f", nullptr);
	pos[19].set("1n1gk1snl/lrs2g3/pp2ppppb/2pp4p/9/4PP1P1/PPPP2P1P/LBKSG1R2/1N3GSNL b - 1 : 60 : -1 : 1g1f", nullptr);

	for (int i = 0; i < batchsize; i++) {
		make_input_features(pos[i], features1 + i, features2 + i);
	}

	NN nn(batchsize);

	nn.load_model(R"(H:\src\DeepLearningShogi\dlshogi\model_rl_val_wideresnet10_110_1)");

	float y1[batchsize][MAX_MOVE_LABEL_NUM * SquareNum];
	float y2[batchsize];
	nn.foward(batchsize, features1, features2, (float*)y1, y2);


	// predict
	np::ndarray ndfeatures1 = np::from_data(
		features1,
		np::dtype::get_builtin<float>(),
		py::make_tuple(batchsize, (int)ColorNum * MAX_FEATURES1_NUM, 9, 9),
		py::make_tuple(sizeof(float)*(int)ColorNum*MAX_FEATURES1_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
		py::object());
	np::ndarray ndfeatures2 = np::from_data(
		features2,
		np::dtype::get_builtin<float>(),
		py::make_tuple(batchsize, MAX_FEATURES2_NUM, 9, 9),
		py::make_tuple(sizeof(float)*MAX_FEATURES2_NUM * 81, sizeof(float) * 81, sizeof(float) * 9, sizeof(float)),
		py::object());
	auto ret = dlshogi_predict(ndfeatures1, ndfeatures2);
	py::tuple ret_list = py::extract<py::tuple>(ret);
	np::ndarray y1_data = py::extract<np::ndarray>(ret_list[0]);
	np::ndarray y2_data = py::extract<np::ndarray>(ret_list[1]);
	float(*logits)[MAX_MOVE_LABEL_NUM * SquareNum] = reinterpret_cast<float(*)[MAX_MOVE_LABEL_NUM * SquareNum]>(y1_data.get_data());
	float *value = reinterpret_cast<float*>(y2_data.get_data());

	for (int i = 0; i < batchsize; i++) {
		// policyの結果
		for (int j = 0; j < MAX_MOVE_LABEL_NUM * SquareNum; j++) {
			cout << y1[i][j] << "\t" << logits[i][j] << endl;
		}
		// valueの結果
		cout << y2[i] << "\t" << value[i] << endl;
	}

	return 0;
}
#endif
