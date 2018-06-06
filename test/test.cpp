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
	Position pos;

	vector<string> sfens = {
		// 詰み
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