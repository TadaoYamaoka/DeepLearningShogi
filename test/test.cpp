#include <iostream>
#include <chrono>

#include "cppshogi.h"

using namespace std;

#if 0
// hcpe作成
#include <fstream>
int main() {
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
#endif

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

#if 0
// 王手生成テスト
int main() {
	initTable();
	Position pos;
	//pos.set("lnsgkgsnl/1r5b1/ppppSpppp/9/9/4L4/PPPPPPPPP/1B5R1/LNSGKG1N1 b p 1"); // 間接王手 銀
	//pos.set("lnsgkgsnl/1r5b1/pppp1pppp/9/9/4N4/PPPPLPPPP/1B5R1/LNSGKGS2 b 2p 1"); // 間接王手 桂馬
	//pos.set("lnsgkgsnl/1r5b1/ppLpppppp/2p6/B8/9/PPPPpPPPP/7R1/LNSGKGSN1 b - 1"); // 間接王手 香車
	//pos.set("lnsgkgsnl/1r1P3b1/ppppPPppp/4pp3/9/9/PPP3PPP/1B5R1/LNSGKGSNL b - 1"); // 歩が成って王手
	pos.set("lnsg1gsnl/1r1P3b1/ppppk1ppp/5P3/4Pp3/4p4/PPP3PPP/1B5R1/LNSGKGSNL b - 1"); // 歩が成って王手

	// 王手生成
	for (MoveList<Check> ml(pos); !ml.end(); ++ml) {
		std::cout << ml.move().toUSI() << std::endl;
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
	//dfpn.set_max_search_node(1000000);
	dfpn.set_maxdepth(31);

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

		cout << ret << "\t";
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
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/1P6L/K1+p4+r1/LN3P1+r1 w SN2P2snl4p 258", 264 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/1+p5+r1/LN3P1+r1 w SN2P2snl4p 260", 264 }, // 262手目で詰み → 詰み
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 263 }, // 262手目で詰み → 持将棋
		{ "2+Bl1k3/1p1p3+P1/5b1N1/1gp2gp2/1gPP5/P2GS1P2/KP6L/L6+r1/1N3P1+r1 w SN3P2snl4p 262", 264 }, // 262手目で詰み → 詰み
		{ "+P2+Rb1gnl/7k1/n1p1+Bp1pp/p5p2/1p1pP2P1/2+s6/PsNGSPP1P/3KG4/L5RNL b SL3Pg 83", 83 + 11 }, // 11手で詰み → 持将棋
		{ "+P2+Rb1gnl/7k1/n1p1+Bp1pp/p5p2/1p1pP2P1/2+s6/PsNGSPP1P/3KG4/L5RNL b SL3Pg 83", 83 + 12 }, // 11手で詰み → 詰み
	};

	for (auto& sfen_draw : sfens) {
		pos.set(sfen_draw.first);
		dfpn.set_maxdepth(sfen_draw.second - pos.gamePly());
		bool ret = dfpn.dfpn(pos);
		cout << ret << endl;
	}

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
	// make_hcpe_by_self_play --threashold 1 --threads 1 --usi_engine E:\game\shogi\apery_wcsc28\bin\apery_wcsc28_bmi2.exe --usi_engine_num 1 --usi_threads 1 --usi_options USI_Ponder:False,Threads:1,Byoyomi_Margin:0 F:\model\model_rl_val_wideresnet10_selfplay_236 R:\hcp\black_win.hcp R:\hcpe 1 800 0 1

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

#if 1
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
