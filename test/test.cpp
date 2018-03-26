#include <iostream>
#include <chrono>

#include "cppshogi.h"

using namespace std;

#if 1
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
	pos.set("lnsgkgsnl/1r5b1/ppLpppppp/2p6/B8/9/PPPPpPPPP/7R1/LNSGKGSN1 b - 1", nullptr); // 間接王手 香車

	// 王手生成
	for (MoveList<Check> ml(pos); !ml.end(); ++ml) {
		std::cout << ml.move().toUSI() << std::endl;
	}

	return 0;
}
#endif