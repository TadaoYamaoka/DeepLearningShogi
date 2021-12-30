/*
  Apery, a USI shogi playing engine derived from Stockfish, a UCI chess playing engine.
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Copyright (C) 2011-2018 Hiraoka Takuya

  Apery is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Apery is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.hpp"
#include "init.hpp"
#include "mt64bit.hpp"
#include "book.hpp"
#include "search.hpp"

namespace {
    // square のマスにおける、障害物を調べる必要がある場所を Bitboard で返す。
    // lance の前方だけを調べれば良さそうだけど、Rank2 ~ Rank8 の状態をそのまま index に使いたいので、
    // 縦方向全て(端を除く)の occupied を全て調べる。
    Bitboard lanceBlockMask(const Square square) {
        return squareFileMask(square) & ~(rankMask<Rank9>() | rankMask<Rank1>());
    }

    // lance の利きを返す。
    // occupied  障害物があるマスが 1 の bitboard
    Bitboard lanceAttackCalc(const Color c, const Square square, const Bitboard& occupied) {
        File file = makeFile(square);
        Bitboard bb{ 0, 0 };
        // 上方向
        for (Rank rank = makeRank(square); rank > Rank1;) {
            rank += DeltaN;
            const Square sq = makeSquare(file, rank);
            bb |= setMaskBB(sq);
            if (occupied.isSet(sq))
                break;
        }
        // 下方向
        for (Rank rank = makeRank(square); rank < Rank9;) {
            rank += DeltaS;
            const Square sq = makeSquare(file, rank);
            bb |= setMaskBB(sq);
            if (occupied.isSet(sq))
                break;
        }

        return bb & inFrontMask(c, makeRank(square));
    }

    // index, bits の情報を元にして、occupied の 1 のbit を いくつか 0 にする。
    // index の値を, occupied の 1のbit の位置に変換する。
    // index   [0, 1<<bits) の範囲のindex
    // bits    bit size
    // blockMask   利きのあるマスが 1 のbitboard
    // result  occupied
    Bitboard indexToOccupied(const int index, const int bits, const Bitboard& blockMask) {
        Bitboard tmpBlockMask = blockMask;
        Bitboard result = allZeroBB();
        for (int i = 0; i < bits; ++i) {
            const Square sq = tmpBlockMask.firstOneFromSQ11();
            if (index & (1 << i))
                result.setBit(sq);
        }
        return result;
    }

    // LanceBlockMask, LanceAttack の値を設定する。
    void initLanceAttacks() {
        for (Color c = Black; c < ColorNum; ++c) {
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                const Bitboard blockMask = lanceBlockMask(sq);
                //const int num1s = blockMask.popCount(); // 常に 7
                const int num1s = 7;
                assert(num1s == blockMask.popCount());
                for (int i = 0; i < (1 << num1s); ++i) {
                    Bitboard occupied = indexToOccupied(i, num1s, blockMask);
                    LanceAttack[c][sq][i] = lanceAttackCalc(c, sq, occupied);
                }
            }
        }
    }

    void initRookAttacks() {
        for (File file = File1; file < FileNum; ++file) {
            for (Rank rank = Rank1; rank < RankNum; ++rank) {
                Bitboard left{ 0, 0 }, right{ 0, 0 };

                // SQのマスから左方向
                for (File file2 = (File)(file + 1); file2 < FileNum; ++file2)
                    left |= setMaskBB(makeSquare(file2, rank));

                // SQのマスから右方向
                for (File file2 = (File)(file - 1); file2 >= File1; --file2)
                    right |= setMaskBB(makeSquare(file2, rank));

                Bitboard rightRev = right.byteReverse();

                Bitboard hi, lo;
                Bitboard::unpack(rightRev, left, hi, lo);

                RookAttackRankToMask[makeSquare(file, rank)][0] = lo;
                RookAttackRankToMask[makeSquare(file, rank)][1] = hi;
            }
        }
    }

    void initBishopAttacks() {
        // 4方向
        constexpr SquareDelta bishopDelta[4] = {
            DeltaNW, // 左上
            DeltaSW, // 左下
            DeltaNE, // 右上
            DeltaSE, // 右下
        };
        for (File file = File1; file < FileNum; ++file) {
            for (Rank rank = Rank1; rank < RankNum; ++rank) {
                // 対象升から
                const Square sq = makeSquare(file, rank);

                // 角の左上、左下、右上、右下それぞれへのstep effect
                Bitboard bishopToBB[4];

                // 4方向の利きをループで求める
                for (int i = 0; i < 4; ++i)
                {
                    Bitboard bb{ 0, 0 };

                    const auto delta = bishopDelta[i];
                    // 壁に突き当たるまで進む
                    Square sq2 = sq;
                    while (true) {
                        if ((delta == DeltaNW || delta == DeltaNE) && makeRank(sq2) == Rank1) break;
                        if ((delta == DeltaSW || delta == DeltaSE) && makeRank(sq2) == Rank9) break;
                        if ((delta == DeltaNW || delta == DeltaSW) && makeFile(sq2) == File9) break;
                        if ((delta == DeltaNE || delta == DeltaSE) && makeFile(sq2) == File1) break;
                        sq2 += delta;
                        bb |= setMaskBB(sq2);
                    }

                    bishopToBB[i] = bb;
                }

                // 右上、右下はbyte reverseしておかないとうまく求められない。(先手の香の利きがうまく求められないのと同様)

                bishopToBB[2] = bishopToBB[2].byteReverse();
                bishopToBB[3] = bishopToBB[3].byteReverse();

                for (int i = 0; i < 2; ++i)
                    BishopAttackToMask[sq][i] = Bitboard256(
                        Bitboard(bishopToBB[0].p(i), bishopToBB[2].p(i)),
                        Bitboard(bishopToBB[1].p(i), bishopToBB[3].p(i))
                    );
            }
        }
    }

    void initKingAttacks() {
        for (Square sq = SQ11; sq < SquareNum; ++sq)
            KingAttack[sq] = rookAttack(sq, allOneBB()) | bishopAttack(sq, allOneBB());
    }

    void initGoldAttacks() {
        for (Color c = Black; c < ColorNum; ++c)
            for (Square sq = SQ11; sq < SquareNum; ++sq)
                GoldAttack[c][sq] = (kingAttack(sq) & inFrontMask(c, makeRank(sq))) | rookAttack(sq, allOneBB());
    }

    void initSilverAttacks() {
        for (Color c = Black; c < ColorNum; ++c)
            for (Square sq = SQ11; sq < SquareNum; ++sq)
                SilverAttack[c][sq] = (kingAttack(sq) & inFrontMask(c, makeRank(sq))) | bishopAttack(sq, allOneBB());
    }

    void initKnightAttacks() {
        for (Color c = Black; c < ColorNum; ++c) {
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                KnightAttack[c][sq] = allZeroBB();
                const Bitboard bb = pawnAttack(c, sq);
                if (bb)
                    KnightAttack[c][sq] = bishopStepAttacks(bb.constFirstOneFromSQ11()) & inFrontMask(c, makeRank(sq));
            }
        }
    }

    void initPawnAttacks() {
        for (Color c = Black; c < ColorNum; ++c)
            for (Square sq = SQ11; sq < SquareNum; ++sq)
                PawnAttack[c][sq] = silverAttack(c, sq) ^ bishopAttack(sq, allOneBB());
    }

    void initSquareRelation() {
        for (Square sq1 = SQ11; sq1 < SquareNum; ++sq1) {
            const File file1 = makeFile(sq1);
            const Rank rank1 = makeRank(sq1);
            for (Square sq2 = SQ11; sq2 < SquareNum; ++sq2) {
                const File file2 = makeFile(sq2);
                const Rank rank2 = makeRank(sq2);
                SquareRelation[sq1][sq2] = DirecMisc;
                if (sq1 == sq2) continue;

                if      (file1 == file2)
                    SquareRelation[sq1][sq2] = DirecFile;
                else if (rank1 == rank2)
                    SquareRelation[sq1][sq2] = DirecRank;
                else if (static_cast<int>(rank1 - rank2) == static_cast<int>(file1 - file2))
                    SquareRelation[sq1][sq2] = DirecDiagNESW;
                else if (static_cast<int>(rank1 - rank2) == static_cast<int>(file2 - file1))
                    SquareRelation[sq1][sq2] = DirecDiagNWSE;
            }
        }
    }

    // 障害物が無いときの利きの Bitboard
    // RookAttack, BishopAttack, LanceAttack を設定してから、この関数を呼ぶこと。
    void initAttackToEdge() {
        for (Square sq = SQ11; sq < SquareNum; ++sq) {
            RookAttackToEdge[sq] = rookAttack(sq, allZeroBB());
            BishopAttackToEdge[sq] = bishopAttack(sq, allZeroBB());
            LanceAttackToEdge[Black][sq] = lanceAttack(Black, sq, allZeroBB());
            LanceAttackToEdge[White][sq] = lanceAttack(White, sq, allZeroBB());
        }
    }

    void initBetweenBB() {
        for (Square sq1 = SQ11; sq1 < SquareNum; ++sq1) {
            for (Square sq2 = SQ11; sq2 < SquareNum; ++sq2) {
                BetweenBB[sq1][sq2] = allZeroBB();
                if (sq1 == sq2) continue;
                const Direction direc = squareRelation(sq1, sq2);
                if      (direc & DirecCross)
                    BetweenBB[sq1][sq2] = rookAttack(sq1, setMaskBB(sq2)) & rookAttack(sq2, setMaskBB(sq1));
                else if (direc & DirecDiag)
                    BetweenBB[sq1][sq2] = bishopAttack(sq1, setMaskBB(sq2)) & bishopAttack(sq2, setMaskBB(sq1));
            }
        }
    }

    void initCheckTable() {
        for (Color c = Black; c < ColorNum; ++c) {
            const Color opp = oppositeColor(c);
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                GoldCheckTable[c][sq] = allZeroBB();
                Bitboard checkBB = goldAttack(opp, sq);
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    GoldCheckTable[c][sq] |= goldAttack(opp, checkSq);
                }
                GoldCheckTable[c][sq].andEqualNot(setMaskBB(sq) | goldAttack(opp, sq));
            }
        }

        for (Color c = Black; c < ColorNum; ++c) {
            const Color opp = oppositeColor(c);
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                SilverCheckTable[c][sq] = allZeroBB();

                Bitboard checkBB = silverAttack(opp, sq);
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    SilverCheckTable[c][sq] |= silverAttack(opp, checkSq);
                }
                const Bitboard TRank123BB = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());
                checkBB = goldAttack(opp, sq);
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    // 移動元が敵陣である位置なら、金に成って王手出来る。
                    SilverCheckTable[c][sq] |= (silverAttack(opp, checkSq) & TRank123BB);
                }

                const Bitboard TRank4BB = (c == Black ? rankMask<Rank4>() : rankMask<Rank6>());
                // 移動先が3段目で、4段目に移動したときも、成ることが出来る。
                checkBB = goldAttack(opp, sq) & TRank123BB;
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    SilverCheckTable[c][sq] |= (silverAttack(opp, checkSq) & TRank4BB);
                }
                SilverCheckTable[c][sq].andEqualNot(setMaskBB(sq) | silverAttack(opp, sq));
            }
        }

        for (Color c = Black; c < ColorNum; ++c) {
            const Color opp = oppositeColor(c);
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                KnightCheckTable[c][sq] = allZeroBB();

                Bitboard checkBB = knightAttack(opp, sq);
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    KnightCheckTable[c][sq] |= knightAttack(opp, checkSq);
                }
                const Bitboard TRank123BB = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());
                checkBB = goldAttack(opp, sq) & TRank123BB;
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    KnightCheckTable[c][sq] |= knightAttack(opp, checkSq);
                }
            }
        }

        for (Color c = Black; c < ColorNum; ++c) {
            const Color opp = oppositeColor(c);
            for (Square sq = SQ11; sq < SquareNum; ++sq) {
                LanceCheckTable[c][sq] = lanceAttackToEdge(opp, sq);

                const Bitboard TRank123BB = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());
                Bitboard checkBB = goldAttack(opp, sq) & TRank123BB;
                while (checkBB) {
                    const Square checkSq = checkBB.firstOneFromSQ11();
                    LanceCheckTable[c][sq] |= lanceAttackToEdge(opp, checkSq);
                }
                LanceCheckTable[c][sq].andEqualNot(setMaskBB(sq) | pawnAttack(opp, sq));
            }
        }

		// 歩
		for (Color c = Black; c < ColorNum; ++c) {
			const Color opp = oppositeColor(c);
			for (Square sq = SQ11; sq < SquareNum; ++sq) {
				// 歩で王手になる可能性のあるものは、敵玉から２つ離れた歩(不成での移動) + ksqに敵の金をおいた範囲(enemyGold)に成りで移動できる
				PawnCheckTable[c][sq] = allZeroBB();

				Bitboard checkBB = pawnAttack(opp, sq);
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					PawnCheckTable[c][sq] |= pawnAttack(opp, checkSq);
				}
				const Bitboard TRank123BB = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());
				checkBB = goldAttack(opp, sq) & TRank123BB;
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					PawnCheckTable[c][sq] |= pawnAttack(opp, checkSq);
				}
				PawnCheckTable[c][sq].andEqualNot(setMaskBB(sq));
			}
		}

		// 角
		for (Color c = Black; c < ColorNum; ++c) {
			const Color opp = oppositeColor(c);
			for (Square sq = SQ11; sq < SquareNum; ++sq) {
				BishopCheckTable[c][sq] = allZeroBB();

				Bitboard checkBB = bishopAttack(sq, allZeroBB());
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					BishopCheckTable[c][sq] |= bishopAttack(checkSq, allZeroBB());
				}
				const Bitboard TRank123BB = (c == Black ? inFrontMask<Black, Rank4>() : inFrontMask<White, Rank6>());
				checkBB = kingAttack(sq) & TRank123BB;
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					// 移動先が敵陣 == 成れる == 王の動き
					BishopCheckTable[c][sq] |= bishopAttack(checkSq, allZeroBB());
				}

				checkBB = kingAttack(sq);
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					// 移動元が敵陣 == 成れる == 王の動き
					BishopCheckTable[c][sq] |= bishopAttack(checkSq, allZeroBB()) & TRank123BB;
				}
				BishopCheckTable[c][sq].andEqualNot(setMaskBB(sq));
			}
		}

		// 馬
		for (Color c = Black; c < ColorNum; ++c) {
			const Color opp = oppositeColor(c);
			for (Square sq = SQ11; sq < SquareNum; ++sq) {
				HorseCheckTable[c][sq] = allZeroBB();

				Bitboard checkBB = horseAttack(sq, allZeroBB());
				while (checkBB) {
					const Square checkSq = checkBB.firstOneFromSQ11();
					HorseCheckTable[c][sq] |= horseAttack(checkSq, allZeroBB());
				}
				HorseCheckTable[c][sq].andEqualNot(setMaskBB(sq));
			}
		}
	}

    void initSquareDistance() {
        for (Square sq0 = SQ11; sq0 < SquareNum; ++sq0) {
            for (Square sq1 = SQ11; sq1 < SquareNum; ++sq1) {
                switch (squareRelation(sq0, sq1)) {
                case DirecMisc:
                    // DirecMisc な関係は全て距離 1 にしてもKPE学習には問題無いんだけれど。
                    SquareDistance[sq0][sq1] = 0;
                    if (knightAttack(Black, sq0).isSet(sq1) || knightAttack(White, sq0).isSet(sq1))
                        SquareDistance[sq0][sq1] = 1;
                    break;
                case DirecFile:
                    SquareDistance[sq0][sq1] = abs(static_cast<int>(sq0 - sq1) / static_cast<int>(DeltaN));
                    break;
                case DirecRank:
                    SquareDistance[sq0][sq1] = abs(static_cast<int>(sq0 - sq1) / static_cast<int>(DeltaE));
                    break;
                case DirecDiagNESW:
                    SquareDistance[sq0][sq1] = abs(static_cast<int>(sq0 - sq1) / static_cast<int>(DeltaNE));
                    break;
                case DirecDiagNWSE:
                    SquareDistance[sq0][sq1] = abs(static_cast<int>(sq0 - sq1) / static_cast<int>(DeltaNW));
                    break;
                default: UNREACHABLE;
                }
            }
        }
    }

    void initNeighbor5x5() {
        for (Square sq = SQ11; sq < SquareNum; ++sq) {
            Neighbor5x5Table[sq] = allZeroBB();
            Bitboard toBB = kingAttack(sq);
            while (toBB) {
                const Square to = toBB.firstOneFromSQ11();
                Neighbor5x5Table[sq] |= kingAttack(to);
            }
            Neighbor5x5Table[sq].andEqualNot(setMaskBB(sq));
        }
    }

}

void initTable() {
    initLanceAttacks();
    initRookAttacks();
    initBishopAttacks();
    initKingAttacks();
    initGoldAttacks();
    initSilverAttacks();
    initPawnAttacks();
    initKnightAttacks();
    initSquareRelation();
    initAttackToEdge();
    initBetweenBB();
    initCheckTable();
    initNeighbor5x5();
    initSquareDistance();

    Book::init();
}
