﻿/*
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
    // square のマスにおける、障害物を調べる必要がある場所を調べて Bitboard で返す。
    Bitboard rookBlockMaskCalc(const Square square) {
        Bitboard result = squareFileMask(square) ^ squareRankMask(square);
        if (makeFile(square) != File9) result &= ~fileMask<File9>();
        if (makeFile(square) != File1) result &= ~fileMask<File1>();
        if (makeRank(square) != Rank9) result &= ~rankMask<Rank9>();
        if (makeRank(square) != Rank1) result &= ~rankMask<Rank1>();
        return result;
    }

    // square のマスにおける、障害物を調べる必要がある場所を調べて Bitboard で返す。
    Bitboard bishopBlockMaskCalc(const Square square) {
        const Rank rank = makeRank(square);
        const File file = makeFile(square);
        Bitboard result = allZeroBB();
        for (Square sq = SQ11; sq < SquareNum; ++sq) {
            const Rank r = makeRank(sq);
            const File f = makeFile(sq);
            if (abs(rank - r) == abs(file - f))
                result.setBit(sq);
        }
        result &= ~(rankMask<Rank9>() | rankMask<Rank1>() | fileMask<File9>() | fileMask<File1>());
        result.clearBit(square);

        return result;
    }

    // square のマスにおける、障害物を調べる必要がある場所を Bitboard で返す。
    // lance の前方だけを調べれば良さそうだけど、Rank2 ~ Rank8 の状態をそのまま index に使いたいので、
    // 縦方向全て(端を除く)の occupied を全て調べる。
    Bitboard lanceBlockMask(const Square square) {
        return squareFileMask(square) & ~(rankMask<Rank9>() | rankMask<Rank1>());
    }

    // Rook or Bishop の利きの範囲を調べて bitboard で返す。
    // occupied  障害物があるマスが 1 の bitboard
    Bitboard attackCalc(const Square square, const Bitboard& occupied, const bool isBishop) {
        const SquareDelta deltaArray[2][4] = {{DeltaN, DeltaS, DeltaE, DeltaW}, {DeltaNE, DeltaSE, DeltaSW, DeltaNW}};
        Bitboard result = allZeroBB();
        for (SquareDelta delta : deltaArray[isBishop]) {
            for (Square sq = square + delta;
                 isInSquare(sq) && abs(makeRank(sq - delta) - makeRank(sq)) <= 1;
                 sq += delta)
            {
                result.setBit(sq);
                if (occupied.isSet(sq))
                    break;
            }
        }

        return result;
    }

    // lance の利きを返す。
    // 香車の利きは常にこれを使っても良いけど、もう少し速くする為に、テーブル化する為だけに使う。
    // occupied  障害物があるマスが 1 の bitboard
    Bitboard lanceAttackCalc(const Color c, const Square square, const Bitboard& occupied) {
        return rookAttack(square, occupied) & inFrontMask(c, makeRank(square));
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

    void initAttacks(const bool isBishop)
    {
        auto* attacks     = (isBishop ? BishopAttack      : RookAttack     );
        auto* attackIndex = (isBishop ? BishopAttackIndex : RookAttackIndex);
        auto* blockMask   = (isBishop ? BishopBlockMask   : RookBlockMask  );
        auto* shift       = (isBishop ? BishopShiftBits   : RookShiftBits  );
#if defined HAVE_BMI2
#else
        auto* magic       = (isBishop ? BishopMagic       : RookMagic      );
#endif
        int index = 0;
        for (Square sq = SQ11; sq < SquareNum; ++sq) {
            blockMask[sq] = (isBishop ? bishopBlockMaskCalc(sq) : rookBlockMaskCalc(sq));
            attackIndex[sq] = index;

            const int num1s = (isBishop ? BishopBlockBits[sq] : RookBlockBits[sq]);
            for (int i = 0; i < (1 << num1s); ++i) {
                const Bitboard occupied = indexToOccupied(i, num1s, blockMask[sq]);
#if defined HAVE_BMI2
                attacks[index + occupiedToIndex(occupied & blockMask[sq], blockMask[sq])] = attackCalc(sq, occupied, isBishop);
#else
                attacks[index + occupiedToIndex(occupied, magic[sq], shift[sq])] = attackCalc(sq, occupied, isBishop);
#endif
            }
            index += 1 << (64 - shift[sq]);
        }
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
    initAttacks(false);
    initAttacks(true);
    initKingAttacks();
    initGoldAttacks();
    initSilverAttacks();
    initPawnAttacks();
    initKnightAttacks();
    initLanceAttacks();
    initSquareRelation();
    initAttackToEdge();
    initBetweenBB();
    initCheckTable();
    initNeighbor5x5();
    initSquareDistance();

    Book::init();
}

#if defined FIND_MAGIC
// square の位置の rook, bishop それぞれのMagic Bitboard に使用するマジックナンバーを見つける。
// isBishop  : true なら bishop, false なら rook のマジックナンバーを見つける。
u64 findMagic(const Square square, const bool isBishop) {
    Bitboard occupied[1<<14];
    Bitboard attack[1<<14];
    Bitboard attackUsed[1<<14];
    Bitboard mask = (isBishop ? bishopBlockMaskCalc(square) : rookBlockMaskCalc(square));
    int num1s = (isBishop ? BishopBlockBits[square] : RookBlockBits[square]);

    // n bit の全ての数字 (利きのあるマスの全ての 0 or 1 の組み合わせ)
    for (int i = 0; i < (1 << num1s); ++i) {
        occupied[i] = indexToOccupied(i, num1s, mask);
        attack[i] = attackCalc(square, occupied[i], isBishop);
    }

    for (u64 k = 0; k < UINT64_C(100000000); ++k) {
        const u64 magic = g_mt64bit.randomFewBits();
        bool fail = false;

        // これは無くても良いけど、少しマジックナンバーが見つかるのが早くなるはず。
        if (count1s((mask.merge() * magic) & UINT64_C(0xfff0000000000000)) < 6)
            continue;

        std::fill(std::begin(attackUsed), std::end(attackUsed), allZeroBB());

        for (int i = 0; !fail && i < (1 << num1s); ++i) {
            const int shiftBits = (isBishop ? BishopShiftBits[square] : RookShiftBits[square]);
            const u64 index = occupiedToIndex(occupied[i], magic, shiftBits);
            if      (attackUsed[index] == allZeroBB())
                attackUsed[index] = attack[i];
            else if (attackUsed[index] != attack[i])
                fail = true;
        }
        if (!fail)
            return magic;
    }

    std::cout << "/***Failed***/\t";
    return 0;
}
#endif // #if defined FIND_MAGIC
