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

#include "generateMoves.hpp"
#include "usi.hpp"

namespace {
	// 角, 飛車の場合
	template <MoveType MT, PieceType PT, Color US, bool ALL>
	FORCE_INLINE ExtMove* generateBishopOrRookMoves(ExtMove* moveList, const Position& pos,
													const Bitboard& target, const Square /*ksq*/)
	{
		Bitboard fromBB = pos.bbOf(PT, US);
		while (fromBB) {
			const Square from = fromBB.firstOneFromSQ11();
			const bool fromCanPromote = canPromote(US, makeRank(from));
			Bitboard toBB = pos.attacksFrom<PT>(US, from) & target;
			FOREACH_BB(toBB, const Square to, {
				const bool toCanPromote = canPromote(US, makeRank(to));
				if (fromCanPromote | toCanPromote) {
					(*moveList++).move = makePromoteMove<MT>(PT, from, to, pos);
					if (/*MT == NonEvasion || */ALL)
						(*moveList++).move = makeNonPromoteMove<MT>(PT, from, to, pos);
				}
				else // 角、飛車は成れるなら成り、不成は生成しない。
					(*moveList++).move = makeNonPromoteMove<MT>(PT, from, to, pos);
			});
		}
		return moveList;
	}

	// 駒打ちの場合
	// 歩以外の持ち駒は、loop の前に持ち駒の種類の数によって switch で展開している。
	// ループの展開はコードが膨れ上がる事によるキャッシュヒット率の低下と、演算回数のバランスを取って決める必要がある。
	// NPSに影響が出ないならシンプルにした方が良さそう。
	template <Color US>
	ExtMove* generateDropMoves(ExtMove* moveList, const Position& pos, const Bitboard& target) {
		const Hand hand = pos.hand(US);
		// まず、歩に対して指し手を生成
		if (hand.exists<HPawn>()) {
			Bitboard toBB = target;
			// 一段目には打てない
			const Rank TRank1 = (US == Black ? Rank1 : Rank9);
			toBB.andEqualNot(rankMask<TRank1>());

			// 二歩の回避
			Bitboard pawnsBB = pos.bbOf(Pawn, US);
			Square pawnsSquare;
			foreachBB(pawnsBB, pawnsSquare, [&](const int part) {
					toBB.set(part, toBB.p(part) & ~squareFileMask(pawnsSquare).p(part));
				});

			// 打ち歩詰めの回避
			const Rank TRank9 = (US == Black ? Rank9 : Rank1);
			const SquareDelta TDeltaS = (US == Black ? DeltaS : DeltaN);

			const Square ksq = pos.kingSquare(oppositeColor(US));
			// 相手玉が九段目なら、歩で王手出来ないので、打ち歩詰めを調べる必要はない。
			if (makeRank(ksq) != TRank9) {
				const Square pawnDropCheckSquare = ksq + TDeltaS;
				assert(isInSquare(pawnDropCheckSquare));
				if (toBB.isSet(pawnDropCheckSquare) && pos.piece(pawnDropCheckSquare) == Empty) {
					if (!pos.isPawnDropCheckMate(US, pawnDropCheckSquare))
						// ここで clearBit だけして MakeMove しないことも出来る。
						// 指し手が生成される順番が変わり、王手が先に生成されるが、後で問題にならないか?
						(*moveList++).move = makeDropMove(Pawn, pawnDropCheckSquare);
					toBB.xorBit(pawnDropCheckSquare);
				}
			}

			Square to;
			FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Pawn, to);
				});
		}

		// 歩 以外の駒を持っているか
		if (hand.exceptPawnExists()) {
			PieceType haveHand[6]; // 歩以外の持ち駒。vector 使いたいけど、速度を求めるので使わない。
			int haveHandNum = 0; // 持ち駒の駒の種類の数

			// 桂馬、香車、それ以外の順番で格納する。(駒を打てる位置が限定的な順)
			if (hand.exists<HKnight>()) haveHand[haveHandNum++] = Knight;
			const int noKnightIdx      = haveHandNum; // 桂馬を除く駒でループするときのループの初期値
			if (hand.exists<HLance >()) haveHand[haveHandNum++] = Lance;
			const int noKnightLanceIdx = haveHandNum; // 桂馬, 香車を除く駒でループするときのループの初期値
			if (hand.exists<HSilver>()) haveHand[haveHandNum++] = Silver;
			if (hand.exists<HGold  >()) haveHand[haveHandNum++] = Gold;
			if (hand.exists<HBishop>()) haveHand[haveHandNum++] = Bishop;
			if (hand.exists<HRook  >()) haveHand[haveHandNum++] = Rook;

			const Rank TRank2 = (US == Black ? Rank2 : Rank8);
			const Rank TRank1 = (US == Black ? Rank1 : Rank9);
			const Bitboard TRank2BB = rankMask<TRank2>();
			const Bitboard TRank1BB = rankMask<TRank1>();

			Bitboard toBB;
			Square to;
			// 桂馬、香車 以外の持ち駒があれば、
			// 一段目に対して、桂馬、香車以外の指し手を生成。
			switch (haveHandNum - noKnightLanceIdx) {
			case 0: break; // 桂馬、香車 以外の持ち駒がない。
			case 1: toBB = target & TRank1BB; FOREACH_BB(toBB, to, { Unroller<1>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightLanceIdx + i], to); }); }); break;
			case 2: toBB = target & TRank1BB; FOREACH_BB(toBB, to, { Unroller<2>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightLanceIdx + i], to); }); }); break;
			case 3: toBB = target & TRank1BB; FOREACH_BB(toBB, to, { Unroller<3>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightLanceIdx + i], to); }); }); break;
			case 4: toBB = target & TRank1BB; FOREACH_BB(toBB, to, { Unroller<4>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightLanceIdx + i], to); }); }); break;
			default: UNREACHABLE;
			}

			// 桂馬以外の持ち駒があれば、
			// 二段目に対して、桂馬以外の指し手を生成。
			switch (haveHandNum - noKnightIdx) {
			case 0: break; // 桂馬 以外の持ち駒がない。
			case 1: toBB = target & TRank2BB; FOREACH_BB(toBB, to, { Unroller<1>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightIdx + i], to); }); }); break;
			case 2: toBB = target & TRank2BB; FOREACH_BB(toBB, to, { Unroller<2>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightIdx + i], to); }); }); break;
			case 3: toBB = target & TRank2BB; FOREACH_BB(toBB, to, { Unroller<3>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightIdx + i], to); }); }); break;
			case 4: toBB = target & TRank2BB; FOREACH_BB(toBB, to, { Unroller<4>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightIdx + i], to); }); }); break;
			case 5: toBB = target & TRank2BB; FOREACH_BB(toBB, to, { Unroller<5>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[noKnightIdx + i], to); }); }); break;
			default: UNREACHABLE;
			}

			// 一、二段目以外に対して、全ての持ち駒の指し手を生成。
			toBB = target & ~(TRank2BB | TRank1BB);
			switch (haveHandNum) {
			case 0: assert(false); break; // 最適化の為のダミー
			case 1: FOREACH_BB(toBB, to, { Unroller<1>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			case 2: FOREACH_BB(toBB, to, { Unroller<2>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			case 3: FOREACH_BB(toBB, to, { Unroller<3>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			case 4: FOREACH_BB(toBB, to, { Unroller<4>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			case 5: FOREACH_BB(toBB, to, { Unroller<5>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			case 6: FOREACH_BB(toBB, to, { Unroller<6>()([&](const int i) { (*moveList++).move = makeDropMove(haveHand[i], to); }); }); break;
			default: UNREACHABLE;
			}
		}

		return moveList;
	}

	// 金, 成り金、馬、竜の指し手生成
	template <MoveType MT, PieceType PT, Color US, bool ALL> struct GeneratePieceMoves {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			static_assert(PT == GoldHorseDragon, "");
			// 金、成金、馬、竜のbitboardをまとめて扱う。
			Bitboard fromBB = (pos.goldsBB() | pos.bbOf(Horse, Dragon)) & pos.bbOf(US);
			while (fromBB) {
				const Square from = fromBB.firstOneFromSQ11();
				// from にある駒の種類を判別
				const PieceType pt = pieceToPieceType(pos.piece(from));
				Bitboard toBB = pos.attacksFrom(pt, US, from) & target;
				FOREACH_BB(toBB, const Square to, {
					(*moveList++).move = makeNonPromoteMove<MT>(pt, from, to, pos);
				});
			}
			return moveList;
		}
	};
	// 歩の場合
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Pawn, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			// Txxx は先手、後手の情報を吸収した変数。数字は先手に合わせている。
			const Rank TRank4 = (US == Black ? Rank4 : Rank6);
			const Bitboard TRank123BB = inFrontMask<US, TRank4>();
			const SquareDelta TDeltaS = (US == Black ? DeltaS : DeltaN);

			Bitboard toBB = pawnAttack<US>(pos.bbOf(Pawn, US)) & target;

			// 成り
			if (MT != NonCaptureMinusPro) {
				Bitboard toOn123BB = toBB & TRank123BB;
				if (toOn123BB) {
					toBB.andEqualNot(TRank123BB);
					Square to;
					FOREACH_BB(toOn123BB, to, {
							const Square from = to + TDeltaS;
							(*moveList++).move = makePromoteMove<MT>(Pawn, from, to, pos);
							if (/*MT == NonEvasion || */ALL) {
								const Rank TRank1 = (US == Black ? Rank1 : Rank9);
								if (makeRank(to) != TRank1)
									(*moveList++).move = makeNonPromoteMove<MT>(Pawn, from, to, pos);
							}
						});
				}
			}
			else
				assert(!(target & TRank123BB));

			// 残り(不成)
			// toBB は 8~4 段目まで。
			Square to;
			FOREACH_BB(toBB, to, {
					const Square from = to + TDeltaS;
					(*moveList++).move = makeNonPromoteMove<MT>(Pawn, from, to, pos);
				});
			return moveList;
		}
	};
	// 香車の場合
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Lance, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			Bitboard fromBB = pos.bbOf(Lance, US);
			while (fromBB) {
				const Square from = fromBB.firstOneFromSQ11();
				Bitboard toBB = pos.attacksFrom<Lance>(US, from) & target;
				do {
					if (toBB) {
						// 駒取り対象は必ず一つ以下なので、toBB のビットを 0 にする必要がない。
						const Square to = (MT == Capture || MT == CapturePlusPro ? toBB.constFirstOneFromSQ11() : toBB.firstOneFromSQ11());
						const bool toCanPromote = canPromote(US, makeRank(to));
						if (toCanPromote) {
							(*moveList++).move = makePromoteMove<MT>(Lance, from, to, pos);
							if (/*MT == NonEvasion || */ALL) {
								if (isBehind<US, Rank1, Rank9>(makeRank(to))) // 1段目の不成は省く
									(*moveList++).move = makeNonPromoteMove<MT>(Lance, from, to, pos);
							}
							else if (MT != NonCapture && MT != NonCaptureMinusPro) { // 駒を取らない3段目の不成を省く
								if (isBehind<US, Rank2, Rank8>(makeRank(to))) // 2段目の不成を省く
									(*moveList++).move = makeNonPromoteMove<MT>(Lance, from, to, pos);
							}
						}
						else
							(*moveList++).move = makeNonPromoteMove<MT>(Lance, from, to, pos);
					}
					// 駒取り対象は必ず一つ以下なので、loop は不要。最適化で do while が無くなると良い。
				} while (!(MT == Capture || MT == CapturePlusPro) && toBB);
			}
			return moveList;
		}
	};
	// 桂馬の場合
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Knight, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			Bitboard fromBB = pos.bbOf(Knight, US);
			while (fromBB) {
				const Square from = fromBB.firstOneFromSQ11();
				Bitboard toBB = pos.attacksFrom<Knight>(US, from) & target;
				FOREACH_BB(toBB, const Square to, {
					const bool toCanPromote = canPromote(US, makeRank(to));
					if (toCanPromote) {
						(*moveList++).move = makePromoteMove<MT>(Knight, from, to, pos);
						if (isBehind<US, Rank2, Rank8>(makeRank(to))) // 1, 2段目の不成は省く
							(*moveList++).move = makeNonPromoteMove<MT>(Knight, from, to, pos);
					}
					else
						(*moveList++).move = makeNonPromoteMove<MT>(Knight, from, to, pos);
				});
			}
			return moveList;
		}
	};
	// 銀の場合
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Silver, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			Bitboard fromBB = pos.bbOf(Silver, US);
			while (fromBB) {
				const Square from = fromBB.firstOneFromSQ11();
				const bool fromCanPromote = canPromote(US, makeRank(from));
				Bitboard toBB = pos.attacksFrom<Silver>(US, from) & target;
				FOREACH_BB(toBB, const Square to, {
					const bool toCanPromote = canPromote(US, makeRank(to));
					if (fromCanPromote | toCanPromote)
						(*moveList++).move = makePromoteMove<MT>(Silver, from, to, pos);
					(*moveList++).move = makeNonPromoteMove<MT>(Silver, from, to, pos);
				});
			}
			return moveList;
		}
	};
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Bishop, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square ksq) {
			return generateBishopOrRookMoves<MT, Bishop, US, ALL>(moveList, pos, target, ksq);
		}
	};
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, Rook, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square ksq) {
			return generateBishopOrRookMoves<MT, Rook, US, ALL>(moveList, pos, target, ksq);
		}
	};
	// 玉の場合
	// 必ず盤上に 1 枚だけあることを前提にすることで、while ループを 1 つ無くして高速化している。
	template <MoveType MT, Color US, bool ALL> struct GeneratePieceMoves<MT, King, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos, const Bitboard& target, const Square /*ksq*/) {
			const Square from = pos.kingSquare(US);
			Bitboard toBB = pos.attacksFrom<King>(US, from) & target;
			FOREACH_BB(toBB, const Square to, {
				(*moveList++).move = makeNonPromoteMove<MT>(King, from, to, pos);
			});
			return moveList;
		}
	};

	// pin は省かない。
	FORCE_INLINE ExtMove* generateRecaptureMoves(ExtMove* moveList, const Position& pos, const Square to, const Color us) {
		Bitboard fromBB = pos.attackersTo(us, to);
		while (fromBB) {
			const Square from = fromBB.firstOneFromSQ11();
			const PieceType pt = pieceToPieceType(pos.piece(from));
			switch (pt) {
			case Empty    : assert(false); break; // 最適化の為のダミー
			case Pawn     : case Lance    : case Knight   : case Silver   : case Bishop   : case Rook     :
				(*moveList++).move = ((canPromote(us, makeRank(to)) || canPromote(us, makeRank(from))) ?
									  makePromoteMove<Capture>(pt, from, to, pos) :
									  makeNonPromoteMove<Capture>(pt, from, to, pos));
				break;
			case Gold     : case King     : case ProPawn  : case ProLance : case ProKnight: case ProSilver: case Horse    : case Dragon   :
				(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
				break;
			default       : UNREACHABLE;
			}
		}
		return moveList;
	}

	// 指し手生成 functor
	// テンプレート引数が複数あり、部分特殊化したかったので、関数ではなく、struct にした。
	// ALL == true のとき、歩、飛、角の不成、香の2段目の不成、香の3段目の駒を取らない不成も生成する。
	template <MoveType MT, Color US, bool ALL = false> struct GenerateMoves {
		ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			static_assert(MT == Capture || MT == NonCapture || MT == CapturePlusPro || MT == NonCaptureMinusPro, "");
			// Txxx は先手、後手の情報を吸収した変数。数字は先手に合わせている。
			const Rank TRank4 = (US == Black ? Rank4 : Rank6);
			const Rank Trank3 = (US == Black ? Rank3 : Rank7);
			const Rank TRank2 = (US == Black ? Rank2 : Rank8);
			const Bitboard TRank123BB = inFrontMask<US, TRank4>();
			const Bitboard TRank4_9BB = inFrontMask<oppositeColor(US), Trank3>();

			const Bitboard targetPawn =
				(MT == Capture           ) ? pos.bbOf(oppositeColor(US))                                             :
				(MT == NonCapture        ) ? pos.emptyBB()                                                           :
				(MT == CapturePlusPro    ) ? pos.bbOf(oppositeColor(US)) | (pos.occupiedBB().notThisAnd(TRank123BB)) :
				(MT == NonCaptureMinusPro) ? pos.occupiedBB().notThisAnd(TRank4_9BB)                                 :
				allOneBB(); // error
			const Bitboard targetOther =
				(MT == Capture           ) ? pos.bbOf(oppositeColor(US)) :
				(MT == NonCapture        ) ? pos.emptyBB()               :
				(MT == CapturePlusPro    ) ? pos.bbOf(oppositeColor(US)) :
				(MT == NonCaptureMinusPro) ? pos.emptyBB()               :
				allOneBB(); // error
			const Square ksq = pos.kingSquare(oppositeColor(US));

			moveList = GeneratePieceMoves<MT, Pawn           , US, ALL>()(moveList, pos, targetPawn, ksq);
			moveList = GeneratePieceMoves<MT, Lance          , US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, Knight         , US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, Silver         , US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, Bishop         , US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, Rook           , US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, GoldHorseDragon, US, ALL>()(moveList, pos, targetOther, ksq);
			moveList = GeneratePieceMoves<MT, King           , US, ALL>()(moveList, pos, targetOther, ksq);

			return moveList;
		}
	};

	// 部分特殊化
	// 駒打ち生成
	template <Color US> struct GenerateMoves<Drop, US> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			const Bitboard target = pos.emptyBB();
			moveList = generateDropMoves<US>(moveList, pos, target);
			return moveList;
		}
	};

	// checkSq にある駒で王手されたとき、玉はその駒の利きの位置には移動できないので、移動できない位置を bannnedKingToBB に格納する。
	// 両王手のときには二度連続で呼ばれるため、= ではなく |= を使用している。
	// 最初に呼ばれたときは、bannedKingToBB == allZeroBB() である。
	// todo: FOECE_INLINE と template 省いてNPS比較
	template <Color THEM>
	FORCE_INLINE void makeBannedKingTo(Bitboard& bannedKingToBB, const Position& pos,
									   const Square checkSq, const Square ksq)
	{
		switch (pos.piece(checkSq)) {
//      case Empty: assert(false); break; // 最適化の為のダミー
		case (THEM == Black ? BPawn      : WPawn):
		case (THEM == Black ? BKnight    : WKnight):
			// 歩、桂馬で王手したときは、どこへ逃げても、その駒で取られることはない。
			// よって、ここでは何もしない。
			assert(
				pos.piece(checkSq) == (THEM == Black ? BPawn   : WPawn) ||
				pos.piece(checkSq) == (THEM == Black ? BKnight : WKnight)
				);
		break;
		case (THEM == Black ? BLance     : WLance):
			bannedKingToBB |= lanceAttackToEdge(THEM, checkSq);
			break;
		case (THEM == Black ? BSilver    : WSilver):
			bannedKingToBB |= silverAttack(THEM, checkSq);
			break;
		case (THEM == Black ? BGold      : WGold):
		case (THEM == Black ? BProPawn   : WProPawn):
		case (THEM == Black ? BProLance  : WProLance):
		case (THEM == Black ? BProKnight : WProKnight):
		case (THEM == Black ? BProSilver : WProSilver):
			bannedKingToBB |= goldAttack(THEM, checkSq);
		break;
		case (THEM == Black ? BBishop    : WBishop):
			bannedKingToBB |= bishopAttackToEdge(checkSq);
			break;
		case (THEM == Black ? BHorse     : WHorse):
			bannedKingToBB |= horseAttackToEdge(checkSq);
			break;
		case (THEM == Black ? BRook      : WRook):
			bannedKingToBB |= rookAttackToEdge(checkSq);
			break;
		case (THEM == Black ? BDragon    : WDragon):
			if (squareRelation(checkSq, ksq) & DirecDiag) {
				// 斜めから王手したときは、玉の移動先と王手した駒の間に駒があることがあるので、
				// dragonAttackToEdge(checkSq) は使えない。
				bannedKingToBB |= pos.attacksFrom<Dragon>(checkSq);
			}
			else {
				bannedKingToBB |= dragonAttackToEdge(checkSq);
			}
			break;
		default:
			UNREACHABLE;
		}
	}

	// 部分特殊化
	// 王手回避生成
	// 王手をしている駒による王手は避けるが、
	// 玉の移動先に敵の利きがある場合と、pinされている味方の駒を動かした場合、非合法手を生成する。
	// そのため、pseudo legal である。
	template <Color US, bool ALL> struct GenerateMoves<Evasion, US, ALL> {
		/*FORCE_INLINE*/ ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			assert(pos.isOK());
			assert(pos.inCheck());

			const Square ksq = pos.kingSquare(US);
			constexpr Color Them = oppositeColor(US);
			const Bitboard checkers = pos.checkersBB();
			Bitboard bb = checkers;
			Bitboard bannedKingToBB = allZeroBB();
			int checkersNum = 0;
			Square checkSq;

			// 玉が逃げられない位置の bitboard を生成する。
			// 絶対に王手が掛かっているので、while ではなく、do while
			do {
				checkSq = bb.firstOneFromSQ11();
				assert(pieceToColor(pos.piece(checkSq)) == Them);
				++checkersNum;
				makeBannedKingTo<Them>(bannedKingToBB, pos, checkSq, ksq);
			} while (bb);

			// 玉が移動出来る移動先を格納。
			bb = bannedKingToBB.notThisAnd(pos.bbOf(US).notThisAnd(kingAttack(ksq)));
			FOREACH_BB(bb, const Square to, {
				// 移動先に相手駒の利きがあるか調べずに指し手を生成する。
				// attackersTo() が重いので、movePicker か search で合法手か調べる。
				(*moveList++).move = makeNonPromoteMove<Capture>(King, ksq, to, pos);
			});

			// 両王手なら、玉を移動するしか回避方法は無い。
			// 玉の移動は生成したので、ここで終了
			if (1 < checkersNum)
				return moveList;

			// 王手している駒を玉以外で取る手の生成。
			// pin されているかどうかは movePicker か search で調べる。
			const Bitboard target1 = betweenBB(checkSq, ksq);
			const Bitboard target2 = target1 | checkers;
			moveList = GeneratePieceMoves<Evasion, Pawn,   US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, Lance,  US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, Knight, US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, Silver, US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, Bishop, US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, Rook,   US, ALL>()(moveList, pos, target2, ksq);
			moveList = GeneratePieceMoves<Evasion, GoldHorseDragon,   US, ALL>()(moveList, pos, target2, ksq);

			if (target1)
				moveList = generateDropMoves<US>(moveList, pos, target1);

			return moveList;
		}
	};

	// 部分特殊化
	// 王手が掛かっていないときの指し手生成
	// これには、玉が相手駒の利きのある地点に移動する自殺手と、pin されている駒を動かす自殺手を含む。
	// ここで生成した手は pseudo legal
	template <Color US> struct GenerateMoves<NonEvasion, US> {
		/*FORCE_INLINE*/ ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			Bitboard target = pos.emptyBB();

			moveList = generateDropMoves<US>(moveList, pos, target);
			target |= pos.bbOf(oppositeColor(US));
			const Square ksq = pos.kingSquare(oppositeColor(US));

			moveList = GeneratePieceMoves<NonEvasion, Pawn           , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, Lance          , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, Knight         , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, Silver         , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, Bishop         , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, Rook           , US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, GoldHorseDragon, US, false>()(moveList, pos, target, ksq);
			moveList = GeneratePieceMoves<NonEvasion, King           , US, false>()(moveList, pos, target, ksq);

			return moveList;
		}
	};

	// 部分特殊化
	// 連続王手の千日手以外の反則手を排除した合法手生成
	// そんなに速度が要求されるところでは呼ばない。
	template <Color US> struct GenerateMoves<Legal, US> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			ExtMove* curr = moveList;
			const Bitboard pinned = pos.pinnedBB();

			moveList = pos.inCheck() ?
				GenerateMoves<Evasion, US>()(moveList, pos) : GenerateMoves<NonEvasion, US>()(moveList, pos);

			// 玉の移動による自殺手と、pinされている駒の移動による自殺手を削除
			while (curr != moveList) {
				if (!pos.pseudoLegalMoveIsLegal<false, false>(curr->move, pinned))
					curr->move = (--moveList)->move;
				else
					++curr;
			}

			return moveList;
		}
	};

	// 部分特殊化
	// Evasion のときに歩、飛、角と、香の2段目の不成も生成する。
	template <Color US> struct GenerateMoves<LegalAll, US> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			ExtMove* curr = moveList;
			const Bitboard pinned = pos.pinnedBB();

			moveList = pos.inCheck() ?
				GenerateMoves<Evasion, US, true>()(moveList, pos) : GenerateMoves<NonEvasion, US>()(moveList, pos);

			// 玉の移動による自殺手と、pinされている駒の移動による自殺手を削除
			while (curr != moveList) {
				if (!pos.pseudoLegalMoveIsLegal<false, false>(curr->move, pinned))
					curr->move = (--moveList)->move;
				else
					++curr;
			}

			return moveList;
		}
	};

	// 王手用
	template <Color US, bool ALL>
	FORCE_INLINE ExtMove* generatCheckMoves(ExtMove* moveList, const PieceType pt, const Position& pos, const Square from, const Square to) {
		switch (pt) {
		case Empty: assert(false); break; // 最適化の為のダミー
		case Pawn:
			if (canPromote(US, makeRank(to))) {
				(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
				// 不成で移動する升
				if (ALL) {
					if (isBehind<US, Rank1, Rank9>(makeRank(to))) // 1段目の不成は省く
						(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
				}
			}
			else
				(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		case Lance:
			if (canPromote(US, makeRank(to))) {
				(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
				// 不成で移動する升
				if (ALL) {
					if (isBehind<US, Rank1, Rank9>(makeRank(to))) // 1段目の不成は省く
						(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
				} else if (isBehind<US, Rank2, Rank8>(makeRank(to))) // 1, 2段目の不成を省く
					(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			}
			else
				(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		case Knight:
			if (canPromote(US, makeRank(to))) {
				(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
				// 不成で移動する升
				if (isBehind<US, Rank2, Rank8>(makeRank(to))) // 1, 2段目の不成は省く
					(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			}
			else
				(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		case Silver:
			if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
				(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
			}
			(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		case Gold: case King: case ProPawn: case ProLance: case ProKnight: case ProSilver: case Horse: case Dragon:
			(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		case Bishop: case Rook:
			if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
				(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
				// 不成で移動する升
				if (ALL) {
					(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
				}
			}
			else
				(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
			break;
		default: UNREACHABLE;
		}
		return moveList;
	}

	// 部分特殊化
	// 王手をかける手を生成する。
	template <Color US, bool ALL> struct GenerateMoves<Check, US, ALL> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			ExtMove* curr = moveList;

			// やねうら王の実装を参考にした
			// https://github.com/yaneurao/YaneuraOu/blob/master/source/movegen.cpp

			// --- 駒の移動による王手

			// 王手になる指し手
			//  1) 成らない移動による直接王手
			//  2) 成る移動による直接王手
			//  3) pinされている駒の移動による間接王手
			// 集合としては1),2) <--> 3)は被覆している可能性があるのでこれを除外できるような指し手生成をしなくてはならない。
			// これを綺麗に実装するのは結構難しい。

			// x = 直接王手となる候補
			// y = 間接王手となる候補

			// ほとんどのケースにおいて y == emptyなのでそれを前提に最適化をする。
			// yと、yを含まないxとに分けて処理する。
			// すなわち、y と (x | y)^y

			constexpr Color opp = oppositeColor(US);
			const Square ksq = pos.kingSquare(opp);

			// 以下の方法だとxとして飛(龍)は100%含まれる。角・馬は60%ぐらいの確率で含まれる。事前条件でもう少し省ければ良いのだが…。
			const Bitboard x =
				(
				(pos.bbOf(Pawn)   & pawnCheckTable(US, ksq)) |
					(pos.bbOf(Lance)  & lanceCheckTable(US, ksq)) |
					(pos.bbOf(Knight) & knightCheckTable(US, ksq)) |
					(pos.bbOf(Silver) & silverCheckTable(US, ksq)) |
					(pos.goldsBB() & goldCheckTable(US, ksq)) |
					(pos.bbOf(Bishop) & bishopCheckTable(US, ksq)) |
					(pos.bbOf(Rook, Dragon)) | // ROOK,DRAGONは無条件全域
					(pos.bbOf(Horse)  & horseCheckTable(US, ksq))
					) & pos.bbOf(US);

			// ここには王を敵玉の8近傍に移動させる指し手も含まれるが、王が近接する形はレアケースなので
			// 指し手生成の段階では除外しなくても良いと思う。

			const Bitboard y = pos.discoveredCheckBB();
			const Bitboard target = ~pos.bbOf(US); // 自駒がない場所が移動対象升

			// yのみ。ただしxかつyである可能性もある。
			auto src = y;
			while (src)
			{
				const Square from = src.firstOneFromSQ11();

				// 両王手候補なので指し手を生成してしまう。

				// いまの敵玉とfromを通る直線上の升と違うところに移動させれば開き王手が確定する。
				const PieceType pt = pieceToPieceType(pos.piece(from));
				Bitboard toBB = pos.attacksFrom(pt, US, from) & target;
				while (toBB) {
					const Square to = toBB.firstOneFromSQ11();
					if (!isAligned<true>(from, to, ksq)) {
						moveList = generatCheckMoves<US, ALL>(moveList, pt, pos, from, to);
					}
					// 直接王手にもなるのでx & fromの場合、直線上の升への指し手を生成。
					else if (x.isSet(from)) {
						const PieceType pt = pieceToPieceType(pos.piece(from));
						switch (pt) {
						case Pawn: // 歩
						{
							if (pawnAttack(US, from).isSet(to)) {
								// 成って王手
								if (canPromote(US, makeRank(to))) {
									(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
									// 成らない手を後に生成
									if (ALL) {
										(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
									}
								}
								else
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							break;
						}
						case Silver: // 銀
						{
							if ((silverAttack(opp, ksq) & silverAttack(US, from)).isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							// 成って王手
							if ((goldAttack(opp, ksq) & silverAttack(US, from)).isSet(to)) {
								if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
									(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								}
							}
							break;
						}
						case Gold: // 金
						case ProPawn: // と金
						case ProLance: // 成香
						case ProKnight: // 成桂
						case ProSilver: // 成銀
						{
							if ((goldAttack(opp, ksq) & goldAttack(US, from)).isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							break;
						}
						case Horse: // 馬
						{
							// 玉が対角上にない場合
							assert(abs(makeFile(ksq) - makeFile(from)) != abs(makeRank(ksq) - makeRank(from)));
							if ((horseAttack(ksq, pos.occupiedBB()) & horseAttack(from, pos.occupiedBB())).isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							break;
						}
						case Dragon: // 竜
						{
							// 玉が直線上にない場合
							assert(makeFile(ksq) != makeFile(from) && makeRank(ksq) != makeRank(from));
							if ((dragonAttack(ksq, pos.occupiedBB()) & dragonAttack(from, pos.occupiedBB())).isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							break;
						}
						case Lance: // 香車
						case Knight: // 桂馬
						case Bishop: // 角
						case Rook: // 飛車
						{
							assert(false);
							break;
						}
						default: UNREACHABLE;
						}
					}
				}
			}

			// yに被覆しないx
			src = (x | y) ^ y;
			while (src)
			{
				const Square from = src.firstOneFromSQ11();

				// 直接王手のみ。
				const PieceType pt = pieceToPieceType(pos.piece(from));
				switch (pt) {
				case Pawn: // 歩
				{
					// 成って王手
					Bitboard toBB = pawnAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						if (canPromote(US, makeRank(to))) {
							(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
							// 成らない手を後に生成
							if (ALL) {
								if (pawnAttack(opp, ksq).isSet(to))
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						}
						else
							(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
					});
					break;
				}
				case Lance: // 香車
				{
					// 玉と筋が異なる場合
					if (makeFile(ksq) != makeFile(from)) {
						Bitboard toBB = goldAttack(opp, ksq) & lanceAttack(US, from, pos.occupiedBB()) & target;
						FOREACH_BB(toBB, const Square to, {
							// 成る
							if (canPromote(US, makeRank(to))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
							}
						});
					}
					// 筋が同じ場合
					else {
						// 間にある駒が一つで、敵駒の場合
						Bitboard dstBB = betweenBB(from, ksq) & pos.occupiedBB();
						if (dstBB.isOneBit() && dstBB & pos.bbOf(opp)) {
							const Square to = dstBB.firstOneFromSQ11();
							// 成れる場合
							if (pawnAttack(opp, ksq).isSet(to) && canPromote(US, makeRank(to))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								// 成らない手を後に生成
								if (ALL) {
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
								}
								else if (isBehind<US, Rank2, Rank8>(makeRank(to))) // 1, 2段目の不成を省く
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
							else {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						}
					}
					break;
				}
				case Knight: // 桂馬
				{
					Bitboard toBB = knightAttack(opp, ksq) & knightAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
					});
					// 成って王手
					toBB = goldAttack(opp, ksq) & knightAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						if (canPromote(US, makeRank(to))) {
							(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
						}
					});
					break;
				}
				case Silver: // 銀
				{
					Bitboard toBB = silverAttack(opp, ksq) & silverAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
					});
					// 成って王手
					toBB = goldAttack(opp, ksq) & silverAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
							(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
						}
					});
					break;
				}
				case Gold: // 金
				case ProPawn: // と金
				case ProLance: // 成香
				case ProKnight: // 成桂
				case ProSilver: // 成銀
				{
					Bitboard toBB = goldAttack(opp, ksq) & goldAttack(US, from) & target;
					FOREACH_BB(toBB, const Square to, {
						(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
					});
					break;
				}
				case Bishop: // 角
				{
					// 玉が対角上にない場合
					if (abs(makeFile(ksq) - makeFile(from)) != abs(makeRank(ksq) - makeRank(from))) {
						Bitboard toBB = horseAttack(ksq, pos.occupiedBB()) & bishopAttack(from, pos.occupiedBB()) & target;
						const Bitboard bishopBB = bishopAttack(ksq, pos.occupiedBB());
						FOREACH_BB(toBB, const Square to, {
							// 成る
							if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								if (ALL) {
									if (bishopBB.isSet(to)) {
										(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
									}

								}
							}
							else if (bishopBB.isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						});
					}
					// 対角上にある場合
					else {
						// 間にある駒が一つで、敵駒の場合
						Bitboard dstBB = betweenBB(from, ksq) & pos.occupiedBB();
						if (dstBB.isOneBit() && dstBB & pos.bbOf(opp)) {
							const Square to = dstBB.firstOneFromSQ11();
							// 成って王手
							if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								// 成らない手を後に生成
								if (ALL) {
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
								}
							}
							else {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						}
					}
					break;
				}
				case Rook: // 飛車
				{
					// 玉が直線上にない場合
					if (makeFile(ksq) != makeFile(from) && makeRank(ksq) != makeRank(from)) {
						Bitboard toBB = dragonAttack(ksq, pos.occupiedBB()) & rookAttack(from, pos.occupiedBB()) & target;
						const Bitboard rookBB = rookAttack(ksq, pos.occupiedBB());
						FOREACH_BB(toBB, const Square to, {
							// 成る
							if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								if (ALL) {
									if (rookBB.isSet(to)) {
										(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
									}
								}
							}
							else if (rookBB.isSet(to)) {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						});
					}
					// 直線上にある場合
					else {
						// 間にある駒が一つで、敵駒の場合
						Bitboard dstBB = betweenBB(from, ksq) & pos.occupiedBB();
						if (dstBB.isOneBit() && dstBB & pos.bbOf(opp)) {
							const Square to = dstBB.firstOneFromSQ11();
							// 成って王手
							if (canPromote(US, makeRank(to)) || canPromote(US, makeRank(from))) {
								(*moveList++).move = makePromoteMove<Capture>(pt, from, to, pos);
								// 成らない手を後に生成
								if (ALL) {
									(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
								}
							}
							else {
								(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
							}
						}
					}
					break;
				}
				case Horse: // 馬
				{
					// 玉が対角上にない場合
					if (abs(makeFile(ksq) - makeFile(from)) != abs(makeRank(ksq) - makeRank(from))) {
						Bitboard toBB = horseAttack(ksq, pos.occupiedBB()) & horseAttack(from, pos.occupiedBB()) & target;
						FOREACH_BB(toBB, const Square to, {
							(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
						});
					}
					// 対角上にある場合
					else {
						// 間にある駒が一つで、敵駒の場合
						Bitboard dstBB = betweenBB(from, ksq) & pos.occupiedBB();
						if (dstBB.isOneBit() && dstBB & pos.bbOf(opp)) {
							const Square to = dstBB.firstOneFromSQ11();
							(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
						}
					}
					break;
				}
				case Dragon: // 竜
				{
					// 玉が直線上にない場合
					if (makeFile(ksq) != makeFile(from) && makeRank(ksq) != makeRank(from)) {
						Bitboard toBB = dragonAttack(ksq, pos.occupiedBB()) & dragonAttack(from, pos.occupiedBB()) & target;
						FOREACH_BB(toBB, const Square to, {
							(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
						});
					}
					// 直線上にある場合
					else {
						Bitboard toBB = kingAttack(ksq) & kingAttack(from) & target;
						// 間にある駒が一つで、敵駒の場合
						const Bitboard dstBB = betweenBB(from, ksq) & pos.occupiedBB();
						if (dstBB.isOneBit() && dstBB & pos.bbOf(opp)) {
							toBB |= dstBB;
						}
						FOREACH_BB(toBB, const Square to, {
							(*moveList++).move = makeNonPromoteMove<Capture>(pt, from, to, pos);
						});
					}
					break;
				}
				default: UNREACHABLE;
				}
			}

			const Bitboard pinned = pos.pinnedBB();

			// pinされている駒の移動による自殺手を削除
			while (curr != moveList) {
				if (!pos.pseudoLegalMoveIsLegal<true, false>(curr->move, pinned))
					curr->move = (--moveList)->move;
				else
					++curr;
			}

			// --- 駒打ちによる王手

			const Bitboard dropTarget = pos.nOccupiedBB(); // emptyBB() ではないので注意して使うこと。
			const Hand ourHand = pos.hand(US);

			// 歩打ち
			if (ourHand.exists<HPawn>()) {
				Bitboard toBB = dropTarget & pawnAttack(opp, ksq);
				// 二歩の回避
				Bitboard pawnsBB = pos.bbOf(Pawn, US);
				Square pawnsSquare;
				foreachBB(pawnsBB, pawnsSquare, [&](const int part) {
					toBB.set(part, toBB.p(part) & ~squareFileMask(pawnsSquare).p(part));
				});

				// 打ち歩詰めの回避
				constexpr Rank TRank9 = (US == Black ? Rank9 : Rank1);
				constexpr SquareDelta TDeltaS = (US == Black ? DeltaS : DeltaN);

				// 相手玉が九段目なら、歩で王手出来ないので、打ち歩詰めを調べる必要はない。
				if (makeRank(ksq) != TRank9) {
					const Square pawnDropCheckSquare = ksq + TDeltaS;
					assert(isInSquare(pawnDropCheckSquare));
					if (toBB.isSet(pawnDropCheckSquare) && pos.piece(pawnDropCheckSquare) == Empty) {
						if (!pos.isPawnDropCheckMate(US, pawnDropCheckSquare))
							// ここで clearBit だけして MakeMove しないことも出来る。
							// 指し手が生成される順番が変わり、王手が先に生成されるが、後で問題にならないか?
							(*moveList++).move = makeDropMove(Pawn, pawnDropCheckSquare);
						toBB.xorBit(pawnDropCheckSquare);
					}
				}

				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Pawn, to);
				});
			}

			// 香車打ち
			if (ourHand.exists<HLance>()) {
				Bitboard toBB = dropTarget & lanceAttack(opp, ksq, pos.occupiedBB());
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Lance, to);
				});
			}

			// 桂馬打ち
			if (ourHand.exists<HKnight>()) {
				Bitboard toBB = dropTarget & knightAttack(opp, ksq);
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Knight, to);
				});
			}

			// 銀打ち
			if (ourHand.exists<HSilver>()) {
				Bitboard toBB = dropTarget & silverAttack(opp, ksq);
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Silver, to);
				});
			}

			// 金打ち
			if (ourHand.exists<HGold>()) {
				Bitboard toBB = dropTarget & goldAttack(opp, ksq);
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Gold, to);
				});
			}

			// 角打ち
			if (ourHand.exists<HBishop>()) {
				Bitboard toBB = dropTarget & bishopAttack(ksq, pos.occupiedBB());
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Bishop, to);
				});
			}

			// 飛車打ち
			if (ourHand.exists<HRook>()) {
				Bitboard toBB = dropTarget & rookAttack(ksq, pos.occupiedBB());
				Square to;
				FOREACH_BB(toBB, to, {
					(*moveList++).move = makeDropMove(Rook, to);
				});
			}

			return moveList;
		}
	};

	// 部分特殊化
	// Check のときに歩、飛、角と、香の2段目の不成も生成する。
	template <Color US> struct GenerateMoves<CheckAll, US> {
		FORCE_INLINE ExtMove* operator () (ExtMove* moveList, const Position& pos) {
			return GenerateMoves<Check, US, true>()(moveList, pos);
		}
	};
}

template <MoveType MT>
ExtMove* generateMoves(ExtMove* moveList, const Position& pos) {
	return (pos.turn() == Black ?
			GenerateMoves<MT, Black>()(moveList, pos) : GenerateMoves<MT, White>()(moveList, pos));
}
template <MoveType MT>
ExtMove* generateMoves(ExtMove* moveList, const Position& pos, const Square to) {
	return generateRecaptureMoves(moveList, pos, to, pos.turn());
}

// 明示的なインスタンス化
// これが無いと、他のファイルから呼んだ時に、
// 実体が無いためにリンクエラーになる。
// ちなみに、特殊化されたテンプレート関数は、明示的なインスタンス化の必要はない。
// 実装を cpp に置くことで、コンパイル時間の短縮が出来る。
//template ExtMove* generateMoves<Capture           >(ExtMove* moveList, const Position& pos);
//template ExtMove* generateMoves<NonCapture        >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<Drop              >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<CapturePlusPro    >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<NonCaptureMinusPro>(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<Evasion           >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<NonEvasion        >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<Legal             >(ExtMove* moveList, const Position& pos);
#if !defined NDEBUG || defined LEARN
template ExtMove* generateMoves<LegalAll          >(ExtMove* moveList, const Position& pos);
#endif
template ExtMove* generateMoves<Recapture         >(ExtMove* moveList, const Position& pos, const Square to);
template ExtMove* generateMoves<Check             >(ExtMove* moveList, const Position& pos);
template ExtMove* generateMoves<CheckAll          >(ExtMove* moveList, const Position& pos);
