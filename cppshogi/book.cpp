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

#include "book.hpp"
#include "position.hpp"
#include "move.hpp"
#include "usi.hpp"
#include "search.hpp"

MT64bit Book::mt64bit_; // 定跡のhash生成用なので、seedは固定でデフォルト値を使う。
Key Book::ZobPiece[PieceNone][SquareNum];
Key Book::ZobHand[HandPieceNum][19]; // 持ち駒の同一種類の駒の数ごと
Key Book::ZobTurn;

void Book::init() {
    for (Piece p = Empty; p < PieceNone; ++p) {
        for (Square sq = SQ11; sq < SquareNum; ++sq)
            ZobPiece[p][sq] = mt64bit_.random();
    }
    for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
        for (int num = 0; num < 19; ++num)
            ZobHand[hp][num] = mt64bit_.random();
    }
    ZobTurn = mt64bit_.random();
}

bool Book::open(const char* fName) {
    fileName_ = "";

    if (is_open())
        close();

    std::ifstream::open(fName, std::ifstream::in | std::ifstream::binary | std::ios::ate);

    if (!is_open())
        return false;

    size_ = tellg() / sizeof(BookEntry);

    if (!good()) {
        std::cerr << "Failed to open book file " << fName  << std::endl;
        exit(EXIT_FAILURE);
    }

    fileName_ = fName;
    return true;
}

void Book::binary_search(const Key key) {
    size_t low = 0;
    size_t high = size_ - 1;
    size_t mid;
    BookEntry entry;

    while (low < high && good()) {
        mid = (low + high) / 2;

        assert(mid >= low && mid < high);

        // std::ios_base::beg はストリームの開始位置を指す。
        // よって、ファイルの開始位置から mid * sizeof(BookEntry) バイト進んだ位置を指す。
        seekg(mid * sizeof(BookEntry), std::ios_base::beg);
        read(reinterpret_cast<char*>(&entry), sizeof(entry));

        if (key <= entry.key)
            high = mid;
        else
            low = mid + 1;
    }

    assert(low == high);

    seekg(low * sizeof(BookEntry), std::ios_base::beg);
}

Key Book::bookKey(const Position& pos) {
    Key key = 0;
    Bitboard bb = pos.occupiedBB();

    while (bb) {
        const Square sq = bb.firstOneFromSQ11();
        key ^= ZobPiece[pos.piece(sq)][sq];
    }
    const Hand hand = pos.hand(pos.turn());
    for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp)
        key ^= ZobHand[hp][hand.numOf(hp)];
    if (pos.turn() == White)
        key ^= ZobTurn;
    return key;
}

Key Book::bookKeyAfter(const Position& pos, const Key key, const Move move) {
    Key key_after = key;
    const Square to = move.to();
    if (move.isDrop()) {
        const Piece pc = colorAndPieceTypeToPiece(pos.turn(), move.pieceTypeDropped());
        key_after ^= ZobPiece[pc][to];
    }
    else {
        const Square from = move.from();
        key_after ^= ZobPiece[pos.piece(from)][from];

        const Piece pc = colorAndPieceTypeToPiece(pos.turn(), move.pieceTypeTo());
        key_after ^= ZobPiece[pc][to];

        if (move.isCapture())
            key_after ^= ZobPiece[pos.piece(to)][to];
    }
    const Hand hand = pos.hand(pos.turn());
    for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp)
        key_after ^= ZobHand[hp][hand.numOf(hp)];
    const Hand hand_after = pos.hand(oppositeColor(pos.turn()));
    for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp)
        key_after ^= ZobHand[hp][hand_after.numOf(hp)];

    key_after ^= ZobTurn;
    return key_after;
}

std::tuple<Move, Score> Book::probe(const Position& pos, const std::string& fName, const bool pickBest) {
    BookEntry entry;
    u16 best = 0;
    u32 sum = 0;
    Move move = Move::moveNone();
    const Key key = bookKey(pos);
    const Score min_book_score = static_cast<Score>(static_cast<int>(pos.searcher()->options["Min_Book_Score"]));
    Score score = ScoreZero;

    if (fileName_ != fName && !open(fName.c_str()))
        return std::make_tuple(Move::moveNone(), ScoreNone);

    binary_search(key);

    // 現在の局面における定跡手の数だけループする。
    while (read(reinterpret_cast<char*>(&entry), sizeof(entry)), entry.key == key && good()) {
        best = std::max(best, entry.count);
        sum += entry.count;

        // 指された確率に従って手が選択される。
        // count が大きい順に並んでいる必要はない。
        if (min_book_score <= entry.score
            && ((!pickBest && random_.random() % sum < entry.count)
                || (pickBest && entry.count == best)))
        {
            const Move tmp = Move(entry.fromToPro);
            const Square to = tmp.to();
            if (tmp.isDrop()) {
                const PieceType ptDropped = tmp.pieceTypeDropped();
                move = makeDropMove(ptDropped, to);
            }
            else {
                const Square from = tmp.from();
                const PieceType ptFrom = pieceToPieceType(pos.piece(from));
                const bool promo = tmp.isPromotion();
                if (promo)
                    move = makeCapturePromoteMove(ptFrom, from, to, pos);
                else
                    move = makeCaptureMove(ptFrom, from, to, pos);
            }
            score = entry.score;
        }
        if (tellg() == size_ * sizeof(BookEntry))
            break;
    }

    return std::make_tuple(move, score);
}

// 千日手の評価値を考慮する
// countの降順にソートされていること
// countが少ない評価値は信頼しない
std::tuple<Move, Score> Book::probeConsideringDraw(const Position& pos, const std::string& fName) {
    BookEntry entry;
    Score best = -ScoreInfinite;
    Move move = Move::moveNone();
    const Key key = bookKey(pos);
    Score score = ScoreZero;
    Score trusted_score = ScoreInfinite;

    if (fileName_ != fName && !open(fName.c_str()))
        return std::make_tuple(Move::moveNone(), ScoreNone);

    binary_search(key);

    // 現在の局面における定跡手の数だけループする。
    while (read(reinterpret_cast<char*>(&entry), sizeof(entry)), entry.key == key && good()) {
        const Move tmp = Move(entry.fromToPro);

        // 回数が少ない評価値は信頼しない
        if (entry.score < trusted_score)
            trusted_score = entry.score;
        entry.score = trusted_score;

        switch (pos.moveIsDraw(tmp)) {
        case RepetitionDraw:
        {
            // 千日手の評価で上書き
            const float drawValue = (pos.turn() == Black
                ? static_cast<int>(pos.searcher()->options["Draw_Value_Black"])
                : static_cast<int>(pos.searcher()->options["Draw_Value_White"])) / 1000.0f;
            const int evalCoef = static_cast<int>(pos.searcher()->options["Eval_Coef"]);
            const Score drawScore = static_cast<Score>(static_cast<int>(-logf(1.0f / drawValue - 1.0f) * evalCoef));
            entry.score = drawScore;
            break;
        }
        case RepetitionWin:
            // 相手の勝ち(自分の負け)
            entry.score = -ScoreInfinite;
            break;
        case RepetitionLose:
            // 相手の負け(自分の勝ち)
            entry.score = ScoreMaxEvaluate;
            break;
        }

        if (entry.score > best)
        {
            best = entry.score;
            const Square to = tmp.to();
            if (tmp.isDrop()) {
                const PieceType ptDropped = tmp.pieceTypeDropped();
                move = makeDropMove(ptDropped, to);
            }
            else {
                const Square from = tmp.from();
                const PieceType ptFrom = pieceToPieceType(pos.piece(from));
                const bool promo = tmp.isPromotion();
                if (promo)
                    move = makeCapturePromoteMove(ptFrom, from, to, pos);
                else
                    move = makeCaptureMove(ptFrom, from, to, pos);
            }
            score = entry.score;
        }
        if (tellg() == size_ * sizeof(BookEntry))
            break;
    }

    return std::make_tuple(move, score);
}

Score Book::getMinMaxBookScore(Position& pos, Score alpha, Score beta, int depth, const Score drawScoreBlack, const Score drawScoreWhite) {
    const Key key = bookKey(pos);
    Score trustedScore = ScoreInfinite;

    binary_search(key);

    std::vector<BookEntry> entries;
    {
        BookEntry entry;
        while (read(reinterpret_cast<char*>(&entry), sizeof(entry)), entry.key == key && good()) {
            entries.emplace_back(entry);
            if (tellg() == size_ * sizeof(BookEntry))
                break;
        }
    }
    if (entries.size() == 0)
        return ScoreNone;

    for (auto& entry : entries) {
        const Move move16 = Move(entry.fromToPro);

        // 回数が少ない評価値は信頼しない
        if (entry.score < trustedScore)
            trustedScore = entry.score;
        entry.score = trustedScore;

        switch (pos.moveIsDraw(move16)) {
        case RepetitionDraw:
        {
            // 千日手の評価で上書き
            entry.score = pos.turn() == Black ? drawScoreBlack : drawScoreWhite;
            break;
        }
        case RepetitionWin:
            // 相手の勝ち(自分の負け)
            entry.score = -ScoreInfinite;
            break;
        case RepetitionLose:
            // 相手の負け(自分の勝ち)
            entry.score = ScoreMaxEvaluate;
            break;
        default:
            if (depth > 0) {
                const Move move = move16toMove(move16, pos);
                StateInfo st;
                pos.doMove(move, st);
                const Score retScore = getMinMaxBookScore(pos, -beta, -alpha, depth - 1, drawScoreBlack, drawScoreWhite);
                if (retScore != ScoreNone)
                    entry.score = -retScore;
                pos.undoMove(move);
            }
            break;
        }

        alpha = std::max(alpha, entry.score);
        if (alpha >= beta) {
            return alpha;
        }

        if (tellg() == size_ * sizeof(BookEntry))
            break;
    }
    return alpha;
}

// 数手先の千日手の評価値を考慮する
// countの降順にソートされていること
// countが少ない評価値は信頼しない
std::tuple<Move, Score> Book::probeConsideringDrawDepth(Position& pos, const std::string& fName) {
    const int evalCoef = static_cast<int>(pos.searcher()->options["Eval_Coef"]);
    const float drawValueBlack = pos.searcher()->options["Draw_Value_Black"] / 1000.0f;
    const float drawValueWhite = pos.searcher()->options["Draw_Value_White"] / 1000.0f;
    const Score drawScoreBlack = static_cast<Score>(static_cast<int>(-logf(1.0f / drawValueBlack - 1.0f) * evalCoef));
    const Score drawScoreWhite = static_cast<Score>(static_cast<int>(-logf(1.0f / drawValueWhite - 1.0f) * evalCoef));
    const int depth = pos.searcher()->options["Book_Consider_Draw_Depth"];

    Score best = -ScoreInfinite;
    Move move = Move::moveNone();
    const Key key = bookKey(pos);
    Score score = ScoreZero;
    Score trusted_score = ScoreInfinite;
    Score alpha = -ScoreInfinite;

    if (fileName_ != fName && !open(fName.c_str()))
        return std::make_tuple(Move::moveNone(), ScoreNone);

    binary_search(key);

    std::vector<BookEntry> entries;
    {
        BookEntry entry;
        while (read(reinterpret_cast<char*>(&entry), sizeof(entry)), entry.key == key && good()) {
            entries.emplace_back(entry);
            if (tellg() == size_ * sizeof(BookEntry))
                break;
        }
    }

    for (auto& entry : entries) {
        const Move tmpMove = move16toMove(Move(entry.fromToPro), pos);

        // 回数が少ない評価値は信頼しない
        if (entry.score < trusted_score)
            trusted_score = entry.score;
        entry.score = trusted_score;

        switch (pos.moveIsDraw(tmpMove)) {
        case RepetitionDraw:
        {
            // 千日手の評価で上書き
            entry.score = pos.turn() == Black ? drawScoreBlack : drawScoreWhite;
            break;
        }
        case RepetitionWin:
            // 相手の勝ち(自分の負け)
            entry.score = -ScoreInfinite;
            break;
        case RepetitionLose:
            // 相手の負け(自分の勝ち)
            entry.score = ScoreMaxEvaluate;
            break;
        default:
            StateInfo st;
            pos.doMove(tmpMove, st);
            const Score retScore = getMinMaxBookScore(pos, -ScoreInfinite, -alpha, depth - 1, drawScoreBlack, drawScoreWhite);
            if (retScore != ScoreNone)
                entry.score = -retScore;
            pos.undoMove(tmpMove);
            break;
        }

        alpha = std::max(alpha, entry.score);

        if (entry.score > best)
        {
            best = entry.score;
            move = tmpMove;
            score = entry.score;
        }
    }

    return std::make_tuple(move, score);
}

inline bool countCompare(const BookEntry& b1, const BookEntry& b2) {
    return b1.count < b2.count;
}

#if !defined MINIMUL
// 以下のようなフォーマットが入力される。
// <棋譜番号> <日付> <先手名> <後手名> <0:引き分け, 1:先手勝ち, 2:後手勝ち> <総手数> <棋戦名前> <戦形>
// <CSA1行形式の指し手>
//
// (例)
// 1 2003/09/08 羽生善治 谷川浩司 2 126 王位戦 その他の戦型
// 7776FU3334FU2726FU4132KI
//
// 勝った方の手だけを定跡として使うこととする。
// 出現回数がそのまま定跡として使う確率となる。
// 基本的には棋譜を丁寧に選別した上で定跡を作る必要がある。
// MAKE_SEARCHED_BOOK を on にしていると、定跡生成に非常に時間が掛かる。
void makeBook(Position& pos, std::istringstream& ssCmd) {
    std::string fileName;
    ssCmd >> fileName;
    std::ifstream ifs(fileName.c_str(), std::ios::binary);
    if (!ifs) {
        std::cout << "I cannot open " << fileName << std::endl;
        return;
    }
    std::string line;
    std::map<Key, std::vector<BookEntry> > bookMap;

    while (std::getline(ifs, line)) {
        std::string elem;
        std::stringstream ss(line);
        ss >> elem; // 棋譜番号を飛ばす。
        ss >> elem; // 対局日を飛ばす。
        ss >> elem; // 先手
        const std::string sente = elem;
        ss >> elem; // 後手
        const std::string gote = elem;
        ss >> elem; // (0:引き分け,1:先手の勝ち,2:後手の勝ち)
        const Color winner = (elem == "1" ? Black : elem == "2" ? White : ColorNum);
        // 勝った方の指し手を記録していく。
        // 又は稲庭戦法側を記録していく。
        const Color saveColor = winner;

        if (!std::getline(ifs, line)) {
            std::cout << "!!! header only !!!" << std::endl;
            return;
        }
        pos.set(DefaultStartPositionSFEN);
        StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));
        while (!line.empty()) {
            const std::string moveStrCSA = line.substr(0, 6);
            const Move move = csaToMove(pos, moveStrCSA);
            if (!move) {
                pos.print();
                std::cout << "!!! Illegal move = " << moveStrCSA << " !!!" << std::endl;
                break;
            }
            line.erase(0, 6); // 先頭から6文字削除
            if (pos.turn() == saveColor) {
                // 先手、後手の内、片方だけを記録する。
                const Key key = Book::bookKey(pos);
                bool isFind = false;
                if (bookMap.find(key) != bookMap.end()) {
                    for (std::vector<BookEntry>::iterator it = bookMap[key].begin();
                         it != bookMap[key].end();
                         ++it)
                    {
                        if (it->fromToPro == move.proFromAndTo()) {
                            ++it->count;
                            if (it->count < 1)
                                --it->count; // 数えられる数の上限を超えたので元に戻す。
                            isFind = true;
                        }
                    }
                }
                if (isFind == false) {
#if defined MAKE_SEARCHED_BOOK
                    states->push_back(StateInfo());
                    pos.doMove(move, states->back());

                    std::istringstream ssCmd("byoyomi 1000");
                    go(pos, ssCmd);

                    pos.undoMove(move);
                    states->pop_back();

                    // doMove してから search してるので点数が反転しているので直す。
                    const Score score = -pos.thisThread()->rootMoves[0].score;
#else
                    const Score score = ScoreZero;
#endif
                    // 未登録の手
                    BookEntry be;
                    be.score = score;
                    be.key = key;
                    be.fromToPro = static_cast<u16>(move.proFromAndTo());
                    be.count = 1;
                    bookMap[key].push_back(be);
                }
            }
            states->push_back(StateInfo());
            pos.doMove(move, states->back());
        }
    }

    // BookEntry::count の値で降順にソート
    for (auto& elem : bookMap) {
        std::sort(elem.second.rbegin(), elem.second.rend(), countCompare);
    }

#if 0
    // 2 回以上棋譜に出現していない手は削除する。
    for (auto& elem : bookMap) {
        auto& second = elem.second;
        auto erase_it = std::find_if(second.begin(), second.end(), [](decltype(*second.begin())& second_elem) { return second_elem.count < 2; });
        second.erase(erase_it, second.end());
    }
#endif

#if 0
    // narrow book
    for (auto& elem : bookMap) {
        auto& second = elem.second;
        auto erase_it = std::find_if(second.begin(), second.end(), [&](decltype(*second.begin())& second_elem) { return second_elem.count < second[0].count / 2; });
        second.erase(erase_it, second.end());
    }
#endif

    std::ofstream ofs("book.bin", std::ios::binary);
    for (auto& elem : bookMap) {
        for (auto& elel : elem.second)
            ofs.write(reinterpret_cast<char*>(&(elel)), sizeof(BookEntry));
    }

    std::cout << "book making was done" << std::endl;
}
#endif
