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

#include "usi.hpp"
#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "book.hpp"

bool CaseInsensitiveLess::operator () (const std::string& s1, const std::string& s2) const {
    for (size_t i = 0; i < s1.size() && i < s2.size(); ++i) {
        const int c1 = tolower(s1[i]);
        const int c2 = tolower(s2[i]);
        if (c1 != c2)
            return c1 < c2;
    }
    return s1.size() < s2.size();
}

namespace {
    // 論理的なコア数の取得
    inline int cpuCoreCount() {
        // std::thread::hardware_concurrency() は 0 を返す可能性がある。
        // HyperThreading が有効なら論理コア数だけ thread 生成した方が強い。
        return std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
    }

    class StringToPieceTypeCSA : public std::map<std::string, PieceType> {
    public:
        StringToPieceTypeCSA() {
            (*this)["FU"] = Pawn;
            (*this)["KY"] = Lance;
            (*this)["KE"] = Knight;
            (*this)["GI"] = Silver;
            (*this)["KA"] = Bishop;
            (*this)["HI"] = Rook;
            (*this)["KI"] = Gold;
            (*this)["OU"] = King;
            (*this)["TO"] = ProPawn;
            (*this)["NY"] = ProLance;
            (*this)["NK"] = ProKnight;
            (*this)["NG"] = ProSilver;
            (*this)["UM"] = Horse;
            (*this)["RY"] = Dragon;
        }
        PieceType value(const std::string& str) const {
            return this->find(str)->second;
        }
        bool isLegalString(const std::string& str) const {
            return (this->find(str) != this->end());
        }
    };
    const StringToPieceTypeCSA g_stringToPieceTypeCSA;
}

void OptionsMap::init(Searcher* s) {
    (*this)["Book_File"]                   = USIOption("book.bin");
    (*this)["Best_Book_Move"]              = USIOption(true);
    (*this)["OwnBook"]                     = USIOption(false);
    (*this)["Min_Book_Score"]              = USIOption(-3000, -ScoreInfinite, ScoreInfinite);
    (*this)["USI_Ponder"]                  = USIOption(false);
    (*this)["Stochastic_Ponder"]           = USIOption(true);
    (*this)["Byoyomi_Margin"]              = USIOption(0, 0, INT_MAX);
    (*this)["Time_Margin"]                 = USIOption(1000, 0, INT_MAX);
    (*this)["MultiPV"]                     = USIOption(1, 1, MaxLegalMoves - 1);
    (*this)["Draw_Ply"]                    = USIOption(0, 0, INT_MAX);
    (*this)["Const_Playout"]               = USIOption(0, 0, INT_MAX);
    (*this)["UCT_Threads"]                 = USIOption(2, 0, 256);
    (*this)["UCT_Threads2"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads3"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads4"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads5"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads6"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads7"]                = USIOption(0, 0, 256);
    (*this)["UCT_Threads8"]                = USIOption(0, 0, 256);
    (*this)["DNN_Model"]                   = USIOption(R"(model.onnx)");
    (*this)["DNN_Model2"]                  = USIOption("");
    (*this)["DNN_Model3"]                  = USIOption("");
    (*this)["DNN_Model4"]                  = USIOption("");
    (*this)["DNN_Model5"]                  = USIOption("");
    (*this)["DNN_Model6"]                  = USIOption("");
    (*this)["DNN_Model7"]                  = USIOption("");
    (*this)["DNN_Model8"]                  = USIOption("");
    (*this)["DNN_Batch_Size"]              = USIOption(128, 1, 256);
    (*this)["DNN_Batch_Size2"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size3"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size4"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size5"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size6"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size7"]             = USIOption(0, 0, 256);
    (*this)["DNN_Batch_Size8"]             = USIOption(0, 0, 256);
    (*this)["Softmax_Temperature"]         = USIOption(174, 1, 500);
    (*this)["Mate_Root_Search"]            = USIOption(33, 0, 37);
#ifdef PV_MATE_SEARCH
    (*this)["PV_Mate_Search_Threads"]      = USIOption(0, 0, 256);
    (*this)["PV_Mate_Search_Depth"]        = USIOption(33, 0, 37);
    (*this)["PV_Mate_Search_Nodes"]        = USIOption(500000, 0, 10000000);
#endif
    (*this)["Resign_Threshold"]            = USIOption(10, 0, 1000);
    (*this)["Draw_Value_Black"]            = USIOption(500, 0, 1000);
    (*this)["Draw_Value_White"]            = USIOption(500, 0, 1000);
    (*this)["C_init"]                      = USIOption(144, 0, 500);
    (*this)["C_base"]                      = USIOption(28288, 10000, 100000);
    (*this)["C_fpu_reduction"]             = USIOption(27, 0, 100);
    (*this)["C_init_root"]                 = USIOption(116, 0, 500);
    (*this)["C_base_root"]                 = USIOption(25617, 10000, 100000);
    (*this)["C_fpu_reduction_root"]        = USIOption(0, 0, 100);
    (*this)["UCT_NodeLimit"]               = USIOption(10000000, 100000, 1000000000); // UCTノードの上限
    (*this)["DfPn_Hash"]                   = USIOption(2048, 64, 4096); // DfPnハッシュサイズ
    (*this)["DfPn_Min_Search_Millisecs"]   = USIOption(300, 0, INT_MAX);
    (*this)["ReuseSubtree"]                = USIOption(true);
    (*this)["Eval_Coef"]                   = USIOption(756, 1, 10000);
    (*this)["Random_Ply"]                  = USIOption(0, 0, 1000);
    (*this)["Random_Temperature"]          = USIOption(10000, 0, 100000);
    (*this)["Random_Temperature_Drop"]     = USIOption(1000, 0, 100000);
    (*this)["Random_Cutoff"]               = USIOption(20, 0, 1000);
#ifdef MAKE_BOOK
    (*this)["PV_Interval"]                 = USIOption(0, 0, INT_MAX);
    (*this)["Save_Book_Interval"]          = USIOption(100, 0, INT_MAX);
    (*this)["Make_Book_Sleep"]             = USIOption(0, 0, INT_MAX);
    (*this)["Use_Book_Policy"]             = USIOption(true);
    (*this)["Use_Interruption"]            = USIOption(true);
    (*this)["Book_Eval_Threshold"]         = USIOption(INT_MAX, 1, INT_MAX);
    (*this)["Book_Visit_Threshold"]        = USIOption(10, 0, 1000);
    (*this)["Make_Book_Color"]             = USIOption("both");
#else
    (*this)["PV_Interval"]                 = USIOption(500, 0, INT_MAX);
#endif // !MAKE_BOOK
    (*this)["DebugMessage"]                = USIOption(false);
#ifdef NDEBUG
    (*this)["Engine_Name"]                 = USIOption("dlshogi");
#else
    (*this)["Engine_Name"]                 = USIOption("dlshogi Debug Build");
#endif
}

USIOption::USIOption(const char* v, Fn* f, Searcher* s) :
    type_("string"), min_(0), max_(0), onChange_(f), searcher_(s)
{
    defaultValue_ = currentValue_ = v;
}

USIOption::USIOption(const bool v, Fn* f, Searcher* s) :
    type_("check"), min_(0), max_(0), onChange_(f), searcher_(s)
{
    defaultValue_ = currentValue_ = (v ? "true" : "false");
}

USIOption::USIOption(Fn* f, Searcher* s) :
    type_("button"), min_(0), max_(0), onChange_(f), searcher_(s) {}

USIOption::USIOption(const int v, const int min, const int max, Fn* f, Searcher* s)
    : type_("spin"), min_(min), max_(max), onChange_(f), searcher_(s)
{
    std::ostringstream ss;
    ss << v;
    defaultValue_ = currentValue_ = ss.str();
}

USIOption& USIOption::operator = (const std::string& v) {
    assert(!type_.empty());

    if ((type_ != "button" && v.empty())
        || (type_ == "check" && v != "true" && v != "false")
        || (type_ == "spin" && (atoi(v.c_str()) < min_ || max_ < atoi(v.c_str()))))
    {
        return *this;
    }

    if (type_ != "button")
        currentValue_ = v;

    if (onChange_ != nullptr)
        (*onChange_)(searcher_, *this);

    return *this;
}

std::ostream& operator << (std::ostream& os, const OptionsMap& om) {
    for (auto& elem : om) {
        const USIOption& o = elem.second;
        os << "\noption name " << elem.first << " type " << o.type_;
        if (o.type_ != "button")
            os << " default " << o.defaultValue_;

        if (o.type_ == "spin")
            os << " min " << o.min_ << " max " << o.max_;
    }
    return os;
}

// 評価値 x を勝率にして返す。
// 係数 600 は Ponanza で採用しているらしい値。
inline double sigmoidWinningRate(const double x) {
    return 1.0 / (1.0 + exp(-x/600.0));
}
inline double dsigmoidWinningRate(const double x) {
    const double a = 1.0/600;
    return a * sigmoidWinningRate(x) * (1 - sigmoidWinningRate(x));
}

#if defined USE_GLOBAL
#else
// 教師局面を増やす為、適当に駒を動かす。玉の移動を多めに。王手が掛かっている時は呼ばない事にする。
void randomMove(Position& pos, std::mt19937& mt) {
    StateInfo state[MaxPly+7];
    StateInfo* st = state;
    const Color us = pos.turn();
    const Color them = oppositeColor(us);
    const Square from = pos.kingSquare(us);
    std::uniform_int_distribution<int> dist(0, 1);
    switch (dist(mt)) {
    case 0: { // 玉の25近傍の移動
        ExtMove legalMoves[MaxLegalMoves]; // 玉の移動も含めた普通の合法手
        ExtMove* pms = &legalMoves[0];
        Bitboard kingToBB = pos.bbOf(us).notThisAnd(neighbor5x5Table(from));
        while (kingToBB) {
            const Square to = kingToBB.firstOneFromSQ11();
            const Move move = makeNonPromoteMove<Capture>(King, from, to, pos);
            if (pos.moveIsPseudoLegal<false>(move)
                && pos.pseudoLegalMoveIsLegal<true, false>(move, pos.pinnedBB()))
            {
                (*pms++).move = move;
            }
        }
        if (&legalMoves[0] != pms) { // 手があったなら
            std::uniform_int_distribution<int> moveDist(0, static_cast<int>(pms - &legalMoves[0] - 1));
            pos.doMove(legalMoves[moveDist(mt)].move, *st++);
            if (dist(mt)) { // 1/2 の確率で相手もランダムに指す事にする。
                MoveList<LegalAll> ml(pos);
                if (ml.size()) {
                    std::uniform_int_distribution<int> moveDist(0, static_cast<int>(ml.size() - 1));
                    pos.doMove((ml.begin() + moveDist(mt))->move, *st++);
                }
            }
        }
        else
            return;
        break;
    }
    case 1: { // 玉も含めた全ての合法手
        bool moved = false;
        for (int i = 0; i < dist(mt) + 1; ++i) { // 自分だけ、または両者ランダムに1手指してみる。
            MoveList<LegalAll> ml(pos);
            if (ml.size()) {
                std::uniform_int_distribution<int> moveDist(0, static_cast<int>(ml.size() - 1));
                pos.doMove((ml.begin() + moveDist(mt))->move, *st++);
                moved = true;
            }
        }
        if (!moved)
            return;
        break;
    }
    default: UNREACHABLE;
    }

    // 違法手が混ざったりするので、一旦 sfen に直して読み込み、過去の手を参照しないようにする。
    std::string sfen = pos.toSFEN();
    std::istringstream ss(sfen);
    setPosition(pos, ss);
}
#endif

Move usiToMoveBody(const Position& pos, const std::string& moveStr) {
    Move move;
    if (g_charToPieceUSI.isLegalChar(moveStr[0])) {
        // drop
        const PieceType ptTo = pieceToPieceType(g_charToPieceUSI.value(moveStr[0]));
        if (moveStr[1] != '*')
            return Move::moveNone();
        const File toFile = charUSIToFile(moveStr[2]);
        const Rank toRank = charUSIToRank(moveStr[3]);
        if (!isInSquare(toFile, toRank))
            return Move::moveNone();
        const Square to = makeSquare(toFile, toRank);
        move = makeDropMove(ptTo, to);
    }
    else {
        const File fromFile = charUSIToFile(moveStr[0]);
        const Rank fromRank = charUSIToRank(moveStr[1]);
        if (!isInSquare(fromFile, fromRank))
            return Move::moveNone();
        const Square from = makeSquare(fromFile, fromRank);
        const File toFile = charUSIToFile(moveStr[2]);
        const Rank toRank = charUSIToRank(moveStr[3]);
        if (!isInSquare(toFile, toRank))
            return Move::moveNone();
        const Square to = makeSquare(toFile, toRank);
        if (moveStr[4] == '\0')
            move = makeNonPromoteMove<Capture>(pieceToPieceType(pos.piece(from)), from, to, pos);
        else if (moveStr[4] == '+') {
            if (moveStr[5] != '\0')
                return Move::moveNone();
            move = makePromoteMove<Capture>(pieceToPieceType(pos.piece(from)), from, to, pos);
        }
        else
            return Move::moveNone();
    }

    if (pos.moveIsPseudoLegal<false>(move)
        && pos.pseudoLegalMoveIsLegal<false, false>(move, pos.pinnedBB()))
    {
        return move;
    }
    return Move::moveNone();
}
#if !defined NDEBUG
// for debug
Move usiToMoveDebug(const Position& pos, const std::string& moveStr) {
    for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
        if (moveStr == ml.move().toUSI())
            return ml.move();
    }
    return Move::moveNone();
}
Move csaToMoveDebug(const Position& pos, const std::string& moveStr) {
    for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
        if (moveStr == ml.move().toCSA())
            return ml.move();
    }
    return Move::moveNone();
}
#endif
Move usiToMove(const Position& pos, const std::string& moveStr) {
    const Move move = usiToMoveBody(pos, moveStr);
    assert(move == usiToMoveDebug(pos, moveStr));
    return move;
}

Move csaToMoveBody(const Position& pos, const std::string& moveStr) {
    if (moveStr.size() != 6)
        return Move::moveNone();
    const File toFile = charCSAToFile(moveStr[2]);
    const Rank toRank = charCSAToRank(moveStr[3]);
    if (!isInSquare(toFile, toRank))
        return Move::moveNone();
    const Square to = makeSquare(toFile, toRank);
    const std::string ptToString(moveStr.begin() + 4, moveStr.end());
    if (!g_stringToPieceTypeCSA.isLegalString(ptToString))
        return Move::moveNone();
    const PieceType ptTo = g_stringToPieceTypeCSA.value(ptToString);
    Move move;
    if (moveStr[0] == '0' && moveStr[1] == '0')
        // drop
        move = makeDropMove(ptTo, to);
    else {
        const File fromFile = charCSAToFile(moveStr[0]);
        const Rank fromRank = charCSAToRank(moveStr[1]);
        if (!isInSquare(fromFile, fromRank))
            return Move::moveNone();
        const Square from = makeSquare(fromFile, fromRank);
        PieceType ptFrom = pieceToPieceType(pos.piece(from));
        if (ptFrom == ptTo)
            // non promote
            move = makeNonPromoteMove<Capture>(ptFrom, from, to, pos);
        else if (ptFrom + PTPromote == ptTo)
            // promote
            move = makePromoteMove<Capture>(ptFrom, from, to, pos);
        else
            return Move::moveNone();
    }

    if (pos.moveIsPseudoLegal<false>(move)
        && pos.pseudoLegalMoveIsLegal<false, false>(move, pos.pinnedBB()))
    {
        return move;
    }
    return Move::moveNone();
}
Move csaToMove(const Position& pos, const std::string& moveStr) {
    const Move move = csaToMoveBody(pos, moveStr);
    assert(move == csaToMoveDebug(pos, moveStr));
    return move;
}

void setPosition(Position& pos, std::istringstream& ssCmd) {
    std::string token;
    std::string sfen;

    ssCmd >> token;

    if (token == "startpos") {
        sfen = DefaultStartPositionSFEN;
        ssCmd >> token; // "moves" が入力されるはず。
    }
    else if (token == "sfen") {
        while (ssCmd >> token && token != "moves")
            sfen += token + " ";
    }
    else
        return;

    pos.set(sfen);
    pos.searcher()->states = StateListPtr(new std::deque<StateInfo>(1));

    while (ssCmd >> token) {
        const Move move = usiToMove(pos, token);
        if (!move) break;
        pos.searcher()->states->push_back(StateInfo());
        pos.doMove(move, pos.searcher()->states->back());
    }
}

bool setPosition(Position& pos, const HuffmanCodedPos& hcp) {
    return pos.set(hcp);
}

void Searcher::setOption(std::istringstream& ssCmd) {
    std::string token;
    std::string name;
    std::string value;

    ssCmd >> token; // "name" が入力されるはず。

    ssCmd >> name;
    // " " が含まれた名前も扱う。
    while (ssCmd >> token && token != "value")
        name += " " + token;

    ssCmd >> value;
    // " " が含まれた値も扱う。
    while (ssCmd >> token)
        value += " " + token;

    if (!options.isLegalOption(name))
        std::cout << "No such option: " << name << std::endl;
    else
        options[name] = value;
}

#if !defined MINIMUL
// for debug
// 指し手生成の速度を計測
void measureGenerateMoves(const Position& pos) {
    pos.print();

    ExtMove legalMoves[MaxLegalMoves];
    for (int i = 0; i < MaxLegalMoves; ++i) legalMoves[i].move = moveNone();
    ExtMove* pms = &legalMoves[0];
    const u64 num = 5000000;
    Timer t = Timer::currentTime();
    if (pos.inCheck()) {
        for (u64 i = 0; i < num; ++i) {
            pms = &legalMoves[0];
            pms = generateMoves<Evasion>(pms, pos);
        }
    }
    else {
        for (u64 i = 0; i < num; ++i) {
            pms = &legalMoves[0];
            pms = generateMoves<CapturePlusPro>(pms, pos);
            pms = generateMoves<NonCaptureMinusPro>(pms, pos);
            pms = generateMoves<Drop>(pms, pos);
//          pms = generateMoves<PseudoLegal>(pms, pos);
//          pms = generateMoves<Legal>(pms, pos);
        }
    }
    const int elapsed = t.elapsed();
    std::cout << "elapsed = " << elapsed << " [msec]" << std::endl;
    if (elapsed != 0)
        std::cout << "times/s = " << num * 1000 / elapsed << " [times/sec]" << std::endl;
    const ptrdiff_t count = pms - &legalMoves[0];
    std::cout << "num of moves = " << count << std::endl;
    for (int i = 0; i < count; ++i)
        std::cout << legalMoves[i].move.toCSA() << ", ";
    std::cout << std::endl;
}
#endif
