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

#ifndef APERY_BOOK_HPP
#define APERY_BOOK_HPP

#include "position.hpp"
#include "mt64bit.hpp"

namespace book {
    // 移動方向
    enum MOVE_DIRECTION {
        UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
        UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
        MOVE_DIRECTION_NUM
    };
    // 指し手を表すインデックスの数
    constexpr int MAX_MOVE_INDEX_NUM = (MOVE_DIRECTION_NUM + HandPieceNum) * SquareNum;
}

struct BookEntry {
    Key key;
    u16 fromToPro;
    u16 count;
    Score score;
};

class Book : private std::ifstream {
public:
    Book() : random_(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) {}
    std::tuple<Move, Score> probe(const Position& pos, const std::string& fName, const bool pickBest);
    std::tuple<Move, Score> probeConsideringDraw(const Position& pos, const std::string& fName);
    std::tuple<Move, Score> probe(const Position& pos, const std::string& fName, const bool pickBest, const bool considerDraw);
    static void init();
    static Key bookKey(const Position& pos);
    static Key bookKeyAfter(const Position& pos, const Key key, const Move move);
    static Key bookKeyConsideringDraw(const Position& pos);

private:
    bool open(const char* fName);
    void binary_search(const Key key);

    static MT64bit mt64bit_; // 定跡のhash生成用なので、seedは固定でデフォルト値を使う。
    MT64bit random_; // 時刻をseedにして色々指すようにする。
    std::string fileName_;
    size_t size_;

    static Key ZobPiece[PieceNone][SquareNum];
    static Key ZobHand[HandPieceNum][19];
    static Key ZobTurn;
    static Key ZobMove[book::MAX_MOVE_INDEX_NUM];
};

void makeBook(Position& pos, std::istringstream& ssCmd);

#endif // #ifndef APERY_BOOK_HPP
