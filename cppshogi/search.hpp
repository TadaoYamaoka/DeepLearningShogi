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

#ifndef APERY_SEARCH_HPP
#define APERY_SEARCH_HPP

#include "move.hpp"
#include "usi.hpp"

// 時間や探索深さの制限を格納する為の構造体
struct LimitsType {
	LimitsType() {
		nodes = time[Black] = time[White] = inc[Black] = inc[White] = movesToGo = moveTime = mate = infinite = ponder = 0;
	}
	bool useTimeManagement() const { return !(mate | moveTime | nodes | infinite); }

	int time[ColorNum], inc[ColorNum], movesToGo, moveTime, mate, infinite, ponder;
	s64 nodes;
	Timer startTime;
};

struct Searcher {
    // static メンバ関数からだとthis呼べないので代わりに thisptr を使う。
    // static じゃないときは this を入れることにする。
    STATIC Searcher* thisptr;
	STATIC LimitsType limits;
	STATIC StateListPtr states;

    STATIC OptionsMap options;

    STATIC void init();

    STATIC void setOption(std::istringstream& ssCmd);
};

#endif // #ifndef APERY_SEARCH_HPP
