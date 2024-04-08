#ifdef MAKE_BOOK

#include "gtest/gtest.h"

#include "cppshogi.h"
#include "make_book.h"

TEST(MakeBookTest, minmax_book_to_cache) {
	initTable();

	Position pos;
	pos.set(DefaultStartPositionSFEN);
	std::unordered_map<Key, std::vector<BookEntry> > bookMap;
	read_book(R"(F:\book\book_pre44_10m-merged.bin)", bookMap);

	std::unordered_map<Key, std::vector<BookEntry> > minmaxBookMap;
	read_book(R"(F:\book\book_pre44_10m-minmax-all.bin)", minmaxBookMap);

	minmax_book_to_cache(pos, bookMap, minmaxBookMap, R"(R:\out.cache)", 1.0 / 0.1);
}

TEST(MakeBookTest, select_priority_book_entry) {
	extern std::tuple<Move, Score> select_priority_book_entry(Position & pos, const Key key, const std::unordered_map<Key, std::vector<BookEntry> >&bookMapBest, double temperature);

	initTable();

	Position pos;
	pos.set(DefaultStartPositionSFEN);
	auto states = StateListPtr(new std::deque<StateInfo>(1));
	std::stringstream ssPosCmd("2g2f 8c8d 2f2e 8d8e 7g7f 4a3b 8h7g 3c3d 7i6h 2b7g+ 6h7g 3a2b 3i3h 2b3c 3g3f 7a6b 4g4f 5a4b 6i7h 7c7d 3h4g 1c1d 1g1f 8a7c 5i6h 6c6d 2i3g 9c9d 9g9f 6b6c 2h2i 6a6b 4i4h 8b8a 4g5f 6c5d 6g6f 4b5b 6h6i 4c4d 6i7i 8a3a 7i8h 3a4a");

	std::string token;
	while (ssPosCmd >> token) {
		const Move move = usiToMove(pos, token);
		if (!move) break;
		pos.doMove(move, states->emplace_back());
	}


	std::unordered_map<Key, std::vector<BookEntry> > bookMapBest;
	read_book(R"(F:\book\book_pre44_10m-policy.bin)", bookMapBest);

	Move move;
	Score score;
	std::tie(move, score) = select_priority_book_entry(pos, Book::bookKey(pos), bookMapBest, 1.0);

	std::cout << move.toUSI() << " " << score << std::endl;
	ASSERT_NE("4h3h", move.toUSI());
}


#endif