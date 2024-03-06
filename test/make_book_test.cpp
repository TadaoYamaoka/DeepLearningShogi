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


#endif