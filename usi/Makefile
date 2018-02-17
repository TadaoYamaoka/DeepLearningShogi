CC = g++
CFLAGS = -std=c++14 -Wextra -Ofast -MMD -MP -fopenmp -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
LDFLAGS = -lpthread -lboost_python3 -lboost_numpy3 -lpython3.6m -flto
INCLUDE = -I../usi -I../cppshogi -I$(PYENV_ROOT)/versions/anaconda3-5.0.1/include/python3.6m
LIB = -L$(PYENV_ROOT)/versions/anaconda3-5.0.1/lib

target = bin/usi
sources = main.cpp mate.cpp Message.cpp UctSearch.cpp ZobristHash.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp evalList.cpp evaluate.cpp generateMoves.cpp hand.cpp init.cpp move.cpp movePicker.cpp mt64bit.cpp pieceScore.cpp position.cpp search.cpp square.cpp thread.cpp timeManager.cpp tt.cpp usi.cpp
objects = $(addprefix obj/, $(sources:.cpp=.o))
cppshogi_objects = $(addprefix obj/, $(cppshogi_sources:.cpp=.o))

$(target): $(objects) $(cppshogi_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

obj/%.o: %.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: ../cppshogi/%.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(target)

clean:
	rm -f $(objects) $(cppshogi_objects) $(target)