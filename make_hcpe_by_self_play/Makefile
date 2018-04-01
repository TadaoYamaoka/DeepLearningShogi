CC = g++
CFLAGS = -std=c++11 -Wextra -Ofast -MMD -MP -fopenmp -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
LDFLAGS = -lpthread -lcuda -lcudnn -lcudart -lcublas -flto
INCLUDE = -I../usi -I../cppshogi -I/usr/local/cuda/include
LIB = -L/usr/local/cuda-9.0/targets/x86_64-linux/lib

target = bin/make_hcpe_by_self_play
sources = self_play.cpp
usi_sources =  mate.cpp ZobristHash.cpp nn.cpp npz.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp evalList.cpp evaluate.cpp generateMoves.cpp hand.cpp init.cpp move.cpp movePicker.cpp mt64bit.cpp pieceScore.cpp position.cpp search.cpp square.cpp thread.cpp timeManager.cpp tt.cpp usi.cpp
objects = $(addprefix obj/, $(sources:.cpp=.o))
usi_objects = $(addprefix obj/, $(usi_sources:.cpp=.o))
cppshogi_objects = $(addprefix obj/, $(cppshogi_sources:.cpp=.o))

$(target): $(objects) $(usi_objects) $(cppshogi_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

obj/%.o: %.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: ../usi/%.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: ../cppshogi/%.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(target)

clean:
	rm -f $(objects) $(usi_objects) $(cppshogi_objects) $(target)