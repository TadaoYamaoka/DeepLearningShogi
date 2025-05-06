CC = g++
MATE_SEARCH_DEPTH = 5
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DONNXRUNTIME -DONNXRUNTIME_CPU -DPV_MATE_SEARCH -DWIN_TYPE_DOUBLE -DMATE_SEARCH_DEPTH=$(MATE_SEARCH_DEPTH) -DTHREAD_POOL -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
ifneq ($(NYUGYOKU_FEATURES),)
CFLAGS += -DNYUGYOKU_FEATURES
endif
LDFLAGS = -lpthread -lonnxruntime -flto
INCLUDE = -I../usi -I../usi_onnxruntime -I../cppshogi -I/work/onnxruntime-linux-x64-1.14.1/include
LIB = -L/work/onnxruntime-linux-x64-1.14.1/lib

target = bin/dlshogi_onnxruntime
target_makebook = bin/makebook
sources = nn_onnxruntime.cpp
usi_sources = main.cpp dfpn.cpp Message.cpp UctSearch.cpp Node.cpp PvMateSearch.cpp make_book.cpp USIBookEngine.cpp make_book_mcts.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp dtype.cpp
objects = $(addprefix obj/, $(sources:.cpp=.o))
usi_objects = $(addprefix obj/, $(usi_sources:.cpp=.o))
cppshogi_objects = $(addprefix obj/, $(cppshogi_sources:.cpp=.o))

$(target): $(objects) $(usi_objects) $(cppshogi_objects) $(cuda_objects)
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

$(target_makebook): $(objects) $(usi_objects) $(cppshogi_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

makebook: CFLAGS += -DMAKE_BOOK
makebook: $(target_makebook)
