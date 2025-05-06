CC = g++
MATE_SEARCH_DEPTH = 5
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DPV_MATE_SEARCH -DWIN_TYPE_DOUBLE -DMATE_SEARCH_DEPTH=$(MATE_SEARCH_DEPTH) -DBOOK_POLICY -DTHREAD_POOL -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
NVCC_CFLAGS =
ifneq ($(NYUGYOKU_FEATURES),)
CFLAGS += -DNYUGYOKU_FEATURES
NVCC_CFLAGS += -DNYUGYOKU_FEATURES
endif
LDFLAGS = -lpthread -lnvinfer -lnvparsers -lnvonnxparser -lcudart -flto
INCLUDE = -I../usi -I../cppshogi -I/usr/local/cuda/include
LIB = -L/usr/local/cuda/lib64

target = bin/usi
target_make_book = bin/make_book
sources = main.cpp dfpn.cpp Message.cpp UctSearch.cpp Node.cpp nn_tensorrt.cpp PvMateSearch.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp dtype.cpp
cuda_sources = unpack.cu
objects = $(addprefix obj/, $(sources:.cpp=.o))
cppshogi_objects = $(addprefix obj/, $(cppshogi_sources:.cpp=.o))
cuda_objects = $(addprefix obj/, $(cuda_sources:.cu=.o))

$(target): $(objects) $(cppshogi_objects) $(cuda_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

obj/%.o: %.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: ../cppshogi/%.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

obj/%.o: ../usi/%.cu
	nvcc -x cu -I../cppshogi -maxrregcount=0 --machine 64 --compile -cudart static -DNDEBUG $(NVCC_CFLAGS) -Xcompiler "-Ofast" -o $@ $<

all: $(target)

clean:
	rm -f $(objects) $(cppshogi_objects) $(target) $(cuda_objects)

$(target_make_book): $(objects) $(cppshogi_objects) $(cuda_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

make_book: CFLAGS += -DMAKE_BOOK
make_book: $(target_make_book)
