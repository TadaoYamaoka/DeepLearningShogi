CC = g++
CFLAGS = -g -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
ifneq ($(NYUGYOKU_FEATURES),)
CFLAGS += -DNYUGYOKU_FEATURES
endif
LDFLAGS = -lpthread -lnvinfer -lnvparsers -lnvonnxparser -lcudart -flto
INCLUDE = -I../usi -I../cppshogi -I../cppshogi -I/usr/local/cuda/include
LIB = -L/usr/local/cuda/lib64

target = bin/test
sources = test.cpp gpubenchmark.cpp
usi_sources = dfpn.cpp Message.cpp UctSearch.cpp Node.cpp nn_tensorrt.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp dtype.cpp
cuda_sources = unpack.cu
objects = $(addprefix obj/, $(sources:.cpp=.o))
usi_objects = $(addprefix obj/, $(usi_sources:.cpp=.o))
cppshogi_objects = $(addprefix obj/, $(cppshogi_sources:.cpp=.o))
cuda_objects = $(addprefix obj/, $(cuda_sources:.cu=.o))

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

obj/%.o: ../usi/%.cu
	nvcc -x cu -I../cppshogi -maxrregcount=0 --machine 64 --compile -cudart static -DNDEBUG -Xcompiler "-Ofast" -o $@ $<

all: $(target)

clean:
	rm -f $(objects) $(usi_objects) $(cppshogi_objects) $(target)
