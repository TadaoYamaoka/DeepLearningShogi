CC = g++
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DMATE_SEARCH_DEPTH=3 -DLEARN -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
LDFLAGS = -lpthread -lnvinfer -lnvparsers -lnvonnxparser -lcudart -flto -lboost_system
INCLUDE = -I../usi -I../cppshogi -I/usr/local/cuda/include -I$(CONDA_PREFIX)/include
LIB = -L/usr/local/cuda/lib64 -L$(CONDA_PREFIX)/lib

target = bin/selfplay
sources = self_play.cpp Node.cpp USIEngine.cpp
usi_sources = dfpn.cpp nn_tensorrt.cpp
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
