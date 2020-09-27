CC = g++
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DLEARN -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
LDFLAGS = -lpthread -lnvinfer -lnvparsers -lnvonnxparser -lcuda -lcudnn -lcudart -lcublas -lz -flto -lboost_system
INCLUDE = -I../usi -I../cppshogi -I/usr/local/cuda/include -I$(CONDA_PREFIX)/include
LIB = -L/usr/local/cuda-10.2/lib64 -L$(CONDA_PREFIX)/lib

target = bin/make_hcpe_by_self_play
sources = self_play.cpp ZobristHash.cpp USIEngine.cpp
usi_sources =  mate.cpp dfpn.cpp nn_fused_wideresnet10.cpp nn_senet10.cpp nn_wideresnet10.cpp nn_wideresnet15.cpp npz.cpp nn_tensorrt.cpp cudnn_dtype.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp
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
