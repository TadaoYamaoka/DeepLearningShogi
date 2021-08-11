CC = g++
MATE_SEARCH_DEPTH = 5
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DLIBTORCH -DPV_MATE_SEARCH -DWIN_TYPE_DOUBLE -DMATE_SEARCH_DEPTH=$(MATE_SEARCH_DEPTH) -DTHREAD_POOL -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2 -D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS = -flto -Wl,-rpath,/home/ubuntu/libtorch_demo/libtorch/lib /home/ubuntu/libtorch_demo/libtorch/lib/libtorch.so /home/ubuntu/libtorch_demo/libtorch/lib/libc10.so /home/ubuntu/libtorch_demo/libtorch/lib/libneuron_op.so -Wl,--no-as-needed,"/home/ubuntu/libtorch_demo/libtorch/lib/libtorch_cpu.so" -Wl,--as-needed /home/ubuntu/libtorch_demo/libtorch/lib/libc10.so -lpthread -Wl,--no-as-needed,"/home/ubuntu/libtorch_demo/libtorch/lib/libtorch.so" -Wl,--as-needed
INCLUDE = -I../usi_libtorch -I../usi -I../cppshogi -I../cppshogi -I/home/ubuntu/libtorch_demo/libtorch/include -I/home/ubuntu/libtorch_demo/libtorch/include/torch/csrc/api/include
LIB = -L/home/ubuntu/libtorch_demo/libtorch/lib

target = bin/usi
target_make_book = bin/make_book
sources = nn_libtorch.cpp
usi_sources = main.cpp dfpn.cpp Message.cpp UctSearch.cpp Node.cpp PvMateSearch.cpp
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
	rm -f $(objects) $(cppshogi_objects) $(target)

$(target_make_book): $(objects) $(cppshogi_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

make_book: CFLAGS += -DMAKE_BOOK -DWIN_TYPE_DOUBLE
make_book: $(target_make_book)
