CC = g++
CFLAGS = -std=c++14 -Wextra -Ofast -MMD -MP -fopenmp -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2 -fPIC
LDFLAGS = -lpthread -lz -flto
INCLUDE = -I/usr/include/python3.6m
LIB = -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -L/usr/lib -lpython3.6m -lboost_python-py36 -lboost_numpy3-py36 -lpthread -ldl -lutil -lm -Xlinker -export-dynamic

target = ../dlshogi/cppshogi.so
sources = cppshogi.cpp python_module.cpp bitboard.cpp book.cpp common.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp
objects = $(addprefix obj/, $(sources:.cpp=.o))

$(target): $(objects)
	$(CC) -shared -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

obj/%.o: %.cpp
	@[ -d obj ] || mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(target)

clean:
	rm -f $(objects) $(target)