# change here if you are using anaconda
PYTHON_PREFIX=/usr

CC = g++
CFLAGS = -std=c++17 -Wextra -Ofast -MMD -MP -fopenmp -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2 -fPIC
LDFLAGS = -lpthread -lz -flto
INCLUDE = -I$(PYTHON_PREFIX)/include/python3.7m -I$(PYTHON_PREFIX)/include
LIB = -L$(PYTHON_PREFIX)/lib -lpython3.7m -lboost_python -lboost_numpy -lpthread -ldl -lutil -lm -Xlinker -export-dynamic

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
