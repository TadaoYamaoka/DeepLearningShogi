OBJECT_DIR ?= obj
.DEFAULT_GOAL := all
CC = g++
MATE_SEARCH_DEPTH = 5
CFLAGS = -std=c++20 -Wextra -Ofast -MMD -MP -fopenmp -DONNXRUNTIME -DONNXRUNTIME_CPU -DPV_MATE_SEARCH -DWIN_TYPE_DOUBLE -DMATE_SEARCH_DEPTH=$(MATE_SEARCH_DEPTH) -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
ifneq ($(NYUGYOKU_FEATURES),)
CFLAGS += -DNYUGYOKU_FEATURES
endif
LDFLAGS = -lpthread -lonnxruntime -flto
INCLUDE = -I../usi -I../usi_onnxruntime -I../cppshogi -I/work/onnxruntime-linux-x64-1.14.1/include
LIB = -L/work/onnxruntime-linux-x64-1.14.1/lib

target = bin/dlshogi_onnxruntime
sources = nn_onnxruntime.cpp
usi_sources = main.cpp dfpn.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp dtype.cpp
objects = $(addprefix $(OBJECT_DIR)/, $(sources:.cpp=.o))
usi_objects = $(addprefix $(OBJECT_DIR)/, $(usi_sources:.cpp=.o))
cppshogi_objects = $(addprefix $(OBJECT_DIR)/, $(cppshogi_sources:.cpp=.o))

include ../lc0_shogi/classic.mk

$(target): $(objects) $(usi_objects) $(cppshogi_objects) $(cuda_objects)
	@[ -d bin ] || mkdir -p bin
	$(CC) -o $@ $^ $(LIB) $(LDFLAGS) $(CFLAGS)

$(OBJECT_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJECT_DIR)/%.o: ../usi/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJECT_DIR)/%.o: ../cppshogi/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(target)

clean:
	rm -f $(objects) $(usi_objects) $(cppshogi_objects) $(target)

-include $(objects:.o=.d) $(usi_objects:.o=.d) $(cppshogi_objects:.o=.d)
