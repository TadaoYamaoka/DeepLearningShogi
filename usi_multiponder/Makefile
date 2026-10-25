OBJECT_DIR ?= obj
.DEFAULT_GOAL := all
CC = g++
MATE_SEARCH_DEPTH = 5
CFLAGS = -std=c++20 -Wextra -Ofast -MMD -MP -fopenmp -DMULTI_PONDER -DPV_MATE_SEARCH -DWIN_TYPE_DOUBLE -DMATE_SEARCH_DEPTH=$(MATE_SEARCH_DEPTH) -DNDEBUG -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -msse4.2 -mbmi2 -DHAVE_AVX2 -mavx2
NVCC_CFLAGS =
ifneq ($(NYUGYOKU_FEATURES),)
CFLAGS += -DNYUGYOKU_FEATURES
NVCC_CFLAGS += -DNYUGYOKU_FEATURES
endif
LDFLAGS = -lpthread -lnvinfer -lnvparsers -lnvonnxparser -lcudart -flto -lboost_system
INCLUDE = -I../usi_multiponder -I../usi -I../cppshogi -I/usr/local/cuda/include -I$(CONDA_PREFIX)/include
LIB = -L/usr/local/cuda/lib64 -L$(CONDA_PREFIX)/lib

target = bin/usi_multiponder
sources = USIPonderEngine.cpp
usi_sources = main.cpp dfpn.cpp nn_tensorrt.cpp
cppshogi_sources = bitboard.cpp book.cpp common.cpp cppshogi.cpp generateMoves.cpp hand.cpp init.cpp move.cpp mt64bit.cpp position.cpp search.cpp square.cpp usi.cpp dtype.cpp
cuda_sources = unpack.cu
objects = $(addprefix $(OBJECT_DIR)/, $(sources:.cpp=.o))
usi_objects = $(addprefix $(OBJECT_DIR)/, $(usi_sources:.cpp=.o))
cppshogi_objects = $(addprefix $(OBJECT_DIR)/, $(cppshogi_sources:.cpp=.o))
cuda_objects = $(addprefix $(OBJECT_DIR)/, $(cuda_sources:.cu=.o))

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

$(OBJECT_DIR)/%.o: ../usi/%.cu
	@mkdir -p $(@D)
	nvcc -x cu -I../cppshogi -maxrregcount=0 --machine 64 --compile -cudart static -DNDEBUG $(NVCC_CFLAGS) -Xcompiler "-Ofast" -o $@ $<

all: $(target)

clean:
	rm -f $(objects) $(usi_objects) $(cppshogi_objects) $(target) $(cuda_objects)

-include $(objects:.o=.d) $(usi_objects:.o=.d) $(cppshogi_objects:.o=.d)
