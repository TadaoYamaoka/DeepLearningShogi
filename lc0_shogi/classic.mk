# Shared, explicit classic-only source boundary. Included by USI Makefiles.
CLASSIC_ROOT := ..
INCLUDE := -I$(CLASSIC_ROOT)/lc0_shogi/compat -I$(CLASSIC_ROOT)/lc0/src -I$(CLASSIC_ROOT)/lc0_shogi $(INCLUDE)
classic_sources := neural/shared_params.cc search/classic/search.cc search/classic/node.cc search/classic/params.cc \
 search/classic/stoppers/timemgr.cc search/classic/stoppers/stoppers.cc \
 utils/optionsdict.cc utils/optionsparser.cc utils/logging.cc utils/random.cc utils/string.cc \
 utils/commandline.cc utils/configfile.cc utils/filesystem.posix.cc
classic_objects := $(addprefix $(OBJECT_DIR)/classic/,$(classic_sources:.cc=.o))
adapter_sources := position_adapter.cpp rules.cpp backend_support.cpp classic_options.cpp backend.cpp search_session.cpp mate_coordinator.cpp
adapter_objects := $(addprefix $(OBJECT_DIR)/shogi/,$(adapter_sources:.cpp=.o))
objects += $(classic_objects) $(adapter_objects)
ifneq ($(filter Message.cpp UctSearch.cpp Node.cpp PvMateSearch.cpp,$(sources) $(usi_sources)),)
$(error Legacy USI search source included in classic build)
endif
$(OBJECT_DIR)/classic/%.o: $(CLASSIC_ROOT)/lc0/src/%.cc
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<
$(OBJECT_DIR)/shogi/%.o: $(CLASSIC_ROOT)/lc0_shogi/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<
