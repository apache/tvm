ROOTDIR = $(CURDIR)

ifndef config
ifneq ("$(wildcard ./config.mk)", "")
	config = config.mk
else
	config = make/config.mk
endif
endif
include $(config)

export LDFLAGS = -pthread -lm
export CFLAGS = -std=c++11 -Wall -O2 -Iinclude -fPIC

ifdef DMLC_CORE_PATH
  CFLAGS += -I$(DMLC_CORE_PATH)/include
else
  CFLAGS += -I$(ROOTDIR)/dmlc-core/include
endif

ifdef DMLC_CORE_PATH
  CFLAGS += -I$(DMLC_CORE_PATH)/include
else
  CFLAGS += -I$(ROOTDIR)/dmlc-core/include
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LFFLAGS += $(ADD_LDFLAGS)
endif

# plugin
PLUGIN_OBJ =
include $(NNVM_PLUGINS)

# specify tensor path
.PHONY: clean all test lint doc cython cython3 cyclean

all: lib/libnnvm.a lib/libnnvm_example.so

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst %.cc, build/%.o, $(SRC))
ALL_DEP = $(ALL_OBJ) $(PLUGIN_OBJ)

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
else
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
endif


include tests/cpp/unittest.mk

test: $(TEST)

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/src/$*.o $< >build/src/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

build/plugin/%.o: plugin/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/plugin/$*.o $< >build/plugin/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnnvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libnnvm_example.so: example/src/operator.cc lib/libnnvm.a
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.cc, $^) $(LDFLAGS) -Wl,${WHOLE_ARCH} lib/libnnvm.a -Wl,${NO_WHOLE_ARCH}

cython:
	cd python; python setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/nnvm/*/*.so python/nnvm/*/*.cpp

lint:
	python2 dmlc-core/scripts/lint.py nnvm cpp include src

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
