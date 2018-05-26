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
CFLAGS += -Itvm/include -Itvm/dlpack/include -Itvm/HalideIR/src

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
	LDFLAGS += $(ADD_LDFLAGS)
endif

# plugin
PLUGIN_OBJ =
include $(NNVM_PLUGINS)

# specify tensor path
.PHONY: clean all test lint pylint doc cython cython3 cyclean

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	SHARED_LIBRARY_SUFFIX := dylib
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
	LDFLAGS += -undefined dynamic_lookup
else
	SHARED_LIBRARY_SUFFIX := so
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
endif

all: lib/libnnvm.a lib/libnnvm_top.$(SHARED_LIBRARY_SUFFIX)

SRC = $(wildcard src/*.cc src/c_api/*.cc src/core/*.cc src/pass/*.cc)
SRC_TOP = $(wildcard src/top/*/*.cc src/compiler/*.cc src/compiler/*/*.cc)
ALL_OBJ = $(patsubst %.cc, build/%.o, $(SRC))
TOP_OBJ = $(patsubst %.cc, build/%.o, $(SRC_TOP))
ALL_DEP = $(ALL_OBJ)

include tests/cpp/unittest.mk

test: $(TEST)

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/src/$*.o $< >build/src/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnnvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libnnvm_top.$(SHARED_LIBRARY_SUFFIX): lib/libnnvm.a ${TOP_OBJ}
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o, $^) $(LDFLAGS) -Wl,${WHOLE_ARCH} lib/libnnvm.a -Wl,${NO_WHOLE_ARCH}

cython:
	cd python; python setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/nnvm/*/*.so python/nnvm/*/*.dylib python/nnvm/*/*.cpp

lint: pylint
	python dmlc-core/scripts/lint.py nnvm cpp include src

pylint:
	pylint python/nnvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
