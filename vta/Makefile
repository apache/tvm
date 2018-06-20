ROOTDIR = $(CURDIR)

export LDFLAGS = -pthread -lm
export CFLAGS = -std=c++11 -Wall -O2 -Iinclude -fPIC

VTA_CONFIG = python make/vta_config.py
CFLAGS += `${VTA_CONFIG} --cflags`
LDFLAGS += `${VTA_CONFIG} --ldflags`
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

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


VTA_LIB_SRC = $(wildcard src/*.cc)

ifeq (${VTA_TARGET}, pynq)
	VTA_LIB_SRC += $(wildcard src/pynq/*.cc)
endif

ifeq (${VTA_TARGET}, sim)
	VTA_LIB_SRC += $(wildcard src/sim/*.cc)
endif

VTA_LIB_OBJ = $(patsubst src/%.cc, build/%.o, $(VTA_LIB_SRC))

all: lib/libvta.so lib/libvta.so.json

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libvta.so.json: lib/libvta.so
	@mkdir -p $(@D)
	${VTA_CONFIG} --cfg-json > $@

lib/libvta.so: $(VTA_LIB_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o, $^) $(LDFLAGS)


lint: pylint cpplint

cpplint:
	python3 tvm/dmlc-core/scripts/lint.py vta cpp include src

pylint:
	python3 -m pylint python/vta --rcfile=$(ROOTDIR)/tests/lint/pylintrc

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o
	$(RM) -rf cat.jpg quantize_graph.json quantize_params.pkl synset.txt


-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
