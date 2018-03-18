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

ifdef NNVM_PATH
	CFLAGS += -I$(NNVM_PATH)/include
else
	NNVM_PATH = $(ROOTDIR)/nnvm
	CFLAGS += -I$(NNVM_PATH)/include
endif

ifdef TVM_PATH
	CFLAGS += -I$(TVM_PATH)/include -I$(TVM_PATH)/dlpack/include -I$(TVM_PATH)/HalideIR/src
else
	TVM_PATH = $(NNVM_PATH)/tvm
	CFLAGS += -I$(TVM_PATH)/include -I$(TVM_PATH)/dlpack/include -I$(TVM_PATH)/HalideIR/src
endif

ifdef DMLC_CORE_PATH
  CFLAGS += -I$(DMLC_CORE_PATH)/include
else
  CFLAGS += -I$(NNVM_PATH)/dmlc-core/include
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

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


all: lib/libvta.$(SHARED_LIBRARY_SUFFIX)

VTA_LIB_SRC = $(wildcard src/*.cc src/tvm/*.cc)
ifeq ($(TARGET), PYNQ_TARGET)
	VTA_LIB_SRC += $(wildcard src/pynq/*.cc)
	LDFLAGS += -L/usr/lib -lsds_lib
	LDFLAGS += -L/opt/python3.6/lib/python3.6/site-packages/pynq/drivers/ -l:libdma.so
endif
VTA_LIB_OBJ = $(patsubst %.cc, build/%.o, $(VTA_LIB_SRC))

test: $(TEST)

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/src/$*.o $< >build/src/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libvta.$(SHARED_LIBRARY_SUFFIX): $(VTA_LIB_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o, $^) $(LDFLAGS)

lint: pylint cpplint

cpplint:
	python nnvm/dmlc-core/scripts/lint.py vta cpp include src

pylint:
	pylint python/vta --rcfile=$(ROOTDIR)/tests/lint/pylintrc

doc:
	doxygen docs/Doxyfile

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o


-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
