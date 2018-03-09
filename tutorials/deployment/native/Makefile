# Makefile to buile deploy_native utility.
TVM_ROOT=$(shell cd ../../..; pwd)
DMLC_CORE=${TVM_ROOT}/dmlc-core

LOCAL_CFLAGS = -std=c++11 -O2 -fPIC \
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\

LOCAL_LDFLAGS = -L${TVM_ROOT}/lib -ldl -ltvm_runtime

# Additional flags for Android
LOCAL_CFLAGS += -pie -Wl,-unresolved-symbols=ignore-in-shared-libs -fpermissive

# Cnpy lib dependency
LOCAL_LDFLAGS += -lcnpy -L$(CNPY_PATH)
LOCAL_CFLAGS += -I$(CNPY_PATH)

.PHONY: clean all

all: deploy_native

# Build rule for deploy_native object file
deploy_native.o: deploy_native.cc
	$(CXX) -c $(LOCAL_CFLAGS) -o $@  $^

# Build rule for deploy_native ulility
deploy_native: deploy_native.o
	$(CXX)  $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o $@ $^

clean:
	rm -rf deploy_native.o deploy_native
