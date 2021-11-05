all: double

TVM_ROOT=/home/sunway/source/tvm
CRT_ROOT=${TVM_ROOT}/build/standalone_crt

DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
PKG_COMPILE_OPTS = -g -Wall -O2 -fPIC
CFLAGS = ${PKG_COMPILE_OPTS} \
	-I${TVM_ROOT}/include \
	-I${DMLC_CORE}/include \
	-I${TVM_ROOT}/3rdparty/dlpack/include \
	-I. \
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
	-ffunction-sections -fdata-sections

LDFLAGS = -static -Wl,-gc-sections

CRT_SRCS = $(shell find $(CRT_ROOT))

MODEL_OBJ = devc.o lib0.o lib1.o

${MODEL_OBJ}:libdouble.py
	python ./libdouble.py
	tar xvf /tmp/libdouble.tar

crt/libcommon.a: $(CRT_SRCS)
	cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath .)/crt CRT_CONFIG=$(abspath crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" common

crt/libmemory.a: $(CRT_SRCS)
	cd $(CRT_ROOT) && make QUIET= BUILD_DIR=$(abspath .)/crt CRT_CONFIG=$(abspath crt_config.h) "EXTRA_CFLAGS=$(PKG_COMPILE_OPTS)" memory

OBJS=double.o tvm_runtime.o
double: ${OBJS} $(MODEL_OBJ) crt/libmemory.a crt/libcommon.a

run:double
	./double

clean:
	-rm ${OBJS}
	-rm ${MODEL_OBJ}
	-rm double
	-rm -rf crt
