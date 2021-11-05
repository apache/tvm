TVM_ROOT=/home/sunway/source/tvm/
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

ifeq ("$(wildcard $(TVM_ROOT))","")
$(error TVM_ROOT ${TVM_ROOT} doesnt exist)
endif

.PHONY: clean all run

APP=double

CPPFLAGS = -std=c++14 -O2 -fPIC \
		-I${TVM_ROOT}/include \
		-I${TVM_ROOT}/src \
		-I${DMLC_CORE}/include \
		-I${TVM_ROOT}/3rdparty/dlpack/include \
		-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
		-g \

CXXFLAGS = -Wvla -ffunction-sections -fdata-sections
LDFLAGS = -L${TVM_ROOT}/build -static -Wl,-Map=${APP}.map -Wl,-gc-sections

LDLIBS = -lstdc++ -lpthread

all: ${APP}

libdouble.o: libdouble.py
	python3 libdouble.py

TVM_SRC= \
   ${TVM_ROOT}/src/runtime/c_runtime_api.cc \
   ${TVM_ROOT}/src/runtime/container.cc \
   ${TVM_ROOT}/src/runtime/cpu_device_api.cc \
   ${TVM_ROOT}/src/runtime/file_utils.cc \
   ${TVM_ROOT}/src/runtime/library_module.cc \
   ${TVM_ROOT}/src/runtime/logging.cc \
   ${TVM_ROOT}/src/runtime/module.cc \
   ${TVM_ROOT}/src/runtime/ndarray.cc \
   ${TVM_ROOT}/src/runtime/object.cc \
   ${TVM_ROOT}/src/runtime/registry.cc \
   ${TVM_ROOT}/src/runtime/system_library.cc \
   ${TVM_ROOT}/src/runtime/thread_pool.cc \
   ${TVM_ROOT}/src/runtime/threading_backend.cc \
   ${TVM_ROOT}/src/runtime/workspace_pool.cc \

TVM_OBJ=$(patsubst %.cc,%.o,$(TVM_SRC))

OBJ=double.o libdouble.o

${APP}: ${OBJ} ${TVM_OBJ}

clean:
	-rm ${OBJ}
	-rm ${TVM_OBJ}
	-rm ${APP}
	-rm ${APP}.map

run:${APP}
	${APP}
