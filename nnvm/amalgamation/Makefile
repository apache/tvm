export NNVM_ROOT=`pwd`/..
export CFLAGS = -std=c++11 -Wall -O2 -Iinclude -fPIC

ifdef DMLC_CORE_PATH
  CFLAGS += -I$(DMLC_CORE_PATH)/include
else
  CFLAGS += -I$(CURDIR)/../3rdparty/dmlc-core/include
endif

.PHONY: all clean

all: libnnvm.a

nnvm.cc:
	python generate.py $@

nnvm.d: nnvm.cc
	${CXX} ${CFLAGS} -M -MT nnvm.o \
		-I ${NNVM_ROOT}/ -I ${NNVM_ROOT}/include \
		-D__MIN__=$(MIN) $+ > nnvm.d

nnvm-all.cc: nnvm.d nnvm.cc
	python ./amalgamation.py $+ $@

nnvm-all.o: nnvm-all.cc
	${CXX} ${CFLAGS} -fPIC -o $@ -c $+

libnnvm.a: nnvm-all.o
	ar rcs $@ $+

clean:
	rm -f *.d *.o *.so *.a nnvm-all.cc nnvm.cc
