export NNVM_ROOT=`pwd`/..
export CFLAGS = -std=c++11 -Wall -O3 -msse2 -Wno-unknown-pragmas -funroll-loops -Iinclude -fPIC

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
