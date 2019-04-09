# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
