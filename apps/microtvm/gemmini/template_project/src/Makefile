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

include $(abs_top_srcdir)/Makefrag

tests_baremetal = $(tests:=-baremetal)

ifeq ($(findstring spike,$(RUNNER)),spike)
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif

RISCV_TESTS = ${TVM_HOME}/3rdparty/gemmini/software/gemmini-rocc-tests/riscv-tests
BENCH_COMMON = ${RISCV_TESTS}/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h
STANDALONE_CRT = $(abs_top_srcdir)/standalone_crt

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-lgcc \
	-I${RISCV_TESTS} \
	-I${RISCV_TESTS}/env \
	-I$(abs_top_srcdir) \
	-I$(abs_top_srcdir)/include \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-DPRINT_TILE=0 \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(abs_top_srcdir)/model/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS) -lm

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/$^

junk += $(tests_baremetal)
