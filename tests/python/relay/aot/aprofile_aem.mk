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

# Makefile to build and run AOT tests against the AArch64
# reference system

CC = clang-16
LD = aarch64-none-elf-gcc

TARGET_ARCH = --target=aarch64-none-elf -march=armv9-a+sme
SYS_ROOT = /opt/arm/gcc-aarch64-none-elf/aarch64-none-elf/

OBJ_FILES := $(build_dir)/test.o $(build_dir)/aprofile_extra_support_routines.o
INCLUDES = -I$(SRC_DIR) \
           -I$(TVM_ROOT)/include \
           -I$(build_dir)/../include

ifneq ($(CODEGEN_ROOT),)
    OBJ_FILES := $(OBJ_FILES) $(wildcard $(CODEGEN_ROOT)/host/lib/*.o)
    INCLUDES := $(INCLUDES) -I$(CODEGEN_ROOT)/host/include
endif

ifneq ($(STANDALONE_CRT_DIR),)
    OBJ_FILES := $(OBJ_FILES) $(build_dir)/stack_allocator.o \
             $(build_dir)/crt_backend_api.o
    INCLUDES := $(INCLUDES) -isystem$(STANDALONE_CRT_DIR)/include
endif

PKG_LDFLAGS = --specs=$(SYS_ROOT)lib/aem-ve.specs --sysroot $(SYS_ROOT)
PKG_CFLAGS = $(INCLUDES) --sysroot $(SYS_ROOT) -c -O3 $(CFLAGS)
PKG_ASFLAGS = $(INCLUDES) --sysroot $(SYS_ROOT) -c

aot_test_runner: $(build_dir)/aot_test_runner

$(build_dir)/aot_test_runner: $(OBJ_FILES)
	$(LD) $(INCLUDES) $(PKG_LDFLAGS) -o $@ $^

$(build_dir)/test.o: $(build_dir)/test.c
	$(CC) $(TARGET_ARCH) $(PKG_CFLAGS) -o $@ $<

# TODO(lhutton1) This is a workaround while __arm_tpidr2_save and
# __arm_tpidr2_restore are not provided with the toolchain. More
# information in aprofile_extra_support_routines.c.
$(build_dir)/aprofile_extra_support_routines.o: ${AOT_TEST_ROOT}/aprofile_extra_support_routines.c
	$(CC) $(TARGET_ARCH) $(PKG_CFLAGS) -o $@ $<

$(build_dir)/stack_allocator.o: $(STANDALONE_CRT_DIR)/src/runtime/crt/memory/stack_allocator.c
	$(CC) $(TARGET_ARCH) $(PKG_CFLAGS) -o $@ $<

$(build_dir)/crt_backend_api.o: $(STANDALONE_CRT_DIR)/src/runtime/crt/common/crt_backend_api.c
	$(CC) $(TARGET_ARCH) $(PKG_CFLAGS) -o $@ $<

run: $(build_dir)/aot_test_runner
	$(FVP_DIR)/FVP_Base_RevC-2xAEMvA \
    -a $(build_dir)/aot_test_runner \
    --plugin $(FVP_DIR)../../plugins/Linux64_GCC-9.3/ScalableVectorExtension.so \
    -C SVE.ScalableVectorExtension.has_sme2=1 \
    -C SVE.ScalableVectorExtension.has_sme=1 \
    -C SVE.ScalableVectorExtension.has_sve2=1 \
    -C SVE.ScalableVectorExtension.enable_at_reset=1 \
    -C cluster0.has_arm_v9-2=1 \
    -C bp.secure_memory=false \
    -C bp.terminal_0.start_telnet=0 \
    -C bp.terminal_1.start_telnet=0 \
    -C bp.terminal_2.start_telnet=0 \
    -C bp.terminal_3.start_telnet=0 \
    -C bp.vis.disable_visualisation=1 \
    -C bp.pl011_uart0.out_file="-" \
    -C bp.pl011_uart0.shutdown_tag=\"EXITTHESIM\" \
    -C semihosting-enable=1

# Note: It's possible to trace instructions running on the FVP by adding the option
# --plugin /opt/arm/fvp/Base_RevC_AEMvA_pkg/plugins/Linux64_GCC-9.3/TarmacTrace.so

clean:
	rm -rf $(build_dir)/crt

cleanall:
	rm -rf $(build_dir)

.SUFFIXES:

.DEFAULT: aot_test_runner

.PHONY: run
