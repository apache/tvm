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

XLEN ?= 64

CC_BAREMETAL := riscv$(XLEN)-unknown-elf-gcc

CC_LINUX_PRESENT := $(shell command -v riscv$(XLEN)-unknown-linux-gnu-gcc 2> /dev/null)

# Support Linux gcc from riscv-gnu-toolchain and from system packages
# riscv64-unknown-linux-gnu-gcc is built from riscv-gnu-toolchain, comes with Firesim's tools
# riscv64-linux-gnu-gcc comes from a system package
ifdef CC_LINUX_PRESENT
    CC_LINUX := riscv$(XLEN)-unknown-linux-gnu-gcc
else
    CC_LINUX := riscv$(XLEN)-linux-gnu-gcc
endif

ENV_P = $(abs_top_srcdir)/riscv-tests/env/p
ENV_V = $(abs_top_srcdir)/riscv-tests/env/v

.PHONY: all clean default

default: all
src_dir = .

clean:
	rm -rf $(junk)
