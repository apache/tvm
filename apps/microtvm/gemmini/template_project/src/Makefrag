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
