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
"""Test various utilities for interaction with compiled binaries.

Specifically, we test the following capabilities:
  - querying the size of a binary section
  - relocating sections within a binary to new addresses
  - reading the contents of a binary section
  - querying the address of a symbol in the binary
"""

import tvm
from tvm import te
import subprocess
from tvm.contrib import util
from tvm.contrib import cc
from tvm.contrib.binutil import *

TOOLCHAIN_PREFIX = ""


def make_binary():
    prog = "int a = 7; \
            int main() { \
                int b = 5; \
                return 0; \
            }"
    tmp_dir = util.tempdir()
    tmp_source = tmp_dir.relpath("source.c")
    tmp_obj = tmp_dir.relpath("obj.obj")
    with open(tmp_source, "w") as f:
        f.write(prog)
    cc.create_executable(tmp_obj, tmp_source, [], cc="{}gcc".format(TOOLCHAIN_PREFIX))
    prog_bin = bytearray(open(tmp_obj, "rb").read())
    return prog_bin


def test_tvm_callback_get_section_size(binary=None):
    if binary is None:
        binary = make_binary()
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)

    def verify():
        print(
            "Text section size: %d"
            % tvm_callback_get_section_size(tmp_bin, "text", TOOLCHAIN_PREFIX)
        )
        print(
            "Data section size: %d"
            % tvm_callback_get_section_size(tmp_bin, "data", TOOLCHAIN_PREFIX)
        )
        print(
            "Bss section size: %d" % tvm_callback_get_section_size(tmp_bin, "bss", TOOLCHAIN_PREFIX)
        )
        print()

    verify()


def test_tvm_callback_relocate_binary():
    binary = make_binary()
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)

    def verify():
        word_size = 8
        text_loc = 0x0
        rodata_loc = 0x10000
        data_loc = 0x20000
        bss_loc = 0x30000
        stack_end = 0x50000
        rel_bin = tvm_callback_relocate_binary(
            tmp_bin, word_size, text_loc, rodata_loc, data_loc, bss_loc, stack_end, TOOLCHAIN_PREFIX
        )
        print("Relocated binary section sizes")
        test_tvm_callback_get_section_size(binary=rel_bin)
        relf = tmp_dir.relpath("rel.bin")
        with open(relf, "wb") as f:
            f.write(rel_bin)
        nm_proc = subprocess.Popen(
            ["nm", "-C", "--defined-only", relf], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out, _) = nm_proc.communicate()
        symbol_entries = out.decode("utf-8").split("\n")
        for entry in symbol_entries:
            if len(entry) == 0:
                continue
            sym_loc, section, sym_name = entry.split(" ")
            sym_loc = int(sym_loc, 16)
            if section == "T":  # text
                assert sym_loc >= text_loc and sym_loc < data_loc
            elif section == "D":  # data
                assert sym_loc >= data_loc and sym_loc < bss_loc
            elif section == "B":  # bss
                assert sym_loc >= bss_loc

    verify()


def test_tvm_callback_read_binary_section():
    binary = make_binary()

    def verify():
        text_bin = tvm_callback_read_binary_section(binary, "text", TOOLCHAIN_PREFIX)
        data_bin = tvm_callback_read_binary_section(binary, "data", TOOLCHAIN_PREFIX)
        bss_bin = tvm_callback_read_binary_section(binary, "bss", TOOLCHAIN_PREFIX)
        print("Read text section part of binary? %r" % (text_bin in binary))
        print("Read data section part of binary? %r" % (data_bin in binary))
        print("Read bss section part of binary? %r" % (bss_bin in binary))
        print()

    verify()


def test_tvm_callback_get_symbol_map():
    binary = make_binary()
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)

    def verify():
        word_size = 8
        text_loc = 0x0
        rodata_loc = 0x10000
        data_loc = 0x20000
        bss_loc = 0x30000
        stack_end = 0x50000
        rel_bin = tvm_callback_relocate_binary(
            tmp_bin, word_size, text_loc, rodata_loc, data_loc, bss_loc, stack_end, TOOLCHAIN_PREFIX
        )
        symbol_map = tvm_callback_get_symbol_map(rel_bin, TOOLCHAIN_PREFIX)
        symbols = set()
        for i, line in enumerate(symbol_map.split("\n")):
            # Every other line is the value the symbol maps to.
            if i % 2 == 0:
                symbols.add(line)
        assert "a" in symbols
        assert "main" in symbols

    verify()


if __name__ == "__main__":
    test_tvm_callback_get_section_size()
    test_tvm_callback_relocate_binary()
    test_tvm_callback_read_binary_section()
    test_tvm_callback_get_symbol_map()
