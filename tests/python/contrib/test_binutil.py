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

import tvm
import subprocess
from tvm.contrib import util
from tvm.contrib import cc
from tvm.contrib.binutil import *


def make_binary():
    prog = "int a = 7; \
            int main() { \
                int b = 5; \
                return 0; \
            }"
    tmp_dir = util.tempdir()
    tmp_source = tmp_dir.relpath("source.c")
    tmp_obj = tmp_dir.relpath("obj.o")
    with open(tmp_source, "w") as f:
        f.write(prog)
    p1 = subprocess.Popen(["gcc", "-c", tmp_source, "-o", tmp_obj],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    p1.communicate()
    prog_bin = bytearray(open(tmp_obj, "rb").read())
    return prog_bin


def test_tvm_callback_get_section_size(binary):
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)
    def verify():
        print("Text section size: %d" % tvm_callback_get_section_size(tmp_bin, "text"))
        print("Data section size: %d" % tvm_callback_get_section_size(tmp_bin, "data"))
        print("Bss section size: %d" % tvm_callback_get_section_size(tmp_bin, "bss"))
        print
    verify()


def test_tvm_callback_relocate_binary(binary):
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)
    def verify():
        text_loc_str = "0x0"
        data_loc_str = "0x10000"
        bss_loc_str = "0x20000"
        rel_bin = tvm_callback_relocate_binary(tmp_bin, text_loc_str, data_loc_str, bss_loc_str)
        print("Relocated binary section sizes")
        test_tvm_callback_get_section_size(rel_bin)
        relf = tmp_dir.relpath("rel.bin")
        with open(relf, "wb") as f:
            f.write(rel_bin)
        nm_proc = subprocess.Popen(["nm", "-C", "--defined-only", relf],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        (out, _) = nm_proc.communicate()
        # Ensure the relocated symbols are within the ranges we specified.
        text_loc = int(text_loc_str, 16)
        data_loc = int(data_loc_str, 16)
        bss_loc = int(bss_loc_str, 16)
        symbol_entries = out.decode("utf-8").split("\n")
        for entry in symbol_entries:
            if len(entry) == 0:
                continue
            sym_loc, section, sym_name = entry.split(' ')
            sym_loc = int(sym_loc, 16)
            if section == 'T':  # text
                assert sym_loc >= text_loc and sym_loc < data_loc
            elif section == 'D':  # data
                assert sym_loc >= data_loc and sym_loc < bss_loc
            elif section == 'B':  # bss
                assert sym_loc >= bss_loc
    verify()


def test_tvm_callback_read_binary_section(binary):
    def verify():
        text_bin = tvm_callback_read_binary_section(binary, "text")
        data_bin = tvm_callback_read_binary_section(binary, "data")
        bss_bin = tvm_callback_read_binary_section(binary, "bss")
        print("Read text section part of binary? %r" % (text_bin in binary))
        print("Read data section part of binary? %r" % (data_bin in binary))
        print("Read bss section part of binary? %r" % (bss_bin in binary))
        print
    verify()


def test_tvm_callback_get_symbol_map(binary):
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("obj.bin")
    with open(tmp_bin, "wb") as f:
        f.write(binary)
    def verify():
        rel_bin = tvm_callback_relocate_binary(tmp_bin, "0x0", "0x10000", "0x20000")
        symbol_map = tvm_callback_get_symbol_map(rel_bin)
        symbols = set()
        for i, line in enumerate(symbol_map.split('\n')):
            # Every other line is the value the symbol maps to.
            if i % 2 == 0:
                symbols.add(line)
        assert "a" in symbols
        assert "main" in symbols
    verify()


if __name__ == "__main__":
    prog_bin = make_binary()
    test_tvm_callback_get_section_size(prog_bin)
    test_tvm_callback_relocate_binary(prog_bin)
    test_tvm_callback_read_binary_section(prog_bin)
    test_tvm_callback_get_symbol_map(prog_bin)
