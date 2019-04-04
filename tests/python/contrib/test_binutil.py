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
        rel_bin = tvm_callback_relocate_binary(tmp_bin, "0x0", "0x10000", "0x20000")
        print("Relocated binary section sizes")
        test_tvm_callback_get_section_size(rel_bin)
        relf = tmp_dir.relpath("rel.bin")
        with open(relf, "wb") as f:
            f.write(rel_bin)
        p1 = subprocess.Popen(["nm", "-C", "--defined-only", relf],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        (out, _) = p1.communicate()
        print("Relocated binary symbols")
        print(out)
        print
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
        print("Obtained symbol map")
        print(symbol_map)
    verify()


if __name__ == "__main__":
    prog_bin = make_binary()
    test_tvm_callback_get_section_size(prog_bin)
    test_tvm_callback_relocate_binary(prog_bin)
    test_tvm_callback_read_binary_section(prog_bin)
    test_tvm_callback_get_symbol_map(prog_bin)
