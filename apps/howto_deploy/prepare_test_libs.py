"""Script to prepare test_addone.so"""
import tvm
import os

def prepare_test_libs(base_path):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    # Compile library as dynamic library
    fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
    dylib_path = os.path.join(base_path, "test_addone_dll.so")
    fadd_dylib.export_library(dylib_path)

    # Compile library in system library mode
    fadd_syslib = tvm.build(s, [A, B], "llvm --system-lib", name="addonesys")
    syslib_path = os.path.join(base_path, "test_addone_sys.o")
    fadd_syslib.save(syslib_path)

if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "./lib"))
