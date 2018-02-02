import tvm
from tvm.contrib import cc, util
import ctypes
import os
import sys
import numpy as np
import subprocess

runtime_py = """
import os
import sys

os.environ["TVM_USE_RUNTIME_LIB"] = "1"
os.environ["TVM_FFI"] = "ctypes"
import tvm
import numpy as np
path_dso = sys.argv[1]
dtype = sys.argv[2]
ff = tvm.module.load(path_dso)
a = tvm.nd.array(np.zeros(10, dtype=dtype))
ff(a)
np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
print("Finish runtime checking...")
"""

def test_dso_module_load():
    if not tvm.module.enabled("llvm"):
        return
    dtype = 'int64'
    temp = util.tempdir()

    def save_object(names):
        n = tvm.var('n')
        Ab = tvm.decl_buffer((n, ), dtype)
        i = tvm.var('i')
        # for i in 0 to n-1:
        stmt = tvm.make.For(
            i, 0, n - 1, 0, 0,
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1))
        fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0, True)
        fapi = tvm.ir_pass.LowerTVMBuiltin(fapi)
        m = tvm.codegen.build_module(fapi, "llvm")
        for name in names:
            m.save(name)

    path_obj = temp.relpath("test.o")
    path_ll = temp.relpath("test.ll")
    path_bc = temp.relpath("test.bc")
    path_dso = temp.relpath("test.so")
    save_object([path_obj, path_ll, path_bc])
    cc.create_shared(path_dso, [path_obj])

    f1 = tvm.module.load(path_dso)
    f2 = tvm.module.load(path_ll)
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f1(a)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f2(a)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))

    path_runtime_py = temp.relpath("runtime.py")
    with open(path_runtime_py, "w") as fo:
        fo.write(runtime_py)

    subprocess.check_call(
        "python %s %s %s" % (path_runtime_py, path_dso, dtype),
        shell=True)


def test_device_module_dump():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 8
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        temp = util.tempdir()
        name = "myadd_%s" % device
        if sys.platform == "darwin" or sys.platform.startswith('linux'):
            f = tvm.build(s, [A, B], device, "llvm -system-lib", name=name)
        elif sys.platform == "win32":
            f = tvm.build(s, [A, B], device, "llvm", name=name)
        else:
            raise ValueError("Unsupported platform")

        path_dso = temp.relpath("dev_lib.so")
        f.export_library(path_dso)

        f1 = tvm.module.load(path_dso)
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        f1(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        if sys.platform != "win32":
            f2 = tvm.module.system_lib()
            f2[name](a, b)
            np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    check_device("cuda")
    check_device("vulkan")
    check_device("opencl")
    check_device("metal")


def test_combine_module_llvm():
    """Test combine multiple module into one shared lib."""
    # graph
    nn = 12
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    def check_llvm():
        ctx = tvm.cpu(0)
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled" )
            return
        temp = util.tempdir()
        fadd1 = tvm.build(s, [A, B], "llvm", name="myadd1")
        fadd2 = tvm.build(s, [A, B], "llvm", name="myadd2")
        path1 = temp.relpath("myadd1.o")
        path2 = temp.relpath("myadd2.o")
        path_dso = temp.relpath("mylib.so")
        fadd1.save(path1)
        fadd2.save(path2)
        # create shared library with multiple functions
        cc.create_shared(path_dso, [path1, path2])
        m = tvm.module.load(path_dso)
        fadd1 = m['myadd1']
        fadd2 = m['myadd2']
        a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
        fadd1(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        fadd2(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    def check_system_lib():
        ctx = tvm.cpu(0)
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled" )
            return
        temp = util.tempdir()
        fadd1 = tvm.build(s, [A, B], "llvm -system-lib", name="myadd1")
        fadd2 = tvm.build(s, [A, B], "llvm -system-lib", name="myadd2")
        path1 = temp.relpath("myadd1.o")
        path2 = temp.relpath("myadd2.o")
        path_dso = temp.relpath("mylib.so")
        fadd1.save(path1)
        fadd2.save(path2)
        cc.create_shared(path_dso, [path1, path2])
        # Load dll, will trigger system library registration
        dll = ctypes.CDLL(path_dso)
        # Load the system wide library
        mm = tvm.module.system_lib()
        a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
        mm['myadd1'](a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        mm['myadd2'](a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    if sys.platform != "win32":
        check_system_lib()
    check_llvm()



if __name__ == "__main__":
    test_combine_module_llvm()
    test_device_module_dump()
    test_dso_module_load()
