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
from tvm import te
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
from tvm import te
import numpy as np
path_dso = sys.argv[1]
dtype = sys.argv[2]
ff = tvm.runtime.load_module(path_dso)
a = tvm.nd.array(np.zeros(10, dtype=dtype))
ff(a)
np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
print("Finish runtime checking...")
"""

def test_dso_module_load():
    if not tvm.runtime.enabled("llvm"):
        return
    dtype = 'int64'
    temp = util.tempdir()

    def save_object(names):
        n = te.size_var('n')
        Ab = tvm.tir.decl_buffer((n, ), dtype)
        i = te.var('i')
        # for i in 0 to n-1:
        stmt = tvm.tir.For(
            i, 0, n - 1, 0, 0,
            tvm.tir.Store(Ab.data,
                           tvm.tir.Load(dtype, Ab.data, i) + 1,
                           i + 1))
        mod = tvm.IRModule.from_expr(
            tvm.tir.PrimFunc([Ab], stmt).with_attr(
                "global_symbol", "main")
        )
        m = tvm.driver.build(mod, target="llvm")
        for name in names:
            m.save(name)

    path_obj = temp.relpath("test.o")
    path_ll = temp.relpath("test.ll")
    path_bc = temp.relpath("test.bc")
    path_dso = temp.relpath("test.so")
    save_object([path_obj, path_ll, path_bc])
    cc.create_shared(path_dso, [path_obj])

    f1 = tvm.runtime.load_module(path_dso)
    f2 = tvm.runtime.load_module(path_ll)
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
        "python3 %s %s %s" % (path_runtime_py, path_dso, dtype),
        shell=True)


def test_device_module_dump():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name='A')
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = te.create_schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 8
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))

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
        # test cross compiler function
        f.export_library(path_dso, cc.cross_compiler("g++"))

        f1 = tvm.runtime.load_module(path_dso)
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        f1(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        if sys.platform != "win32":
            f2 = tvm.runtime.system_lib()
            f2[name](a, b)
            np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    def check_stackvm(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        temp = util.tempdir()
        name = "myadd_%s" % device
        f = tvm.build(s, [A, B], device, "stackvm", name=name)
        path_dso = temp.relpath("dev_lib.stackvm")
        f.export_library(path_dso)
        f1 = tvm.runtime.load_module(path_dso)
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        f(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    for device in ["cuda", "vulkan", "opencl", "metal"]:
        check_device(device)
        check_stackvm(device)

def test_combine_module_llvm():
    """Test combine multiple module into one shared lib."""
    # graph
    nn = 12
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name='A')
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = te.create_schedule(B.op)

    def check_llvm():
        ctx = tvm.cpu(0)
        if not tvm.runtime.enabled("llvm"):
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
        m = tvm.runtime.load_module(path_dso)
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
        if not tvm.runtime.enabled("llvm"):
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
        mm = tvm.runtime.system_lib()
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
