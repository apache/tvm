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
from tvm.contrib import cc, utils, popen_pool
import sys
import numpy as np
import subprocess
import tvm.testing
import pytest

runtime_py = """
import os
import sys

os.environ["TVM_USE_RUNTIME_LIB"] = "1"
import tvm
from tvm import te
import numpy as np
path_dso = sys.argv[1]
dtype = sys.argv[2]
ff = tvm.runtime.load_module(path_dso)
a = tvm.nd.array(np.zeros(10, dtype=dtype))
ff(a)
np.testing.assert_equal(a.numpy(), np.arange(a.shape[0]))
print("Finish runtime checking...")
"""


@tvm.testing.requires_llvm
@pytest.mark.parametrize("target", ["llvm", "llvm -jit=mcjit"])
def test_dso_module_load(target):
    dtype = "int64"
    temp = utils.tempdir()

    def save_object(names):
        n = te.size_var("n")
        Ab = tvm.tir.decl_buffer((n,), dtype)
        i = te.var("i")
        # for i in 0 to n-1:
        stmt = tvm.tir.For(
            i,
            0,
            n - 1,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.BufferStore(Ab, tvm.tir.BufferLoad(Ab, [i]) + 1, [i + 1]),
        )
        mod = tvm.IRModule.from_expr(
            tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "main")
        )
        m = tvm.tir.build(mod, target=target)
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
    np.testing.assert_equal(a.numpy(), np.arange(a.shape[0]))
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f2(a)
    np.testing.assert_equal(a.numpy(), np.arange(a.shape[0]))

    path_runtime_py = temp.relpath("runtime.py")
    with open(path_runtime_py, "w") as fo:
        fo.write(runtime_py)

    proc = subprocess.run(
        [sys.executable, path_runtime_py, path_dso, dtype],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.returncode == 0, f"{proc.args} exited with {proc.returncode}: {proc.stdout}"


@tvm.testing.requires_gpu
def test_device_module_dump():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")

    sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
    # create iter var and assign them tags.
    num_thread = 8
    bx, tx = sch.split(sch.get_loops("B")[0], factors=[None, num_thread])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        temp = utils.tempdir()
        f = tvm.compile(sch.mod, target=device)

        path_dso = temp.relpath("dev_lib.so")
        # test cross compiler function
        f.export_library(path_dso, fcompile=cc.cross_compiler("g++"))

        def popen_check():
            import tvm

            f1 = tvm.runtime.load_module(path_dso)
            a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
            b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
            f1(a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        # system lib should be loaded in different process
        worker = popen_pool.PopenWorker()
        worker.send(popen_check)
        worker.recv()

    def check_c(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        f = tvm.compile(sch.mod, target=tvm.target.Target(device, host="c"))
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
        f["main"](a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    for device in ["cuda", "vulkan", "opencl", "metal"]:
        check_device(device)
        check_c(device)


@tvm.testing.requires_llvm
def test_combine_module_llvm():
    """Test combine multiple module into one shared lib."""
    # graph
    nn = 12
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    mod1 = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "myadd1"))
    mod2 = tvm.IRModule.from_expr(te.create_prim_func([A, B]).with_attr("global_symbol", "myadd2"))

    def check_llvm():
        dev = tvm.cpu(0)
        temp = utils.tempdir()
        fadd1 = tvm.tir.build(mod1, "llvm")
        fadd2 = tvm.tir.build(mod2, "llvm")
        path1 = temp.relpath("myadd1.o")
        path2 = temp.relpath("myadd2.o")
        path_dso = temp.relpath("mylib.so")
        fadd1.save(path1)
        fadd2.save(path2)
        # create shared library with multiple functions
        cc.create_shared(path_dso, [path1, path2])
        m = tvm.runtime.load_module(path_dso)
        fadd1 = m["myadd1"]
        fadd2 = m["myadd2"]
        a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), dev)
        fadd1(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)
        fadd2(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    def check_system_lib():
        dev = tvm.cpu(0)
        if not tvm.testing.device_enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        temp = utils.tempdir()
        print("Running popen check")
        fadd1 = tvm.tir.build(mod1.with_attr("system_lib_prefix", ""), "llvm")
        fadd2 = tvm.tir.build(mod2.with_attr("system_lib_prefix", ""), "llvm")
        path1 = temp.relpath("myadd1.o")
        path2 = temp.relpath("myadd2.o")
        path_dso = temp.relpath("mylib.so")
        fadd1.save(path1)
        fadd2.save(path2)
        cc.create_shared(path_dso, [path1, path2])

        def popen_check():
            import tvm.runtime
            import ctypes

            # Load dll, will trigger system library registration
            ctypes.CDLL(path_dso)
            # Load the system wide library
            mm = tvm.runtime.system_lib()
            a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), dev)
            b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), dev)
            mm["myadd1"](a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)
            mm["myadd2"](a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        # system lib should be loaded in different process
        worker = popen_pool.PopenWorker()
        worker.send(popen_check)
        worker.recv()

    if sys.platform != "win32":
        check_system_lib()
    check_llvm()


if __name__ == "__main__":
    test_combine_module_llvm()
