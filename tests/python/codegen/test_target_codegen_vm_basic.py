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
import tvm.testing
from tvm import te
from tvm.script import tir as T, ir as I

import numpy as np


def run_jit(fapi, check):
    for target in ["llvm", "stackvm"]:
        if not tvm.testing.device_enabled(target):
            continue
        f = tvm.driver.build(fapi, target=target)
        s = f.get_source()
        check(f)


def test_stack_vm_basic():
    a = tvm.nd.array(np.zeros(10, dtype="float32"))

    @tvm.register_func
    def tvm_call_back_get_shape(shape0):
        print(shape0)
        assert shape0 == a.shape[0]

    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((n,), "float32")
    stmt = tvm.tir.Evaluate(tvm.tir.call_packed("tvm_call_back_get_shape", Ab.shape[0]))

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "print_shape")
    )

    run_jit(mod, lambda f: f(a))


@tvm.register_func
def tvm_stack_vm_print(*x):
    print(x)


def test_stack_vm_loop():
    dtype = "int64"
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((n,), dtype)
    i = te.size_var("i")

    ib = tvm.tir.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    with ib.for_range(0, n - 1, "i") as i:
        A[i + 1] = A[i] + 1
        ib.emit(tvm.tir.call_packed("tvm_stack_vm_print", i))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "ramp"))
    a = tvm.nd.array(np.zeros(10, dtype=dtype))

    def check(f):
        f(a)
        np.testing.assert_equal(a.numpy(), np.arange(a.shape[0]))

    run_jit(mod, check)


def test_stack_vm_cond():
    dtype = "int64"
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((n,), dtype)

    ib = tvm.tir.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    with ib.for_range(0, n - 1, "i") as i:
        with ib.if_scope(tvm.tir.EQ(i, 4)):
            A[i + 1] = A[i] + 1
        with ib.else_scope():
            A[i + 1] = A[i] + 2

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "test"))

    def check(f):
        a = tvm.nd.array(np.zeros(10, dtype=dtype))
        f(a)
        y = np.arange(a.shape[0]) * 2
        y[5:] -= 1
        np.testing.assert_equal(a.numpy(), y)

    run_jit(mod, check)


def test_vm_parallel():
    dtype = "int64"
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((n,), dtype)
    i = te.size_var("i")
    ib = tvm.tir.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    with ib.for_range(0, n, "i", kind="parallel") as i:
        A[i] = A[i] + 1
    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "test"))

    def check(f):
        a = tvm.nd.array(np.zeros(10, dtype=dtype))
        f(a)
        np.testing.assert_equal(a.numpy(), np.ones(a.shape[0]))

    run_jit(mod, check)


def test_codegen_decl_buffer():
    """The codegen should accept DeclBuffer nodes in its input"""

    @I.ir_module
    class mod:
        @T.prim_func
        def kernel(A_data: T.handle("float32")):
            T.func_attr({"global_symbol": "kernel"})
            A_buf = T.decl_buffer([256], dtype="float32", scope="global", data=A_data)

    target = tvm.target.Target("stackvm")
    stackvm_codegen = tvm.get_global_func("target.build.stackvm")
    stackvm_codegen(mod, target)


if __name__ == "__main__":
    tvm.testing.main()
