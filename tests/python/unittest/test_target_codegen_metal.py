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
import numpy as np

import tvm
import tvm.script
import tvm.testing
from tvm import te
from tvm.script import tir as T


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_metal_inf_nan():
    target = "metal"

    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        prim_func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(prim_func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float16")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float16")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_unaligned_vectorize():
    @tvm.script.ir_module
    class IRModule:
        @T.prim_func
        def main(A: T.Buffer((2, 3), "float32"), B: T.Buffer((6,), "float32")):
            T.func_attr({"global_symbol": "main"})
            for i0_1 in T.thread_binding(3, thread="threadIdx.x"):
                for i0_0 in T.vectorized(2):
                    with T.block("block"):
                        vi0 = T.axis.spatial(6, i0_0 * 3 + i0_1)
                        B[vi0] = A[vi0 // 3, vi0 % 3]

    target = "metal"
    dev = tvm.metal()

    a = (np.arange(6).reshape(2, 3)).astype("float32")
    a_nd = tvm.nd.array(a, dev)
    b_nd = tvm.nd.empty((6,), "float32", dev)
    f = tvm.build(IRModule, target=target)
    f(a_nd, b_nd)
    np.testing.assert_allclose(b_nd.numpy(), a.reshape(6), atol=1e-5, rtol=1e-5)


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_metal_erf():
    target = "metal"

    def check_erf(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        C = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="C")
        func = te.create_prim_func([A, C])
        sch = tvm.tir.Schedule(func)
        (x,) = sch.get_loops(sch.get_block("C"))
        sch.bind(x, "threadIdx.x")
        fun = tvm.build(sch.mod, target=target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_ramp():
    target = "metal"

    @tvm.script.ir_module
    class IRModule:
        @T.prim_func
        def main(A: T.Buffer((1, 2), "int32")):
            T.func_attr({"global_symbol": "main"})
            for i in T.thread_binding(1, thread="threadIdx.x"):
                with T.block("block"):
                    tx = T.axis.spatial(1, i)
                    r = T.ramp(tx, 3, 2)
                    A[0, T.ramp(0, 1, 2)] = r

    f = tvm.build(IRModule, target=target)
    dev = tvm.metal()
    a_nd = tvm.nd.empty((1, 2), "int32", dev)
    f(a_nd)
    assert tuple(a_nd.numpy()[0, :]) == (0, 3)


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_select_vectorize():
    @tvm.script.ir_module
    class IRModule:
        @T.prim_func
        def main(A: T.Buffer((6), "float32"), B: T.Buffer((6,), "float32")):
            T.func_attr({"global_symbol": "main"})
            for i0_1 in T.thread_binding(3, thread="threadIdx.x"):
                for i0_0 in T.vectorized(2):
                    with T.block("block"):
                        vi0 = T.axis.spatial(6, i0_0 * 3 + i0_1)
                        B[vi0] = T.Select((vi0 % 2) == 0, A[vi0], T.float32(0))

    target = "metal"
    dev = tvm.metal()
    a = np.arange(6).astype("float32")
    a_nd = tvm.nd.array(a, dev)
    b_nd = tvm.nd.empty((6,), "float32", dev)
    f = tvm.build(IRModule, target=target)
    f(a_nd, b_nd)
    a.reshape(3, 2)[:, 1] = 0
    np.testing.assert_allclose(b_nd.numpy(), a, atol=1e-5, rtol=1e-5)


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_vectorized_uint8():
    @T.prim_func
    def func(A: T.Buffer((16), "uint8"), B: T.Buffer((16), "float32")):
        for i in T.thread_binding(4, thread="threadIdx.x"):
            for j in T.vectorized(4):
                with T.block("block"):
                    vi = T.axis.spatial(16, i * 4 + j)
                    B[vi] = T.Cast("float32", A[vi])

    dev = tvm.metal()
    a = np.arange(16).astype("uint8")
    a_nd = tvm.nd.array(a, dev)
    b_nd = tvm.nd.empty((16,), "float32", dev)
    f = tvm.build(func, target="metal")
    f(a_nd, b_nd)
    np.testing.assert_allclose(b_nd.numpy(), a.astype("float32"), atol=1e-5, rtol=1e-5)


@tvm.testing.requires_metal(support_required="compile-only")
def test_func_with_trailing_pod_params():
    from tvm.contrib import xcode  # pylint: disable=import-outside-toplevel

    @T.prim_func
    def func(A: T.Buffer((16), "float32"), B: T.Buffer((16), "float32"), x: T.float32):
        for i in T.thread_binding(16, thread="threadIdx.x"):
            with T.block("block"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + x

    @tvm.register_func("tvm_callback_metal_compile")
    def compile_metal(src, target):
        return xcode.compile_metal(src)

    mod = tvm.IRModule({"main": func})

    f = tvm.build(mod, target="metal")
    src: str = f.imported_modules[0].get_source()
    occurrences = src.count("struct func_kernel_args_t")
    assert occurrences == 1, occurrences


if __name__ == "__main__":
    tvm.testing.main()
