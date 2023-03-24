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
import numpy as np

from tvm.contrib.nvcc import have_fp16, have_int8, have_bf16
from tvm.contrib import nvcc
import tvm.testing
import tvm.script
from tvm.script import tir as T

tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_metal_inf_nan():
    target = "metal"

    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], target)
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
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], target)
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


if __name__ == "__main__":
    test_ramp()
    test_metal_inf_nan()
    test_metal_erf()
