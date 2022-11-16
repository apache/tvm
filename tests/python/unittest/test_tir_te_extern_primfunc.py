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

import sys
import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T


# TODO(csullivan): Additional tests cases needed:
# - PrimFunc with 1 arg, inplace update
# - PrimFunc with buffer that uses custom storage_scope


@T.prim_func
def func_1(A: T.Buffer[(16,), "float32"], C: T.Buffer[(1,), "float32"]):
    for i in T.serial(
        0,
        16,
    ):
        with T.block():
            B = T.alloc_buffer((1,), dtype="float32")
            with T.block():
                B[0] = A[i] * T.float32(2)
            with T.block():
                C[0] = C[0] + A[i] + B[0] + T.float32(1)
                A[i] = B[0] + T.float32(1)


def verify_func_1(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    a = tvm.nd.array(a_np, device=tvm.cpu(0))
    c = tvm.nd.array(c_np, device=tvm.cpu(0))

    module(a, c)
    tvm.testing.assert_allclose(c_np + np.sum(3 * a_np + 1), c.numpy(), rtol=1e-4)
    # also test in place update
    tvm.testing.assert_allclose(a_np * 2 + 1, a.numpy(), rtol=1e-4)


@T.prim_func
def func_2(
    C: T.Buffer[(1,), "float32"], A: T.Buffer[(16,), "float32"], D: T.Buffer[(2,), "float32"]
):
    for i in T.serial(
        0,
        16,
    ):
        with T.block():
            B = T.alloc_buffer((1,), dtype="float32")
            with T.block():
                B[0] = A[i] * T.float32(2)
            with T.block():
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_2(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    a = tvm.nd.array(a_np, device=tvm.cpu(0))
    d = tvm.nd.array(d_np, device=tvm.cpu(0))
    c = tvm.nd.array(c_np, device=tvm.cpu(0))

    module(c, a, d)
    tvm.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)


@T.prim_func
def func_3(
    C: T.Buffer[(1,), "float32"],
    A: T.Buffer[(16,), "float32"],
    D: T.Buffer[(2,), "float32"],
    E: T.Buffer[(16,), "float32"],
    F: T.Buffer[(16,), "float32"],
):
    for i in T.serial(
        0,
        16,
    ):
        with T.block():
            B = T.alloc_buffer((1,), dtype="float32")
            with T.block():
                B[0] = A[i] * T.float32(2)
            with T.block():
                E[i] = A[i]
                F[i] = E[i] + 1.0
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_3(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    e_np = np.zeros((16,), dtype=np.float32)
    f_np = np.zeros((16,), dtype=np.float32)
    a = tvm.nd.array(a_np, device=tvm.cpu(0))
    d = tvm.nd.array(d_np, device=tvm.cpu(0))
    c = tvm.nd.array(c_np, device=tvm.cpu(0))
    e = tvm.nd.array(e_np, device=tvm.cpu(0))
    f = tvm.nd.array(f_np, device=tvm.cpu(0))

    module(c, a, d, e, f)
    tvm.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np, e.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np + 1, f.numpy(), rtol=1e-4)


@T.prim_func
def func_4(
    C: T.Buffer[(1,), "float32"],
    A: T.Buffer[(16,), "float32"],
    F: T.Buffer[(16,), "float32"],
    D: T.Buffer[(2,), "float32"],
    E: T.Buffer[(16,), "float32"],
):
    for i in T.serial(
        0,
        16,
    ):
        with T.block():
            B = T.alloc_buffer((1,), dtype="float32")
            with T.block():
                B[0] = A[i] * T.float32(2)
            with T.block():
                E[i] = A[i]
                F[i] = E[i] + 1.0
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_4(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    e_np = np.zeros((16,), dtype=np.float32)
    f_np = np.zeros((16,), dtype=np.float32)
    a = tvm.nd.array(a_np, device=tvm.cpu(0))
    d = tvm.nd.array(d_np, device=tvm.cpu(0))
    c = tvm.nd.array(c_np, device=tvm.cpu(0))
    e = tvm.nd.array(e_np, device=tvm.cpu(0))
    f = tvm.nd.array(f_np, device=tvm.cpu(0))

    module(c, a, f, d, e)
    tvm.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np, e.numpy(), rtol=1e-4)
    tvm.testing.assert_allclose(a_np + 1, f.numpy(), rtol=1e-4)


class TestPrimFuncs:
    func, params, verify = tvm.testing.parameters(
        [func_1, ("A"), verify_func_1],
        [func_2, ("C", "D"), verify_func_2],
        [func_3, ("C", "A", "D", "E"), verify_func_3],
        [func_4, ("C", "A", "D", "E"), verify_func_4],
    )

    def test_primfunc_call(self, func, verify):
        target = tvm.target.Target("llvm")
        func = tvm.build(func, target=target)
        verify(func)

    def test_te_extern_call(self, func, params, verify):
        ir_mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        prim_func = ir_mod["main"]

        buf_name_map = {buf.name: buf for buf in func.buffer_map.values()}
        input_tensors = [te.placeholder(buf_name_map[name].shape) for name in params]
        output = te.extern_primfunc(input_tensors, prim_func)
        rt_prim_func = te.create_prim_func(tensors_from_extern_op(output, prim_func))
        tvm.ir.assert_structural_equal(tvm.lower(prim_func), tvm.lower(rt_prim_func))

        target = tvm.target.Target("llvm")
        func = tvm.build(rt_prim_func, target=target)
        verify(func)


def tensors_from_extern_op(extern, func):
    if isinstance(extern, list):
        output_tensors = extern
    else:
        output_tensors = [extern]
    output_buffers = []
    input_buffers = []
    input_tensors = []
    for ext in output_tensors:
        output_buffers.extend(ext.op.output_placeholders)
        input_buffers.extend(ext.op.input_placeholders)
        input_tensors.extend(ext.op.input_tensors)
    input_binds = dict(zip(input_buffers, input_tensors))
    output_binds = dict(zip(output_buffers, output_tensors))
    buffer_to_tensor = {**input_binds, **output_binds}
    ordered_tensors = []
    for var in func.params:
        buf = func.buffer_map[var]
        ordered_tensors.append(buffer_to_tensor[buf])
    return ordered_tensors


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
