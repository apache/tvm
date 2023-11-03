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
import tvm.testing
from tvm.script import tir as T


@T.prim_func
def get_valid_counts(
    data: T.handle,
    valid_count: T.handle,
    out: T.handle,
    out_indices: T.handle,
    score_threshold: T.float32,
    id_index: T.int32,
    score_index: T.int32,
) -> None:

    data_buf = T.match_buffer(data, (1, 2500, 6), "float32")
    valid_count_buf = T.match_buffer(valid_count, (1,), "int32")
    out_buf = T.match_buffer(out, (1, 2500, 6), "float32")
    out_indices_buf = T.match_buffer(out_indices, (1, 2500), "int32")

    with T.block("init"):
        vi = T.axis.S(1, 0)
        valid_count_buf[vi] = T.int32(0)
        for j in range(2500):
            with T.block("update"):
                vj = T.axis.S(2500, j)
                T.reads([data_buf[vi, vj, 6]])
                T.writes([valid_count_buf[vi], out_indices_buf[vi, vj], out_buf[vi, vj, 6]])
                if (data_buf[vi, vj, score_index] > score_threshold) and (
                    (id_index < 0) or (data_buf[vi, vj, id_index] >= T.float32(0))
                ):
                    for k in T.serial(0, 6):
                        out_buf[vi, valid_count_buf[vi], k] = data_buf[vi, vj, k]
                    out_indices_buf[vi, valid_count_buf[vi]] = vj
                    valid_count_buf[vi] = valid_count_buf[vi] + 1
                if vj >= valid_count_buf[vi]:
                    for k in T.serial(0, 6):
                        out_buf[vi, vj, k] = T.float32(-1)
                    out_indices_buf[vi, vj] = T.int32(-1)


def _check_get_valid_counts_with_numpy(f, dshape, score_threshold, id_index, score_index):
    dtype = "float32"
    ctx = tvm.cpu()
    batch_size, num_anchor, elem_length = dshape
    np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
    np_out1 = np.zeros(shape=(batch_size,), dtype="int32")
    np_out2 = np.zeros(shape=dshape).astype(dtype)
    np_out3 = np.zeros(shape=(batch_size, num_anchor), dtype="int32")
    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor):
            score = np_data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                for k in range(elem_length):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                np_out3[i, inter_idx] = j
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(elem_length):
                    np_out2[i, j, k] = -1.0
                np_out3[i, j] = -1

    in_data = tvm.nd.array(np_data, ctx)
    out1 = tvm.nd.array(np_out1, ctx)
    out2 = tvm.nd.array(np_out2, ctx)
    out3 = tvm.nd.array(np_out3, ctx)
    f(in_data, out1, out2, out3, score_threshold, id_index, score_index)
    tvm.testing.assert_allclose(out1.numpy(), np_out1, rtol=1e-5)
    tvm.testing.assert_allclose(out2.numpy(), np_out2, rtol=1e-5)
    tvm.testing.assert_allclose(out3.numpy(), np_out3, rtol=1e-5)
    print("test get_valid_counts end")


def test_get_valid_counts_script_func():
    device = "llvm"
    # check lowering
    print(get_valid_counts.script())
    mod = tvm.ir.IRModule({"get_valid_counts": get_valid_counts})
    print(mod.script())
    # check building
    f = tvm.build(mod["get_valid_counts"], target=device)
    _check_get_valid_counts_with_numpy(f, (1, 2500, 6), 0.0, 0, 1)


@T.prim_func
def alloc_zero_dim_buffer(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [], dtype="float32")
    B = T.match_buffer(b, [], dtype="float32")
    # body
    # tir.with block("root")
    C = T.alloc_buffer([], dtype="float32")
    A[()] = T.float32(2)
    C[()] = A[()] + B[()]
    B[()] = C[()]


@T.prim_func
def alloc_zero_dim_buffer_block(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (), "float32")
    B = T.match_buffer(b, (), "float32")
    with T.block("root"):
        T.reads([])
        T.writes([])
        C = T.alloc_buffer((), "float32")
        A[()] = T.float32(2)
        C[()] = A[()] + B[()]
        B[()] = C[()]


def _check_alloc_zero_dim_buffer(f):
    dtype = "float32"
    ctx = tvm.cpu()

    np_data = np.zeros(shape=()).astype(dtype)
    np_out = np.zeros(shape=()).astype(dtype)
    tvm_data = tvm.nd.array(np_data, ctx)
    tvm_out = tvm.nd.array(np_out, ctx)

    # np func exection
    np_inter = np.array(1)
    np_data[()] = 2.0
    np_inter[()] = np_data[()] + np_out[()]
    np_out[()] = np_inter[()]

    # tvm func execution
    f(tvm_data, tvm_out)
    tvm.testing.assert_allclose(tvm_out.numpy(), np_out, rtol=1e-5)


def test_alloc_zero_dim_buffer_round_trip():
    func = alloc_zero_dim_buffer
    func_with_block = alloc_zero_dim_buffer_block
    rt_func = tvm.script.from_source(func.script())
    rt_func_with_block = tvm.script.from_source(func_with_block.script())
    rt_mod = tvm.build(rt_func, "llvm")
    rt_mod_with_block = tvm.build(rt_func_with_block, "llvm")
    tvm.ir.assert_structural_equal(
        func.with_attr("global_symbol", "main"), func_with_block.with_attr("global_symbol", "main")
    )
    tvm.ir.assert_structural_equal(
        rt_func.with_attr("global_symbol", "main"),
        rt_func_with_block.with_attr("global_symbol", "main"),
    )
    _check_alloc_zero_dim_buffer(rt_mod)
    _check_alloc_zero_dim_buffer(rt_mod_with_block)


@T.prim_func
def ceildiv_test(A: T.Buffer(16, "int32")):
    for i in range(16):
        A[i] = T.ceildiv(A[i], 4)


@tvm.testing.requires_llvm
def test_ceildiv():
    f = tvm.build(ceildiv_test, "llvm")
    a = tvm.nd.array(np.arange(16).astype("int32"))
    f(a)
    ref = (np.arange(16) + 3) // 4
    tvm.testing.assert_allclose(a.numpy(), ref)


@T.prim_func
def slice_op_test(
    A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32"), C: T.Buffer((10,), "uint32")
):
    B[0:5] = A[0:5] + B[0:5]
    B[0:5] = A[0:5] - B[0:5]
    B[0:5] = A[0:5] * B[0:5]
    B[0:5] = A[0:5] / B[0:5]
    C[0:5] = C[0:5] % T.broadcast(T.uint32(5), 5)
    B[0:5] = -B[0:5]
    C[0:5] = C[0:5] >> 4
    C[0:5] = C[0:5] << 4
    C[0:5] = C[0:5] << C[0:5]
    C[0:5] = C[0:5] >> C[0:5]
    T.evaluate(A[0:5] > B[0:5])
    T.evaluate(A[0:5] > 5)
    T.evaluate(A[0:5] >= B[0:5])
    T.evaluate(A[0:5] >= 5)
    T.evaluate(A[0:5] < B[0:5])
    T.evaluate(A[0:5] < 5)
    T.evaluate(A[0:5] <= B[0:5])
    T.evaluate(A[0:5] <= 5)
    T.evaluate(A[0:5] == B[0:5])
    T.evaluate(A[0:5] == 5)
    T.evaluate(A[0:5] != B[0:5])
    T.evaluate(A[0:5] != 5)
    T.evaluate((A[0:5] > 0) and (B[0:5] > 0))
    T.evaluate((A[0:5] > 0) or (B[0:5] > 0))
    T.evaluate((A[0:5] < 0) and (1 > 0))
    T.evaluate((A[0:5] > 0) or (1 > 0))


@T.prim_func
def slice_op_test_ref(
    A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32"), C: T.Buffer((10,), "uint32")
):
    B[0:5] = A[0:5] + B[0:5]
    B[0:5] = A[0:5] - B[0:5]
    B[0:5] = A[0:5] * B[0:5]
    B[0:5] = A[0:5] / B[0:5]
    C[0:5] = C[0:5] % T.Broadcast(T.uint32(5), 5)
    B[0:5] = B[0:5] * T.Broadcast(T.float32(-1), 5)
    C[0:5] = T.shift_right(C[0:5], T.Broadcast(T.uint32(4), 5))
    C[0:5] = T.shift_left(C[0:5], T.Broadcast(T.uint32(4), 5))
    C[0:5] = T.shift_left(C[0:5], C[0:5])
    C[0:5] = T.shift_right(C[0:5], C[0:5])
    T.evaluate(A[0:5] > B[0:5])
    T.evaluate(A[0:5] > T.Broadcast(T.float32(5), 5))
    T.evaluate(A[0:5] >= B[0:5])
    T.evaluate(A[0:5] >= T.Broadcast(T.float32(5), 5))
    T.evaluate(A[0:5] < B[0:5])
    T.evaluate(A[0:5] < T.Broadcast(T.float32(5), 5))
    T.evaluate(A[0:5] <= B[0:5])
    T.evaluate(A[0:5] <= T.Broadcast(T.float32(5), 5))
    T.evaluate(A[0:5] == B[0:5])
    T.evaluate(A[0:5] == T.Broadcast(T.float32(5), 5))
    T.evaluate(A[0:5] != B[0:5])
    T.evaluate(A[0:5] != T.Broadcast(T.float32(5), 5))
    T.bitwise_and(A[0:5] > T.Broadcast(T.float32(0), 5), B[0:5] > T.Broadcast(T.float32(0), 5))
    T.bitwise_or(A[0:5] > T.Broadcast(T.float32(0), 5), B[0:5] > T.Broadcast(T.float32(0), 5))
    T.bitwise_and(A[0:5] < T.Broadcast(T.float32(0), 5), T.Broadcast(T.bool(1), 5))
    T.bitwise_or(A[0:5] > T.Broadcast(T.float32(0), 5), T.Broadcast(T.bool(1), 5))


def test_slice_op():
    tvm.ir.assert_structural_equal(
        slice_op_test.with_attr("global_symbol", "main"),
        slice_op_test_ref.with_attr("global_symbol", "main"),
    )


if __name__ == "__main__":
    test_get_valid_counts_script_func()
    test_alloc_zero_dim_buffer_round_trip()
    test_slice_op()
