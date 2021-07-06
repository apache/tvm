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
from tvm.script import ty
from tvm import te, tir
import numpy as np
import tvm.testing


@tvm.script.tir
def get_valid_counts(
    data: ty.handle,
    valid_count: ty.handle,
    out: ty.handle,
    out_indices: ty.handle,
    score_threshold: ty.float32,
    id_index: ty.int32,
    score_index: ty.int32,
) -> None:

    data_buf = tir.match_buffer(data, (1, 2500, 6), "float32")
    valid_count_buf = tir.match_buffer(valid_count, (1,), "int32")
    out_buf = tir.match_buffer(out, (1, 2500, 6), "float32")
    out_indices_buf = tir.match_buffer(out_indices, (1, 2500), "int32")

    with tir.block([1], "init") as [vi]:
        valid_count_buf[vi] = tir.int32(0)
        with tir.block([2500], "update") as [vj]:
            tir.reads([data_buf[vi, vj, 6]])
            tir.writes([valid_count_buf[vi], out_indices_buf[vi, vj], out_buf[vi, vj, 6]])
            if (data_buf[vi, vj, score_index] > score_threshold) and (
                (id_index < 0) or (data_buf[vi, vj, id_index] >= tir.float32(0))
            ):
                for k in tir.serial(0, 6):
                    out_buf[vi, valid_count_buf[vi], k] = data_buf[vi, vj, k]
                out_indices_buf[vi, valid_count_buf[vi]] = vj
                valid_count_buf[vi] = valid_count_buf[vi] + 1
            if vj >= valid_count_buf[vi]:
                for k in tir.serial(0, 6):
                    out_buf[vi, vj, k] = tir.float32(-1)
                out_indices_buf[vi, vj] = tir.int32(-1)


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
    score_threshold_data = tvm.nd.array(np.array([score_threshold], dtype=dtype), ctx)
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
    print(tvm.script.asscript(get_valid_counts))
    mod = tvm.script.create_module({"get_valid_counts": get_valid_counts})
    print(tvm.script.asscript(mod))
    # check building
    f = tvm.build(mod["get_valid_counts"], target=device)
    _check_get_valid_counts_with_numpy(f, (1, 2500, 6), 0.0, 0, 1)


@tvm.script.tir
def ops_with_buffer_slice_indices(data: ty.handle, index: ty.handle) -> None:
    data_buf = tir.match_buffer(data, (1, 5, 6), "float16")
    index_buf = tir.match_buffer(index, (1,), "int32")
    copy_buf = tir.alloc_buffer((1, 5), "float16")

    with tir.block([1, 5], "init") as [vi, vj]:
        copy_buf[vi, vj] = data_buf[vi, vj, index_buf[0]]


@tvm.script.tir
def substituted_ops_with_buffer_slice_indices(data: ty.handle, index: ty.handle) -> None:
    index_buf = tir.match_buffer(
        index, [1], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    data_buf = tir.match_buffer(
        data, [1, 5, 6], dtype="float16", elem_offset=0, align=128, offset_factor=1
    )

    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        copy_buf = tir.alloc_buffer(
            [1, 5], dtype="float16", elem_offset=0, align=128, offset_factor=1
        )
        for i0, i1 in tir.grid(1, 5):
            with tir.block([1, 5], "init") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                tir.reads([data_buf[vi, vj, 0:6], index_buf[0]])
                tir.writes([copy_buf[vi, vj]])
                copy_buf[vi, vj] = data_buf[vi, vj, index_buf[0]]


def test_buffer_slice_indices_func():
    new_func = tvm.script.from_source(tvm.script.asscript(ops_with_buffer_slice_indices))
    tvm.ir.assert_structural_equal(new_func, ops_with_buffer_slice_indices)


if __name__ == "__main__":
    test_get_valid_counts_script_func()
    test_buffer_slice_indices_func()
