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
# pylint: disable=missing-function-docstring
import ml_dtypes
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TileLayout

ml_dtypes_dict = {
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
    "bfloat16": ml_dtypes.bfloat16,
    "int4": ml_dtypes.int4,
}


@pytest.mark.parametrize(
    "task",
    [
        (
            (4, 32),  # a_shape
            TileLayout(S[4, 32]),  # layoutA
            tvm.cuda(0),
        ),
        (
            (4, 64),  # a_shape
            TileLayout(S[4, 64]),  # layoutA
            tvm.cuda(0),
        ),
        (
            (3, 64),  # a_shape
            TileLayout(S[3, 64]),  # layoutA
            tvm.cuda(0),
        ),
        (
            (9, 64),  # a_shape
            TileLayout(S[9, 64]),  # layoutA
            tvm.cuda(0),
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "float16", "int32"])
def test_vectorized_permute_dims_2d(task, dtype):
    a_shape, layoutA, dev = task
    list(slice(None) for _ in range(len(a_shape)))

    # fmt: off
    @Tx.prim_func
    def permute_dims(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=layoutA)

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([32])
            with Tx.cta():
                with Tx.warp():
                    Tx.permute_dims(A, [1, 0])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": permute_dims})

        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, a_shape)

        A = tvm.runtime.tensor(A_np, dev)
        mod(A)
        A_ref = np.transpose(A_np, (1, 0)).reshape(a_shape)
        np.testing.assert_allclose(A_ref.flatten(), A.numpy().flatten())


@pytest.mark.parametrize(
    "task",
    [
        (
            (1, 4, 32),  # a_shape
            TileLayout(S[1, 4, 32]),  # layoutA
            [0, 0, 0],
            [1, 4, 32],
            tvm.cuda(0),
        ),
        (
            (2, 2, 8, 64),  # a_shape
            TileLayout(S[2, 2, 8, 64]),  # layoutA
            [1, 1, 0, 0],
            [1, 1, 8, 64],
            tvm.cuda(0),
        ),
        ((1, 10, 40), TileLayout(S[1, 10, 40]), [0, 5, 3], [1, 4, 32], tvm.cuda(0)),
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "float16", "int32"])
def test_vectorized_permute_dims_nd(task, dtype):
    a_shape, layoutA, st, extent, dev = task
    ndim = len(a_shape)
    region = list(slice(st[i], st[i] + extent[i]) for i in range(ndim))
    order = [*list(range(ndim - 2)), ndim - 1, ndim - 2]

    # fmt: off
    @Tx.prim_func
    def permute_dims(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=layoutA)

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([32])
            with Tx.cta():
                with Tx.warp():
                    Tx.permute_dims(A[tuple(region)], order)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": permute_dims})

        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, a_shape)

        A = tvm.runtime.tensor(A_np, dev)
        mod(A)
        A_ref = A_np.copy()
        A_ref[tuple(region)] = np.transpose(A_np[tuple(region)], order).reshape(extent)
        np.testing.assert_allclose(A_ref.flatten(), A.numpy().flatten())


if __name__ == "__main__":
    tvm.testing.main()
