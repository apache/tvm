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
"""Test flop calculation"""

import tvm
import numpy as np

from tvm.autotvm.task.task import compute_flop

def random_dtypes():
    """Return pair of (input, accumulator) dtypes"""
    candidates = [("float32", "float32"), ("float16", "float32"), ("int8", "int32")]
    return candidates[np.random.choice(len(candidates))]

def test_conv():
    for i in range(5):
        N, H, W, CO, CI, KH, KW = [np.random.randint(10, 32) for _ in range(7)]
        (input_dtype, acc_dtype) = random_dtypes()
        D = tvm.placeholder((N, CI, H, W), dtype=input_dtype)
        K = tvm.placeholder((CO, CI, KH, KW), dtype=input_dtype)

        KH = min(H, KH)
        KW = min(W, KW)

        ci = tvm.reduce_axis((0, CI))
        kh = tvm.reduce_axis((0, KH))
        kw = tvm.reduce_axis((0, KW))

        OH = (H - KH) + 1
        OW = (W - KW) + 1

        C = tvm.compute((N, CO, OH, OW), lambda n, co, h, w:
        tvm.sum(D[n][ci][h][w].astype(acc_dtype) * K[co][ci][h][w].astype(acc_dtype),
                axis=[ci, kh, kw]))

        s = tvm.create_schedule([C.op])

        assert compute_flop(s) == 2 * N * CO * OH * OW * CI * KH * KW

def test_pack_gemm():
    for i in range(5):
        N, L, M = [np.random.randint(10, 128) * 4 for _ in range(3)]
        (input_dtype, acc_dtype) = random_dtypes()
        A = tvm.placeholder((N, L), dtype=input_dtype)
        B = tvm.placeholder((M, L), dtype=input_dtype)
        k = tvm.reduce_axis((0, L))

        bn = 4
        A_pack = tvm.compute((N // bn, L, bn), lambda i, j, k: A[i * bn + k][j])
        B_pack = tvm.compute((M // bn, L, bn), lambda i, j, k: B[i * bn + k][j])
        C_pack = tvm.compute((N // bn, M // bn, bn, bn), lambda i, j, ii, jj:
        tvm.sum(A_pack[i, k, ii].astype(acc_dtype) * B_pack[j, k, jj].astype(acc_dtype), axis=[k]))
        C = tvm.compute((N, M), lambda i, j: C_pack[i // bn][j // bn][i % bn][j % bn])

        s = tvm.create_schedule([C.op])
        assert compute_flop(s) == 2 * N * L * M

def test_outer_dot():
    for i in range(5):
        N, M = [np.random.randint(10, 128) * 4 for _ in range(2)]
        (input_dtype, acc_dtype) = random_dtypes()
        A = tvm.placeholder((N,), dtype=input_dtype)
        B = tvm.placeholder((M,), dtype=input_dtype)

        C = tvm.compute((N, M), lambda i, j: A[i].astype(acc_dtype) * B[j].astype(acc_dtype))

        s = tvm.create_schedule([C.op])
        assert compute_flop(s) == N * M

def test_max_pool():
    for i in range(5):
        N, H, W, CO, CI, KH, KW = [np.random.randint(10, 32) for _ in range(7)]
        (input_dtype, _) = random_dtypes()
        D = tvm.placeholder((N, CI, H, W), dtype=input_dtype)

        KH = min(H, KH)
        KW = min(W, KW)

        kh = tvm.reduce_axis((0, KH))
        kw = tvm.reduce_axis((0, KW))

        OH = (H - KH) + 1
        OW = (W - KW) + 1

        C = tvm.compute(
            (N, CO, OH, OW),
            lambda n, co, h, w: tvm.max(D[n][co][h + kh][w + kw], axis=[kh, kw]))

        s = tvm.create_schedule([C.op])

        assert compute_flop(s) == N * CO * OH * OW * KH * KW

def test_average_pool():
    for i in range(5):
        N, H, W, CO, CI, KH, KW = [np.random.randint(10, 32) for _ in range(7)]
        (input_dtype, acc_dtype) = random_dtypes()
        D = tvm.placeholder((N, CI, H, W), dtype=input_dtype)

        KH = min(H, KH)
        KW = min(W, KW)

        kh = tvm.reduce_axis((0, KH))
        kw = tvm.reduce_axis((0, KW))

        OH = (H - KH) + 1
        OW = (W - KW) + 1

        C = tvm.compute(
            (N, CO, OH, OW),
            lambda n, co, h, w: tvm.sum(D[n][co][h + kh][w + kw].astype(acc_dtype) / (KW * KH), axis=[kh, kw]))

        s = tvm.create_schedule([C.op])

        assert compute_flop(s) == 2 * N * CO * OH * OW * KH * KW

def test_move():
    """No float number operation in simple move. So the estimator should raise an error """
    N = 1024

    A = tvm.placeholder((N,))
    C = tvm.compute((N,), lambda i: A[i])
    s = tvm.create_schedule([C.op])

    try:
        compute_flop(s)
        assert False
    except RuntimeError:
        pass

if __name__ == '__main__':
    test_conv()
    test_pack_gemm()
    test_outer_dot()
    test_move()
