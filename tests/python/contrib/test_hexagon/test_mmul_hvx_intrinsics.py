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

from numpy.random import default_rng
from tvm.script import tir as T
from tvm.tir.function import TensorIntrin

from tests.python.contrib.test_hexagon.mmul_unit8_hvx_intrin import get_mm_uint8_intrin
from tests.python.contrib.test_hexagon.quantization_utils import quantize_array, quantize_uint8

UNROLL_FACTOR = 4  # This must match the hard-coded unrolling in mm_uint8_intrinsic().

def can_tensorize(n, m, k):
    return m % (4 * UNROLL_FACTOR) == k % 32 == 0

def blockify_matrix(B):
    """
    inputs
    ------
    B        : numpy 2D array (of M x K ) to be blockified

    outputs
    -------
    BB       : blockified B as array of dimensions (M/4) x K x 4

    blockification is in preparation for HVX ops on 128B 'vectors'
    assuming input is of type int8 or uint8, and matrix multiplication uses vrmpy to accumulate to int32
    - once for each block - then K/32 such vectors will be required to carry out the operation.

    if B is height x width = M x K, then output is array of K blocks in x direction,
    i.e. as [block1, block2, ... blockK ] where each block has dimensions M/4 x 4.
    Specifically:

     B = [ B[1,1]     B[1,2]      ...   B[1,K] ]
         [ B[2,1]     B[2,2]      ...   B[2,K] ]
                         .
                         .
                         .
         [ B[M,1]     B[M,2]      ...   B[M,K] ]


     BB =                                 |                                   |     |
                                          |                                   |     |
     [  B[1,1]   B[2,1]   B[3,1]   B[4,1] | B[1,2]   B[2,2]   B[3,2]   B[4,2] | ... | B[1,K]   B[2,K]   B[3,K]   B[4,K] ]
     [  B[5,1]   B[6,1]   B[7,1]   B[8,1] | B[5,2]   B[6,2]   B[7,2]   B[8,2] | ... | B[5,K]   B[6,K]   B[7,K]   B[8,K] ]
                        .                 |                 .                 |  .  |                 .
                        .                 |                 .                 |  .  |                 .
                        .                 |                 .                 |  .  |                 .
     [  B[M-3,1] B[M-2,1] B[M-1,1] B[M,1] | B[M-3,2] B[M-2,2] B[M-1,2] B[M,2] | ... | B[M-3,K] B[M-2,K] B[M-1,K] B[M,K] ]
                                          |                                   |     |
                                          |                                   |     |
                                          |                                   |     |
                                          |                                   |     |
                      ^                   |                 ^                 |     |                 ^
                      |                   |                 |                 |     |                 |
                      |                   |                 |                 |     |                 |
                      |                   |                 |                 |     |                 |
                    block 1                               block 2             |     |               block K

    """
    BT, M, K = B.shape
    assert M % 4 == 0
    out_height = M // 4
    out_shape = tuple((BT, out_height, K, 4))
    BB = np.zeros(out_shape).astype(B.dtype)  # block form of B
    for bt in range(BT):
        for j in range(0, out_height):
            for k in range(0, K):
                for b in range(0, 4):
                    y = b + 4 * j
                    BB[bt, j, k, b] = B[bt, y, k]
    return BB

def setup_test(b, m, n, k):
    a_shape = (b, n, m)
    b_shape = (b, m, k)

    rng = default_rng()
    a = rng.integers(1, 16, a_shape, dtype="uint8")
    b = rng.integers(1, 16, b_shape, dtype="uint8")

    a_q, a_min, a_max = quantize_array(a.reshape(a.size), a.size)
    b_q, b_min, b_max = quantize_array(b.reshape(b.size), b.size)
    a_q = np.array(a_q, dtype="uint8").reshape(a_shape)
    b_q = np.array(b_q, dtype="uint8").reshape(b_shape)
    a_offset = quantize_uint8(0.0, a_min, a_max)
    b_offset = quantize_uint8(0.0, b_min, b_max)
    if can_tensorize(n, m, k):
        bb = blockify_matrix(b_q)  # blockification is only used by the tensorized version
        bb = bb.reshape(-1)  # go via 1D rep in case there are stride / offset issues
        bb = bb.reshape(b_shape)
    else:
        bb = []

    a_f = np.array(a_q, dtype="int32").reshape(a_shape)
    b_f = np.array(b_q, dtype="int32").reshape(b_shape)
    expected_output = np.matmul(a_f, b_f)

    intrin_name = "mm.uint8_{}x{}x{}".format(m, n, k)
    try:
        TensorIntrin.register(intrin_name, *get_mm_uint8_intrin(m, n, k))
    except:
        print("Intrinsic already registered.")

    return a_q, b_q, bb, a_offset, b_offset, intrin_name, expected_output

class TestMatMulVec:

    batches, m, n, k = tvm.testing.parameters(
        (1, 128, 768, 768),
        (1, 128, 768, 3072),
        (1, 128, 3072, 768),
        (1, 128, 128, 64),
        (1, 128, 64, 128),
    )

    @tvm.testing.requires_hexagon
    def test_matmul_intrinsics(self, hexagon_session, batches, m, n, k):
        
        out_shape = (batches, n, k) 

        a_q, b_q, bb, a_offset, b_offset, intrin_name, out_ref = setup_test(batches, m, n, k)

        @T.prim_func
        def operator(a: T.handle, b: T.handle, c: T.handle, offsets: T.handle) -> None:
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [batches, n, m], dtype="uint8")
            B = T.match_buffer(b, [batches, m, k], dtype="uint8")
            C = T.match_buffer(c, [batches, n, k], dtype="int32")
            OFFSETS = T.match_buffer(offsets, [2], dtype="uint8")
            # body
            with T.block("root"):
                for i0, i1, i2, i3 in T.grid(batches, m, n, k):
                    with T.block("C"):
                        batch, y, x, j = T.axis.remap("SSSR", [i0, i1, i2, i3])
                        C[batch, y, x] = C[batch, y, x] + T.cast(A[batch, y, j] - OFFSETS[0], "int32") * T.cast(B[batch, j, x] - OFFSETS[1], "int32")

        ir_module = operator
        sch = tvm.tir.Schedule(ir_module, debug_mask="all")
        
        block = sch.get_block("C")
        _, y, _, _ = sch.get_loops(block)
        sch.tensorize(y, intrin_name)
        
        A = tvm.tir.decl_buffer(a_q.shape, name="A", dtype="uint8")
        B = tvm.tir.decl_buffer(b_q.shape, name="B", dtype="uint8")
        C = tvm.tir.decl_buffer(out_shape, name="C", dtype="int32")
        OFFSETS = tvm.tir.decl_buffer((2), name="OFFSETS", dtype="uint8")

        target_hexagon = tvm.target.hexagon("v68", link_params=True)
        func_tir = tvm.build(sch.mod, [A, B, C, OFFSETS], tvm.target.Target(target_hexagon, host=target_hexagon), name="qmmul_vrmpy")
        module = hexagon_session.load_module(func_tir)

        c = np.zeros(out_shape, dtype="int32")
        offsets = np.array([a_offset, b_offset], dtype="uint8")
        
        a_hexagon = tvm.runtime.ndarray.array(a_q, device=hexagon_session.device)
        b_hexagon = tvm.runtime.ndarray.array(bb, device=hexagon_session.device)
        c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device)
        offsets_hexagon = tvm.runtime.ndarray.array(offsets, device=hexagon_session.device)
        
        module(a_hexagon, b_hexagon, c_hexagon, offsets_hexagon)
        evaluator = module.time_evaluator(module.entry_name, hexagon_session.device, number=1)
        time_ms = evaluator(a_hexagon, b_hexagon, c_hexagon, offsets_hexagon).mean * 1e3
        print("Input Shape: {}. Conv time elapsed: {} ms".format((batches, m, n, k), time_ms))

        out = c_hexagon.numpy()
        out_a = out.reshape(batches * n * k)
        out_req, _, _ = quantize_array(out_a, batches * n * k)
        out_req = np.array(out_req).reshape(batches, n, k)

        out_ref_a = out_ref.reshape(batches * n * k)
        out_ref_q, _, _ = quantize_array(out_ref_a, batches * n * k)
        out_ref_q = np.array(out_ref_q).reshape(batches, n, k)

        tvm.testing.assert_allclose(out_req, out_ref_q, atol=2.0, rtol=0.0)
        