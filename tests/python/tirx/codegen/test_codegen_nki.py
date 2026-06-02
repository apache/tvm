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
from tvm.script import tirx as Tx

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def lower_and_get_source(func):
    with target:
        mod = tvm.IRModule({"main": func})
        mod = tvm.compile(mod, tir_pipeline="trn")
        src = mod.mod.imports[0].inspect_source()
        return src


def compare_strings_ignore_whitespace(s1, s2):
    # Remove all whitespace by splitting and joining the string back together
    return "".join(s1.split()) == "".join(s2.split())


def test_nki_add_1():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer((128, 512)), B: Tx.Buffer((128, 512))):
        Tx.func_attr({"num_inputs": 1})
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        B_sbuf = Tx.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        with Tx.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    Tx.nki.load(A_sbuf[i, j], A[i, j])
        with Tx.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    Tx.nki.tensorscalar(B_sbuf[i, j], A_sbuf[i, j], Tx.float32(1.0), "add")
        with Tx.attr(0, "tensorized_nki_instruction", 1):
            for i in range(0, 128):
                for j in range(0, 512):
                    Tx.nki.store(B[i, j], B_sbuf[i, j])
        # fmt: on
    src = lower_and_get_source(func)
    print(src)
    expected = """# Function: func_kernel
import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace
import numpy as np
import neuronxcc.nki.isa as nisa
import math
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt
import neuronxcc.nki.compiler as ncc
@nki.compiler.enable_stack_allocator
@nki.compiler.skip_middle_end_transformations
@baremetal(experimental_flags='enable-mutable-parameter', additional_compile_opt='--internal-skip-backend-allocation-opt-nki')
def func_kernel(A_ptr, B_ptr: nt.mutable_tensor, ):
  B_ptr_buffer = B_ptr.reshape([65536])
  A_ptr_buffer = A_ptr.reshape([65536])
  A_sbuf_ptr = nl.ndarray(shape=[128, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=0))
  B_sbuf_ptr = nl.ndarray(shape=[128, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=2048))
  i = nl.arange(128)
  j = nl.arange(512)
  A_sbuf_ptr[i[:, None, ], j[None, :, ]] = nl.load(A_ptr_buffer[((i[:, None, ] * 512) + j[None, :, ])])
  i_1 = nl.arange(128)
  j_1 = nl.arange(512)
  B_sbuf_ptr[i_1[:, None, ], j_1[None, :, ]] = nisa.tensor_scalar(A_sbuf_ptr[i_1[:, None, ], j_1[None, :, ]], operand0=1.000000e+00, op0=nki.language.add, reverse0=False)
  i_2 = nl.arange(128)
  j_2 = nl.arange(512)
  nl.store(B_ptr_buffer[((i_2[:, None, ] * 512) + j_2[None, :, ])], B_sbuf_ptr[i_2[:, None, ], j_2[None, :, ]])
  return B_ptr
  """  # noqa: E501
    assert compare_strings_ignore_whitespace(src, expected)


def test_nki_add_2():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer((128, 2048)), B: Tx.Buffer((128, 2048))):
        Tx.func_attr({"num_inputs": 1})
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        B_sbuf = Tx.alloc_buffer((128, 512), "float32", scope="trn.sbuf",)
        for k in range(0, 4):
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        Tx.nki.load(A_sbuf[i, j], A[i, 512*k+j])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        Tx.nki.tensorscalar(B_sbuf[i, j], A_sbuf[i, j], Tx.float32(1.0), "add")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for i in range(0, 128):
                    for j in range(0, 512):
                        Tx.nki.store(B[i, 512*k+j], B_sbuf[i, j])

        # fmt: on
    src = lower_and_get_source(func)
    print(src)
    expected = """# Function: func_kernel
import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace
import numpy as np
import neuronxcc.nki.isa as nisa
import math
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt
import neuronxcc.nki.compiler as ncc
@nki.compiler.enable_stack_allocator
@nki.compiler.skip_middle_end_transformations
@baremetal(experimental_flags='enable-mutable-parameter', additional_compile_opt='--internal-skip-backend-allocation-opt-nki')
def func_kernel(A_ptr, B_ptr: nt.mutable_tensor, ):
  B_ptr_buffer = B_ptr.reshape([262144])
  A_ptr_buffer = A_ptr.reshape([262144])
  A_sbuf_ptr = nl.ndarray(shape=[128, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=0))
  B_sbuf_ptr = nl.ndarray(shape=[128, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=2048))
  for k in nl.sequential_range(4, body_no_reorder=True):
    i = nl.arange(128)
    j = nl.arange(512)
    A_sbuf_ptr[i[:, None, ], j[None, :, ]] = nl.load(A_ptr_buffer[(((i[:, None, ] * 2048) + (k * 512)) + j[None, :, ])])
    i_1 = nl.arange(128)
    j_1 = nl.arange(512)
    B_sbuf_ptr[i_1[:, None, ], j_1[None, :, ]] = nisa.tensor_scalar(A_sbuf_ptr[i_1[:, None, ], j_1[None, :, ]], operand0=1.000000e+00, op0=nki.language.add, reverse0=False)
    i_2 = nl.arange(128)
    j_2 = nl.arange(512)
    nl.store(B_ptr_buffer[(((i_2[:, None, ] * 2048) + (k * 512)) + j_2[None, :, ])], B_sbuf_ptr[i_2[:, None, ], j_2[None, :, ]])
  return B_ptr"""  # noqa: E501
    assert compare_strings_ignore_whitespace(src, expected)


def test_nki_matmul_1():
    TILES_IN_BLOCK_M = 16
    TILES_IN_BLOCK_N = 1
    TILES_IN_BLOCK_K = 8
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    K = 1024
    M = 4096
    N = 2048
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K
    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    @Tx.prim_func
    def func(
        lhsT: Tx.Buffer((K, M), "float16"),
        rhs: Tx.Buffer((K, N), "float16"),
        result: Tx.buffer((M, N), "float16"),
    ):
        Tx.func_attr({"num_inputs": 2})
        with Tx.thread():
            result_tiles = Tx.alloc_buffer(
                (TILE_M, NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_N),
                "float32",
                scope="trn.sbuf",
            )
            rhs_tiles = Tx.alloc_buffer(
                (TILE_K, TILES_IN_BLOCK_K, BLOCK_N), "float16", scope="trn.sbuf"
            )
            lhsT_tiles = Tx.alloc_buffer(
                (TILE_K, TILES_IN_BLOCK_K, BLOCK_M), "float16", scope="trn.sbuf"
            )
            res_tile = Tx.alloc_buffer((1, TILE_M, TILE_N), "float32", scope="trn.psum")
            result_packed = Tx.alloc_buffer((TILE_K, BLOCK_N), "float32", scope="trn.sbuf")
            for n in range(NUM_BLOCK_N):
                with Tx.attr(0, "tensorized_nki_instruction", 1):
                    for i0 in range(TILE_M):
                        for i1 in range(NUM_BLOCK_M):
                            for i2 in range(TILES_IN_BLOCK_M):
                                for i3 in range(TILES_IN_BLOCK_N):
                                    for i4 in range(TILE_N):
                                        Tx.nki.memset(
                                            result_tiles[i0, i1, i2, i3, i4], Tx.float32(0.0)
                                        )
                for k in range(NUM_BLOCK_K):
                    for bk_r in range(TILES_IN_BLOCK_K):
                        with Tx.attr(0, "tensorized_nki_instruction", 1):
                            for i in range(TILE_K):
                                for j in range(BLOCK_N):
                                    Tx.nki.load(
                                        rhs_tiles[i, bk_r, j],
                                        rhs[
                                            (TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i,
                                            n * BLOCK_N + j,
                                        ],
                                    )
                    for m in range(NUM_BLOCK_M):
                        for bk_l in range(TILES_IN_BLOCK_K):
                            with Tx.attr(0, "tensorized_nki_instruction", 1):
                                for i in range(TILE_K):
                                    for j in range(BLOCK_M):
                                        Tx.nki.load(
                                            lhsT_tiles[i, bk_l, j],
                                            lhsT[
                                                (TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i,
                                                m * BLOCK_M + j,
                                            ],
                                        )
                        for bn in range(TILES_IN_BLOCK_N):
                            for bm in range(TILES_IN_BLOCK_M):
                                with Tx.attr(0, "tensorized_nki_instruction", 1):
                                    for i in range(TILE_M):
                                        for j in range(TILE_N):
                                            Tx.nki.memset(res_tile[0, i, j], Tx.float32(0.0))
                                for bk in range(TILES_IN_BLOCK_K):
                                    with Tx.attr(0, "tensorized_nki_instruction", 1):
                                        for i in range(TILE_M):
                                            for j in range(TILE_N):
                                                for k in range(TILE_K):
                                                    Tx.nki.matmul(
                                                        res_tile[0, i, j],
                                                        lhsT_tiles[k, bk, bm * TILE_M + i],
                                                        rhs_tiles[k, bk, bn * TILE_N + j],
                                                        1,
                                                    )
                                with Tx.attr(0, "tensorized_nki_instruction", 1):
                                    for i in range(TILE_M):
                                        for j in range(TILE_N):
                                            Tx.nki.tensortensor(
                                                result_tiles[i, m, bm, bn, j],
                                                result_tiles[i, m, bm, bn, j],
                                                res_tile[0, i, j],
                                                "add",
                                            )
                for m in range(NUM_BLOCK_M):
                    for bm in range(TILES_IN_BLOCK_M):
                        for bn in range(TILES_IN_BLOCK_N):
                            with Tx.attr(0, "tensorized_nki_instruction", 1):
                                for i in range(TILE_K):
                                    for j in range(TILE_N):
                                        Tx.nki.tensor_copy(
                                            result_packed[i, bn * TILE_N + j],
                                            result_tiles[i, m, bm, bn, j],
                                        )
                        with Tx.attr(0, "tensorized_nki_instruction", 1):
                            for i in range(TILE_K):
                                for j in range(BLOCK_N):
                                    Tx.nki.store(
                                        result[m * BLOCK_M + bm * TILE_M + i, n * BLOCK_N + j],
                                        result_packed[i, j],
                                    )

    # fmt: on

    src = lower_and_get_source(func)
    print(src)
    expected = """# Function: func_kernel
import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace
import numpy as np
import neuronxcc.nki.isa as nisa
import math
import neuronxcc.nki as nki
import neuronxcc.nki.typing as nt
import neuronxcc.nki.compiler as ncc
@nki.compiler.enable_stack_allocator
@nki.compiler.skip_middle_end_transformations
@baremetal(experimental_flags='enable-mutable-parameter', additional_compile_opt='--internal-skip-backend-allocation-opt-nki')
def func_kernel(lhsT_ptr, rhs_ptr, result_ptr: nt.mutable_tensor, ):
  result_ptr_buffer = result_ptr.reshape([8388608])
  rhs_ptr_buffer = rhs_ptr.reshape([2097152])
  lhsT_ptr_buffer = lhsT_ptr.reshape([4194304])
  result_tiles_ptr = nl.ndarray(shape=[128, 2, 16, 1, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=0))
  rhs_tiles_ptr = nl.ndarray(shape=[128, 8, 512], dtype=np.float16, buffer=ncc.sbuf.mod_alloc(base_addr=65536))
  lhsT_tiles_ptr = nl.ndarray(shape=[128, 8, 2048], dtype=np.float16, buffer=ncc.sbuf.mod_alloc(base_addr=73728))
  res_tile_ptr = nl.ndarray(shape=[1, nl.par_dim(128), 512], dtype=np.float32, buffer=nl.psum)
  result_packed_ptr = nl.ndarray(shape=[128, 512], dtype=np.float32, buffer=ncc.sbuf.mod_alloc(base_addr=106496))
  for n in nl.sequential_range(4, body_no_reorder=True):
    i0 = nl.arange(128)
    i1 = nl.arange(2)
    i2 = nl.arange(16)
    i4 = nl.arange(512)
    result_tiles_ptr[i0[:, None, None, None, ], i1[None, :, None, None, ], i2[None, None, :, None, ], 0, i4[None, None, None, :, ]] = 0.000000e+00
    for bk_r in nl.sequential_range(8):
      i = nl.arange(128)
      j = nl.arange(512)
      rhs_tiles_ptr[i[:, None, ], bk_r, j[None, :, ]] = nl.load(rhs_ptr_buffer[((((bk_r * 262144) + (i[:, None, ] * 2048)) + (n * 512)) + j[None, :, ])])
    for m in nl.sequential_range(2):
      for bk_l in nl.sequential_range(8):
        i_1 = nl.arange(128)
        j_1 = nl.arange(2048)
        lhsT_tiles_ptr[i_1[:, None, ], bk_l, j_1[None, :, ]] = nl.load(lhsT_ptr_buffer[((((bk_l * 524288) + (i_1[:, None, ] * 4096)) + (m * 2048)) + j_1[None, :, ])])
      for bm in nl.sequential_range(16):
        i_2 = nl.arange(128)
        j_2 = nl.arange(512)
        res_tile_ptr[0, i_2[:, None, ], j_2[None, :, ]] = 0.000000e+00
        for bk in nl.sequential_range(8):
          i_3 = nl.arange(128)
          j_3 = nl.arange(512)
          k = nl.arange(128)
          res_tile_ptr[0, i_3[:, None, ], j_3[None, :, ]] += nisa.nc_matmul(lhsT_tiles_ptr[k[:, None, ], bk, ((bm * 128) + i_3[None, :, ])],rhs_tiles_ptr[k[:, None, ], bk, j_3[None, :, ]])
        i_4 = nl.arange(128)
        j_4 = nl.arange(512)
        result_tiles_ptr[i_4[:, None, ], m, bm, 0, j_4[None, :, ]] = nisa.tensor_tensor(result_tiles_ptr[i_4[:, None, ], m, bm, 0, j_4[None, :, ]], res_tile_ptr[0, i_4[:, None, ], j_4[None, :, ]], op=nki.language.add)
    for m_1 in nl.sequential_range(2):
      for bm_1 in nl.sequential_range(16):
        i_5 = nl.arange(128)
        j_5 = nl.arange(512)
        result_packed_ptr[i_5[:, None, ], j_5[None, :, ]] = nisa.tensor_copy(result_tiles_ptr[i_5[:, None, ], m_1, bm_1, 0, j_5[None, :, ]])
        i_6 = nl.arange(128)
        j_6 = nl.arange(512)
        nl.store(result_ptr_buffer[(((((m_1 * 4194304) + (bm_1 * 262144)) + (i_6[:, None, ] * 2048)) + (n * 512)) + j_6[None, :, ])], result_packed_ptr[i_6[:, None, ], j_6[None, :, ]])
  return result_ptr"""  # noqa: E501
    assert compare_strings_ignore_whitespace(src, expected)


if __name__ == "__main__":
    tvm.testing.main()
