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
"""The compile-time tcgen05 descriptor encoders must agree with the C ones.

``encode_smem_descriptor_base_uint64`` and
``encode_instr_descriptor_block_scaled_uint32`` re-derive, in Python, bit
layouts that otherwise live only in a C struct (``cuda/cpp/descriptors.py``). A
kernel whose descriptor inputs are all compile-time constants uses them to bake
a literal instead of calling the runtime encoder, which is an opaque helper in
the generated CUDA.

Re-deriving a bit layout by hand is exactly the kind of thing that is silently
wrong, so these run both encoders on the same inputs and compare.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_block_scaled_uint32,
    encode_smem_descriptor_base_uint64,
)
from tvm.script import tirx as T
from tvm.testing import env

TARGET = tvm.target.Target("cuda")


def _compile_and_run(kernel, n_out):
    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=TARGET, tir_pipeline="tirx")
    result = {}

    def go():
        dev = tvm.cuda(0)
        out = tvm.runtime.tensor(np.zeros(n_out, dtype="uint64"), device=dev)
        mod(out)
        result["out"] = out.numpy()

    tvm.testing.run_with_gpu_lock(go)
    return result["out"]


# One case per axis the SMEM encoder branches on: every swizzle enum, and
# offsets that exercise both 14-bit fields.
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "ldo,sdo,swizzle",
    [(0, 8, 0), (0, 8, 3), (64, 128, 2), (32, 64, 1), (16, 16, 4)],
)
def test_smem_descriptor_matches_runtime_encoder(ldo, sdo, swizzle):
    """`base | (addr >> 4)` must reproduce the C bitfield fill exactly."""

    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (2,), "uint64")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((64,), "uint32", scope="shared")
        smem[tx] = T.uint32(0)
        smem[tx + 32] = T.uint32(0)
        if tx == 0:
            desc = T.local_scalar("uint64")
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc), smem.ptr_to([0]), ldo=ldo, sdo=sdo, swizzle=swizzle
            )
            out[0] = desc
            # The shared address is the encoder's only runtime input; hand it
            # back so the comparison can OR it into the Python constant.
            out[1] = T.cast(T.cuda.cvta_generic_to_shared(smem.ptr_to([0])), "uint64")

    got = _compile_and_run(kernel, 2)
    want = encode_smem_descriptor_base_uint64(ldo, sdo, swizzle) | ((int(got[1]) >> 4) & 0x3FFF)
    assert int(got[0]) == want, (
        f"runtime {int(got[0]):#018x} != compile-time {want:#018x} "
        f"for ldo={ldo} sdo={sdo} swizzle={swizzle}"
    )


# (M, N, K, a_dtype, b_dtype, trans_a, trans_b, cta_group)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "m,n,k,a_dtype,b_dtype,trans_a,trans_b,cta_group",
    [
        (128, 224, 32, "float8_e4m3fn", "float8_e4m3fn", False, False, 1),
        (128, 128, 32, "float8_e4m3fn", "float8_e4m3fn", True, True, 1),
        (256, 256, 32, "float8_e4m3fn", "float8_e5m2", False, True, 2),
        (128, 64, 64, "float4_e2m1fn", "float4_e2m1fn", False, False, 1),
    ],
)
def test_instr_descriptor_block_scaled_matches_runtime_encoder(
    m, n, k, a_dtype, b_dtype, trans_a, trans_b, cta_group
):
    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint64")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            desc = T.local_scalar("uint32")
            T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc),
                d_dtype="float32",
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=0,
                sfb_tmem_addr=0,
                M=m,
                N=n,
                K=k,
                trans_a=trans_a,
                trans_b=trans_b,
                n_cta_groups=cta_group,
            )
            out[0] = T.cast(desc, "uint64")

    got = _compile_and_run(kernel, 1)
    want = encode_instr_descriptor_block_scaled_uint32(
        M=m,
        N=n,
        K=k,
        d_dtype="float32",
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype="float8_e8m0fnu",
        trans_a=trans_a,
        trans_b=trans_b,
        cta_group=cta_group,
    )
    assert int(got[0]) == want, (
        f"runtime {int(got[0]):#010x} != compile-time {want:#010x} "
        f"for M={m} N={n} a={a_dtype} b={b_dtype} trans=({trans_a},{trans_b}) "
        f"cta_group={cta_group}"
    )


if __name__ == "__main__":
    tvm.testing.main()
