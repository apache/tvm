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

"""Executable examples and contract tests for the explicit PTX tile prototype."""

from types import SimpleNamespace

import numpy as np
import pytest

import tvm
from tvm.backend.cuda.axis import laneid
from tvm.backend.cuda.ptx import instruction_name
from tvm.ir import Var, assert_structural_equal
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.tile.dispatch import run_dispatch

LD = "ld.global.v4.b32"
LDMATRIX = "ldmatrix.sync.aligned.m8n8.x2.shared.b16"


def global_kernel(gapped=False):
    original = TileLayout(S[(15, 2, 7, 2) : (4, 1, 60, 2)])
    dst_layout = TileLayout(S[(15, 2, 7, 2) : (8, 2, 120, 4)]) if gapped else original

    @T.function
    def kernel(
        A: T.Tensor((15, 2, 7, 2), "float32", layout=original, align=16),
        Out: T.Tensor((420,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        R = T.alloc_tensor(
            ty_args=[T.Tensor((15, 2, 7, 2), "float32", scope="local", layout=dst_layout)]
        )
        A_std = A.rearrange("i b j d -> j i (d b)")
        R_std = R.rearrange("i b j d -> j i (d b)")
        T.cuda.tile.apply(T.ptx[LD], R_std, A_std, scope="thread")
        for j, i, d, b in T.grid(7, 15, 2, 2):
            Out[60 * j + 4 * i + 2 * d + b] = R[i, b, j, d]

    return kernel


def matrix_kernel(partial=False):
    shared = TileLayout(S[(16, 4, 2) : (8, 1, 128)])
    registers = TileLayout(S[(2, 2, 8, 4) : (8, 2, 4 @ laneid, 1 @ laneid)])

    @T.function
    def kernel(A: T.Tensor((16, 4, 2), "uint32"), Out: T.Tensor((32, 12), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        SM = T.alloc_tensor(
            ty_args=[T.Tensor((16, 4, 2), "uint32", scope="shared", layout=shared, align=16)]
        )
        for q in range(4):
            n = lane + 32 * q
            SM[n // 8, (n // 2) % 4, n % 2] = A[n // 8, (n // 2) % 4, n % 2]
        T.cuda.cta_sync()
        raw = T.alloc_tensor(ty_args=[T.Tensor((12,), "uint32", scope="local", layout="default")])
        for i in T.unroll(12):
            raw[i] = T.uint32(4294966519)  # bit pattern of int32(-777)
        R = T.decl_tensor(
            raw.ptr_to([0]),
            ty_args=[T.Tensor((2, 2, 8, 4), "uint32", scope="local", layout=registers)],
        )
        S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
        if T.constexpr(partial):
            if lane < 16:
                T.cuda.tile.apply(T.ptx[LDMATRIX], R, S_std, scope="warp")
        else:
            T.cuda.tile.apply(T.ptx[LDMATRIX], R, S_std, scope="warp")
        for i in T.unroll(12):
            Out[lane, i] = raw[i]

    return kernel


def compile_kernel(kernel):
    target = tvm.target.Target(
        {"kind": "cuda", "backend_config": {"cuda": {"target_format": "ptx"}}}
    )
    with target:
        return tvm.tirx.compile(kernel, target=target, pipeline="tirx")


@pytest.mark.parametrize("factory", [global_kernel, matrix_kernel])
def test_roundtrip(factory):
    kernel = factory()
    assert_structural_equal(kernel, tvm.ir.load_json(tvm.ir.save_json(kernel)))
    reparsed = tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    assert_structural_equal(kernel, reparsed)


def test_instruction_name():
    assert instruction_name(T.ptx["ld.b32.v4.global"]) == LD
    assert instruction_name(T.ptx[LDMATRIX]) == LDMATRIX
    with pytest.raises(ValueError, match="not fully specified"):
        instruction_name(T.ptx.ld)


def _dispatch(
    src_shape=(2, 4),
    src_layout=None,
    dst_shape=None,
    dst_layout=None,
    opcode=LD,
    scope="thread",
    align=16,
    elem_offset=0,
    active=32,
):
    matrix = opcode.startswith("ldmatrix")
    src = Var(
        "src",
        T.Tensor(
            src_shape,
            "uint32",
            scope="shared" if matrix else "global",
            layout=src_layout or TileLayout(S[src_shape]),
            align=align,
            elem_offset=elem_offset,
        ),
    )
    dst_shape = dst_shape or src_shape
    dst = Var(
        "dst",
        T.Tensor(dst_shape, "uint32", scope="local", layout=dst_layout or TileLayout(S[dst_shape])),
    )
    ctx = SimpleNamespace(
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        scope_kind=scope,
        var_range_map={},
        intra={"laneid": (active, 0)} if matrix else {},
    )
    return run_dispatch(T.cuda.tile.apply(T.ptx[opcode], dst, src, scope=scope), ctx)


def test_contract_rejections():
    with pytest.raises(RuntimeError, match="consecutive words"):
        _dispatch(src_layout=TileLayout(S[(2, 4) : (8, 2)]))
    with pytest.raises(RuntimeError, match="16-byte boundary"):
        _dispatch(elem_offset=1)
    with pytest.raises(RuntimeError, match="known 16-byte-aligned base"):
        _dispatch(align=4)
    with pytest.raises(RuntimeError, match="Repeat shapes must agree"):
        _dispatch(dst_shape=(3, 4))
    with pytest.raises(RuntimeError, match="No explicit tile pattern"):
        _dispatch(opcode="ld.global.v2.b32")
    with pytest.raises(RuntimeError, match="nonoverlapping storage strides"):
        _dispatch(dst_layout=TileLayout(S[(2, 4) : (4, 0)]))


def test_matrix_contract_rejections():
    shape = (2, 2, 8, 4)
    reg = TileLayout(S[shape : (8, 2, 4 @ laneid, 1 @ laneid)])
    shared = TileLayout(S[shape : (128, 8, 16, 1)])
    kw = dict(src_shape=shape, src_layout=shared, dst_layout=reg, opcode=LDMATRIX, scope="warp")
    _dispatch(**kw)
    # Last word axis becomes offsets 0,2,1,3 after splitting into two bits.
    with pytest.raises(RuntimeError, match="consecutive b32 words"):
        _dispatch(**{**kw, "src_layout": TileLayout(S[(2, 2, 8, 2, 2) : (128, 8, 16, 1, 2)])})
    with pytest.raises(RuntimeError, match="owner lane"):
        _dispatch(**{**kw, "dst_layout": TileLayout(S[shape : (8, 2, 1 @ laneid, 8 @ laneid)])})
    with pytest.raises(RuntimeError, match="16-byte boundary"):
        _dispatch(**{**kw, "src_layout": TileLayout(S[shape : (128, 8, 17, 1)])})
    with pytest.raises(RuntimeError, match="all 32 lanes"):
        _dispatch(**{**kw, "active": 16})
    with pytest.raises(RuntimeError, match="requires scope='warp'"):
        _dispatch(**{**kw, "scope": "thread"})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
@pytest.mark.parametrize("gapped", [False, True])
def test_global_gpu(gapped):
    compiled = compile_kernel(global_kernel(gapped))
    ptx = compiled.mod.imports[0].inspect_source("ptx")
    assert ptx.count("ld.global.v4.b32") == 105
    a = np.arange(420, dtype="float32").reshape(15, 2, 7, 2) + 0.25
    out = tvm.runtime.tensor(np.zeros(420, dtype="float32"), tvm.cuda())
    compiled(tvm.runtime.tensor(a, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), a.ravel())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
def test_matrix_gpu():
    compiled = compile_kernel(matrix_kernel())
    ptx = compiled.mod.imports[0].inspect_source("ptx")
    assert ptx.count(LDMATRIX) == 2
    values = np.arange(128, dtype="uint32").reshape(16, 4, 2)
    a = ((values + 1000) << 16) | (values + 1)
    expected = np.full((32, 12), 4294966519, dtype="uint32")
    for lane in range(32):
        for k in range(2):
            for m in range(2):
                expected[lane, 8 * k + 2 * m] = a[2 * (lane // 4) + m, lane % 4, k]
    out = tvm.runtime.tensor(np.zeros_like(expected), tvm.cuda())
    compiled(tvm.runtime.tensor(a, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_partial_warp_rejected_in_pipeline():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target, pytest.raises((RuntimeError, ValueError), match="all 32 lanes"):
        tvm.tirx.transform.TileDispatch()(tvm.IRModule({"main": matrix_kernel(partial=True)}))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
def test_region_starts_and_element_offsets_gpu():
    @T.function
    def kernel(A: T.Tensor((32,), "uint32"), Out: T.Tensor((32,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        raw = T.alloc_tensor(ty_args=[T.Tensor((32,), "uint32", scope="local", layout="default")])
        for i in T.unroll(32):
            raw[i] = T.uint32(777)
        src = T.decl_tensor(
            A.data, ty_args=[T.Tensor((3, 8), "uint32", elem_offset=4, align=16, layout="default")]
        )
        dst = T.decl_tensor(
            raw.data,
            ty_args=[T.Tensor((3, 8), "uint32", elem_offset=4, scope="local", layout="default")],
        )
        T.cuda.tile.apply(T.ptx[LD], dst[1:3, 2:6], src[0:2, 4:8])
        for i in T.unroll(32):
            Out[i] = raw[i]

    compiled = compile_kernel(kernel)
    a = np.arange(32, dtype="uint32") + 100
    expected = np.full(32, 777, dtype="uint32")
    expected[14:18] = a[8:12]
    expected[22:26] = a[16:20]
    out = tvm.runtime.tensor(np.zeros_like(expected), tvm.cuda())
    compiled(tvm.runtime.tensor(a, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), expected)
