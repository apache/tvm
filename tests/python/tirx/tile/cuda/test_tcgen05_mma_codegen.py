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

"""Native tcgen05.mma descriptor and datapath contracts."""

import re

import pytest

import tvm
from tvm.backend.cuda.axis import TCol, TLane
from tvm.backend.cuda.layout import mma_shared_layout, sf_tmem_layout, tmem_datapath_layout
from tvm.script import tirx as T
from tvm.tirx.layout import ComposeLayout, R, S, TileLayout


def _compile(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        return (
            tvm.tirx.compile(func, target=target, pipeline="tirx").mod.imports[0].inspect_source()
        )


def _lower(func):
    with tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        return tvm.tirx.transform.TileDispatch()(tvm.IRModule({"main": func}))["main"]


def _calls(src, ws=False, cta=1):
    prefix = f"tvm_builtin_ptx_tcgen05_mma{'_ws' if ws else ''}_"
    return [
        line
        for line in src.splitlines()
        if line.strip().startswith(prefix) and f"cta_group__{cta}" in line
    ]


def _descriptors(lines):
    return [
        int(re.search(r", \(uint\)(\d+), \(uint\)0, \(uint\)0", line).group(1)) for line in lines
    ]


def _stage_fixture(n=128, ws=False, instruction_shape=None, **params):
    """Keep the original three-stage A/B and C[:,256:384] region."""
    instruction_shape = instruction_shape if instruction_shape is not None else (128, n, 16)
    layout = mma_shared_layout("float16", 3, (3, 128, 64))
    opcode = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::1.kind::f16"

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        A = T.alloc_shared((3, 128, 64), "float16", layout=layout, align=1024)
        B = T.alloc_shared((3, 128, 64), "float16", layout=layout, align=1024)
        D = T.cuda.decl_tmem(
            0,
            ty_args=[
                T.Tensor(
                    (128, 512),
                    "float32",
                    scope="tmem",
                    layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        av = (
            A.sub[1:2, :, :].view(128, 64).rearrange("m (kr k) -> kr m k", k=16).view(1, 4, 128, 16)
        )
        bv = B.sub[2:3, :, :].view(128, 64).rearrange("(nr n) (kr k) -> nr kr n k", n=n, k=16)
        dv = D.sub[:, 256:384].rearrange("m (nr n) -> nr m n", n=n).view(1, 128 // n, 128, n)
        if T.cuda.thread_idx("x") == 0:
            T.cuda.tile.apply(
                T.ptx[opcode],
                dv,
                av,
                bv,
                instruction_shape=instruction_shape,
                a_descriptor=(16, 1024, 3),
                b_descriptor=(16, 1024, 3),
                **params,
            )

    return kernel


def _shared_fixture(
    M,
    N,
    K,
    dtype,
    A_layout,
    B_layout,
    *,
    a_descriptor,
    b_descriptor,
    a_major="k",
    b_major="mn",
    cta=1,
    ws=False,
    instruction_k=16,
    C_layout=None,
    a_format=None,
    b_format=None,
    kind="f16",
    trans_b=True,
    issue_elect=False,
    **params,
):
    """The original A[M,K], B[K,N] fixtures; only their operand views change."""
    C_layout = C_layout or tmem_datapath_layout("F" if M == 64 else "D", M, N)
    opcode = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::{cta}.kind::{kind}"
    formats = {}
    if a_format is not None:
        formats["a_format"] = a_format
    if b_format is not None:
        formats["b_format"] = b_format

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta, block=128, cluster=cta))
        A = T.alloc_shared((M, K), dtype, layout=A_layout, align=1024)
        B = T.alloc_shared(
            (K, N // cta) if trans_b else (N // cta, K), dtype, layout=B_layout, align=1024
        )
        D = T.cuda.decl_tmem(
            400, ty_args=[T.Tensor((M, N), "float32", scope="tmem", layout=C_layout)]
        )
        av = A.rearrange("m (kr k) -> kr m k", k=instruction_k).view(
            1, K // instruction_k, M, instruction_k
        )
        if T.constexpr(trans_b):
            bmatrix = B.permute(1, 0)
        else:
            bmatrix = B
        bv = bmatrix.rearrange("n (kr k) -> kr n k", k=instruction_k).view(
            1, K // instruction_k, N // cta, instruction_k
        )
        issue = T.cuda.elect_sync() if T.constexpr(issue_elect) else T.cuda.thread_idx("x") == 0
        if (T.cuda.warp_id() == 0) & issue:
            T.cuda.tile.apply(
                T.ptx[opcode],
                D.view(1, 1, M, N),
                av,
                bv,
                instruction_shape=(M * cta, N, instruction_k),
                a_major=a_major,
                b_major=b_major,
                a_descriptor=a_descriptor,
                b_descriptor=b_descriptor,
                **formats,
                **params,
            )

    return kernel


def _tmem_fixture(*, cta=1, ws=True, c_form="packed", a_form="banked", n=128, split_n=128, M=64):
    """Keep the original FlashMLA A[2,64,16] and packed/banked C footprints."""
    K = 16 if cta == 1 else 128
    B_layout = mma_shared_layout("bfloat16", 1 if cta == 1 else 3, (n // cta, K))
    a_shape = (2, M, K) if a_form == "banked" else (M, K)
    if a_form == "banked":
        a_layout = TileLayout(S[(2, M, K) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])
    else:
        a_spec = S[(M, K) : (1 @ TLane, 1 @ TCol)]
        a_layout = TileLayout(a_spec + R[2 : 64 @ TLane] if a_form == "replica" else a_spec)
    c_shape = (2, M, n // 2) if c_form == "banked" else (M, n)
    if c_form == "banked":
        c_layout = TileLayout(S[(2, M, n // 2) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])
    elif c_form == "identity":
        c_layout = TileLayout(S[(M, n) : (1 @ TLane, 1 @ TCol)])
    else:
        c_layout = TileLayout(S[(M, 2, n // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    opcode = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::{cta}.kind::f16"

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta, block=128, cluster=cta))
        B = T.alloc_shared((n // cta, K), "bfloat16", layout=B_layout, align=1024)
        A = T.cuda.decl_tmem(
            256, ty_args=[T.Tensor(a_shape, "bfloat16", scope="tmem", layout=a_layout)]
        )
        D = T.cuda.decl_tmem(
            400, ty_args=[T.Tensor(c_shape, "float32", scope="tmem", layout=c_layout)]
        )
        if T.constexpr(a_form == "banked"):
            av = A.rearrange("bank m (kr k) -> kr bank m k", k=16).view(1, K // 16, 2, M, 16)
        else:
            av = A.rearrange("m (kr k) -> kr m k", k=16).view(1, K // 16, M, 16)
        if T.constexpr(c_form == "banked"):
            dmatrix = D.permute(1, 0, 2).view(M, n)
        else:
            dmatrix = D
        dv = dmatrix.rearrange("m (nr n) -> nr m n", n=split_n).view(1, n // split_n, M, split_n)
        bv = B.rearrange("(nr n) (kr k) -> nr kr n k", n=split_n // cta, k=16)
        if T.cuda.thread_idx("x") == 0:
            T.cuda.tile.apply(
                T.ptx[opcode],
                dv,
                av,
                bv,
                instruction_shape=(M * cta, split_n, 16),
                b_descriptor=(16, 256 if cta == 1 else 1024, 1 if cta == 1 else 3),
            )

    return kernel


def test_gemm_tcgen05_cta_group_2_accepts_replicated_tmem_a_codegen():
    src = _compile(_tmem_fixture(cta=2, ws=False, a_form="replica"))
    assert "tcgen05.mma.cta_group::2.kind::f16" in src
    assert "tcgen05.mma.ws" not in src
    assert "tvm_builtin_ptx_tcgen05_mma_ts_" in src


def test_gemm_tcgen05_cta_group_2_rejects_flat_tmem_a_codegen():
    with pytest.raises(Exception, match="replicas disagree"):
        _compile(_tmem_fixture(cta=2, ws=False, a_form="flat"))


def test_gemm_tcgen05_explicit_mma_tile_changes_physical_instructions():
    wide, narrow = (_calls(_compile(_stage_fixture(n))) for n in (128, 64))
    assert len(wide) == 4
    assert len(narrow) == 8
    assert {(v >> 17) & 0x3F for v in _descriptors(wide)} == {128 // 8}
    assert {(v >> 17) & 0x3F for v in _descriptors(narrow)} == {64 // 8}


def test_gemm_tcgen05_weight_stationary_codegen():
    src = _compile(_stage_fixture(ws=True))
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src


def test_gemm_tcgen05_cta1_m64_accepts_packed_c_layout_ws():
    src = _compile(_tmem_fixture())
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src
    assert len(_calls(src, ws=True)) == 1


def test_gemm_tcgen05_cta1_m64_accepts_batched_c_layout_ws():
    packed = _compile(_tmem_fixture())
    banked = _compile(_tmem_fixture(c_form="banked"))
    assert [re.sub(r"v_\d+", "v", x) for x in _calls(packed, ws=True)] == [
        re.sub(r"v_\d+", "v", x) for x in _calls(banked, ws=True)
    ]


@pytest.mark.parametrize("a_form", ["flat", "banked"])
def test_gemm_tcgen05_cta1_m64_ws_requires_layout_e_c(a_form):
    message = "A: expected trailing atom shape" if a_form == "flat" else "D: layout disagrees"
    with pytest.raises(Exception, match=message):
        _compile(_tmem_fixture(c_form="identity", a_form=a_form))


def test_gemm_tcgen05_cta1_m64_ws_requires_batched_a():
    with pytest.raises(Exception, match="A: expected trailing atom shape"):
        _compile(_tmem_fixture(a_form="flat"))


def test_gemm_tcgen05_batched_a_rejects_unproven_datapath():
    with pytest.raises(Exception, match="Repeat shapes must agree"):
        _compile(_tmem_fixture(M=128, c_form="identity", ws=False))


def test_gemm_tcgen05_cta1_m64_packed_c_requires_explicit_weight_stationary():
    """The old inference case now chooses .ws explicitly and rejects plain MMA."""
    assert "tcgen05.mma.ws" in _compile(_tmem_fixture())
    with pytest.raises(Exception, match="D: layout disagrees"):
        _compile(_tmem_fixture(ws=False, a_form="flat"))


def test_gemm_tcgen05_dense_descI_rejected():
    with pytest.raises(Exception, match="unknown operands.*descI"):
        _compile(_stage_fixture(descI=T.uint32(0x04410490)))


def test_gemm_tcgen05_dense_descI_rejected_at_dispatch():
    with pytest.raises(Exception, match="unknown operands.*descI"):
        _lower(_stage_fixture(descI=T.uint32(0x04410490)))


@pytest.mark.parametrize("mode", ["hoist", "local_hoist", "encode", "recompute"])
def test_gemm_smem_desc_modes_codegen(mode):
    """All four former policies share one explicit address walk now."""
    src = _compile(_stage_fixture())
    assert len(_calls(src)) == 4
    assert "encode_matrix_descriptor" not in src
    assert "shfl.sync" not in src
    with pytest.raises(Exception, match="unknown operands.*smem_desc"):
        _lower(_stage_fixture(smem_desc=mode))


@pytest.mark.parametrize("n", [24, 32])
def test_gemm_tcgen05_instr_desc_fold_mirrors_runtime_shape_rules(n):
    fixture = _shared_fixture(
        128,
        n,
        64,
        "bfloat16",
        mma_shared_layout("bfloat16", 3, (128, 64)),
        mma_shared_layout("bfloat16", 3, (n, 64)),
        a_descriptor=(16, 1024, 3),
        b_descriptor=(16, 1024, 3),
        b_major="k",
        trans_b=False,
    )
    if n == 24:
        with pytest.raises(Exception, match="Invalid matrix shape"):
            _lower(fixture)
    else:
        assert "tcgen05.mma" in _compile(fixture)


@pytest.mark.parametrize("gap", [4096, 2048])
def test_gemm_tcgen05_rejects_non_uniform_atom_grid(gap):
    fixture = _shared_fixture(
        64,
        256,
        64,
        "bfloat16",
        ComposeLayout(3, 3, 3, TileLayout(S[(2, 4, 8, 64) : (gap, 512, 64, 1)])),
        mma_shared_layout("bfloat16", 3, (64, 256)),
        a_descriptor=(16, 1024, 3),
        b_descriptor=(8192, 1024, 3),
    )
    if gap == 4096:
        with pytest.raises(Exception, match="A: view disagrees"):
            _lower(fixture)
    else:
        assert "tcgen05.mma" in _compile(fixture)


def test_gemm_tcgen05_rejects_tf32_mn_major():
    fixture = _shared_fixture(
        64,
        256,
        32,
        "float32",
        mma_shared_layout("float32", 3, (64, 32)),
        mma_shared_layout("float32", 3, (32, 256)),
        a_descriptor=(16, 1024, 3),
        b_descriptor=(4096, 1024, 3),
        a_format="tf32",
        b_format="tf32",
        kind="tf32",
        instruction_k=8,
    )
    with pytest.raises(Exception, match="32-bit MN-major requires"):
        _lower(fixture)
    positive = _shared_fixture(
        64,
        256,
        64,
        "bfloat16",
        mma_shared_layout("bfloat16", 3, (64, 64)),
        mma_shared_layout("bfloat16", 3, (64, 256)),
        a_descriptor=(16, 1024, 3),
        b_descriptor=(8192, 1024, 3),
    )
    assert "tcgen05.mma" in _compile(positive)


@pytest.mark.parametrize("rows,instruction_m", [(64, 128), (128, 256)])
def test_gemm_tcgen05_explicit_mma_m_is_cluster_total_for_cta_group_2(rows, instruction_m):
    layout = tmem_datapath_layout("B" if rows == 64 else "A", rows, 64)
    fixture = _shared_fixture(
        rows,
        64,
        16,
        "float16",
        mma_shared_layout("float16", 0, (rows, 16)),
        mma_shared_layout("float16", 0, (32, 16)),
        a_descriptor=(rows * 16, 128, 0),
        b_descriptor=(512, 128, 0),
        C_layout=layout,
        b_major="k",
        trans_b=False,
        cta=2,
    )
    calls = _calls(_compile(fixture), cta=2)
    assert len(calls) == 1
    assert {(v >> 24) & 0x1F for v in _descriptors(calls)} == {instruction_m // 16}


def _packed_fixture(a_form="packed", dtype="bfloat16", M=64, cta=2):
    """The old column-major alias is explicitly unfolded to packed 16B lines."""
    N, K = 256, 64
    e16 = 128 // tvm.DataType(dtype).bits
    a_layout = (
        TileLayout(S[(M, K) : (1, M)])
        if a_form == "column_major"
        else TileLayout(S[(M, K // e16, e16) : (e16, M * e16, 1)])
    )
    # Original B[K,N/2] has K contiguous in each swizzle atom.
    b_layout = ComposeLayout(3, 3, 3, TileLayout(S[(64, N // cta) : (1, 64)]))
    if cta == 1:
        b_layout = mma_shared_layout(dtype, 3, (K, N))
    b_major = "mn" if cta == 1 else "k"
    b_descriptor = (8192 if cta == 1 else 16, 1024, 3)
    c_layout = tmem_datapath_layout("B" if cta == 2 else "F", M, N)
    k = 16 if dtype == "bfloat16" else 32
    kind = "f16" if dtype == "bfloat16" else "f8f6f4"

    b_shape = (K, N // cta)

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta, block=128, cluster=cta))
        A = T.alloc_shared((M, K), dtype, layout=a_layout, align=1024)
        B = T.alloc_shared(b_shape, dtype, layout=b_layout, align=1024)
        D = T.cuda.decl_tmem(
            0, ty_args=[T.Tensor((M, N), "float32", scope="tmem", layout=c_layout)]
        )
        if T.constexpr(a_form == "column_major"):
            apacked = A.permute(1, 0).view(K // e16, M, e16).permute(1, 0, 2).view(M, K)
        else:
            apacked = A
        av = apacked.rearrange("m (kr k) -> kr m k", k=k).view(1, K // k, M, k)
        bmatrix = B.permute(1, 0)
        bv = bmatrix.rearrange("n (kr k) -> kr n k", k=k).view(1, K // k, N // cta, k)
        if T.cuda.thread_idx("x") == 0:
            T.cuda.tile.apply(
                T.ptx[f"tcgen05.mma.cta_group::{cta}.kind::{kind}"],
                D.view(1, 1, M, N),
                av,
                bv,
                instruction_shape=(M * cta, N, k),
                a_descriptor=(M * 16, 128, 0),
                b_major=b_major,
                b_descriptor=b_descriptor,
            )

    return kernel


@pytest.mark.parametrize("a_layout_kind", ["column_major", "packed_16b"])
def test_gemm_tcgen05_no_swizzle_smem_descriptor_codegen(a_layout_kind):
    src = _compile(_packed_fixture("column_major" if a_layout_kind == "column_major" else "packed"))
    assert len(_calls(src, cta=2)) == 4
    assert "encode_instr_descriptor" not in src
    assert {(v >> 15) & 1 for v in _descriptors(_calls(src, cta=2))} == {0}


def test_gemm_tcgen05_no_swizzle_col_major_requires_explicit_packed_view():
    """The old stride fiction cannot be silently consumed as packed K-major."""
    for rows in (128, 64):
        fixture = _shared_fixture(
            rows,
            256,
            64,
            "bfloat16",
            TileLayout(S[(rows, 64) : (1, rows)]),
            mma_shared_layout("bfloat16", 3, (64, 256)),
            a_descriptor=(rows * 16, 128, 0),
            b_descriptor=(8192, 1024, 3),
        )
        with pytest.raises(Exception, match="A: view disagrees"):
            _lower(fixture)
    # Preserve the original in-domain alias by making its byte interpretation explicit.
    assert len(_calls(_compile(_packed_fixture("column_major")), cta=2)) == 4


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_gemm_tcgen05_no_swizzle_packed_16b_explicit_byte_strides(dtype):
    """The old fp8-only rejection is unnecessary with correct 128B SBO."""
    src = _compile(_packed_fixture(dtype=dtype, cta=1))
    assert f"kind::{'f16' if dtype == 'bfloat16' else 'f8f6f4'}" in src


def _scale_fixture(split_n=False, lane_offsets=False):
    M, N, K = 128, 128 if split_n else 64, 32 if split_n else 64
    cta, n, vec = (2, 64, 1) if split_n else (1, 64, 4)
    dtype = "float8_e4m3fn" if split_n else "float4_e2m1fn"
    sf_dtype = "float8_e8m0fnu" if split_n else "float8_e4m3fn"
    a_layout = mma_shared_layout(dtype, 1, (M, K))
    b_layout = mma_shared_layout(dtype, 1, (N // cta, K))
    sa = sf_tmem_layout(M, SF_K=vec, sf_per_mma=vec)
    sb = sf_tmem_layout(N, SF_K=vec, sf_per_mma=vec)
    if lane_offsets:
        sa = TileLayout.from_iters(sa.shard, sa.replica, {TLane: 1})
        sb = TileLayout.from_iters(sb.shard, sb.replica, {TLane: 2})
    opcode = (
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"
        if split_n
        else "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
    )

    b_shape = (K, N // cta)

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta, block=128, cluster=cta))
        A = T.alloc_shared((M, K), dtype, layout=a_layout, align=1024)
        B = T.alloc_shared((N // cta, K), dtype, layout=b_layout, align=1024)
        D = T.cuda.decl_tmem(
            0,
            ty_args=[
                T.Tensor((M, N), "float32", scope="tmem", layout=tmem_datapath_layout("D", M, N))
            ],
        )
        SA = T.cuda.decl_tmem(
            N if split_n else 256, ty_args=[T.Tensor((M, vec), sf_dtype, scope="tmem", layout=sa)]
        )
        SB = T.cuda.decl_tmem(
            N + 4 if split_n else 320,
            ty_args=[T.Tensor((N, vec), sf_dtype, scope="tmem", layout=sb)],
        )
        dv = D.rearrange("m (nr n) -> nr m n", n=n).view(1, N // n, M, n)
        bv = B.view(N // n, 1, n // cta, K)
        sv = SB.view(N // n, 1, n, vec)
        if T.cuda.thread_idx("x") == 0:
            T.cuda.tile.apply(
                T.ptx[opcode],
                dv,
                A.view(1, 1, M, K),
                bv,
                SA.view(1, 1, M, vec),
                sv,
                instruction_shape=(M * cta, n, K),
                a_descriptor=(16, 256, 1),
                b_descriptor=(16, 256, 1),
            )

    return kernel


def test_gemm_tcgen05_block_scaled_explicit_mma_tile_splits_n():
    src = _compile(_scale_fixture(split_n=True))
    assert (
        len(
            [
                line
                for line in src.splitlines()
                if line.strip().startswith("tvm_builtin_ptx_tcgen05_mma_block_scale_ss_")
            ]
        )
        == 2
    )


def test_gemm_tcgen05_preserves_block_scale_tmem_lane_bases():
    src = _compile(_scale_fixture(lane_offsets=True))
    calls = [
        line
        for line in src.splitlines()
        if line.strip().startswith("tvm_builtin_ptx_tcgen05_mma_block_scale_ss_")
    ]
    assert len(calls) == 1
    assert re.search(r"get_tmem_addr\([^;]+, 1, 0\)", calls[0])
    assert re.search(r"get_tmem_addr\([^;]+, 2, 0\)", calls[0])


@pytest.mark.parametrize(
    "shape,message",
    [
        ((128, 64), "three positive integers"),
        ((128,), "three positive integers"),
        ((0, 64, 16), "three positive integers"),
        ((128, -1, 16), "three positive integers"),
        ((True, 64, 16), "expected trailing atom shape"),
        ((128.0, 64, 16), "three positive integers"),
        ((64, 64, 16), "expected trailing atom shape"),
        ((128, 96, 16), "expected trailing atom shape"),
        ((128, 8, 16), "expected trailing atom shape"),
    ],
)
def test_gemm_tcgen05_explicit_mma_tile_rejects_invalid_config(shape, message):
    with pytest.raises(Exception, match=message):
        _compile(_stage_fixture(n=64, instruction_shape=shape))


@pytest.mark.parametrize("scope_kind", ["thread", "warp"])
def test_gemm_tcgen05_hoisted_descriptor_uniformization(scope_kind):
    """The caller selects one issuer; descriptors have no hidden cross-lane shuffle."""
    fixture = _shared_fixture(
        64,
        256,
        64,
        "bfloat16",
        mma_shared_layout("bfloat16", 3, (64, 64)),
        mma_shared_layout("bfloat16", 3, (64, 256)),
        a_descriptor=(16, 1024, 3),
        b_descriptor=(8192, 1024, 3),
        issue_elect=scope_kind == "warp",
    )
    src = _compile(fixture)
    assert "shfl.sync" not in src
    assert ("elect.sync" in src) == (scope_kind == "warp")
    assert len(_calls(src)) == 4


def test_gemm_tcgen05_preserves_explicit_tmem_lane_bases():
    """The original lane-one D and TMEM-A origins reach the actual PTX call."""
    base = tmem_datapath_layout("F", 64, 64)
    dl = TileLayout.from_iters(base.shard, base.replica, {TLane: 1})
    fixture = _shared_fixture(
        64,
        64,
        16,
        "bfloat16",
        mma_shared_layout("bfloat16", 1, (64, 16)),
        mma_shared_layout("bfloat16", 1, (16, 64)),
        a_descriptor=(16, 256, 1),
        b_descriptor=(512, 256, 1),
        C_layout=dl,
    )
    src = _compile(fixture)
    assert re.search(r"get_tmem_addr\([^;]+, 1, 0\)", _calls(src)[0])

    al = TileLayout.from_iters(
        TileLayout(S[(64, 16) : (1 @ TLane, 1 @ TCol)]).shard, [], {TLane: 1}
    )
    bl = mma_shared_layout("bfloat16", 1, (16, 64))

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        A = T.cuda.decl_tmem(256, ty_args=[T.Tensor((64, 16), "bfloat16", scope="tmem", layout=al)])
        D = T.cuda.decl_tmem(
            400, ty_args=[T.Tensor((64, 64), "float32", scope="tmem", layout=base)]
        )
        B = T.alloc_shared((16, 64), "bfloat16", layout=bl, align=1024)
        if T.cuda.thread_idx("x") == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                D.view(1, 1, 64, 64),
                A.view(1, 1, 64, 16),
                B.permute(1, 0).view(1, 1, 64, 16),
                instruction_shape=(64, 64, 16),
                b_major="mn",
                b_descriptor=(512, 256, 1),
            )

    src = _compile(kernel)
    assert re.search(r"get_tmem_addr\([^;]+, 1, 0\)", _calls(src)[0])
