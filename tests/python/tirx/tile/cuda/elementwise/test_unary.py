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

"""Original unary regions with caller-defined scalar atoms and explicit views."""

import math
import re

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.backend.cuda.axis import laneid, tid_in_wg, tx, warpid, wid_in_wg
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


def _scalar(operation, src_dtype, dst_dtype, with_bias=False):
    source, destination = tvm.ir.PrimType(src_dtype), tvm.ir.PrimType(dst_dtype)

    @T.function(private=True)
    def atom(x: source, bias: source) -> destination:
        # Conversion and composition are explicit parts of this scalar atom.
        if T.constexpr(with_bias):
            value = T.cast(x, "float32") * T.float32(1.5) + T.cast(bias, "float32")
        else:
            value = T.cast(x, "float32")
        if T.constexpr(operation == "zero"):
            return T.cast(0, dst_dtype)
        elif T.constexpr(operation == "sqrt"):
            return T.cast(T.sqrt(value), dst_dtype)
        elif T.constexpr(operation == "reciprocal"):
            return T.cast(T.float32(1) / value, dst_dtype)
        elif T.constexpr(operation == "exp"):
            return T.cast(T.exp(value), dst_dtype)
        elif T.constexpr(operation == "exp2"):
            return T.cast(T.exp2(value), dst_dtype)
        elif T.constexpr(operation == "log2"):
            return T.cast(T.log2(value), dst_dtype)
        else:
            return T.cast(value / (T.float32(1) + T.exp(-value)), dst_dtype)

    return atom


def _reference(operation, x):
    return {
        "zero": np.zeros_like,
        "sqrt": np.sqrt,
        "reciprocal": lambda v: 1 / v,
        "exp": np.exp,
        "exp2": np.exp2,
        "log2": np.log2,
        "silu": lambda v: v / (1 + np.exp(-v.astype("float32"))),
    }[operation](x)


def _run(kernel, arrays, expected, output=0, atol=1e-3, rtol=1e-7, target="cuda"):
    with tvm.target.Target(target):
        module = tvm.tirx.compile(kernel, target=target, pipeline="tirx")

    def check():
        args = [tvm.runtime.tensor(a, tvm.cuda()) for a in arrays]
        module(*args)
        tvm.testing.assert_allclose(args[output].numpy(), expected, atol=atol, rtol=rtol)

    tvm.testing.run_with_gpu_lock(check)
    return module


def _shared_case(input, operation, src_dtype, dst_dtype, bias_type=None, target="cuda"):
    shape, start, end, extent, result_extent, threads, device = input
    size, count = math.prod(shape), math.prod(extent)
    in_place = src_dtype == dst_dtype
    bias_constant = 0.88 if bias_type == "const" else 0
    atom = _scalar(operation, src_dtype, dst_dtype, with_bias=bias_type is not None)
    layout = TileLayout(S[shape])

    def allocate_output(source):
        if in_place:
            return source
        return T.alloc_shared(shape, dst_dtype, layout=layout)

    @T.function
    def kernel(
        A: T.Tensor(shape, src_dtype, layout=layout),
        B: T.Tensor(shape, dst_dtype, layout=layout),
        Bias: T.Tensor(shape, src_dtype, layout=layout),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=threads))
        tid = T.cuda.thread_idx("x")
        AS = T.alloc_shared(shape, src_dtype, layout=layout)
        DS = T.meta_var(allocate_output(AS))
        if T.constexpr(bias_type is not None):
            BS = T.alloc_shared(shape, src_dtype, layout=layout)
            bsf, bf = BS.view(size), Bias.view(size)
        af, sf, df, outf = A.view(size), AS.view(size), DS.view(size), B.view(size)
        for i in T.serial(T.ceildiv(size, threads)):
            q = i * threads + tid
            if q < size:
                sf[q] = af[q]
                if T.constexpr(not in_place):
                    df[q] = outf[q]
                if T.constexpr(bias_type is not None):
                    bsf[q] = bf[q]
        T.cuda.cta_sync()
        a = T.alloc_local((1,), src_dtype, layout="default")
        b = T.alloc_local((1,), src_dtype, layout="default")
        d = T.alloc_local((1,), dst_dtype, layout="default")
        for i in T.serial(T.ceildiv(count, threads)):
            q = i * threads + tid
            if q < count:
                c = T.meta_var(
                    tuple(q // math.prod(extent[k + 1 :]) % extent[k] for k in range(len(extent)))
                )
                src_index = T.meta_var(tuple(start[k] + c[k] for k in range(len(extent))))
                dst_index = T.meta_var(tuple(end[k] + c[k] for k in range(len(extent))))
                a[0] = AS[src_index]
                if T.constexpr(bias_type == "region"):
                    b[0] = BS[src_index]
                else:
                    b[0] = T.cast(bias_constant, src_dtype)
                if T.constexpr(operation == "zero"):
                    T.cuda.tile.apply(T.ptx["mov.b16"], d.view("uint16"), T.uint16(0))
                else:
                    T.cuda.tile.ewise(atom, d, a, b)
                DS[dst_index] = d[0]
        T.cuda.cta_sync()
        for i in T.serial(T.ceildiv(size, threads)):
            q = i * threads + tid
            if q < size:
                if T.constexpr(in_place):
                    af[q] = df[q]
                else:
                    outf[q] = df[q]

    rng = np.random.RandomState(0)
    a = np.abs(rng.rand(*shape).astype(src_dtype)) + 0.1
    bias = rng.rand(*shape).astype(src_dtype)
    out = np.zeros(shape, dtype=dst_dtype)
    ss = tuple(slice(s, s + e) for s, e in zip(start, extent))
    ds = tuple(slice(s, s + e) for s, e in zip(end, result_extent))
    value = a[ss]
    if bias_type is not None:
        value = value * 1.5 + (bias[ss] if bias_type == "region" else 0.88)
    reference = a.copy() if in_place else out.copy()
    reference[ds] = _reference(operation, value).astype(dst_dtype)
    atol = (
        (
            1e-1
            if src_dtype == "float16" and operation == "exp"
            else 1e-2
            if src_dtype == "float16"
            else 1e-3
        )
        if bias_type is not None
        else 1e-3
    )
    if not in_place:
        atol = 1e-1 if bias_type is not None else 1e-2
    return _run(
        kernel,
        [a, out, bias],
        reference,
        output=0 if in_place else 1,
        atol=atol,
        rtol=1e-2 if not in_place else 1e-7,
        target=target,
    )


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["zero", "sqrt"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype", [("float16", "float16"), ("float32", "float16"), ("float32", "bfloat16")]
)
def test_unary_op_shared(input, op_type, src_dtype, dst_dtype):
    _shared_case(input, op_type, src_dtype, dst_dtype)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
@pytest.mark.parametrize("bias_type", ["const", "region"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("float16", "float16"),
        ("float32", "float32"),
        ("float32", "float16"),
        ("float32", "bfloat16"),
    ],
)
def test_unary_op_shared_with_bias_scale(input, op_type, bias_type, src_dtype, dst_dtype):
    _shared_case(input, op_type, src_dtype, dst_dtype, bias_type)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("exec_scope", ["warp", "warpgroup"])
def test_unary_op_shared_subcta_scope(exec_scope):
    warps = 4 if exec_scope == "warpgroup" else 1
    shape = (warps * 32, 8)
    group = (warps, 32, 8, 1) if exec_scope == "warpgroup" else (32, 8, 1)
    layout = (
        TileLayout(S[(4, 32, 8, 1) : (1 @ wid_in_wg, 1 @ laneid, 1, 1)])
        if exec_scope == "warpgroup"
        else TileLayout(S[(32, 8, 1) : (1 @ laneid, 1, 1)])
    )
    size = math.prod(shape)

    @T.function
    def kernel(A: T.Tensor(shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        wg = T.cuda.warpgroup_id()
        AS = T.alloc_shared(shape, "float16", layout="default")
        sf, af = AS.view(size), A.view(size)
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                sf[q] = af[q]
        T.cuda.cta_sync()
        r = T.alloc_local(group, "uint16", layout=layout)
        if T.constexpr(exec_scope == "warp"):
            if warp == 5:
                T.cuda.tile.apply(T.ptx["mov.b16"], r, T.uint16(0), scope="warp")
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"], AS.view("uint16").view(*group), r, scope="warp"
                )
        else:
            if wg == 1:
                T.cuda.tile.apply(T.ptx["mov.b16"], r, T.uint16(0), scope="warpgroup")
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"], AS.view("uint16").view(*group), r, scope="warpgroup"
                )
        T.cuda.cta_sync()
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                af[q] = sf[q]

    values = np.random.RandomState(0).rand(*shape).astype("float16")
    _run(kernel, [values], np.zeros_like(values))


@pytest.mark.parametrize(
    "input",
    [
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            1,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            4,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
        (
            "wgmma",  # layout
            2,  # N_GROUPS
            8,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["reciprocal", "exp", "exp2", "log2"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype", [("float16", "float16"), ("float32", "float16"), ("float32", "bfloat16")]
)
def test_unary_op_local(input, op_type, src_dtype, dst_dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, device_id = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = TileLayout(S[g_shape_a])
    acc_shape = red_shape = (16, NUM_COL)

    scalar_atom = _scalar(op_type, src_dtype, dst_dtype, with_bias=False)

    @T.function
    def test_unary(
        A: T.Tensor(g_shape_a, src_dtype, layout=g_layout_a),
        B: T.Tensor(g_shape_b, dst_dtype, layout=g_layout_b),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=N_GROUPS * 128))
        bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
        wg_id = T.cuda.warpgroup_id()
        warp_id_in_wg = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()
        if T.cuda.warp_id() < N_WARPS:
            # acc layout
            atom = T.TileLayout(T.S[(1, 2) : (2, 1)])
            warp_layout = T.TileLayout(T.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
            warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
            tile = T.TileLayout(T.S[(2, NUM_COL // 8) : (1, 2)])
            acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
            acc = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [2, NUM_COL // 4],
                        src_dtype,
                        scope="local",
                        layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                    )
                ]
            )
            res = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [2, NUM_COL // 4],
                        dst_dtype,
                        scope="local",
                        layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                    )
                ]
            )
            for i in T.serial(NUM_COL // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        acc[j, i * 2 + vec] = A[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ]

                # unary op
            acc_view = acc.view(*acc_shape, layout=acc_layout)
            res_view = res.view(*red_shape, layout=acc_layout)
            T.cuda.tile.ewise(
                scalar_atom,
                res_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                acc_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                T.cast(0, src_dtype),
                scope="warp",
            )

            # write res into B
            for i in T.serial(NUM_COL // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        B[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ] = res[j, i * 2 + vec]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_unary})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        np.random.seed(0)
        A_np = np.abs(np.random.rand(*g_shape_a).astype(src_dtype)) + 0.1
        B_np = np.zeros(g_shape_b, dtype=dst_dtype)

        # find ref result
        if op_type == "reciprocal":
            B_ref = (1 / A_np).astype(dst_dtype)
        elif op_type == "exp":
            B_ref = np.exp(A_np).astype(dst_dtype)
        elif op_type == "exp2":
            B_ref = np.exp2(A_np).astype(dst_dtype)
        elif op_type == "log2":
            B_ref = np.log2(A_np).astype(dst_dtype)
        else:
            raise ValueError(f"op_type={op_type} is not supported")

        def run_and_check():
            dev = tvm.cuda(device_id)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            tvm.testing.assert_allclose(B_ref, B.numpy(), atol=1e-2, rtol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "input",
    [
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            1,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
        (
            "wgmma",  # layout
            1,  # N_GROUPS
            4,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
        (
            "wgmma",  # layout
            2,  # N_GROUPS
            8,  # N_WARPS
            32,  # thread_cnt
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
@pytest.mark.parametrize("bias_type", ["const", "region"])
@pytest.mark.parametrize(
    "src_dtype,dst_dtype", [("float32", "float32"), ("float32", "float16"), ("float32", "bfloat16")]
)
def test_unary_op_local_with_bias_scale(input, op_type, bias_type, src_dtype, dst_dtype):
    layout, N_GROUPS, N_WARPS, thread_cnt, device_id = input
    assert layout == "wgmma", "logical tensor which is not WGMMA layout is not supported"

    # get shape info
    NUM_COL = 128
    g_shape_a = g_shape_b = g_shape_bias = (16 * N_WARPS, NUM_COL)
    g_layout_a = g_layout_b = g_layout_bias = TileLayout(S[g_shape_a])
    acc_shape = red_shape = bias_shape = (16, NUM_COL)

    scale = T.float16(1.5) if src_dtype == "float16" else T.float32(1.5)
    const_bias = T.float16(0.88) if src_dtype == "float16" else T.float32(0.88)

    scalar_atom = _scalar(op_type, src_dtype, dst_dtype, with_bias=True)

    @T.function
    def test_unary_with_bias(
        A: T.Tensor(g_shape_a, src_dtype, layout=g_layout_a),
        B: T.Tensor(g_shape_b, dst_dtype, layout=g_layout_b),
        bias: T.Tensor(g_shape_bias, src_dtype, layout=g_layout_bias),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=N_GROUPS * 128))
        bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
        wg_id = T.cuda.warpgroup_id()
        warp_id_in_wg = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()
        if T.cuda.warp_id() < N_WARPS:
            # acc layout
            atom = T.TileLayout(T.S[(1, 2) : (2, 1)])
            warp_layout = T.TileLayout(T.S[(8, 4) : (4 @ laneid, 1 @ laneid)])
            warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
            tile = T.TileLayout(T.S[(2, NUM_COL // 8) : (1, 2)])
            acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
            acc = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [2, NUM_COL // 4],
                        src_dtype,
                        scope="local",
                        layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                    )
                ]
            )
            bias_local = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [2, NUM_COL // 4],
                        src_dtype,
                        scope="local",
                        layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                    )
                ]
            )
            res = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [2, NUM_COL // 4],
                        dst_dtype,
                        scope="local",
                        layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
                    )
                ]
            )
            for i in T.serial(NUM_COL // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        acc[j, i * 2 + vec] = A[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ]
                # load bias into bias_local
            for i in T.serial(NUM_COL // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        bias_local[j, i * 2 + vec] = bias[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ]

                # unary op
            acc_view = acc.view(*acc_shape, layout=acc_layout)
            res_view = res.view(*red_shape, layout=acc_layout)
            bias_view = bias_local.view(*bias_shape, layout=acc_layout)
            if T.constexpr(bias_type == "const"):
                T.cuda.tile.ewise(
                    scalar_atom,
                    res_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                    acc_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                    const_bias,
                    scope="warp",
                )
            else:
                T.cuda.tile.ewise(
                    scalar_atom,
                    res_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                    acc_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                    bias_view.rearrange("(j r) (c l v) -> (r l) c j v", j=2, r=8, c=16, l=4, v=2),
                    scope="warp",
                )

                # write res into B
            for i in T.serial(NUM_COL // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        B[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ] = res[j, i * 2 + vec]

    def get_ref(A_np, bias_np):
        A_ref = A_np.copy()
        if bias_type == "region":
            if op_type == "sqrt":
                A_ref = np.sqrt(A_np * scale.value + bias_np)
            elif op_type == "exp":
                A_ref = np.exp(A_np * scale.value + bias_np)
        elif bias_type == "const":
            if op_type == "sqrt":
                A_ref = np.sqrt(A_np * scale.value + const_bias.value)
            elif op_type == "exp":
                A_ref = np.exp(A_np * scale.value + const_bias.value)
        else:
            raise ValueError(f"bias_type={bias_type} is not supported")
        return A_ref.astype(dst_dtype)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(src_dtype)
        bias_np = np.random.rand(*g_shape_bias).astype(src_dtype)
        B_np = np.zeros(g_shape_b, dtype=dst_dtype)
        mod = tvm.IRModule({"main": test_unary_with_bias})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        B_ref = get_ref(A_np, bias_np)
        atol = 1e-3 if src_dtype == dst_dtype else 2e-2

        def run_and_check():
            dev = tvm.cuda(device_id)
            A = tvm.runtime.tensor(A_np, dev)
            bias = tvm.runtime.tensor(bias_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B, bias)
            tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("shape", [(128, 8), (128, 4, 16), (128, 5, 5)])
@pytest.mark.parametrize("op_type", ["fill"])
@pytest.mark.parametrize("exec_scope", ["thread", "cta"])
@pytest.mark.parametrize("storage_scope", ["local", "shared"])
def test_unary_op_vectorized(shape, op_type, exec_scope, storage_scope):
    n = math.prod(shape[1:])
    bits = int(np.float16(7.89).view("uint16"))
    group_layout = TileLayout(S[(4, 32, n, 1) : (1 @ warpid, 1 @ laneid, 1, 1)])

    @T.function
    def kernel(A: T.Tensor(shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        r = T.alloc_local(shape[1:], "float16", layout="default")
        if T.constexpr(exec_scope == "cta"):
            values = r.view(4, 32, n, 1, layout=group_layout).view("uint16")
            T.cuda.tile.apply(T.ptx["mov.b16"], values, T.uint16(bits), scope="cta")
        else:
            T.cuda.tile.apply(T.ptx["mov.b16"], r.view("uint16").view(n, 1), T.uint16(bits))
        rf, af = r.view(n), A.view(128, n)
        if T.constexpr(storage_scope == "shared"):
            sm = T.alloc_shared(shape, "float16", layout="default")
            sf = sm.view(128, n)
            for i in range(n):
                sf[tid, i] = rf[i]
            T.cuda.cta_sync()
            for i in range(n):
                af[tid, i] = sf[tid, i]
        else:
            for i in range(n):
                af[tid, i] = rf[i]

    x = np.random.RandomState(0).rand(*shape).astype("float16")
    _run(kernel, [x], np.full(shape, np.float16(7.89)), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["zero", "sqrt", "reciprocal", "exp", "silu"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_unary_op_local_thread_wise(op_type, dtype):
    """Test unary ops in thread scope with local tensors (trivial layout)."""
    shape = (64, 32)
    local_shape = shape[1:]

    scalar_atom = _scalar(op_type, dtype, dtype)

    @T.function
    def kernel(A: T.Tensor(shape, dtype, layout=TileLayout(S[shape]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        a_local = T.alloc_tensor(
            ty_args=[T.Tensor(local_shape, dtype, scope="local", layout=TileLayout(S[local_shape]))]
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            a_local.view("uint32").view(4, 4),
            A.sub[tid, :].view("uint32").view(4, 4),
        )
        if T.constexpr(op_type == "zero"):
            T.cuda.tile.apply(T.ptx["mov.b16"], a_local.view("uint16").view(32, 1), T.uint16(0))
        else:
            T.cuda.tile.ewise(scalar_atom, a_local, a_local, T.float16(0))
        T.cuda.tile.apply(
            T.ptx["st.global.v4.b32"],
            A.sub[tid, :].view("uint32").view(4, 4),
            a_local.view("uint32").view(4, 4),
        )

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.abs(np.random.rand(*shape).astype(dtype)) + 0.1
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        if op_type == "zero":
            A_ref = np.zeros_like(A_np)
        elif op_type == "sqrt":
            A_ref = np.sqrt(A_np)
        elif op_type == "reciprocal":
            A_ref = (1.0 / A_np).astype(dtype)
        elif op_type == "exp":
            A_ref = np.exp(A_np)
        elif op_type == "silu":
            A_ref = (A_np / (1.0 + np.exp(-A_np.astype("float32")))).astype(dtype)

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            mod(A)
            tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-2, rtol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("shape", [(8,), (16, 16), (5, 5)])
@pytest.mark.parametrize("A_dtype", ["float16", "float32"])
@pytest.mark.parametrize("B_dtype", ["float16", "float32"])
def test_cast_thread_local(shape, A_dtype, B_dtype):
    A_ref = np.random.rand(*shape).astype(A_dtype)
    B_ref = np.random.rand(*shape).astype(B_dtype)
    B_ref = A_ref.astype(B_dtype)

    words = {"float16": "uint16", "float32": "float32"}
    load = {"float16": "ld.global.b16", "float32": "ld.global.f32"}[A_dtype]
    store = {"float16": "st.global.b16", "float32": "st.global.f32"}[B_dtype]
    opcode = {
        ("float16", "float32"): "cvt.f32.f16",
        ("float32", "float16"): "cvt.rn.f16.f32",
        ("float16", "float16"): "mov.b16",
        ("float32", "float32"): "mov.f32",
    }[A_dtype, B_dtype]
    count = math.prod(shape)

    @T.function
    def test_cast(
        A: T.Tensor(shape, A_dtype, layout=TileLayout(S[shape])),
        B: T.Tensor(shape, B_dtype, layout=TileLayout(S[shape])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(256,)))
        cta_id = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        A_local = T.alloc_local(shape, dtype=A_dtype, layout=TileLayout(S[shape]))
        B_local = T.alloc_local(shape, dtype=B_dtype, layout=TileLayout(S[shape]))
        # One thread owns this entire tensor; other launched threads stay idle.
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx[load],
                A_local.view(words[A_dtype]).view(count, 1),
                A.view(words[A_dtype]).view(count, 1),
            )
            T.cuda.tile.apply(
                T.ptx[opcode],
                B_local.view(words[B_dtype]).view(count, 1),
                A_local.view(words[A_dtype]).view(count, 1),
            )
            T.cuda.tile.apply(
                T.ptx[store],
                B.view(words[B_dtype]).view(count, 1),
                B_local.view(words[B_dtype]).view(count, 1),
            )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_warpgroup_local_view(A_dtype, B_dtype):
    """T.cast in warpgroup scope with offset (tid_in_wg + layout offset). Covers offset/tid_in_wg/warpgroup scope."""  # noqa: E501
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    use_offset = True
    if use_offset:
        from tvm.tirx.layout import Axis, Iter

        m_axis = Axis.get("m")
        shard = [Iter(N_THREADS, 1, tid_in_wg), Iter(LOCAL_LEN, 1, m_axis)]
        cast_layout = TileLayout.from_iters(shard, [], {m_axis: 0})
    else:
        cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tid_in_wg, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tid_in_wg, i]
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.view("uint16").view(4, 32, LOCAL_LEN, 1),
            reg_src_view.view(4, 32, LOCAL_LEN, 1),
            scope="warpgroup",
        )
        for i in T.serial(LOCAL_LEN):
            B[tid_in_wg, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_cta_local_view(A_dtype, B_dtype):
    """T.cast with view+layout in CTA scope (128 threads, register->register)."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N_THREADS,)))
        cta_id = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tx_var, i]
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.view("uint16").view(4, 32, LOCAL_LEN, 1),
            reg_src_view.view(4, 32, LOCAL_LEN, 1),
            scope="cta",
        )
        for i in T.serial(LOCAL_LEN):
            B[tx_var, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_warpgroup_src_layout_to_flat_uses_vec2_intrinsic(A_dtype, B_dtype):
    """Regression: GEMM-epilogue cast pattern must emit the packed vec2 cuda intrinsic.

    Pattern: both sides have ``wg_local_layout`` (per-thread 1xK row). dst is
    allocated per-chunk to keep both operands wg-distributed — the dispatch
    requires layout-symmetric operands (no flat-vs-wg asymmetry).
    """
    from tvm.backend.cuda.layout import wg_local_layout

    N_THREADS, LOCAL_LEN, N_CHUNKS = 128, 8, 4
    g_shape = (N_THREADS, LOCAL_LEN * N_CHUNKS)
    g_layout = TileLayout(S[g_shape])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16x2.f32" if B_dtype == "float16" else "cvt.rn.bf16x2.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        for no in T.unroll(N_CHUNKS):
            reg_src = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (LOCAL_LEN,),
                        A_dtype,
                        scope="local",
                        layout="default",
                    )
                ],
            )
            Dreg_chunk = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (LOCAL_LEN,),
                        B_dtype,
                        scope="local",
                        layout="default",
                    )
                ],
            )
            for i in T.serial(LOCAL_LEN):
                reg_src[i] = A[tid, no * LOCAL_LEN + i]
            reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=wg_local_layout(LOCAL_LEN))
            Dreg_chunk_view = Dreg_chunk.view(
                N_THREADS, LOCAL_LEN, layout=wg_local_layout(LOCAL_LEN)
            )
            pairs = reg_src_view.view(4, 32, LOCAL_LEN // 2, 2)
            # PTX writes its first f32 input into the high halfword.
            T.cuda.tile.apply(
                T.ptx[opcode],
                Dreg_chunk_view.view("uint32").view(4, 32, LOCAL_LEN // 2, 1),
                pairs[:, :, :, 1:2],
                pairs[:, :, :, 0:1],
                scope="warpgroup",
            )
            for i in T.serial(LOCAL_LEN):
                B[tid, no * LOCAL_LEN + i] = Dreg_chunk[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        assert opcode in src, src

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
@pytest.mark.parametrize("slice_start,slice_end", [(0, 4), (2, 6), (4, 8)])
def test_cast_local_view_sliced(A_dtype, B_dtype, slice_start, slice_end):
    """An explicit CTA conversion writes only the selected local slice."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref[:, slice_start:slice_end] = A_ref[:, slice_start:slice_end].astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def kernel(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N_THREADS,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tx, i]
            reg_dst[i] = T.cast(0, B_dtype)
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.sub[:, slice_start:slice_end]
            .view("uint16")
            .view(4, 32, slice_end - slice_start, 1),
            reg_src_view.sub[:, slice_start:slice_end].view(4, 32, slice_end - slice_start, 1),
            scope="cta",
        )
        for i in T.serial(LOCAL_LEN):
            B[tx, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_ref, dev)
        B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
        mod(A, B)
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("slice_start,slice_end", [(0, 2), (2, 4)])
def test_cast_mixed_axes_and_subregion(slice_start, slice_end):
    """Test cast with mixed axes and subregion."""

    N_WARPS, LANES = 2, 32
    LOCAL_LEN = 4
    full_shape = (8, N_WARPS, 4, LOCAL_LEN)
    g_layout = TileLayout(S[full_shape])
    cast_layout = TileLayout(S[full_shape : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)])

    A_ref = np.zeros(full_shape, dtype="float32")
    for j in range(full_shape[0]):
        for w in range(full_shape[1]):
            for k in range(full_shape[2]):
                for i in range(full_shape[3]):
                    A_ref[j, w, k, i] = float(j * 1000 + w * 100 + k * 10 + i)
    B_ref = np.zeros(full_shape, dtype="float16")
    B_ref[:, :, :, slice_start:slice_end] = A_ref[:, :, :, slice_start:slice_end].astype("float16")

    @T.function
    def kernel(
        A: T.Tensor(full_shape, "float32", layout=g_layout),
        B: T.Tensor(full_shape, "float16", layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=N_WARPS * 32))
        cta_id = T.cuda.block_idx("x")
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        reg_src = T.alloc_tensor(
            ty_args=[T.Tensor((LOCAL_LEN,), "float32", scope="local", layout="default")]
        )
        reg_dst = T.alloc_tensor(
            ty_args=[T.Tensor((LOCAL_LEN,), "float16", scope="local", layout="default")]
        )
        j, k = lane_id // 4, lane_id % 4
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[j, warp_id, k, i]
            reg_dst[i] = T.float16(0)
        reg_src_view = reg_src.view(*full_shape, layout=cast_layout)
        reg_dst_view = reg_dst.view(*full_shape, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            reg_dst_view.sub[:, :, :, slice_start:slice_end]
            .permute(1, 0, 2, 3)
            .view("uint16")
            .view(2, 32, slice_end - slice_start, 1),
            reg_src_view.sub[:, :, :, slice_start:slice_end]
            .permute(1, 0, 2, 3)
            .view(2, 32, slice_end - slice_start, 1),
            scope="cta",
        )
        j_1, k_1 = lane_id // 4, lane_id % 4
        for i in T.serial(LOCAL_LEN):
            B[j_1, warp_id, k_1, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_ref, dev)
        B = tvm.runtime.tensor(np.zeros(full_shape, dtype="float16"), dev)
        mod(A, B)
        tvm.testing.assert_allclose(
            B.numpy(),
            B_ref,
            atol=1e-2,
            rtol=0,
        )

    tvm.testing.run_with_gpu_lock(run_and_check)


def test_cast_validate_extent_mismatch_rejected():
    """Validation rejects when src and dst layouts have same thread positions but different extents."""  # noqa: E501

    view_shape = (2, 8, 4, 8)
    g_layout = TileLayout(S[view_shape])
    src_layout = TileLayout(S[view_shape : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)])
    dst_layout = TileLayout(
        S[view_shape : (2 @ warpid, 8 @ laneid, 1 @ laneid, 1)]
    )  # dim1 extent 8 != 4

    @T.function
    def kernel(
        A: T.Tensor(view_shape, "float32", layout=g_layout),
        B: T.Tensor(view_shape, "float16", layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 32))
        cta_id = T.cuda.block_idx("x")
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        reg_src = T.alloc_tensor(
            ty_args=[T.Tensor((8,), "float32", scope="local", layout="default")]
        )
        reg_dst = T.alloc_tensor(
            ty_args=[T.Tensor((8,), "float16", scope="local", layout="default")]
        )
        j, k = lane_id // 4, lane_id % 4
        for i in T.serial(8):
            reg_src[i] = A[warp_id, j, k, i]
        reg_src_view = reg_src.view(*view_shape, layout=src_layout)
        reg_dst_view = reg_dst.view(*view_shape, layout=dst_layout)
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            reg_dst_view.view("uint16").view(2, 32, 8, 1),
            reg_src_view.view(2, 32, 8, 1),
            scope="cta",
        )
        j_1, k_1 = lane_id // 4, lane_id % 4
        for i in T.serial(8):
            B[warp_id, j_1, k_1, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        # The mismatched dst also fails the scope-level check (thread axes don't
        # span the full CTA), which fires first — either rejection is fine.
        with pytest.raises(
            Exception,
            match="incorrect .* owner",
        ):
            tvm.tirx.compile(mod, target=target, pipeline="tirx")


def test_unary_exp_f16_shared_scalar_fallback_dispatch():
    """The caller's scalar exp composite stages shared data explicitly."""
    module = _shared_case(
        ((64, 32), (0, 0), (0, 0), (64, 32), (64, 32), 64, 0),
        "exp",
        "float16",
        "float16",
        target={
            "kind": "cuda",
            "arch": "sm_80",
            "backend_config": {"cuda": {"target_format": "ptx"}},
        },
    )
    assert re.search(r"exp\(|expf", module.mod.imports[0].inspect_source())


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,intrinsic",
    [
        ("float32", "float16", "__float22half2_rn"),
        ("float16", "float32", "__half22float2"),
    ],
)
def test_cast_vec2_packed_dispatch(src_dtype, dst_dtype, intrinsic):
    """Packed narrowing and scalar widening preserve the original pair values."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    load = "ld.global.f32" if src_dtype == "float32" else "ld.global.b16"
    store = "st.global.f32" if dst_dtype == "float32" else "st.global.b16"
    src_word = "float32" if src_dtype == "float32" else "uint16"
    dst_word = "float32" if dst_dtype == "float32" else "uint16"

    @T.function
    def k(
        A: T.Tensor(shape, src_dtype, layout=lay), B: T.Tensor(shape, dst_dtype, layout=lay)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ra = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], src_dtype, scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rb = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], dst_dtype, scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        T.cuda.tile.apply(
            T.ptx[load], ra.view(src_word).view(32, 1), A.sub[tx, :].view(src_word).view(32, 1)
        )
        if T.constexpr(src_dtype == "float32"):
            pairs = ra.view(16, 2)
            T.cuda.tile.apply(
                T.ptx["cvt.rn.f16x2.f32"],
                rb.view("uint32").view(16, 1),
                pairs[:, 1:2],
                pairs[:, 0:1],
            )
        else:
            # __half22float2 expands to two scalar PTX conversions; there is
            # no native cvt.f32x2.f16x2 instruction.
            T.cuda.tile.apply(T.ptx["cvt.f32.f16"], rb.view(32, 1), ra.view("uint16").view(32, 1))
        T.cuda.tile.apply(
            T.ptx[store], B.sub[tx, :].view(dst_word).view(32, 1), rb.view(dst_word).view(32, 1)
        )

    target = tvm.target.Target(
        {"kind": "cuda", "arch": "sm_80", "backend_config": {"cuda": {"target_format": "ptx"}}}
    )
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    opcode = "cvt.rn.f16x2.f32" if src_dtype == "float32" else "cvt.f32.f16"
    assert opcode in src, src
    rng = np.random.RandomState(0)
    a = rng.rand(*shape).astype(src_dtype)
    _run(
        k,
        [a, np.zeros(shape, dtype=dst_dtype)],
        a.astype(dst_dtype),
        output=1,
        target=target,
        atol=0,
        rtol=0,
    )


_SL_ROWS, _SL_COLS = 128, 8


def _sl_compile(fn):
    target = tvm.target.Target("cuda")
    with target:
        tvm.tirx.compile(tvm.IRModule({"main": fn}), target=target, pipeline="tirx")


def test_cast_wg_rejects_thread_local_view():
    """warpgroup conversion on a .local() (thread-axis-stripped) view is rejected."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.local().view("uint16").view(8, 1),
            src.local().view(8, 1),
            scope="warpgroup",
        )
        dst_row = dst.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_cta_rejects_thread_local_view():
    """CTA conversion on a .local() view is rejected (cta -> tx)."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(_SL_ROWS,)))
        _bx = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tx, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tx, 1)]),
                )
            ]
        )
        src_row = src.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tx_var, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.local().view("uint16").view(8, 1),
            src.local().view(8, 1),
            scope="cta",
        )
        dst_row = dst.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            B[tx_var, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_wg_rejects_partial_thread_coverage():
    """A tid_in_wg layout covering only 64 of the 128 wg threads is rejected."""
    half = 64

    @T.function
    def kernel(
        A: T.Tensor((half, _SL_COLS), "float32", layout=TileLayout(S[half, _SL_COLS])),
        B: T.Tensor((half, _SL_COLS), "float16", layout=TileLayout(S[half, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (half, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(half, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (half, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(half, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view("uint16").view(-1, 32, _SL_COLS, 1),
            src.view(-1, 32, _SL_COLS, 1),
            scope="warpgroup",
        )
        dst_row = dst.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_wg_accepts_wg_level_layout():
    """warpgroup conversion on a wg-level (tid_in_wg-distributed) layout compiles."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view("uint16").view(-1, 32, _SL_COLS, 1),
            src.view(-1, 32, _SL_COLS, 1),
            scope="warpgroup",
        )
        dst_row = dst.local(_SL_COLS)
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    _sl_compile(kernel)


def test_cast_thread_accepts_local_view():
    """thread scope is exempt: a thread-axis-free local tile still compiles."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(_SL_ROWS,)))
        _bx = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor((_SL_COLS,), "float32", scope="local", layout=TileLayout(S[(_SL_COLS,)]))
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor((_SL_COLS,), "float16", scope="local", layout=TileLayout(S[(_SL_COLS,)]))
            ]
        )
        for i in T.serial(_SL_COLS):
            src[i] = A[tx_var, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"], dst.view("uint16").view(_SL_COLS, 1), src.view(_SL_COLS, 1)
        )
        for i in T.serial(_SL_COLS):
            B[tx_var, i] = dst[i]

    _sl_compile(kernel)


# --- tcgen05 split-laneid atom (B00011 / tf32 A-cast path) -----------------
_TCGEN05_M, _TCGEN05_K = 64, 64
_TCGEN05_REGS_PER_THREAD = 32  # ``.16x256b`` fp32 atom on (64, 64)


def _tcgen05_16x256b_row_col(tid_wg: T.int32, lane: T.int32, reg_idx: T.int32):
    """Map ``(tid_in_wg, reg)`` → logical ``(row, col)`` for ``.16x256b`` fp32 atom."""
    t0 = lane & T.int32(3)
    t1 = lane >> 2
    v0p = reg_idx & T.int32(1)
    va = (reg_idx >> 1) & T.int32(1)
    vb = reg_idx >> 2
    wid = tid_wg >> 5
    row = t1 + T.int32(8) * va + T.int32(16) * wid
    col = v0p + T.int32(2) * t0 + T.int32(8) * vb
    return row, col


def _tcgen05_cast_atom_layout():
    from tvm.backend.cuda.layout import tcgen05_atom_layout

    # Production tf32_hc_prenorm_gemm pattern: bf16 payload in fp32 atom slots.
    return tcgen05_atom_layout("16x256b", (_TCGEN05_M, _TCGEN05_K), "float32")


def _tcgen05_cast_warpgroup_kernel():
    """Scatter gmem bf16 → reg, ``warpgroup conversion``, gather fp32 → gmem."""
    atom_layout = _tcgen05_cast_atom_layout()
    m, k = _TCGEN05_M, _TCGEN05_K

    @T.function
    def kernel(A: T.Tensor((m, k), "bfloat16"), B: T.Tensor((m, k), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _lane = T.cuda.lane_id()
        tid_wg = T.cuda.thread_in_warpgroup()
        lane = T.cuda.lane_id()
        a_bf16 = T.alloc_tensor(
            ty_args=[T.Tensor((m, k), "bfloat16", scope="local", layout=atom_layout)]
        )
        a_fp32 = T.alloc_tensor(
            ty_args=[T.Tensor((m, k), "float32", scope="local", layout=atom_layout)]
        )
        reg_in = a_bf16.local(_TCGEN05_REGS_PER_THREAD)
        reg_out = a_fp32.local(_TCGEN05_REGS_PER_THREAD)
        for r in T.serial(_TCGEN05_REGS_PER_THREAD):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            reg_in[r] = A[row, col]
        dst_view = a_fp32.rearrange(
            "(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2
        ).view(4, 32, 8, 2, 2, 1)
        src_view = (
            a_bf16.rearrange("(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2)
            .view("uint16")
            .view(4, 32, 8, 2, 2, 1)
        )
        T.cuda.tile.apply(T.ptx["cvt.f32.bf16"], dst_view, src_view, scope="warpgroup")
        for r in T.serial(_TCGEN05_REGS_PER_THREAD):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            B[row, col] = reg_out[r]

    return kernel


def test_cast_tcgen05_atom_warpgroup_reg_dispatch_compiles():
    """Regression: tf32 A-cast ``warpgroup conversion(a_fp32, a_bf16)`` on tcgen05 atom compiles."""
    _sl_compile(_tcgen05_cast_warpgroup_kernel())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cast_tcgen05_atom_warpgroup_gpu():
    """GPU: bf16→fp32 cast on tcgen05 ``.16x256b`` atom must preserve per-slot layout."""
    m, k = _TCGEN05_M, _TCGEN05_K
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.tirx.compile(
            tvm.IRModule({"main": _tcgen05_cast_warpgroup_kernel()}),
            target=target,
            pipeline="tirx",
        )

    np.random.seed(0)
    a_np = (np.random.randn(m, k).astype("float32") * 4.0).astype("bfloat16")
    b_ref = a_np.astype("float32")

    def run_and_check():
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(np.zeros((m, k), dtype="float32"), dev)
        mod(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_ref, rtol=0, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


def _view_address_proof(layout, shape, permutation, expected):
    @T.function
    def views():
        T.device_entry()
        original = T.alloc_local(shape, "float32", layout=layout)
        canonical = original.permute(*permutation)
        T.evaluate(canonical.data)

    tensors = []

    def collect(node):
        if isinstance(node, tvm.ir.Bind) and isinstance(node.var.ty, tvm.tirx.TensorType):
            tensors.append(node.var)

    tvm_ffi.structural_walk(views.body, collect)
    view = tensors[-1]
    analyzer = tvm.sym.Analyzer()
    for coord in np.ndindex(shape):
        before = layout.apply(*coord, shape=shape)
        after = view.ty.layout.apply(*(coord[i] for i in permutation), shape=view.ty.shape)
        reference = {k: v for k, v in expected(*coord).items() if v != 0}
        for physical in (before, after):
            actual = {str(k): int(analyzer.simplify(v)) for k, v in physical.items()}
            assert {k: v for k, v in actual.items() if v != 0} == reference
    # Replica metadata also survives: a view cannot invent/remove a broadcast.
    assert list(layout.replica) == list(view.ty.layout.replica)


def test_cast_layout_partition_and_validation():
    """All seven original layout samples keep their physical map after a view.

    The old helper's layout-only accept/reject booleans are retired. Instruction
    applicability is checked by the scope/ownership rejection tests below.
    """
    from tvm.tirx.layout import Iter

    cases = [
        (TileLayout(S[(128, 8) : (1 @ tx, 1)]), (128, 8), (0, 1)),
        (TileLayout(S[(128, 8) : (1 @ tid_in_wg, 1)]), (128, 8), (0, 1)),
        (
            TileLayout.from_iters([Iter(4, 16, "m"), Iter(8, 2, tx), Iter(2, 1, "m")]),
            (4, 8, 2),
            (1, 0, 2),
        ),
        (
            TileLayout(S[(2, 8, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)]),
            (2, 8, 4, 2),
            (0, 1, 2, 3),
        ),
        (TileLayout(S[(64, 8) : (1, 1)]), (64, 8), (0, 1)),
        (TileLayout(S[(8, 8) : (1 @ tx, 1 @ laneid)]), (8, 8), (0, 1)),
        (
            TileLayout.from_iters([Iter(128, 1, tx), Iter(8, 1, "m")], [Iter(2, 1, laneid)]),
            (128, 8),
            (0, 1),
        ),
    ]
    references = [
        lambda thread, value: {"tx": thread, "m": value},
        lambda thread, value: {"tid_in_wg": thread, "m": value},
        lambda outer, thread, inner: {"tx": 2 * thread, "m": 16 * outer + inner},
        lambda warp, group, thread, value: {
            "warpid": 2 * warp,
            "laneid": 4 * group + thread,
            "m": value,
        },
        lambda row, col: {"m": row + col},
        lambda row, col: {"tx": row, "laneid": col},
        lambda thread, value: {"tx": thread, "m": value},
    ]
    for (layout, shape, permutation), expected in zip(cases, references):
        _view_address_proof(layout, shape, permutation, expected)


def test_cast_joint_decomposition_extents_order():
    """The caller's view keeps both gapped ownership axes in logical order."""
    layout = TileLayout(S[(2, 32, 4) : (2 @ warpid, 32 @ laneid, 1)])
    _view_address_proof(
        layout,
        (2, 32, 4),
        (0, 1, 2),
        lambda warp, lane, value: {"warpid": 2 * warp, "laneid": 32 * lane, "m": value},
    )


if __name__ == "__main__":
    tvm.testing.main()
