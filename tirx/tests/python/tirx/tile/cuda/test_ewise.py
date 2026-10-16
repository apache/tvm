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

"""Caller-defined scalar compositions and explicit memory staging."""

import math
import re

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.axis import laneid
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


def _scalar(operation, src_dtype, dst_dtype, with_bias=False):
    source, destination = tvm_tirx.ir.PrimType(src_dtype), tvm_tirx.ir.PrimType(dst_dtype)

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
    with tvm_tirx.target.Target(target):
        module = tvm_tirx.tirx.compile(kernel, target=target, pipeline="tirx")

    def check():
        args = [tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()) for a in arrays]
        module(*args)
        tvm_tirx.testing.assert_allclose(args[output].numpy(), expected, atol=atol, rtol=rtol)

    tvm_tirx.testing.run_with_gpu_lock(check)
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

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": test_unary})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

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
            dev = tvm_tirx.cuda(device_id)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            B = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A, B)
            tvm_tirx.testing.assert_allclose(B_ref, B.numpy(), atol=1e-2, rtol=1e-2)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


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

    target = tvm_tirx.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(src_dtype)
        bias_np = np.random.rand(*g_shape_bias).astype(src_dtype)
        B_np = np.zeros(g_shape_b, dtype=dst_dtype)
        mod = tvm_tirx.IRModule({"main": test_unary_with_bias})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        B_ref = get_ref(A_np, bias_np)
        atol = 1e-3 if src_dtype == dst_dtype else 2e-2

        def run_and_check():
            dev = tvm_tirx.cuda(device_id)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            bias = tvm_tirx.runtime.tensor(bias_np, dev)
            B = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A, B, bias)
            tvm_tirx.testing.assert_allclose(B_ref, B.numpy(), atol=atol)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


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

    target = tvm_tirx.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.abs(np.random.rand(*shape).astype(dtype)) + 0.1
        mod = tvm_tirx.IRModule({"main": kernel})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
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
            dev = tvm_tirx.cuda(0)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            mod(A)
            tvm_tirx.testing.assert_allclose(A_ref, A.numpy(), atol=1e-2, rtol=1e-2)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


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


if __name__ == "__main__":
    tvm_tirx.testing.main()
