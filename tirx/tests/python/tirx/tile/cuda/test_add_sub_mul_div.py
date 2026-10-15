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

"""Native arithmetic instructions, broadcasts, and caller-composed half division."""

import math
import re
import struct

import numpy as np
import pytest

import tvm
import tvm.testing
from tirx.layout_fixtures import tcgen05_register_layout
from tvm.backend.cuda.axis import Axis as CUDAAxis
from tvm.backend.cuda.axis import laneid, warpid, wid_in_wg
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


@T.function(private=True)
def half_divide(a: T.float16, b: T.float16) -> T.float16:
    # PTX has no native half division. The caller supplies the scalar composite.
    return T.cast(T.cast(a, "float32") / T.cast(b, "float32"), "float16")


def _run(kernel, arrays, expected, output=0, atol=1e-3, target="cuda", opcode=None):
    with tvm.target.Target(target):
        mod = tvm.tirx.compile(kernel, target=target, pipeline="tirx")
    if opcode is not None:
        assert opcode in mod.mod.imports[0].inspect_source()

    def check():
        args = [tvm.runtime.tensor(a, tvm.cuda()) for a in arrays]
        mod(*args)
        tvm.testing.assert_allclose(args[output].numpy(), expected, atol=atol)

    tvm.testing.run_with_gpu_lock(check)
    return mod


def _reference(operation, a, b):
    return {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[operation](
        a, b
    )


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (32, 32),  # g_shape
            (0, 0),  # st_a
            (0, 0),  # st_b
            (0, 0),  # st_res
            (32, 32),  # extent_a
            (32, 32),  # extent_b
            (32, 32),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
        ######### offset test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (14, 1, 4),  # st_b
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (5, 6, 7),  # extent_b
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
        ######### broadcast test #########
        (
            (32, 8, 12),  # g_shape
            (10, 0, 3),  # st_a
            (14, 1, 4),  # st_b
            (20, 0, 2),  # st_res
            (5, 6, 7),  # extent_a
            (1, 6, 1),  # extent_b
            (5, 6, 7),  # extent_res
            64,  # thread_cnt
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("operands_type", ["region_region", "region_const", "const_region"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_shared(input, op_type, operands_type, dtype):
    g_shape, st_a, st_b, st_res, ext_a, ext_b, ext_res, thread_cnt, device_id = input
    size, count = math.prod(g_shape), math.prod(ext_a)
    layout = TileLayout(S[g_shape])

    @T.function
    def kernel(
        A: T.Tensor(g_shape, dtype, layout=layout), B: T.Tensor(g_shape, dtype, layout=layout)
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=thread_cnt))
        tid = T.cuda.thread_idx("x")
        AS = T.alloc_tensor(ty_args=[T.Tensor(g_shape, dtype, scope="shared", layout=layout)])
        BS = T.alloc_tensor(ty_args=[T.Tensor(g_shape, dtype, scope="shared", layout=layout)])
        af, bf, sf, tf = A.view(size), B.view(size), AS.view(size), BS.view(size)
        for step in T.serial(T.ceildiv(size, thread_cnt)):
            i = step * thread_cnt + tid
            if i < size:
                sf[i] = af[i]
                tf[i] = bf[i]
        T.cuda.cta_sync()
        a, b, out = (
            T.alloc_local((1,), dtype, layout="default"),
            T.alloc_local((1,), dtype, layout="default"),
            T.alloc_local((1,), dtype, layout="default"),
        )
        for step in T.serial(T.ceildiv(count, thread_cnt)):
            i = step * thread_cnt + tid
            if i < count:
                coord = T.meta_var(
                    tuple(i // math.prod(ext_a[k + 1 :]) % ext_a[k] for k in range(len(ext_a)))
                )
                ia = T.meta_var(tuple(st_a[k] + coord[k] for k in range(len(ext_a))))
                ib = T.meta_var(tuple(st_b[k] + coord[k] % ext_b[k] for k in range(len(ext_a))))
                dst_index = T.meta_var(tuple(st_res[k] + coord[k] for k in range(len(ext_a))))
                a[0] = AS[ia]
                if T.constexpr(operands_type == "region_region"):
                    b[0] = BS[ib]
                else:
                    b[0] = T.float16(3)
                if T.constexpr(operands_type == "const_region"):
                    lhs, rhs = b, a
                else:
                    lhs, rhs = a, b
                if T.constexpr(op_type == "fdiv"):
                    T.cuda.tile.ewise(half_divide, out, lhs, rhs)
                else:
                    T.cuda.tile.apply(
                        T.ptx[f"{op_type}.rn.f16"],
                        out.view("uint16"),
                        lhs.view("uint16"),
                        rhs.view("uint16"),
                    )
                AS[dst_index] = out[0]
        T.cuda.cta_sync()
        for step in T.serial(T.ceildiv(size, thread_cnt)):
            i = step * thread_cnt + tid
            if i < size:
                af[i] = sf[i]

    rng = np.random.RandomState(0)
    a, b = rng.rand(*g_shape).astype(dtype), rng.rand(*g_shape).astype(dtype)
    sa = tuple(slice(s, s + e) for s, e in zip(st_a, ext_a))
    sb = tuple(slice(s, s + e) for s, e in zip(st_b, ext_b))
    sr = tuple(slice(s, s + e) for s, e in zip(st_res, ext_res))
    lhs, rhs = a[sa], b[sb] if operands_type == "region_region" else np.float16(3)
    if operands_type == "const_region":
        lhs, rhs = rhs, lhs
    expected = a.copy()
    expected[sr] = _reference(op_type, lhs, rhs)
    _run(kernel, [a, b], expected, atol=1e-2 if op_type == "fdiv" else 1e-3)


@pytest.mark.parametrize("op_type", ["sub"])
def test_binary_non_commutative_const_lhs(op_type):
    # A constant left operand preserves subtraction order.
    test_binary_op_shared(
        ((16, 16), (0, 0), (0, 0), (0, 0), (16, 16), (16, 16), (16, 16), 64, 0),
        op_type,
        "const_region",
        "float16",
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("thread_scope", ["warp", "warpgroup"])
@pytest.mark.parametrize("op_type", ["add", "mul"])
def test_binary_op_shared_subcta_scope(thread_scope, op_type):
    n_warps = 4 if thread_scope == "warpgroup" else 1
    shape = (n_warps * 32, 8)
    size = math.prod(shape)
    # The old test executes warp 5 / warpgroup 1 inside a 256-thread block.
    group_shape = (n_warps, 32, 8, 1) if thread_scope == "warpgroup" else (32, 8, 1)
    reg_layout = (
        TileLayout(S[(n_warps, 32, 8, 1) : (1 @ wid_in_wg, 1 @ laneid, 1, 1)])
        if thread_scope == "warpgroup"
        else TileLayout(S[(32, 8, 1) : (1 @ laneid, 1, 1)])
    )

    @T.function
    def kernel(A: T.Tensor(shape, "float16"), B: T.Tensor(shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
        tid = T.cuda.thread_idx("x")
        wid = T.cuda.warp_id()
        wg = T.cuda.warpgroup_id()
        AS = T.alloc_shared(shape, "float16", layout="default")
        BS = T.alloc_shared(shape, "float16", layout="default")
        af, bf, sf, tf = A.view(size), B.view(size), AS.view(size), BS.view(size)
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                sf[q], tf[q] = af[q], bf[q]
        T.cuda.cta_sync()
        a = T.alloc_local(group_shape, "float16", layout=reg_layout)
        b = T.alloc_local(group_shape, "float16", layout=reg_layout)
        if T.constexpr(thread_scope == "warpgroup"):
            if wg == 1:
                T.cuda.tile.apply(
                    T.ptx["ld.shared.b16"],
                    a.view("uint16"),
                    AS.view(*group_shape).view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx["ld.shared.b16"],
                    b.view("uint16"),
                    BS.view(*group_shape).view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx[f"{op_type}.rn.f16"],
                    a.view("uint16"),
                    a.view("uint16"),
                    b.view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"],
                    AS.view(*group_shape).view("uint16"),
                    a.view("uint16"),
                    scope=thread_scope,
                )
        else:
            if wid == 5:
                T.cuda.tile.apply(
                    T.ptx["ld.shared.b16"],
                    a.view("uint16"),
                    AS.view(*group_shape).view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx["ld.shared.b16"],
                    b.view("uint16"),
                    BS.view(*group_shape).view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx[f"{op_type}.rn.f16"],
                    a.view("uint16"),
                    a.view("uint16"),
                    b.view("uint16"),
                    scope=thread_scope,
                )
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"],
                    AS.view(*group_shape).view("uint16"),
                    a.view("uint16"),
                    scope=thread_scope,
                )
        T.cuda.cta_sync()
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                af[q] = sf[q]

    rng = np.random.RandomState(0)
    a, b = rng.rand(*shape).astype("float16"), rng.rand(*shape).astype("float16")
    _run(kernel, [a, b], _reference(op_type, a, b))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("thread_scope", ["cta", "warpgroup", "warp"])
@pytest.mark.parametrize("rhs_kind", ["region", "broadcast", "const"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
def test_binary_op_local_subcta_trivial(thread_scope, rhs_kind, op_type):
    dtype = "float16"
    m, n = 4, 8
    n_threads = 256 if thread_scope == "cta" else (128 if thread_scope == "warpgroup" else 32)
    # in this test, use warp3/warpgroup1 to test
    thr_str = 0 if thread_scope == "cta" else (128 if thread_scope == "warpgroup" else 32 * 3)
    a_shape = (n_threads, m, n)
    b_n = n if rhs_kind == "region" else 1
    b_shape = (n_threads, m, b_n)
    c_shape = a_shape
    const = T.float16(1.25)

    def tx_op(dst, lhs, rhs):
        if op_type == "fdiv":
            return T.cuda.tile.ewise(half_divide, dst, lhs, rhs)
        return T.cuda.tile.apply(
            T.ptx[f"{op_type}.rn.f16"],
            dst.view("uint16").view(m, n, 1),
            lhs.view("uint16").view(m, n, 1),
            rhs.view("uint16").view(m, n, 1) if isinstance(rhs, tvm.ir.Var) else T.uint16(0x3D00),
        )

    tid_in_scope_fn = {
        "cta": T.cuda.linear_thread_id,
        "warpgroup": T.cuda.thread_in_warpgroup,
        "warp": T.cuda.lane_id,
    }[thread_scope]

    @T.function
    def kernel(
        A: T.Tensor(a_shape, dtype, layout=TileLayout(S[a_shape])),
        B: T.Tensor(b_shape, dtype, layout=TileLayout(S[b_shape])),
        C: T.Tensor(c_shape, dtype, layout=TileLayout(S[c_shape])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(256,)))
        wg_id = T.cuda.warpgroup_id()
        warp_id = T.cuda.warp_id()
        _bx = T.cuda.block_idx("x")
        _tid = T.cuda.thread_idx("x")
        tid_in_scope = tid_in_scope_fn()
        A_local = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), dtype, scope="local", layout=TileLayout(S[(m, n)]))]
        )
        C_local = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), dtype, scope="local", layout=TileLayout(S[(m, n)]))]
        )
        B_local = T.alloc_tensor(
            ty_args=[T.Tensor((m, b_n), dtype, scope="local", layout=TileLayout(S[(m, b_n)]))]
        )

        B_values = B_local.view(m, n, layout=TileLayout(S[(m, n) : (b_n, 1 if b_n == n else 0)]))
        if thr_str <= _tid and _tid < thr_str + n_threads:
            for i in T.serial(m):
                for j in T.serial(n):
                    A_local[i, j] = A[tid_in_scope, i, j]
            if T.constexpr(rhs_kind != "const"):
                for i in T.serial(m):
                    for j in T.serial(b_n):
                        B_local[i, j] = B[tid_in_scope, i, j]

        if T.constexpr(thread_scope == "cta"):
            if T.constexpr(rhs_kind == "const"):
                tx_op(C_local, A_local, const)
            else:
                tx_op(C_local, A_local, B_values)
        elif T.constexpr(thread_scope == "warpgroup"):
            if wg_id == 1:
                if T.constexpr(rhs_kind == "const"):
                    tx_op(C_local, A_local, const)
                else:
                    tx_op(C_local, A_local, B_values)
        else:
            if warp_id == 3:
                if T.constexpr(rhs_kind == "const"):
                    tx_op(C_local, A_local, const)
                else:
                    tx_op(C_local, A_local, B_values)
                # T.cuda.cta_sync()

        if thr_str <= _tid and _tid < thr_str + n_threads:
            for i in T.serial(m):
                for j in T.serial(n):
                    C[tid_in_scope, i, j] = C_local[i, j]

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        C_np = np.zeros(c_shape, dtype=dtype)
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[op_type]
        if rhs_kind == "region":
            C_ref = np_op(A_np, B_np)
        elif rhs_kind == "broadcast":
            C_ref = np_op(A_np, np.repeat(B_np, n, axis=2))
        else:
            C_ref = np_op(A_np, const.value)
        atol = 1e-2 if op_type == "fdiv" else 1e-3

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            C = tvm.runtime.tensor(C_np, dev)
            mod(A, B, C)
            tvm.testing.assert_allclose(C_ref, C.numpy(), atol=atol)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (64, 32),  # a_shape
            (64, 32),  # b_shape
            (64, 32),  # res_shape
            64,  # thread_cnt
            0,  # device_id
        ),
        ######### broadcast test #########
        (
            (32, 5, 4),  # a_shape
            (32, 1, 4),  # b_shape
            (32, 5, 4),  # res_shape
            32,  # thread_cnt (≥ warp size so sctx.intra at cta scope models cleanly)
            0,  # device_id
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("storage_scope", ["shared", "local"])
@pytest.mark.parametrize("thread_scope", ["cta", "thread"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_vectorized(input, storage_scope, thread_scope, op_type, dtype):
    a_shape, b_shape, res_shape, thread_cnt, device_id = input
    row_a, row_b = a_shape[1:], b_shape[1:]
    n, m = math.prod(row_a), math.prod(row_b)
    # Half pairs are adjacent, including in the m-axis broadcast.
    broadcast = TileLayout(S[row_a]) if row_a == row_b else TileLayout(S[(5, 4) : (0, 1)])
    a_strides = tuple(math.prod(row_a[k + 1 :]) for k in range(len(row_a)))
    b_strides = a_strides if row_a == row_b else (0, 1)
    cta_a_layout = TileLayout(
        S[(thread_cnt // 32, 32, *row_a) : (1 @ warpid, 1 @ laneid, *a_strides)]
    )
    cta_b_layout = TileLayout(
        S[(thread_cnt // 32, 32, *row_a) : (1 @ warpid, 1 @ laneid, *b_strides)]
    )

    @T.function
    def kernel(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=thread_cnt))
        tid = T.cuda.thread_idx("x")
        a = T.alloc_local(row_a, dtype, layout="default")
        b = T.alloc_local(row_b, dtype, layout="default")
        af, bf = A.view(thread_cnt, n), B.view(thread_cnt, m)
        al, bl = a.view(n), b.view(m)
        if T.constexpr(storage_scope == "shared"):
            AS = T.alloc_shared(a_shape, dtype, layout="default")
            BS = T.alloc_shared(b_shape, dtype, layout="default")
            sf, tf = AS.view(thread_cnt, n), BS.view(thread_cnt, m)
            for j in range(n):
                sf[tid, j] = af[tid, j]
            for j in range(m):
                tf[tid, j] = bf[tid, j]
            T.cuda.cta_sync()
            for j in range(n):
                al[j] = sf[tid, j]
            for j in range(m):
                bl[j] = tf[tid, j]
        else:
            for j in range(n):
                al[j] = af[tid, j]
            for j in range(m):
                bl[j] = bf[tid, j]
        bv = b.view(*row_a, layout=broadcast)
        if T.constexpr(thread_scope == "cta"):
            # Expose the implicit per-thread register arrays as an explicit
            # CTA view. These declarations share their original storage.
            av = a.view(thread_cnt // 32, 32, *row_a, layout=cta_a_layout)
            bv_cta = b.view(thread_cnt // 32, 32, *row_a, layout=cta_b_layout)
            if T.constexpr(op_type == "fdiv"):
                T.cuda.tile.ewise(half_divide, av, av, bv_cta, scope="cta")
            else:
                T.cuda.tile.apply(
                    T.ptx[f"{op_type}.rn.f16x2"],
                    av.view("uint32").view(thread_cnt // 32, 32, n // 2, 1),
                    av.view("uint32").view(thread_cnt // 32, 32, n // 2, 1),
                    bv_cta.view("uint32").view(thread_cnt // 32, 32, n // 2, 1),
                    scope="cta",
                )
        elif T.constexpr(op_type == "fdiv"):
            T.cuda.tile.ewise(half_divide, a, a, bv)
        else:
            T.cuda.tile.apply(
                T.ptx[f"{op_type}.rn.f16x2"],
                a.view("uint32").view(n // 2, 1),
                a.view("uint32").view(n // 2, 1),
                bv.view("uint32").view(n // 2, 1),
            )
        if T.constexpr(storage_scope == "shared"):
            for j in range(n):
                sf[tid, j] = al[j]
            T.cuda.cta_sync()
            for j in range(n):
                af[tid, j] = sf[tid, j]
        else:
            for j in range(n):
                af[tid, j] = al[j]

    rng = np.random.RandomState(0)
    a, b = rng.rand(*a_shape).astype(dtype), rng.rand(*b_shape).astype(dtype)
    _run(
        kernel,
        [a, b],
        _reference(op_type, a, b),
        atol=1e-2 if op_type == "fdiv" else 1e-3,
        opcode=None if op_type == "fdiv" else f"{op_type}.rn.f16x2",
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_type", ["add", "sub", "mul"])
def test_binary_op_packed_f32x2_explicit(op_type):
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits:
        pytest.skip(f"cannot parse target arch: {arch}")
    sm_version = int(sm_digits)
    if sm_version < 100:
        pytest.skip(f"packed_f32x2 requires sm_100+, got {arch}")

    a_shape, b_shape = (64, 32), (64, 32)
    dtype = "float32"

    @T.function
    def test_func(
        A: T.Tensor(a_shape, dtype, layout=TileLayout(S[a_shape])),
        B: T.Tensor(b_shape, dtype, layout=TileLayout(S[b_shape])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        A_local = T.alloc_tensor(
            ty_args=[T.Tensor(a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]]))]
        )
        B_local = T.alloc_tensor(
            ty_args=[T.Tensor(b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]]))]
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            A_local.view("uint32").view(-1, 4),
            A.sub[tx, :].view("uint32").view(-1, 4),
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            B_local.view("uint32").view(-1, 4),
            B.sub[tx, :].view("uint32").view(-1, 4),
        )
        if T.constexpr(op_type == "add"):
            T.cuda.tile.apply(
                T.ptx["add.rn.ftz.f32x2"],
                A_local.view("uint64").view(16, 1),
                A_local.view("uint64").view(16, 1),
                B_local.view("uint64").view(16, 1),
            )
        elif T.constexpr(op_type == "sub"):
            T.cuda.tile.apply(
                T.ptx["sub.rn.ftz.f32x2"],
                A_local.view("uint64").view(16, 1),
                A_local.view("uint64").view(16, 1),
                B_local.view("uint64").view(16, 1),
            )
        elif T.constexpr(op_type == "mul"):
            T.cuda.tile.apply(
                T.ptx["mul.rn.ftz.f32x2"],
                A_local.view("uint64").view(16, 1),
                A_local.view("uint64").view(16, 1),
                B_local.view("uint64").view(16, 1),
            )
        T.cuda.tile.apply(
            T.ptx["st.global.v4.b32"],
            A.sub[tx, :].view("uint32").view(-1, 4),
            A_local.view("uint32").view(-1, 4),
        )

    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        ptx_pat = {
            "add": r"add\.[a-z]+\.ftz\.f32x2",
            "sub": r"sub\.[a-z]+\.ftz\.f32x2",
            "mul": r"mul\.[a-z]+\.ftz\.f32x2",
        }[op_type]
        assert re.search(ptx_pat, src), src
        if op_type == "add":
            A_ref = A_np + B_np
        elif op_type == "sub":
            A_ref = A_np - B_np
        elif op_type == "mul":
            A_ref = A_np * B_np

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_binary_op_warpgroup_thread_row_layout(op_name):
    dtype = "float32"
    rows, cols = 128, 16
    target = tvm.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        B: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        C: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()

        lhs = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        rhs = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        out = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        lhs_row = lhs.view(cols, layout="default")
        rhs_row = rhs.view(cols, layout="default")
        out_row = out.view(cols, layout="default")
        for i in T.serial(cols):
            lhs_row[i] = A[tid, i]
            rhs_row[i] = B[tid, i]
            out_row[i] = T.float32(0)
        if T.constexpr(op_name == "add"):
            T.cuda.tile.apply(
                T.ptx["add.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        elif T.constexpr(op_name == "sub"):
            T.cuda.tile.apply(
                T.ptx["sub.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        elif T.constexpr(op_name == "mul"):
            T.cuda.tile.apply(
                T.ptx["mul.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        out_row_1 = out.view(cols, layout="default")
        for i in T.serial(cols):
            C[tid, i] = out_row_1[i]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.random.rand(rows, cols).astype(dtype)
        C_np = np.zeros((rows, cols), dtype=dtype)

        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        if op_name == "add":
            C_ref = A_np + B_np
        elif op_name == "sub":
            C_ref = A_np - B_np
        else:
            C_ref = A_np * B_np

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            C = tvm.runtime.tensor(C_np, dev)
            mod(A, B, C)
            tvm.testing.assert_allclose(C_ref, C.numpy(), atol=1e-5)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("op_name,ptx_op", [("add", "add"), ("sub", "sub"), ("mul", "mul")])
def test_binary_op_warpgroup_wg_local_emits_packed_f32x2(op_name, ptx_op):
    """The explicit packed instruction survives a warpgroup-local row view."""
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits or int(sm_digits) < 100:
        pytest.skip(f"packed_f32x2 wg-local path requires sm_100+, got {arch}")

    dtype = "float32"
    rows, cols = 128, 16

    @T.function
    def test_func(
        A: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        B: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        C: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()

        lhs = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        rhs = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        out = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        lhs_row = lhs.view(cols, layout="default")
        rhs_row = rhs.view(cols, layout="default")
        out_row = out.view(cols, layout="default")
        for i in T.serial(cols):
            lhs_row[i] = A[tid, i]
            rhs_row[i] = B[tid, i]
            out_row[i] = T.float32(0)
        if T.constexpr(op_name == "add"):
            T.cuda.tile.apply(
                T.ptx["add.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        elif T.constexpr(op_name == "sub"):
            T.cuda.tile.apply(
                T.ptx["sub.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        else:
            T.cuda.tile.apply(
                T.ptx["mul.rn.ftz.f32x2"],
                out.view("uint64").view(4, 32, cols // 2, 1),
                lhs.view("uint64").view(4, 32, cols // 2, 1),
                rhs.view("uint64").view(4, 32, cols // 2, 1),
                scope="warpgroup",
            )
        out_row_1 = out.view(cols, layout="default")
        for i in T.serial(cols):
            C[tid, i] = out_row_1[i]

    with target:
        mod = tvm.IRModule({"main": test_func})
        ex = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    # Codegen must use the packed f32x2 path, not scalar fallback.
    assert re.search(rf"{ptx_op}\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed f32x2 PTX for op={op_name}, source preview:\n{src[:2000]}"
    )


def test_fma_warpgroup_wg_local_emits_packed_f32x2():
    """The packed FMA consumes explicit splatted b64 coefficients."""
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits or int(sm_digits) < 100:
        pytest.skip(f"packed_f32x2 wg-local path requires sm_100+, got {arch}")

    dtype = "float32"
    rows, cols = 128, 16

    @T.function
    def test_func(
        A: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        C: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()

        buf = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        buf_row = buf.view(cols, layout="default")
        for i in T.serial(cols):
            buf_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["fma.rn.ftz.f32x2"],
            buf.view("uint64").view(4, 32, cols // 2, 1),
            buf.view("uint64").view(4, 32, cols // 2, 1),
            T.uint64(0x4000000040000000),
            T.uint64(0x3F0000003F000000),
            scope="warpgroup",
        )
        buf_row_1 = buf.view(cols, layout="default")
        for i in T.serial(cols):
            C[tid, i] = buf_row_1[i]

    with target:
        mod = tvm.IRModule({"main": test_func})
        ex = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert re.search(r"fma\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed f32x2 fma PTX, source preview:\n{src[:2000]}"
    )


def test_binary_add_f32_sm100_packed_f32x2_dispatch():
    """The selected opcode remains packed through code generation."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    @T.function
    def k(
        A: T.Tensor(shape, "float32", layout=lay), B: T.Tensor(shape, "float32", layout=lay)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ra = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rb = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            ra.view("uint32").view(-1, 4),
            A.sub[tx, :].view("uint32").view(-1, 4),
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            rb.view("uint32").view(-1, 4),
            B.sub[tx, :].view("uint32").view(-1, 4),
        )
        T.cuda.tile.apply(
            T.ptx["add.rn.ftz.f32x2"],
            ra.view("uint64").view(16, 1),
            ra.view("uint64").view(16, 1),
            rb.view("uint64").view(16, 1),
        )
        T.cuda.tile.apply(
            T.ptx["st.global.v4.b32"],
            A.sub[tx, :].view("uint32").view(-1, 4),
            ra.view("uint32").view(-1, 4),
        )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert re.search(r"add\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed add_f32x2; got:\n{src[:2000]}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_binary_maximum_reg():
    N = 128

    @T.function
    def relu_max(
        A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
        tid = T.cuda.thread_in_warpgroup()
        a_reg = T.alloc_local((1,), "float32", layout="default")
        b_reg = T.alloc_local((1,), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], a_reg[:], A[tid : tid + 1])
        T.cuda.tile.apply(T.ptx["ld.global.f32"], b_reg[:], B[tid : tid + 1])
        T.cuda.tile.apply(T.ptx["max.f32"], a_reg[:], a_reg[:], b_reg[:])
        T.cuda.tile.apply(T.ptx["st.global.f32"], C[tid : tid + 1], a_reg[:])

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": relu_max}), target=target, pipeline="tirx")
    np.random.seed(0)
    a = np.random.randn(N).astype("float32")
    b = np.random.randn(N).astype("float32")
    c = np.zeros(N, dtype="float32")

    def run_and_check():
        dev = tvm.cuda(0)
        a_t, b_t, c_t = (tvm.runtime.tensor(x, dev) for x in (a, b, c))
        mod["main"](a_t, b_t, c_t)
        np.testing.assert_allclose(c_t.numpy(), np.maximum(a, b), atol=0, rtol=0)

    tvm.testing.run_with_gpu_lock(run_and_check)


def test_binary_add_f16_scalar_fallback_dispatch():
    """The caller can request scalar half arithmetic even when packed is legal."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    @T.function
    def k(
        A: T.Tensor(shape, "float16", layout=lay), B: T.Tensor(shape, "float16", layout=lay)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ra = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float16", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rb = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float16", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            ra.view("uint32").view(-1, 4),
            A.sub[tx, :].view("uint32").view(-1, 4),
        )
        T.cuda.tile.apply(
            T.ptx["ld.global.v4.b32"],
            rb.view("uint32").view(-1, 4),
            B.sub[tx, :].view("uint32").view(-1, 4),
        )
        T.cuda.tile.apply(
            T.ptx["add.rn.f16"],
            ra.view("uint16").view(32, 1),
            ra.view("uint16").view(32, 1),
            rb.view("uint16").view(32, 1),
        )
        T.cuda.tile.apply(
            T.ptx["st.global.v4.b32"],
            A.sub[tx, :].view("uint32").view(-1, 4),
            ra.view("uint32").view(-1, 4),
        )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert "add.rn.f16" in src, f"expected scalar half add; got:\n{src[:2000]}"
    assert not re.search(r"add\.[a-z]+\.ftz\.f(32|16)x2", src), (
        f"unexpected packed f32x2/f16x2 add in scalar-fallback path; got:\n{src[:2000]}"
    )


def test_mul_tcgen05_16x256b_atom_warpgroup_dispatch():
    """Caller views expose the split lane/warp register atom as packed pairs."""
    rows, cols = 64, 64
    regs_per_thread = 32
    atom_layout = tcgen05_register_layout("16x256b", (rows, cols), "float32")

    @T.function
    def kernel(
        A: T.Tensor((128, regs_per_thread), "float32"),
        B: T.Tensor((128, regs_per_thread), "float32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _lane = T.cuda.lane_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[T.Tensor((rows, cols), "float32", scope="local", layout=atom_layout)]
        )
        dst = T.alloc_tensor(
            ty_args=[T.Tensor((rows, cols), "float32", scope="local", layout=atom_layout)]
        )
        src_local = src.view(regs_per_thread, layout="default")
        dst_local = dst.view(regs_per_thread, layout="default")
        for i in T.serial(regs_per_thread):
            src_local[i] = A[tid, i]
        src_pairs = src.rearrange(
            "(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2
        ).view("uint64")
        dst_pairs = dst.rearrange(
            "(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2
        ).view("uint64")
        T.cuda.tile.apply(
            T.ptx["mul.rn.ftz.f32x2"],
            dst_pairs,
            src_pairs,
            T.uint64(0x4040000040400000),
            scope="warpgroup",
        )
        for i in T.serial(regs_per_thread):
            B[tid, i] = dst_local[i]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": kernel}), target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert re.search(r"mul\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed mul_f32x2 for tcgen05 atom; got:\n{src[:2000]}"
    )

    rng = np.random.RandomState(0)
    values = rng.rand(128, regs_per_thread).astype("float32")
    result = np.zeros_like(values)
    _run(kernel, [values, result], values * 3, output=1, target=target)


def _splat_f32(value):
    """Prepare the two equal f32 fields of a native b64 packed operand."""
    if isinstance(value, tvm.ir.prim.FloatImm):
        bits = struct.unpack("I", struct.pack("f", value.value))[0]
        return T.uint64(bits) | (T.uint64(bits) << T.uint64(32))
    bits = T.cast(T.reinterpret(value, ty="uint32"), "uint64")
    return bits | (bits << T.uint64(32))


def _get_sm_version():
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        return 0
    digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    return int(digits) if digits else 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_mul_scalar_broadcast():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed mul requires sm_100+, got sm_{sm}")

    N = 16
    dtype = "float32"
    target = tvm.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
        Scale: T.Tensor((1,), dtype, layout=TileLayout(S[1])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        a_local = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        s_local = T.alloc_tensor(
            ty_args=[T.Tensor((1,), dtype, scope="local", layout=TileLayout(S[1]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], a_local.view(4, 4), A.view(4, 4))
        T.cuda.tile.apply(T.ptx["ld.global.f32"], s_local, Scale[0:1])
        T.cuda.tile.apply(
            T.ptx["mul.rn.ftz.f32x2"],
            a_local.view("uint64").view(8, 1),
            a_local.view("uint64").view(8, 1),
            _splat_f32(s_local[0]),
        )
        T.cuda.tile.apply(T.ptx["st.global.v4.f32"], A.view(4, 4), a_local.view(4, 4))

    with target:
        A_np = np.random.rand(N).astype(dtype)
        S_np = np.array([2.5], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            S_dev = tvm.runtime.tensor(S_np, dev)
            mod(A_dev, S_dev)
            expected = A_np * S_np[0]
            tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_add_rounding_mode():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed add with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm.target.Target("cuda")

    round_const = float(2**23 + 2**22)

    @T.function
    def test_func(A: T.Tensor((N,), dtype, layout=TileLayout(S[N]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], buf, A[0:N])
        T.cuda.tile.apply(
            T.ptx["add.rm.ftz.f32x2"],
            buf.view("uint64"),
            buf.view("uint64"),
            _splat_f32(T.float32(round_const)),
        )
        T.cuda.tile.apply(T.ptx["st.global.v2.f32"], A[0:N], buf)

    with target:
        A_np = np.array([1.3, 2.7], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        # Check that the PTX uses the rounding mode
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"add\.rm\.ftz\.f32x2", src), (
            f"Expected packed add with rm rounding in PTX:\n{src}"
        )

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            mod(A_dev)
            # Both exact sums lie above these integers; rm rounds down.
            expected = np.array([round_const + 1, round_const + 2], dtype="float32")
            np.testing.assert_array_equal(expected, A_dev.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_sub_tensor_tensor_rounding():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed sub with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
        B: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        a_buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        b_buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], a_buf, A[0:N])
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], b_buf, B[0:N])
        T.cuda.tile.apply(
            T.ptx["sub.rn.ftz.f32x2"],
            a_buf.view("uint64"),
            a_buf.view("uint64"),
            b_buf.view("uint64"),
        )
        T.cuda.tile.apply(T.ptx["st.global.v2.f32"], A[0:N], a_buf)

    with target:
        A_np = np.array([3.14, 2.71], dtype=dtype)
        B_np = np.array([1.41, 0.57], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"sub\.rn\.ftz\.f32x2", src), (
            f"Expected packed sub with rn rounding in PTX:\n{src}"
        )

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            B_dev = tvm.runtime.tensor(B_np, dev)
            mod(A_dev, B_dev)
            expected = A_np - B_np
            tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-6)

        tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
