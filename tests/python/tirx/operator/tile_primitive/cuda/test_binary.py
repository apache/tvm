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
import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TileLayout, wg_local_layout


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
            tvm.cuda(0),  # dev
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
            tvm.cuda(0),  # dev
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
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("operands_type", ["region_region", "region_const", "const_region"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_shared(input, op_type, operands_type, dtype):
    # skip test
    if op_type in ["sub", "fdiv"] and operands_type == "const_region":
        return

    g_shape, st_a, st_b, st_res, ext_a, ext_b, ext_res, thread_cnt, dev = input
    g_layout = s_layout = TileLayout(S[g_shape])

    copy_slice = list(slice(None) for i in range(len(g_shape)))
    map_slice_a = list(slice(st_a[i], st_a[i] + ext_a[i]) for i in range(len(g_shape)))
    map_slice_b = list(slice(st_b[i], st_b[i] + ext_b[i]) for i in range(len(g_shape)))
    map_slice_res = list(slice(st_res[i], st_res[i] + ext_res[i]) for i in range(len(g_shape)))

    const = Tx.float16(3.0) if dtype == "float16" else Tx.float32(3.0)

    # fmt: off
    @Tx.prim_func
    def binary_op_region_region(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([thread_cnt])

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)
                B_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                Tx.copy(B_smem[tuple(copy_slice)], B[tuple(copy_slice)])
                if op_type == "add":
                    Tx.add(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "sub":
                    Tx.sub(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "mul":
                    Tx.mul(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                elif op_type == "fdiv":
                    Tx.fdiv(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], B_smem[tuple(map_slice_b)])  # noqa: E501
                Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])

    @Tx.prim_func
    def binary_op_const_region_or_region_const(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        _B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([thread_cnt])

            with Tx.cta():
                A_smem = Tx.alloc_buffer(g_shape, dtype, scope="shared", layout=s_layout)

                Tx.copy(A_smem[tuple(copy_slice)], A[tuple(copy_slice)])
                if op_type == "add":
                    if operands_type == "const_region":
                        Tx.add(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.add(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "sub":
                    if operands_type == "const_region":
                        Tx.sub(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.sub(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "mul":
                    if operands_type == "const_region":
                        Tx.mul(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.mul(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                elif op_type == "fdiv":
                    if operands_type == "const_region":
                        Tx.fdiv(A_smem[tuple(map_slice_res)], const, A_smem[tuple(map_slice_a)])
                    elif operands_type == "region_const":
                        Tx.fdiv(A_smem[tuple(map_slice_res)], A_smem[tuple(map_slice_a)], const)
                Tx.copy(A[tuple(copy_slice)], A_smem[tuple(copy_slice)])
    # fmt: on

    def get_prim_func(operands_type):
        if operands_type == "region_region":
            return binary_op_region_region
        elif operands_type in ["const_region", "region_const"]:
            return binary_op_const_region_or_region_const
        raise ValueError(f"operands_type={operands_type} is not supported")

    def get_ref(A_np, B_np):
        A_ref = A_np.copy()
        if op_type == "add":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] + B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] + 3.0
        elif op_type == "sub":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] - B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] - 3.0
        elif op_type == "mul":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] * B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] * 3.0
        elif op_type == "fdiv":
            if operands_type == "region_region":
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] / B_np[tuple(map_slice_b)]
            elif operands_type in ["const_region", "region_const"]:
                A_ref[tuple(map_slice_res)] = A_np[tuple(map_slice_a)] / 3.0

        return A_ref

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func(operands_type)})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B)

        A_ref = get_ref(A_np, B_np)
        atol = 1e-3
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=atol)


@pytest.mark.parametrize("op_type", ["sub", "fdiv"])
def test_binary_non_commutative_const_lhs_rejected(op_type):
    dtype = "float16"
    shape = (16, 16)
    layout = TileLayout(S[shape])
    const = Tx.float16(3.0)

    with pytest.raises(Exception):

        @Tx.prim_func
        def bad_kernel() -> None:
            with Tx.kernel():
                _bx = Tx.cta_id([1])
                _tid = Tx.thread_id([64])
                with Tx.cta():
                    A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=layout)
                    if op_type == "sub":
                        Tx.sub(A_smem, const, A_smem)
                    elif op_type == "fdiv":
                        Tx.fdiv(A_smem, const, A_smem)

        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": bad_kernel})
            tvm.compile(mod, target=target, tir_pipeline="tirx")


@pytest.mark.parametrize("exec_scope", ["warp", "warpgroup"])
@pytest.mark.parametrize("op_type", ["add", "mul"])
def test_binary_op_shared_subcta_scope(exec_scope, op_type):
    """Test binary ops in warp/warpgroup scope with shared memory."""
    dtype = "float16"
    n_warps = 4 if exec_scope == "warpgroup" else 1
    g_shape = (n_warps * 32, 8)
    dev = tvm.cuda(0)
    tx_op = {"add": Tx.add, "mul": Tx.mul}[op_type]

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=TileLayout(S[g_shape]))
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=TileLayout(S[g_shape]))
        with Tx.kernel():
            warp_id = Tx.warp_id([(256) // 32])
            wg_id = Tx.warpgroup_id([(256) // 128])
            _bx = Tx.cta_id([1])
            _tid = Tx.thread_id([256])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(
                    g_shape, dtype, scope="shared", layout=TileLayout(S[g_shape])
                )
                B_smem = Tx.alloc_buffer(
                    g_shape, dtype, scope="shared", layout=TileLayout(S[g_shape])
                )
                Tx.copy(A_smem, A)
                Tx.copy(B_smem, B)
                if exec_scope == "warp":
                    if Tx.filter(warp_id, 5, 6):
                        with Tx.warp():
                            tx_op(A_smem, A_smem, B_smem)
                elif exec_scope == "warpgroup":
                    if Tx.filter(wg_id, 1, 2):
                        with Tx.warpgroup():
                            tx_op(A_smem, A_smem, B_smem)
                Tx.cuda.cta_sync()
                Tx.copy(A, A_smem)

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(dtype)
        B_np = np.random.rand(*g_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        np_op = {"add": np.add, "mul": np.multiply}[op_type]
        A_ref = np_op(A_np, B_np).astype(dtype)
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)


@pytest.mark.parametrize("exec_scope", ["cta", "warpgroup", "warp"])
@pytest.mark.parametrize("rhs_kind", ["region", "broadcast", "const"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
def test_binary_op_local_subcta_trivial(exec_scope, rhs_kind, op_type):
    dtype = "float16"
    m, n = 4, 8
    n_threads = 256 if exec_scope == "cta" else (128 if exec_scope == "warpgroup" else 32)
    # in this test, use warp3/warpgroup1 to test
    thr_str = 0 if exec_scope == "cta" else (128 if exec_scope == "warpgroup" else 32 * 3)
    a_shape = (n_threads, m, n)
    b_shape = (n_threads, m, n if rhs_kind == "region" else 1)
    c_shape = a_shape
    const = Tx.float16(1.25)
    dev = tvm.cuda(0)
    tx_op = {"add": Tx.add, "sub": Tx.sub, "mul": Tx.mul, "fdiv": Tx.fdiv}[op_type]
    tid_in_scope_fn = {"cta": Tx.thread_id, "warpgroup": Tx.thread_id_in_wg, "warp": Tx.lane_id}[
        exec_scope
    ]

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))
        C = Tx.match_buffer(C_ptr, c_shape, dtype, layout=TileLayout(S[c_shape]))

        with Tx.kernel():
            wg_id = Tx.warpgroup_id([(256) // 128])
            warp_id = Tx.warp_id([(256) // 32])
            _bx = Tx.cta_id([1])
            _tid = Tx.thread_id([256])
            tid_in_scope = tid_in_scope_fn([n_threads])

            with Tx.cta():
                b_n = Tx.meta_var(n if rhs_kind == "region" else 1)
                A_local = Tx.alloc_buffer(
                    (m, n), dtype, scope="local", layout=TileLayout(S[(m, n)])
                )
                C_local = Tx.alloc_buffer(
                    (m, n), dtype, scope="local", layout=TileLayout(S[(m, n)])
                )
                B_local = Tx.alloc_buffer(
                    (m, b_n), dtype, scope="local", layout=TileLayout(S[(m, b_n)])
                )

                if Tx.filter(_tid, thr_str, thr_str + n_threads):
                    with Tx.thread():
                        for i in Tx.serial(m):
                            for j in Tx.serial(n):
                                A_local[i, j] = A[tid_in_scope, i, j]
                        if rhs_kind != "const":
                            for i in Tx.serial(m):
                                for j in Tx.serial(b_n):
                                    B_local[i, j] = B[tid_in_scope, i, j]
                # Tx.cuda.cta_sync()

                if exec_scope == "cta":
                    with Tx.cta():
                        if rhs_kind == "const":
                            tx_op(C_local, A_local, const)
                        else:
                            tx_op(C_local, A_local, B_local)
                elif exec_scope == "warpgroup":
                    if Tx.filter(wg_id, 1, 2):
                        with Tx.warpgroup():
                            if rhs_kind == "const":
                                tx_op(C_local, A_local, const)
                            else:
                                tx_op(C_local, A_local, B_local)
                else:
                    if Tx.filter(warp_id, 3, 4):
                        with Tx.warp():
                            if rhs_kind == "const":
                                tx_op(C_local, A_local, const)
                            else:
                                tx_op(C_local, A_local, B_local)
                # Tx.cuda.cta_sync()

                if Tx.filter(_tid, thr_str, thr_str + n_threads):
                    with Tx.thread():
                        for i in Tx.serial(m):
                            for j in Tx.serial(n):
                                C[tid_in_scope, i, j] = C_local[i, j]

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        C_np = np.zeros(c_shape, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        C = tvm.runtime.tensor(C_np, dev)

        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B, C)

        np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[op_type]
        if rhs_kind == "region":
            C_ref = np_op(A_np, B_np)
        elif rhs_kind == "broadcast":
            C_ref = np_op(A_np, np.repeat(B_np, n, axis=2))
        else:
            C_ref = np_op(A_np, const.value)
        atol = 1e-2 if op_type == "fdiv" else 1e-3
        tvm.testing.assert_allclose(C_ref, C.numpy(), atol=atol)


@pytest.mark.parametrize(
    "input",
    [
        ######### basic test #########
        (
            (64, 32),  # a_shape
            (64, 32),  # b_shape
            (64, 32),  # res_shape
            64,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
        ######### broadcast test #########
        (
            (16, 5, 4),  # a_shape
            (16, 1, 4),  # b_shape
            (16, 5, 4),  # res_shape
            16,  # thread_cnt
            tvm.cuda(0),  # dev
        ),
    ],
)
@pytest.mark.parametrize("storage_scope", ["shared", "local"])
@pytest.mark.parametrize("exec_scope", ["cta", "thread"])
@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "fdiv"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_binary_op_vectorized(input, storage_scope, exec_scope, op_type, dtype):
    a_shape, b_shape, res_shape, thread_cnt, dev = input
    tx_op = {"add": Tx.add, "sub": Tx.sub, "mul": Tx.mul, "fdiv": Tx.fdiv}[op_type]

    # fmt: off
    @Tx.prim_func
    def test_binary_cta(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            tx = Tx.thread_id([thread_cnt])
            with Tx.cta():
                if storage_scope == "shared":
                    A_smem = Tx.alloc_buffer(
                        a_shape, dtype, scope="shared", layout=TileLayout(S[a_shape])
                    )
                    B_smem = Tx.alloc_buffer(
                        b_shape, dtype, scope="shared", layout=TileLayout(S[b_shape])
                    )
                    Tx.copy(A_smem, A)
                    Tx.copy(B_smem, B)
                    tx_op(A_smem, A_smem, B_smem)
                    Tx.copy(A, A_smem)
            with Tx.thread():
                if storage_scope == "local":
                    A_local = Tx.alloc_buffer(
                        a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                    )
                    B_local = Tx.alloc_buffer(
                        b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                    )
                    Tx.copy(A_local, A[tx])
                    Tx.copy(B_local, B[tx])
                    with Tx.cta():
                        tx_op(A_local, A_local, B_local)
                    Tx.copy(A[tx], A_local)

    @Tx.prim_func
    def test_binary_thread(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            tx = Tx.thread_id([thread_cnt])

            with Tx.thread():
                if storage_scope == "shared":
                    A_smem = Tx.alloc_buffer(
                        a_shape, dtype, scope="shared", layout=TileLayout(S[a_shape])
                    )
                    B_smem = Tx.alloc_buffer(
                        b_shape, dtype, scope="shared", layout=TileLayout(S[b_shape])
                    )
                    Tx.copy(A_smem, A)
                    Tx.copy(B_smem, B)
                    tx_op(A_smem, A_smem, B_smem)
                    Tx.copy(A, A_smem)
                elif storage_scope == "local":
                    A_local = Tx.alloc_buffer(
                        a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                    )
                    B_local = Tx.alloc_buffer(
                        b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                    )
                    Tx.copy(A_local, A[tx])
                    Tx.copy(B_local, B[tx])
                    tx_op(A_local, A_local, B_local)
                    Tx.copy(A[tx], A_local)
    # fmt: on

    def get_prim_func():
        if exec_scope == "cta":
            return test_binary_cta
        elif exec_scope == "thread":
            return test_binary_thread
        else:
            raise ValueError(f"exec_scope={exec_scope} is not supported")

    target = tvm.target.Target("cuda")
    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": get_prim_func()})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        print(f"compiled source code: {mod.mod.imports[0].inspect_source()}")
        mod(A, B)

        np_op = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "fdiv": np.divide}[op_type]
        A_ref = np_op(A_np, B_np)
        atol = 1e-2 if op_type == "fdiv" else 1e-3
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=atol)


@pytest.mark.parametrize("op_type", ["add", "sub", "mul"])
def test_binary_op_packed_f32x2_auto_dispatch(op_type):
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits:
        pytest.skip(f"cannot parse target arch: {arch}")
    sm_version = int(sm_digits)
    if sm_version < 100:
        pytest.skip(f"packed_f32x2 auto-dispatch requires sm_100+, got {arch}")

    a_shape, b_shape = (64, 32), (64, 32)
    dtype = "float32"
    dev = tvm.cuda(0)

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, a_shape, dtype, layout=TileLayout(S[a_shape]))
        B = Tx.match_buffer(B_ptr, b_shape, dtype, layout=TileLayout(S[b_shape]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            tx = Tx.thread_id([64])
            with Tx.thread():
                A_local = Tx.alloc_buffer(
                    a_shape[1:], dtype, scope="local", layout=TileLayout(S[a_shape[1:]])
                )
                B_local = Tx.alloc_buffer(
                    b_shape[1:], dtype, scope="local", layout=TileLayout(S[b_shape[1:]])
                )
                Tx.copy(A_local, A[tx])
                Tx.copy(B_local, B[tx])
                if op_type == "add":
                    Tx.add(A_local, A_local, B_local)
                elif op_type == "sub":
                    Tx.sub(A_local, A_local, B_local)
                elif op_type == "mul":
                    Tx.mul(A_local, A_local, B_local)
                Tx.copy(A[tx], A_local)

    with target:
        np.random.seed(0)
        A_np = np.random.rand(*a_shape).astype(dtype)
        B_np = np.random.rand(*b_shape).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        ptx_pat = {
            "add": r"add\.[a-z]+\.ftz\.f32x2",
            "sub": r"sub\.[a-z]+\.ftz\.f32x2",
            "mul": r"mul\.[a-z]+\.ftz\.f32x2",
        }[op_type]
        builtin_pat = {
            "add": r"tvm_builtin_ptx_add_packed_",
            "sub": r"tvm_builtin_ptx_sub_packed_",
            "mul": r"tvm_builtin_ptx_mul_packed_",
        }[op_type]
        assert re.search(ptx_pat, src) or re.search(builtin_pat, src), src
        mod(A, B)

        if op_type == "add":
            A_ref = A_np + B_np
        elif op_type == "sub":
            A_ref = A_np - B_np
        elif op_type == "mul":
            A_ref = A_np * B_np
        tvm.testing.assert_allclose(A_ref, A.numpy(), atol=1e-3)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_binary_op_warpgroup_wg_local_layout(op_name):
    dtype = "float32"
    rows, cols = 128, 16
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        B = Tx.match_buffer(B_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        C = Tx.match_buffer(C_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([1])
            tid = Tx.thread_id_in_wg([rows])

            lhs = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
            rhs = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
            out = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))

            with Tx.thread():
                lhs_row = lhs.local(cols)
                rhs_row = rhs.local(cols)
                out_row = out.local(cols)
                for i in Tx.serial(cols):
                    lhs_row[i] = A[tid, i]
                    rhs_row[i] = B[tid, i]
                    out_row[i] = Tx.float32(0)

            with Tx.warpgroup():
                if op_name == "add":
                    Tx.add(out, lhs, rhs)
                elif op_name == "sub":
                    Tx.sub(out, lhs, rhs)
                elif op_name == "mul":
                    Tx.mul(out, lhs, rhs)

            with Tx.thread():
                out_row = out.local(cols)
                for i in Tx.serial(cols):
                    C[tid, i] = out_row[i]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.random.rand(rows, cols).astype(dtype)
        C_np = np.zeros((rows, cols), dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        C = tvm.runtime.tensor(C_np, dev)

        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B, C)

        if op_name == "add":
            C_ref = A_np + B_np
        elif op_name == "sub":
            C_ref = A_np - B_np
        else:
            C_ref = A_np * B_np
        tvm.testing.assert_allclose(C_ref, C.numpy(), atol=1e-5)


@pytest.mark.parametrize("op_name,ptx_op", [("add", "add"), ("sub", "sub"), ("mul", "mul")])
def test_binary_op_warpgroup_wg_local_emits_packed_f32x2(op_name, ptx_op):
    """Warpgroup-scope binary on a wg-local fp32 view must lower to packed
    f32x2 PTX on SM100+, mirroring the thread-scope packed dispatch.

    Regression test for the fa4 perf path: rescale-style ``Tx.{add,sub,mul}``
    calls in warpgroup scope used to fall through to scalar codegen because
    ``_emit_binary_local_view`` only emitted ``op_func(...)`` per element.
    """
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits or int(sm_digits) < 100:
        pytest.skip(f"packed_f32x2 wg-local path requires sm_100+, got {arch}")

    dtype = "float32"
    rows, cols = 128, 16

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        B = Tx.match_buffer(B_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        C = Tx.match_buffer(C_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            _wg_id = Tx.warpgroup_id([1])
            tid = Tx.thread_id_in_wg([rows])

            lhs = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
            rhs = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
            out = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))

            with Tx.thread():
                lhs_row = lhs.local(cols)
                rhs_row = rhs.local(cols)
                out_row = out.local(cols)
                for i in Tx.serial(cols):
                    lhs_row[i] = A[tid, i]
                    rhs_row[i] = B[tid, i]
                    out_row[i] = Tx.float32(0)

            with Tx.warpgroup():
                if op_name == "add":
                    Tx.add(out, lhs, rhs)
                elif op_name == "sub":
                    Tx.sub(out, lhs, rhs)
                else:
                    Tx.mul(out, lhs, rhs)

            with Tx.thread():
                out_row = out.local(cols)
                for i in Tx.serial(cols):
                    C[tid, i] = out_row[i]

    with target:
        mod = tvm.IRModule({"main": test_func})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    # Codegen must use the packed f32x2 path, not scalar fallback.
    assert re.search(rf"{ptx_op}\.[a-z]+\.ftz\.f32x2", src) or re.search(
        rf"tvm_builtin_ptx_{ptx_op}_packed_[a-z]+_f32x2", src
    ), f"expected packed f32x2 PTX for op={op_name}, source preview:\n{src[:2000]}"


def test_fma_warpgroup_wg_local_emits_packed_f32x2():
    """Same regression coverage as the binary case but for ``Tx.fma``."""
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        pytest.skip(f"unknown target arch: {arch}")
    sm_digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    if not sm_digits or int(sm_digits) < 100:
        pytest.skip(f"packed_f32x2 wg-local path requires sm_100+, got {arch}")

    dtype = "float32"
    rows, cols = 128, 16

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        C = Tx.match_buffer(C_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))

        with Tx.kernel():
            _bx = Tx.cta_id([1])
            _wg_id = Tx.warpgroup_id([1])
            tid = Tx.thread_id_in_wg([rows])

            buf = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))

            with Tx.thread():
                buf_row = buf.local(cols)
                for i in Tx.serial(cols):
                    buf_row[i] = A[tid, i]

            with Tx.warpgroup():
                Tx.fma(buf, buf, Tx.float32(2.0), Tx.float32(0.5))

            with Tx.thread():
                buf_row = buf.local(cols)
                for i in Tx.serial(cols):
                    C[tid, i] = buf_row[i]

    with target:
        mod = tvm.IRModule({"main": test_func})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert re.search(r"fma\.[a-z]+\.ftz\.f32x2", src) or re.search(
        r"tvm_builtin_ptx_fma_packed_[a-z]+_f32x2", src
    ), f"expected packed f32x2 fma PTX, source preview:\n{src[:2000]}"


if __name__ == "__main__":
    tvm.testing.main()
