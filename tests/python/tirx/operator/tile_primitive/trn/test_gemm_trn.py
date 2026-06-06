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
import pytest

import tvm
import tvm.testing
from tvm.ir import assert_structural_equal as _assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import F, P, S, TileLayout
from tvm.tirx.stmt_functor import ir_transform

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def _strip_exec_scope_stmt(stmt):
    def _postorder(node):
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "tirx.device_entry":
            return node.body
        return node

    return ir_transform(
        stmt,
        preorder=lambda _node: None,
        postorder=_postorder,
        only_enable=["tirx.AttrStmt"],
    )


def assert_structural_equal(lhs, rhs, *args, **kwargs):
    if isinstance(lhs, tvm.tirx.PrimFunc):
        lhs = lhs.with_body(_strip_exec_scope_stmt(lhs.body))
    if isinstance(rhs, tvm.tirx.PrimFunc):
        rhs = rhs.with_body(_strip_exec_scope_stmt(rhs.body))
    _assert_structural_equal(lhs, rhs, *args, **kwargs)


def test_simple_gemm():
    A_layout = TileLayout(S[(128, 128) : (1 @ F, 1 @ P)])
    B_layout = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)])

    C_layout = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((128, 128), "float32", scope="trn.psum", layout=C_layout)
        Tx.gemm(C_psum, A_sbuf, B_sbuf, C_psum)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 128), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 128), scope="trn.sbuf")
        C_psum = T.alloc_buffer((1, 128, 128), scope="trn.psum")
        for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(1, 1, 1):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[0, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, lhs_f_loop], B_sbuf[p_loop, rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_larger_gemm():
    A_layout = TileLayout(S[(2, 128, 4, 128) : (512 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(2, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((256, 512), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((256, 256), "float32", scope="trn.psum", layout=C_layout)
        Tx.gemm(C_psum, A_sbuf, B_sbuf, C_psum)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        C_psum = T.alloc_buffer((1, 128, 512), scope="trn.psum")
        for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 1, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 256, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[0, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, lhs_b_loop * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, reduction_b_loop * 256 + rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_in_a_loop():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_psum[256 * i : 256 * i + 256, :],
                )

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 2, 1, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 256, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_with_stride():
    A_layout = TileLayout(S[(4, 128, 128, 8) : (1024 @ F, 1 @ F, 1 @ P, 128 @ F)])
    B_layout = TileLayout(S[(128, 8, 2, 128) : (1 @ P, 512 @ F, 256 @ F, 2 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 512, 2), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((512, 2, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, :, k],
                    B_sbuf[:, k, :],
                    C_psum[256 * i : 256 * i + 256, :],
                )

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 4095), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 2, 1, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 256, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + reduction_b_loop * 256 + k * 128 + lhs_f_loop], B_sbuf[p_loop, reduction_b_loop * 1024 + k * 512 + rhs_f_loop * 2], True)  # noqa: E501
            # fmt: on

    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_swap_lhs_rhs():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_psum[256 * i : 256 * i + 256, :],
                )

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 2, 2, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[i, lhs_f_loop, rhs_b_loop * 256 + lhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_with_sbuf_output():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)])

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_sbuf[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_sbuf[256 * i : 256 * i + 256, :],
                )
    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        buffer = T.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        for i, k, lhs_b_loop, rhs_b_loop in T.grid(2, 2, 2, 2):
            for reduction_b_loop in range(4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                  for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                    for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                        T.nki.matmul(buffer[lhs_b_loop * 2 + rhs_b_loop, lhs_f_loop, rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], True)  # noqa: E501
            T.attr(0, "tensorized_nki_instruction", 1)
            for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                T.nki.tensor_copy(C_sbuf[lhs_f_loop, i * 512 + rhs_b_loop * 256 + lhs_b_loop * 128 + rhs_f_loop], buffer[lhs_b_loop * 2 + rhs_b_loop, lhs_f_loop, rhs_f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_different_shape():
    A_layout = TileLayout(S[(2, 4, 128, 8, 128) : (4096 @ F, 1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((2, 512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[1, 256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_psum[256 * i : 256 * i + 256, :],
                )

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 8192), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 2, 2, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[i, lhs_f_loop, rhs_b_loop * 256 + lhs_b_loop * 128 + rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop + 4096], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_too_large_f_size():
    A_layout = TileLayout(S[(256, 128) : (1 @ F, 1 @ P)])
    B_layout = TileLayout(S[(128, 1024) : (1 @ P, 1 @ F)])

    C_layout = TileLayout(S[(2, 128, 1024) : (1024 @ F, 1 @ P, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((256, 128), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((128, 1024), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((256, 1024), "float32", scope="trn.psum", layout=C_layout)
        Tx.gemm(C_psum, A_sbuf, B_sbuf, C_psum)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 256), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        C_psum = T.alloc_buffer((4, 128, 512), scope="trn.psum")
        for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 1):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 512, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[lhs_b_loop * 2 + rhs_b_loop, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, lhs_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, rhs_b_loop * 512 + rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_sbuf_output_with_workspace():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)])

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
        C_psum = T.alloc_buffer((1, 128, 512), "float32", scope="trn.psum", allocated_addr=(0, 0))
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_sbuf[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_sbuf[256 * i : 256 * i + 256, :],
                    workspace={"acc_psum": C_psum}
                )
    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        C_psum = T.alloc_buffer((1, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
        for i, k, lhs_b_loop, rhs_b_loop in T.grid(2, 2, 2, 2):
            for reduction_b_loop in range(4):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                  for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                    for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                        T.nki.matmul(C_psum[0, lhs_f_loop, rhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, i * 2048 + rhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + rhs_f_loop], True)  # noqa: E501
            T.attr(0, "tensorized_nki_instruction", 1)
            for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                T.nki.tensor_copy(C_sbuf[lhs_f_loop, i * 512 + rhs_b_loop * 256 + lhs_b_loop * 128 + rhs_f_loop], C_psum[0, lhs_f_loop, rhs_f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_pf_mismatch_fail():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(2, 128, 8, 128) : (128 @ F, 1 @ F, 256 @ F, 1 @ P)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((256, 1024), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[:, 512 * k : 512 * k + 512],
                    C_psum[256 * i : 256 * i + 256, :],
                )
        # fmt: on
    with pytest.raises(Exception):
        with target:
            mod = tvm.IRModule({"main": gemm})
            mod = tvm.tirx.transform.LowerTIRx()(mod)


def test_gemm_transpose_AB():
    A_layout = TileLayout(S[(8, 128, 4, 128) : (128 @ F, 1 @ P, 1024 @ F, 1 @ F)])
    B_layout = TileLayout(S[(2, 128, 8, 128) : (128 @ F, 1 @ F, 256 @ F, 1 @ P)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((1024, 512), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((256, 1024), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for i in range(2):
            for k in range(2):
                Tx.gemm(
                    C_psum[256 * i : 256 * i + 256, :],
                    A_sbuf[512 * k : 512 * k + 512, 256 * i : 256 * i + 256],
                    B_sbuf[:, 512 * k : 512 * k + 512],
                    C_psum[256 * i : 256 * i + 256, :],
                    transpose_A=True,
                    transpose_B=True,
                )

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(2, 2, 2, 1, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 256, annotations={"nki_dim":"rhs_F"}):
                    T.nki.matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + rhs_f_loop], True)  # noqa: E501

            #fmt: off
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_guard():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)])

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_sbuf = T.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Tx.gemm(
                        C_sbuf[0: 256 * i, 0: 128 * (j + 1)],
                        A_sbuf[0: 256 * i, 0: 512 * (k + 1)],
                        B_sbuf[0: 512 * (k + 1), 0: 128 * (j + 1)],
                        C_sbuf[0: 256 * i, 0: 128 * (j + 1)],
                    )
    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        acc_psum = T.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_sbuf = T.alloc_buffer((128, 1024), scope="trn.sbuf")
        for i, j, k, lhs_b_loop, rhs_b_loop in T.grid(2, 2, 2, 2, 2):
            for reduction_b_loop in range(8):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                  for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                    for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"rhs_F"}):
                        if reduction_b_loop - k * 4 < 4 and lhs_b_loop - j < 1 and 0 < i and reduction_b_loop - k * 4 < 4:  # noqa: E501
                            T.nki.matmul(acc_psum[lhs_b_loop * 2 + rhs_b_loop, lhs_f_loop, rhs_f_loop], B_sbuf[p_loop, reduction_b_loop * 256 + lhs_b_loop * 128 + lhs_f_loop], A_sbuf[p_loop, rhs_b_loop * 1024 + reduction_b_loop * 128 + rhs_f_loop], True)  # noqa: E501
            T.attr(0, "tensorized_nki_instruction", 1)
            for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for rhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"F"}):
                if 0 < i and lhs_b_loop - j < 1:
                    T.nki.tensor_copy(C_sbuf[lhs_f_loop, rhs_b_loop * 256 + lhs_b_loop * 128 + rhs_f_loop], acc_psum[lhs_b_loop * 2 + rhs_b_loop, lhs_f_loop, rhs_f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm_guard2():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 128 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)]).to_psum()

    # fmt: off
    @T.prim_func
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_buffer((512, 256), "float32", scope="trn.psum", layout=C_layout)
        for j in range(4):
            for i in range(2):
                for k in range(2):
                    Tx.gemm(
                        C_psum[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + (j+1) * 128],
                        B_sbuf[512 * k : 512 * k + (j+1) * 128, :],
                        C_psum[256 * i : 256 * i + 256, :],
                    )
    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        C_psum = T.alloc_buffer((2, 128, 512), scope="trn.psum")
        for j, i, k, lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(4, 2, 2, 2, 1, 4):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
              for lhs_f_loop in T.serial(0, 128, annotations={"nki_dim":"lhs_F"}):
                for rhs_f_loop in T.serial(0, 256, annotations={"nki_dim":"rhs_F"}):
                    if reduction_b_loop - j < 1 and reduction_b_loop - j < 1:
                        T.nki.matmul(C_psum[i, lhs_f_loop, lhs_b_loop * 256 + rhs_f_loop], A_sbuf[p_loop, i * 2048 + lhs_b_loop * 1024 + k * 512 + reduction_b_loop * 128 + lhs_f_loop], B_sbuf[p_loop, k * 1024 + reduction_b_loop * 256 + rhs_f_loop], True)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
