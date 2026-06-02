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
from tvm.ir import assert_structural_equal as _assert_structural_equal
from tvm.script import tirx as Tx
from tvm.tirx.layout import F, P, S, TileLayout
from tvm.tirx.stmt_functor import ir_transform

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def _strip_exec_scope_stmt(stmt):
    def _postorder(node):
        if isinstance(node, tvm.tirx.ExecScopeStmt):
            return node.body
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "tirx.device_entry":
            return node.body
        return node

    return ir_transform(
        stmt,
        preorder=lambda _node: None,
        postorder=_postorder,
        only_enable=["tirx.ExecScopeStmt", "tirx.AttrStmt"],
    )


def assert_structural_equal(lhs, rhs, *args, **kwargs):
    if isinstance(lhs, tvm.tirx.PrimFunc):
        lhs = lhs.with_body(_strip_exec_scope_stmt(lhs.body))
    if isinstance(rhs, tvm.tirx.PrimFunc):
        rhs = rhs.with_body(_strip_exec_scope_stmt(rhs.body))
    _assert_structural_equal(lhs, rhs, *args, **kwargs)


def test_simple_copy():
    src_shape = [128, 512]
    src_layout = Tx.TileLayout(Tx.S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.copy(A_sbuf, A)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (128, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((65536,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.load(A_sbuf[p_loop, f_loop], A_1[p_loop * 512 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_copy_2():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 4, 128) : (512, 128, 1)])

    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 4, 128) : (4 @ F, 1 @ F, 1 @ P)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.copy(A_sbuf, A)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (128, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((65536,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 512):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 1, annotations={"nki_dim": "F"}):
                        Tx.nki.load(A_sbuf[p_loop, b_loop], A_1[b_loop * 128 + p_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop():
    src_shape = [512, 512]
    src_layout = Tx.TileLayout(Tx.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.copy(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (512, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((262144,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for i, b_loop in Tx.grid(4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.load(
                            A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop]
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop_2():
    src_shape = [512, 512]
    src_layout = Tx.TileLayout(Tx.S[(128, 2048) : (2048, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        A_sbuf_view = A_sbuf.view(128, 4, 512)
        A_view = A.view(128, 4, 512)
        for i in range(4):
            Tx.copy(A_sbuf_view[:, i, :], A_view[:, i, :])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (512, 512), layout=None)
        with Tx.thread():
            _A_flat = Tx.decl_buffer((262144,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            A_sbuf_view = Tx.decl_buffer(
                (128, 2048), data=A_sbuf.data, scope="trn.sbuf", layout=None
            )
            A_view = Tx.decl_buffer((262144,), data=A.data, layout=None)
            for i, b_loop in Tx.grid(4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.load(
                            A_sbuf_view[p_loop, i * 512 + f_loop],
                            A_view[p_loop * 2048 + i * 512 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod.show()
        assert_structural_equal(mod["main"], expected)


def test_copy_transpose():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(2048, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.copy(B_sbuf, A_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for b_loop in range(16):
                for extend_b_loop in range(1):
                    Tx.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                        for lhs_f_loop in Tx.serial(128, annotations={"nki_dim": "lhs_F"}):
                            for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "rhs_F"}):
                                Tx.nki.matmul(acc_psum[b_loop % 8, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, b_loop * 128 + lhs_f_loop], identity[p_loop, rhs_f_loop], Tx.bool(True))  # noqa: E501
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(B_sbuf[p_loop, f_loop * 16 + b_loop], acc_psum[b_loop % 8, p_loop, f_loop])  # noqa: E501
                # fmt: on

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_transpose_2():
    src_shape = [65536]
    src_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
    dst_shape = [4, 65536]
    dst_layout = TileLayout(S[(4, 128, 128, 4) : (4 @ F, 16 @ F, 1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.copy(B_sbuf[i, :], A_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for i in range(4):
                for b_loop in range(4):
                    for extend_b_loop in range(1):
                        Tx.attr(0, "tensorized_nki_instruction", 1)
                        for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                            for lhs_f_loop in Tx.serial(128, annotations={"nki_dim": "lhs_F"}):
                                for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "rhs_F"}):
                                    Tx.nki.matmul(acc_psum[b_loop, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, lhs_f_loop * 4 + b_loop], identity[p_loop, rhs_f_loop], Tx.bool(True))  # noqa: E501
                    Tx.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                            Tx.nki.tensor_copy(B_sbuf[p_loop, f_loop * 16 + i * 4 + b_loop], acc_psum[b_loop, p_loop, f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_f():
    src_shape = [512, 64]
    src_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 16 @ F, 4 @ F, 1 @ F)])
    dst_shape = [512, 64]
    dst_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 4 @ F, 16 @ F, 1 @ F)])

    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.copy(B_sbuf, A_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 256), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 256), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 64):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 4, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(
                            B_sbuf[
                                p_loop,
                                b_loop // 16 * 64 + b_loop % 4 * 16 + b_loop % 16 // 4 * 4 + f_loop,
                            ],
                            A_sbuf[p_loop, b_loop * 4 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape():
    src_shape = [512, 64]
    src_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 16 @ F, 4 @ F, 1 @ F)])
    dst_shape = [4, 128, 4]
    dst_layout = TileLayout(S[(4, 128, 4) : (4 @ F, 1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        B_sbuf_view = B_sbuf.view(512, 4)
        Tx.copy(B_sbuf_view, A_sbuf[:, 0:4])

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 256), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 16), scope="trn.sbuf")
            _B_sbuf_view = Tx.decl_buffer(
                (128, 16), data=B_sbuf.data, scope="trn.sbuf", layout=None
            )
            for b_loop in Tx.serial(0, 4):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 4, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(
                            B_sbuf[p_loop, b_loop * 4 + f_loop],
                            A_sbuf[p_loop, b_loop * 64 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_irregular_shape():
    src_shape = [128, 10000]
    src_layout = TileLayout(S[(128, 10000) : (10000, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.copy(A[:, i * 512 : i * 512 + 512], A_sbuf)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (128, 10000), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((1280000,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for i, b_loop in Tx.grid(4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.store(A_1[p_loop * 10000 + i * 512 + f_loop], A_sbuf[p_loop, f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape_dim():
    src_shape = [32, 128, 512]
    src_layout = TileLayout(S[(32, 128, 512) : (128 * 512, 128, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(32):
            Tx.copy(A_sbuf, A[i, :, :])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (32, 128, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((2097152,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for i, b_loop in Tx.grid(32, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.load(A_sbuf[p_loop, f_loop], A_1[i * 65536 + p_loop * 128 + f_loop])
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_offset():
    src_shape = [256, 512]
    src_layout = TileLayout(S[(256, 512) : (512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(2):
            Tx.copy(A_sbuf[i * 256 : i * 256 + 256, :], A)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (256, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((131072,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for i, b_loop in Tx.grid(2, 2):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.load(
                            A_sbuf[p_loop, i * 1024 + b_loop * 512 + f_loop],
                            A_1[b_loop * 65536 + p_loop * 512 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_large_dma_copy():
    src_shape = [512, 4096]
    src_layout = Tx.TileLayout(Tx.S[(4, 128, 4096) : (4096 * 128, 4096, 1)])
    dst_shape = [512, 4096]
    dst_layout = TileLayout(S[(4, 128, 4096) : (4096 @ F, 1 @ P, 1 @ F)])

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.copy(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (512, 4096), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((2097152,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            for i, b_loop in Tx.grid(4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 4096, annotations={"nki_dim": "F"}):
                        Tx.nki.load(
                            A_sbuf[p_loop, i * 4096 + f_loop],
                            A_1[i * 524288 + p_loop * 4096 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_inst_size_limit():
    src_shape = [512, 4096]
    src_layout = dst_layout = TileLayout(S[(4, 128, 4096) : (4096 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        Tx.device_entry()
        B_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.copy(A_sbuf[i * 128 : i * 128 + 128, :], B_sbuf[i * 128 : i * 128 + 128, :])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            B_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            for i, b_loop in Tx.grid(4, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(
                            A_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                            B_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                        )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_complex_index():
    A_shape = [4096, 4096]
    A_layout = Tx.TileLayout(Tx.S[(4096, 4096) : (1, 4096)])
    A_sbuf_shape = (2, 2048, 1024)
    A_sbuf_layout = TileLayout(S[(2, 2048, 8, 128) : (16384 @ F, 1 @ F, 2048 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle, ) -> None:
        A = Tx.match_buffer(A_ptr, A_shape, "float32", layout=A_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
        Tx.copy(A_sbuf[1, 0:2048, 0:1024], A[2048: 4096, 3072:4096])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (4096, 4096), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((16777216,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 32768), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 2048, annotations={"nki_dim":"F"}):
                        Tx.nki.load(A_sbuf[p_loop, b_loop * 2048 + f_loop + 16384], A_1[b_loop * 524288 + p_loop * 4096 + f_loop + 12584960])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_complex_index_2():
    A_sbuf_shape = [4096, 4096]
    A_sbuf_layout = Tx.TileLayout(Tx.S[(4096, 32, 128) : (1 @ F, 4096 @ F, 1 @ P)])
    A_shape = (2, 2048, 1024)
    A_layout = Tx.TileLayout(Tx.S[(2, 2048, 1024) : (2048 * 1024, 1, 2048)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle, ) -> None:
        A = Tx.match_buffer(A_ptr, A_shape, "float32", layout=A_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
        Tx.copy(A_sbuf[2048: 4096, 3072:4096], A[1, 0:2048, 0:1024])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (2, 2048, 1024), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((4194304,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 131072), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 2048, annotations={"nki_dim":"F"}):
                        Tx.nki.load(A_sbuf[p_loop, b_loop * 4096 + f_loop + 100352], A_1[b_loop * 262144 + p_loop * 2048 + f_loop + 2097152])  # noqa: E501
                # fmt: on

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_transpose_with_workspace():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(2048, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        identity = Tx.alloc_buffer((128, 128), "float32", scope="trn.sbuf")
        acc_psum = Tx.alloc_buffer((1, 128, 512), "float32", scope="trn.psum", allocated_addr=(0, 0))  # noqa: E501
        with Tx.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                for rhs_f_loop in Tx.serial(0, 128, annotations={"nki_dim":"F"}):
                    Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
        Tx.copy(B_sbuf, A_sbuf, workspace={"identity": identity, "acc_psum": acc_psum})

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((1, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            for b_loop in range(16):
                for extend_b_loop in range(1):
                    Tx.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                        for lhs_f_loop in Tx.serial(128, annotations={"nki_dim": "lhs_F"}):
                            for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "rhs_F"}):
                                Tx.nki.matmul(acc_psum[0, lhs_f_loop, extend_b_loop * 128 + rhs_f_loop], A_sbuf[p_loop, b_loop * 128 + lhs_f_loop], identity[p_loop, rhs_f_loop], Tx.bool(True))  # noqa: E501
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(B_sbuf[p_loop, f_loop * 16 + b_loop], acc_psum[0, p_loop, f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_guard():
    src_shape = [512, 512]
    src_layout = Tx.TileLayout(Tx.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for j in range(4):
            for i in range(4):
                Tx.copy(A_sbuf[i * 128 : i * 128 + 128, 0:128*j], A[i * 128 : i * 128 + 128, 0:128*j])  # noqa: E501

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (512, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((262144,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for j, i, b_loop in Tx.grid(4, 4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 384, annotations={"nki_dim":"F"}):
                        if f_loop < j * 128:
                            Tx.nki.load(A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_guard_2():
    src_shape = [512, 512]
    src_layout = Tx.TileLayout(Tx.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for j in range(4):
            for i in range(4):
                Tx.copy(A_sbuf[0:128*j, 0:128*i], A[0:128*j, 0:128*i])

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        A = Tx.match_buffer(A_ptr, (512, 512), layout=None)
        with Tx.thread():
            A_1 = Tx.decl_buffer((262144,), data=A.data, layout=None)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for j, i, b_loop in Tx.grid(4, 4, 3):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 384, annotations={"nki_dim":"F"}):
                        if b_loop - j < 0 and f_loop < i * 128:
                            Tx.nki.load(A_sbuf[p_loop, b_loop * 512 + f_loop], A_1[b_loop * 65536 + p_loop * 512 + f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_transpose_with_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(2048, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def copy() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            for j in range(4):
                Tx.copy(B_sbuf[i * 128 : i * 128 + 128, 0:128*j], A_sbuf[i * 128 : i * 128 + 128, 0:128*j])  # noqa: E501

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for i, j, b_loop in Tx.grid(4, 4, 3):
                for extend_b_loop in range(1):
                    Tx.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                        for lhs_f_loop in Tx.serial(128, annotations={"nki_dim": "lhs_F"}):
                            for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "rhs_F"}):
                                if b_loop - j < 0:
                                    Tx.nki.matmul(acc_psum[b_loop, lhs_f_loop, rhs_f_loop], A_sbuf[p_loop, i * 512 + b_loop * 128 + lhs_f_loop], identity[p_loop, rhs_f_loop], Tx.bool(True))  # noqa: E501
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        if b_loop - j < 0:
                            Tx.nki.tensor_copy(B_sbuf[p_loop, i * 512 + f_loop * 4 + b_loop], acc_psum[b_loop, p_loop, f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_specified_max_inst_size():
    src_shape = [128, 512]
    src_layout = "PF"
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.copy(A_sbuf, B_sbuf, max_inst_size=128)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf", layout=None)
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf", layout=None)
            for b_loop in Tx.serial(0, 4):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(A_sbuf[p_loop, b_loop * 128 + f_loop], B_sbuf[p_loop, b_loop * 128 + f_loop])  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_transpose_with_extended_f():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer((128, 2048), "float32", scope="trn.sbuf", layout="PF")
        B_sbuf = Tx.alloc_buffer((128, 2048), "float32", scope="trn.sbuf", layout="FP")
        Tx.copy(B_sbuf, A_sbuf)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "copy"})

        with Tx.thread():
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for b_loop in range(4):
                for extend_b_loop in range(4):
                    Tx.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                        for lhs_f_loop in Tx.serial(128, annotations={"nki_dim": "lhs_F"}):
                            for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "rhs_F"}):
                                Tx.nki.matmul(acc_psum[b_loop, lhs_f_loop, extend_b_loop * 128 + rhs_f_loop], A_sbuf[p_loop, b_loop * 512 + extend_b_loop * 128 + lhs_f_loop], identity[p_loop, rhs_f_loop], Tx.bool(True))  # noqa: E501
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(512, annotations={"nki_dim": "F"}):
                        Tx.nki.tensor_copy(B_sbuf[p_loop, b_loop * 512 + f_loop], acc_psum[b_loop, p_loop, f_loop])  # noqa: E501

                # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
