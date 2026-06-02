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


def test_select():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def select() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "select"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.affine_select(B_sbuf[p_loop, f_loop], p_loop < f_loop, A_sbuf[p_loop, f_loop], Tx.float32(0.0))  # noqa: E501
                # fmt: on

    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_in_loop():
    src_shape = [32, 128, 512]
    src_layout = TileLayout(S[(32, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def select() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(2):
            Tx.select(B_sbuf, A_sbuf[i*16, :, :], 0.0, lambda a, b: (i+1)* a < b)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "select"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for i, b_loop in Tx.grid(2, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.affine_select(B_sbuf[p_loop, f_loop], (i + 1) * p_loop < f_loop, A_sbuf[p_loop, i * 8192 + f_loop], Tx.float32(0.0))  # noqa: E501

                # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_expr_affine():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @Tx.prim_func
    def select() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "select"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 4):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], Tx.float32(0.0))  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_with_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @Tx.prim_func
    def select() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            for j in range(4):
                Tx.select(B_sbuf[0: (i+1) * 128, 0: (j+1) * 128], A_sbuf[0: (i+1) * 128, 0: (j+1) * 128], 0.0, lambda a, b: a < b)  # noqa: E501

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "select"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            for i, j, b_loop in Tx.grid(4, 4, 4):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        if b_loop - i < 1 and f_loop < j * 128 + 128:
                            Tx.nki.affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], Tx.float32(0.0))  # noqa: E501
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": select})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
