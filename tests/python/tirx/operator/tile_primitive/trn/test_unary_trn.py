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


Tx_func_map = {"reciprocal": Tx.reciprocal, "sqrt": Tx.sqrt, "memset": Tx.memset, "exp": Tx.exp}


@pytest.mark.parametrize("op_type", ["reciprocal", "memset"])
def test_simple_unary(op_type):
    src_shape = [128, 512]
    src_layout = Tx.TileLayout(Tx.S[(128, 512) : (1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = Tx.TileLayout(Tx.S[(128, 512) : (1 @ P, 1 @ F)])
    tx_func = Tx_func_map[op_type]

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        if op_type == "memset":
            tx_func(B_sbuf, Tx.float32(0.0))
        else:
            tx_func(B_sbuf, A_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        if op_type == "reciprocal":
                            Tx.nki.reciprocal(
                                B_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop]
                            )
                        elif op_type == "memset":
                            Tx.nki.memset(B_sbuf[p_loop, f_loop], 0.0)
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["reciprocal", "memset"])
def test_unary_in_a_loop(op_type):
    src_shape = [1024, 512]
    src_layout = Tx.TileLayout(Tx.S[(128, 4096) : (1 @ P, 1 @ F)])
    dst_shape = [512, 512]
    dst_layout = Tx.TileLayout(Tx.S[(128, 2048) : (1 @ P, 1 @ F)])

    Tx_func = Tx_func_map[op_type]

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        A_sbuf_view = A_sbuf.view(128, 8, 512)
        B_sbuf_view = B_sbuf.view(128, 4, 512)
        for i in range(4):
            if op_type == "memset":
                Tx_func(B_sbuf_view[:, i, :], Tx.float32(0.0))
            else:
                Tx_func(B_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :])

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            A_sbuf_view = Tx.decl_buffer((128, 4096), data=A_sbuf.data, scope="trn.sbuf", layout=None)  # noqa: E501
            B_sbuf_view = Tx.decl_buffer((128, 2048), data=B_sbuf.data, scope="trn.sbuf", layout=None)  # noqa: E501
            for i, b_loop in Tx.grid(4, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        if op_type == "reciprocal":
                            Tx.nki.reciprocal(B_sbuf_view[p_loop, i * 512 + f_loop], A_sbuf_view[p_loop, i * 1024 + f_loop])  # noqa: E501
                        elif op_type == "memset":
                            Tx.nki.memset(B_sbuf[p_loop, i * 512 + f_loop], 0.0)
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_complex1():
    dst_layout = TileLayout(S[(32, 128, 256) : (256 @ F, 1 @ P, 1 @ F)])
    dst_shape = [4096, 256]

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.memset(A_sbuf, Tx.float32(0.0))

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 8192), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 16):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.memset(A_sbuf[p_loop, b_loop * 512 + f_loop], Tx.float32(0.0))
                # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
def test_unary_with_bias_scale(op_type):
    src_shape = [512, 1024]
    src_layout = TileLayout(S[(128, 4096) : (1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout
    bias_shape = [512, 1]
    bias_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])
    scale = Tx.float32(2.0)
    tx_func = Tx_func_map[op_type]

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(bias_shape, "float32", scope="trn.sbuf", layout=bias_layout)
        C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        tx_func(C_sbuf, A_sbuf, bias=B_sbuf, scale=scale)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 4), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        Tx.nki.activation(C_sbuf[p_loop, b_loop * 512  + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], op_type, B_sbuf[p_loop, b_loop//2], Tx.float32(2.0))  # noqa: E501
                # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["sqrt", "exp"])
def test_unary_with_bias_scale_2(op_type):
    src_shape = [512, 1024]
    src_layout = TileLayout(S[(128, 4096) : (1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout
    bias = Tx.float32(1.0)
    scale = Tx.float32(2.0)
    tx_func = Tx_func_map[op_type]

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        tx_func(C_sbuf, A_sbuf, bias=bias, scale=scale)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            const_bias = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(512, annotations={"nki_dim": "F"}):
                        Tx.nki.memset(const_bias[p_loop, f_loop], Tx.float32(1.0))
            A_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(512, annotations={"nki_dim": "F"}):
                        Tx.nki.activation(C_sbuf[p_loop, b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], op_type, const_bias[p_loop, f_loop], Tx.float32(2.0))  # noqa: E501
                # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.trn.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_unary_with_guard():
    src_shape = [512, 1024]
    src_layout = TileLayout(S[(4, 128, 1024) : (1024 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout
    bias_shape = [512, 1]
    bias_layout = TileLayout(S[(4, 128) : (1 @ F, 1 @ P)])
    scale = Tx.float32(2.0)

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        Tx.device_entry()
        A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = Tx.alloc_buffer(bias_shape, "float32", scope="trn.sbuf", layout=bias_layout)
        C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            for j in range(4):
                Tx.sqrt(C_sbuf[0: (i+1) * 128, 0: (j+1)*256], A_sbuf[0: (i+1) * 128, 0: (j+1)*256], bias=B_sbuf[0: (i+1) * 128, 0], scale=scale)  # noqa: E501

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})

        with Tx.thread():
            A_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 4), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 4096), scope="trn.sbuf")
            for i, j, b_loop in Tx.grid(4, 4, 8):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        if b_loop // 2 - i < 1 and b_loop % 2 * 512 + f_loop < j * 256 + 256:
                            Tx.nki.activation(C_sbuf[p_loop, b_loop * 512 + f_loop], A_sbuf[p_loop, b_loop * 512 + f_loop], "sqrt", B_sbuf[p_loop, b_loop // 2], Tx.float32(2.0))  # noqa: E501
                 # fmt: off
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
