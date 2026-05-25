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
    return ir_transform(
        stmt,
        preorder=lambda _node: None,
        postorder=lambda node: node.body,
        only_enable=["tirx.ExecScopeStmt"],
    )


def assert_structural_equal(lhs, rhs, *args, **kwargs):
    if isinstance(lhs, tvm.tirx.PrimFunc):
        lhs = lhs.with_body(_strip_exec_scope_stmt(lhs.body))
    if isinstance(rhs, tvm.tirx.PrimFunc):
        rhs = rhs.with_body(_strip_exec_scope_stmt(rhs.body))
    _assert_structural_equal(lhs, rhs, *args, **kwargs)


Tx_func_map = {"add": Tx.add, "sub": Tx.sub, "mul": Tx.mul, "min": Tx.minimum, "max": Tx.maximum}


@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "min", "max"])
@pytest.mark.parametrize(
    "operands_type",
    [
        "region_region",
        "const_region",
        "region_const",
        "region_broadcast_lhs",
        "region_broadcast_rhs",
    ],
)
def test_simple_binary(op_type, operands_type):
    const = Tx.float32(3.0)
    src1_shape = [128, 512] if operands_type != "region_broadcast_lhs" else [128, 1]
    src1_layout = TileLayout(S[src1_shape : (1 @ P, 1 @ F)])
    src2_shape = [128, 512] if operands_type != "region_broadcast_rhs" else [128, 1]
    src2_layout = TileLayout(S[src2_shape : (1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
    Tx_func = Tx_func_map[op_type]

    # fmt: off
    @Tx.prim_func
    def binary() ->None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            if operands_type == "region_region" or operands_type.startswith("region_broadcast"):
                Tx_func(C_sbuf, A_sbuf, B_sbuf)
            elif operands_type == "const_region":
                Tx_func(C_sbuf, const, A_sbuf)
            elif operands_type == "region_const":
                Tx_func(C_sbuf, A_sbuf, const)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer(src2_shape, scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer(dst_shape, scope="trn.sbuf")
            for b_loop in Tx.serial(0, 1):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 512, annotations={"nki_dim":"F"}):
                        if operands_type == "region_region":
                            Tx.nki.tensortensor(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], B_sbuf[p_loop, f_loop], op_type)  # noqa: E501
                        elif operands_type == "region_const":
                            Tx.nki.tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], Tx.float32(3.0), op_type, Tx.bool(False))  # noqa: E501
                        elif operands_type == "const_region":
                            Tx.nki.tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], Tx.float32(3.0), op_type, Tx.bool(True))  # noqa: E501
                        elif operands_type == "region_broadcast_rhs":
                            Tx.nki.tensorscalar(C_sbuf[p_loop, f_loop], A_sbuf[p_loop, f_loop], B_sbuf[p_loop, 0], op_type, Tx.bool(False))  # noqa: E501
                        elif operands_type == "region_broadcast_lhs":
                            Tx.nki.tensorscalar(C_sbuf[p_loop, f_loop], B_sbuf[p_loop, f_loop], A_sbuf[p_loop, 0], op_type, Tx.bool(True))  # noqa: E501
        # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("op_type", ["add", "sub", "mul", "min", "max"])
@pytest.mark.parametrize(
    "operands_type",
    [
        "region_region",
        "const_region",
        "region_const",
        "region_broadcast_lhs",
        "region_broadcast_rhs",
    ],
)
def test_binary_complex(op_type, operands_type):
    src1_shape = [1024, 512] if operands_type != "region_broadcast_lhs" else [1024, 4]
    src1_layout_data_iter = (128, 4096) if operands_type != "region_broadcast_lhs" else (128, 32)
    src1_layout = TileLayout(S[src1_layout_data_iter : (1 @ P, 1 @ F)])
    src2_shape = [512, 512] if operands_type != "region_broadcast_rhs" else [128, 512]
    src2_layout_data_iter = (128, 2048) if operands_type != "region_broadcast_rhs" else (128, 512)
    src2_layout = TileLayout(S[src2_layout_data_iter : (1 @ P, 1 @ F)])

    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])
    const = Tx.float32(3.0)
    Tx_func = Tx_func_map[op_type]

    src1_view_shape = [128, 8, 512]
    src2_view_shape = [128, 4, 512] if operands_type != "region_broadcast_rhs" else [128, 1, 512]
    dst_view_shape = [128, 4, 512]
    if operands_type == "region_broadcast_lhs":
        src1_view_shape = [128, 8, 4, 1]
        src2_view_shape = [128, 4, 4, 128]
        dst_view_shape = [128, 4, 4, 128]

    # fmt: off
    @Tx.prim_func
    def binary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            A_sbuf_view = A_sbuf.view(*src1_view_shape)
            B_sbuf_view = B_sbuf.view(*src2_view_shape)
            C_sbuf_view = C_sbuf.view(*dst_view_shape)
            for i in range(4):
                if operands_type == "region_region":
                    Tx_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], B_sbuf_view[:, i, :])
                elif operands_type == "region_const":
                    Tx_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], const)
                elif operands_type == "const_region":
                    Tx_func(C_sbuf_view[:, i, :], const, A_sbuf_view[:, i * 2, :])
                elif operands_type == "region_broadcast_rhs":
                    Tx_func(C_sbuf_view[:, i, :], A_sbuf_view[:, i * 2, :], B_sbuf_view[:, 0, :])
                elif operands_type == "region_broadcast_lhs":
                    Tx_func(C_sbuf_view[:, i, :, :], A_sbuf_view[:, i*2,:, :], B_sbuf_view[:, i, :, :])  # noqa: E501

    f_extent = 128 if operands_type == "region_broadcast_lhs" else 512
    b_extent = 4 if operands_type == "region_broadcast_lhs" else 1

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_layout_data_iter, scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer(src2_layout_data_iter, scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 2048), scope="trn.sbuf")
            A_sbuf_view = Tx.decl_buffer(src1_layout_data_iter, data=A_sbuf.data, scope="trn.sbuf", layout=None)  # noqa: E501
            B_sbuf_view = Tx.decl_buffer(src2_layout_data_iter, data=B_sbuf.data, scope="trn.sbuf", layout=None)  # noqa: E501
            C_sbuf_view = Tx.decl_buffer((128, 2048), data=C_sbuf.data, scope="trn.sbuf", layout=None)  # noqa: E501
            for i, b_loop in Tx.grid(4, b_extent):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, f_extent, annotations={"nki_dim":"F"}):
                        if operands_type == "region_region":
                            Tx.nki.tensortensor(C_sbuf_view[p_loop, i * 512 + f_loop], A_sbuf_view[p_loop, i * 1024 + f_loop], B_sbuf_view[p_loop, i * 512 + f_loop], op_type)  # noqa: E501
                        elif operands_type == "const_region":
                            Tx.nki.tensorscalar(C_sbuf_view[p_loop, i * 512 + f_loop], A_sbuf_view[p_loop, i * 1024 + f_loop], Tx.float32(3.0), op_type, Tx.bool(True))  # noqa: E501
                        elif operands_type == "region_const":
                            Tx.nki.tensorscalar(C_sbuf_view[p_loop, i * 512 + f_loop], A_sbuf_view[p_loop, i * 1024 + f_loop], Tx.float32(3.0), op_type, Tx.bool(False))  # noqa: E501
                        elif operands_type == "region_broadcast_lhs":
                            Tx.nki.tensorscalar(C_sbuf_view[p_loop, i * 512 + b_loop * 128 + f_loop], B_sbuf_view[p_loop, i * 512 + b_loop * 128 + f_loop], A_sbuf_view[p_loop, i * 8 + b_loop], op_type, Tx.bool(True))  # noqa: E501
                        elif operands_type == "region_broadcast_rhs":
                            Tx.nki.tensortensor(C_sbuf_view[p_loop, i * 512 + f_loop], A_sbuf_view[p_loop, i * 1024 + f_loop], B_sbuf_view[p_loop, f_loop], op_type)  # noqa: E501

        # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast1():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(S[(32, 128, 4, 128) : (1 @ F, 32 @ F, 32 * 128 @ F, 1 @ P)])
    src2_shape = [128, 512]
    src2_layout = TileLayout(S[(512, 128) : (1 @ F, 1 @ P)])
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @Tx.prim_func
    def binary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.add(C_sbuf, A_sbuf, B_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 512):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 32, annotations={"nki_dim":"F"}):
                        Tx.nki.tensorscalar(C_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 32 + f_loop], B_sbuf[p_loop, b_loop], "add", Tx.bool(False))  # noqa: E501
        # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast2():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(S[(32, 128, 4, 128) : (128 @ F, 1 @ F, 32 * 128 @ F, 1 @ P)])
    src2_shape = [128, 512]
    src2_layout = TileLayout(S[(128, 4, 128) : (1 @ F, 128 @ F, 1 @ P)])
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @Tx.prim_func
    def binary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.add(C_sbuf, A_sbuf, B_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 128):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 128, annotations={"nki_dim":"F"}):
                        Tx.nki.tensortensor(C_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 128 + f_loop], A_sbuf[p_loop, b_loop % 4 * 4096 + b_loop // 4 * 128 + f_loop], B_sbuf[p_loop, b_loop % 4 * 128 + f_loop], "add")  # noqa: E501
        # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_broadcast3():
    src1_shape = [128, 512]
    src1_layout = TileLayout(S[(128, 4, 128) : (1 @ F, 128 @ F, 1 @ P)])
    src2_shape = [32, 128, 512]
    src2_layout = TileLayout(S[(32, 128, 4, 128) : (128 @ F, 1 @ F, 32 * 128 @ F, 1 @ P)])
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @Tx.prim_func
    def binary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.add(C_sbuf, A_sbuf, B_sbuf[0])

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            for b_loop in Tx.serial(0, 4):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 128, annotations={"nki_dim":"F"}):
                        Tx.nki.tensortensor(C_sbuf[p_loop, b_loop * 128 + f_loop], A_sbuf[p_loop, b_loop * 128 + f_loop], B_sbuf[p_loop, b_loop * 4096 + f_loop], "add")  # noqa: E501
        # fmt: on

    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_with_guard():
    src1_shape = [32, 128, 512]
    src1_layout = TileLayout(S[(32, 128, 4, 128) : (128 @ F, 1 @ F, 32 * 128 @ F, 1 @ P)])
    src2_shape = [128, 512]
    src2_layout = TileLayout(S[(128, 4, 128) : (1 @ F, 128 @ F, 1 @ P)])
    dst_shape = src1_shape
    dst_layout = src1_layout

    # fmt: off
    @Tx.prim_func
    def binary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(src2_shape, "float32", scope="trn.sbuf", layout=src2_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            for j in range(4):
                Tx.add(C_sbuf[:, :, 0:j*128], A_sbuf[:, :, 0:j*128], B_sbuf[:, 0:j*128])

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "binary"})

        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            B_sbuf = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            C_sbuf = Tx.alloc_buffer((128, 16384), scope="trn.sbuf")
            for j, b_loop in Tx.grid(4, 96):
                Tx.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in Tx.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in Tx.serial(0, 128, annotations={"nki_dim":"F"}):
                        if b_loop % 3 - j < 0:
                            Tx.nki.tensortensor(C_sbuf[p_loop, b_loop % 3 * 4096 + b_loop // 3 * 128 + f_loop], A_sbuf[p_loop, b_loop % 3 * 4096 + b_loop // 3 * 128 + f_loop], B_sbuf[p_loop, b_loop % 3 * 128 + f_loop], "add")  # noqa: E501

        # fmt: on
    with target:
        mod = tvm.IRModule({"main": binary})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
