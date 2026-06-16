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


opcode_map = {"sum": "add", "max": "max", "min": "min"}

Tx_func_map = {"sum": Tx.sum, "max": Tx.max, "min": Tx.min}


@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
def test_simple_reduction(op_type):
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
    dst_shape = [128, 1]
    dst_layout = TileLayout(S[(128, 1) : (1 @ P, 1 @ F)])

    opcode = opcode_map[op_type]
    tx_func = Tx_func_map[op_type]

    # fmt: off
    @T.prim_func
    def reduction() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        tx_func(B_sbuf, A_sbuf, axes=-1)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        A_sbuf = T.alloc_buffer((128, 512), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 1), scope="trn.sbuf")
        for b_loop in range(1):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.tensorreduce(B_sbuf[p_loop, 0], A_sbuf[p_loop, f_loop], opcode, False, -1)

            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_with_multiple_axes():
    src_shape = [128, 512, 4]
    src_layout = TileLayout(S[(128, 512, 4) : (1 @ P, 1 @ F, 512 @ F)])
    dst_shape = [128]
    dst_layout = TileLayout(S[128 : 1 @ P])

    # fmt: off
    @T.prim_func
    def reduction():
        T.device_entry()
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.sum(B_sbuf, A_sbuf, axes=(1, 2), max_inst_size=2048)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 1), scope="trn.sbuf")
        for b_loop in range(1):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                    T.nki.tensorreduce(B_sbuf[p_loop, 0], A_sbuf[p_loop, f_loop], "add", False, -1)

            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_in_loop():
    src_shape = [128, 512, 4]
    src_layout = TileLayout(S[(128, 512, 4) : (1 @ P, 4 @ F, 1 @ F)])
    dst_shape = [128, 4]
    dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.prim_func
    def reduction():
        T.device_entry()
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            Tx.sum(B_sbuf[:, i], A_sbuf[:, :, i], axes=-2)

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        A_sbuf = T.alloc_buffer((128, 2048), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
        for i, b_loop in T.grid(4, 1):
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.tensorreduce(B_sbuf[p_loop, i], A_sbuf[p_loop, f_loop * 4 + i], "add", False, -1)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(S[(128, 32 * 32 * 4) : (1 @ P, 1 @ F)])
    dst_shape = [128, 4]
    dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.prim_func
    def reduction():
        T.device_entry()
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.sum(B_sbuf, A_sbuf, axes=(1, 3))

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        intermediate_buffer = T.alloc_buffer((128, 32), scope="trn.sbuf")
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
        for b_loop in range(4):
            for reduction_b_loop in range(32):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki.tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, reduction_b_loop * 128 + b_loop * 32 + f_loop], "add", False, -1)  # noqa: E501
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                    T.nki.tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", False, -1)  # noqa: E501

            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.trn.transform.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_with_guard():
    src_shape = [512, 2048]
    src_layout = TileLayout(S[(4, 128, 2048) : (2048 @ F, 1 @ P, 1 @ F)])
    dst_shape = [512, 1]
    dst_layout = TileLayout(S[(4, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @T.prim_func
    def reduction() -> None:
        T.device_entry()
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            for j in range(4):
                Tx.sum(B_sbuf[0: (i+1) * 128, 0], A_sbuf[0: (i+1) * 128, 0: (j+1) * 256], max_inst_size=512)  # noqa: E501

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        intermediate_buffer = T.alloc_buffer((128, 2), scope="trn.sbuf")
        A_sbuf = T.alloc_buffer((128, 8192), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
        for i, j in T.grid(4, 4):
            for b_loop in range(4):
                for reduction_b_loop in range(2):
                    T.attr(0, "tensorized_nki_instruction", 1)
                    for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                        for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                            if (
                                b_loop - i < 1
                                and reduction_b_loop * 512 + f_loop < j * 256 + 256
                            ):
                                T.nki.tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, b_loop * 2048 + reduction_b_loop * 512 + f_loop], "add", T.bool(False), -1)  # noqa: E501
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in T.serial(2, annotations={"nki_dim": "F"}):
                        if b_loop - i < 1 and f_loop * 2 - j < 1:
                            T.nki.tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", T.bool(False), -1)  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.trn.transform.TrnPrivateBufferAlloc()(mod)
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage_workspace():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(S[(128, 32 * 32 * 4) : (1 @ P, 1 @ F)])
    dst_shape = [128, 4]
    dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.prim_func
    def reduction():
        T.device_entry()
        intermediate_buffer = T.alloc_buffer((128, 64), scope="trn.sbuf")
        A_sbuf = T.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        Tx.sum(B_sbuf, A_sbuf, axes=(1, 3), workspace={"partial_reduce": intermediate_buffer})

    @T.prim_func
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        intermediate_buffer = T.alloc_buffer((128, 64), scope="trn.sbuf")
        A_sbuf = T.alloc_buffer((128, 4096), scope="trn.sbuf")
        B_sbuf = T.alloc_buffer((128, 4), scope="trn.sbuf")
        for b_loop in range(4):
            for reduction_b_loop in range(32):
                T.attr(0, "tensorized_nki_instruction", 1)
                for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                    for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                        T.nki.tensorreduce(intermediate_buffer[p_loop, reduction_b_loop], A_sbuf[p_loop, reduction_b_loop * 128 + b_loop * 32 + f_loop], "add", False, -1)  # noqa: E501
            T.attr(0, "tensorized_nki_instruction", 1)
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 32, annotations={"nki_dim":"F"}):
                    T.nki.tensorreduce(B_sbuf[p_loop, b_loop], intermediate_buffer[p_loop, f_loop], "add", False, -1)  # noqa: E501

            # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
