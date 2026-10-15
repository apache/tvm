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

import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.trn.axis import F, P
from tvm_tirx.ir import assert_structural_equal as _assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import S, TileLayout

target = tvm_tirx.target.Target("aws/trn1/trn1.2xlarge")


def _strip_thread_scope_stmt(stmt):
    def _strip_region(node: tvm_tirx.ir.RegionStmt):
        if node.op.same_as(tvm_tirx.ir.Op.get("tirx.device_entry")):
            return node.body
        return node

    return tvm_ffi.structural_map(
        stmt,
        (tvm_tirx.ir.RegionStmt, _strip_region),
        order="post",
    )


def assert_structural_equal(lhs, rhs, *args, **kwargs):
    if isinstance(lhs, tvm_tirx.tirx.Function):
        lhs = lhs.with_body(_strip_thread_scope_stmt(lhs.body))
    if isinstance(rhs, tvm_tirx.tirx.Function):
        rhs = rhs.with_body(_strip_thread_scope_stmt(rhs.body))
    _assert_structural_equal(lhs, rhs, *args, **kwargs)


def test_select():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.function
    def select() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.affine_select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)

    @T.function
    def expected():
        T.func_attr({"global_symbol": "select"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.affine_select(B_sbuf[p_loop, f_loop], p_loop < f_loop, A_sbuf[p_loop, f_loop], T.float32(0.0))  # noqa: E501
            # fmt: on

    with target:
        mod = tvm_tirx.IRModule({"main": select})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_in_loop():
    src_shape = [32, 128, 512]
    src_layout = TileLayout(S[(32, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.function
    def select() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(2):
            T.trn.tile.affine_select(B_sbuf, A_sbuf[i*16, :, :], 0.0, lambda a, b: (i+1)* a < b)

    @T.function
    def expected():
        T.func_attr({"global_symbol": "select"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 16384), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(2, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.affine_select(B_sbuf[p_loop, f_loop], (i + 1) * p_loop < f_loop, A_sbuf[p_loop, i * 8192 + f_loop], T.float32(0.0))  # noqa: E501

            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": select})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_expr_affine():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @T.function
    def select() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.affine_select(B_sbuf, A_sbuf, 0.0, lambda i, j: i < j)

    @T.function
    def expected():
        T.func_attr({"global_symbol": "select"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], T.float32(0.0))  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": select})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_select_with_guard():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @T.function
    def select() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(4):
            for j in range(4):
                T.trn.tile.affine_select(B_sbuf[0: (i+1) * 128, 0: (j+1) * 128], A_sbuf[0: (i+1) * 128, 0: (j+1) * 128], 0.0, lambda a, b: a < b)  # noqa: E501

    @T.function
    def expected():
        T.func_attr({"global_symbol": "select"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for i, j, b_loop in T.grid(4, 4, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    if b_loop - i < 1 and f_loop < j * 128 + 128:
                        T.nki.affine_select(B_sbuf[p_loop, b_loop * 512 + f_loop], b_loop * 128 + p_loop < f_loop, A_sbuf[p_loop, b_loop * 512 + f_loop], T.float32(0.0))  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": select})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
