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
import tvm_ffi

import tvm
import tvm.testing
from tvm.ir import assert_structural_equal as _assert_structural_equal
from tvm.script import tirx as T
from tvm.tirx.layout import F, P, S, TileLayout

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def _strip_exec_scope_stmt(stmt):
    def _strip_region(node: tvm.ir.RegionStmt):
        if node.op.same_as(tvm.ir.Op.get("tirx.device_entry")):
            return node.body
        return node

    return tvm_ffi.structural_map(
        stmt,
        (tvm.ir.RegionStmt, _strip_region),
        order="post",
    )


def assert_structural_equal(lhs, rhs, *args, **kwargs):
    if isinstance(lhs, tvm.tirx.Function):
        lhs = lhs.with_body(_strip_exec_scope_stmt(lhs.body))
    if isinstance(rhs, tvm.tirx.Function):
        rhs = rhs.with_body(_strip_exec_scope_stmt(rhs.body))
    _assert_structural_equal(lhs, rhs, *args, **kwargs)


def test_simple_copy():
    src_shape = [128, 512]
    src_layout = T.TileLayout(T.S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.load(A_sbuf, A)

    @T.function
    def expected(A: T.Tensor((128, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((65536,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf")
        for b_loop in T.serial(0, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(A_sbuf[p_loop, f_loop], A_1[p_loop * 512 + f_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_copy_2():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 4, 128) : (512, 128, 1)])

    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 4, 128) : (4 @ F, 1 @ F, 1 @ P)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.load(A_sbuf, A)

    @T.function
    def expected(A: T.Tensor((128, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((65536,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf")
        for b_loop in T.serial(0, 512):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 1, annotations={"nki_dim": "F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop], A_1[b_loop * 128 + p_loop])

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.function
    def expected(A: T.Tensor((512, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((262144,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 2048), scope="trn.sbuf")
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop]
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop_2():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(128, 2048) : (2048, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        A_sbuf_view = A_sbuf.view(128, 4, 512)
        A_view = A.view(128, 4, 512)
        for i in range(4):
            T.trn.tile.load(A_sbuf_view[:, i, :], A_view[:, i, :])

    @T.function
    def expected(A: T.Tensor((512, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        _A_flat = T.decl_tensor((262144,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 2048), scope="trn.sbuf")
        A_sbuf_view = T.decl_tensor((128, 2048), data=A_sbuf.data, scope="trn.sbuf", layout=None)
        A_view = T.decl_tensor((262144,), data=_A_flat.data, layout=None)
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf_view[p_loop, i * 512 + f_loop],
                        A_view[p_loop * 2048 + i * 512 + f_loop],
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod.show()
        assert_structural_equal(mod["main"], expected)


def test_copy_different_f():
    src_shape = [512, 64]
    src_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 16 @ F, 4 @ F, 1 @ F)])
    dst_shape = [512, 64]
    dst_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 4 @ F, 16 @ F, 1 @ F)])

    @T.function
    def copy() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.tensor_copy(B_sbuf, A_sbuf)

    @T.function
    def expected():
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor((128, 256), scope="trn.sbuf")
        B_sbuf = T.alloc_tensor((128, 256), scope="trn.sbuf")
        for b_loop in T.serial(0, 64):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 4, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(
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

    @T.function
    def copy() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        B_sbuf_view = B_sbuf.view(512, 4)
        T.trn.tile.tensor_copy(B_sbuf_view, A_sbuf[:, 0:4])

    @T.function
    def expected():
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor((128, 256), scope="trn.sbuf")
        B_sbuf = T.alloc_tensor((128, 16), scope="trn.sbuf")
        B_sbuf_view = T.decl_tensor((128, 16), data=B_sbuf.data, scope="trn.sbuf", layout=None)
        for b_loop in T.serial(0, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 4, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(
                        B_sbuf_view[p_loop, b_loop * 4 + f_loop],
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

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            T.trn.tile.store(A[:, i * 512 : i * 512 + 512], A_sbuf)

    @T.function
    def expected(A: T.Tensor((128, 10000), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((1280000,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf")
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.store(A_1[p_loop * 10000 + i * 512 + f_loop], A_sbuf[p_loop, f_loop])

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
    @T.function
    def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(32):
            T.trn.tile.load(A_sbuf, A[i, :, :])

    @T.function
    def expected(A: T.Tensor((32, 128, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((2097152,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf")
        for i, b_loop in T.grid(32, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, f_loop], A_1[i * 65536 + p_loop * 128 + f_loop])
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

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(2):
            T.trn.tile.load(A_sbuf[i * 256 : i * 256 + 256, :], A)

    @T.function
    def expected(A: T.Tensor((256, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((131072,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 2048), scope="trn.sbuf")
        for i, b_loop in T.grid(2, 2):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf[p_loop, i * 1024 + b_loop * 512 + f_loop],
                        A_1[b_loop * 65536 + p_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_large_dma_copy():
    src_shape = [512, 4096]
    src_layout = T.TileLayout(T.S[(4, 128, 4096) : (4096 * 128, 4096, 1)])
    dst_shape = [512, 4096]
    dst_layout = TileLayout(S[(4, 128, 4096) : (4096 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.function
    def expected(A: T.Tensor((512, 4096), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((2097152,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 16384), scope="trn.sbuf")
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 4096, annotations={"nki_dim": "F"}):
                    T.nki.load(
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

    @T.function
    def copy(A_ptr: T.handle) -> None:
        T.device_entry()
        B_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for i in range(4):
            T.trn.tile.tensor_copy(
                A_sbuf[i * 128 : i * 128 + 128, :], B_sbuf[i * 128 : i * 128 + 128, :]
            )

    @T.function
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        B_sbuf = T.alloc_tensor((128, 16384), scope="trn.sbuf")
        A_sbuf = T.alloc_tensor((128, 16384), scope="trn.sbuf")
        for i, b_loop in T.grid(4, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(
                        A_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                        B_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_complex_index():
    A_shape = [4096, 4096]
    A_layout = T.TileLayout(T.S[(4096, 4096) : (1, 4096)])
    A_sbuf_shape = (2, 2048, 1024)
    A_sbuf_layout = TileLayout(S[(2, 2048, 8, 128) : (16384 @ F, 1 @ F, 2048 @ F, 1 @ P)])

    # fmt: off
    @T.function
    def copy(A: T.Tensor(A_shape, 'float32', layout=A_layout), ) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
        T.trn.tile.load(A_sbuf[1, 0:2048, 0:1024], A[2048: 4096, 3072:4096])

    @T.function
    def expected(A: T.Tensor((4096, 4096), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((16777216,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 32768), scope="trn.sbuf")
        for b_loop in T.serial(0, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop * 2048 + f_loop + 16384], A_1[b_loop * 524288 + p_loop * 4096 + f_loop + 12584960])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_complex_index_2():
    A_sbuf_shape = [4096, 4096]
    A_sbuf_layout = T.TileLayout(T.S[(4096, 32, 128) : (1 @ F, 4096 @ F, 1 @ P)])
    A_shape = (2, 2048, 1024)
    A_layout = T.TileLayout(T.S[(2, 2048, 1024) : (2048 * 1024, 1, 2048)])

    # fmt: off
    @T.function
    def copy(A: T.Tensor(A_shape, 'float32', layout=A_layout), ) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)
        T.trn.tile.load(A_sbuf[2048: 4096, 3072:4096], A[1, 0:2048, 0:1024])

    @T.function
    def expected(A: T.Tensor((2, 2048, 1024), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((4194304,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 131072), scope="trn.sbuf")
        for b_loop in T.serial(0, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop * 4096 + f_loop + 100352], A_1[b_loop * 262144 + p_loop * 2048 + f_loop + 2097152])  # noqa: E501
            # fmt: on

    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_guard():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    # fmt: off
    @T.function
    def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for j in range(4):
            for i in range(4):
                T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, 0:128*j], A[i * 128 : i * 128 + 128, 0:128*j])  # noqa: E501

    @T.function
    def expected(A: T.Tensor((512, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((262144,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 2048), scope="trn.sbuf")
        for j, i, b_loop in T.grid(4, 4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 384, annotations={"nki_dim":"F"}):
                    if f_loop < j * 128:
                        T.nki.load(A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_guard_2():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    # fmt: off
    @T.function
    def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        for j in range(4):
            for i in range(4):
                T.trn.tile.load(A_sbuf[0:128*j, 0:128*i], A[0:128*j, 0:128*i])

    @T.function
    def expected(A: T.Tensor((512, 512), layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor((262144,), data=A.data, layout=None)
        A_sbuf = T.alloc_tensor((128, 2048), scope="trn.sbuf")
        for j, i, b_loop in T.grid(4, 4, 3):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 384, annotations={"nki_dim":"F"}):
                    if b_loop - j < 0 and f_loop < i * 128:
                        T.nki.load(A_sbuf[p_loop, b_loop * 512 + f_loop], A_1[b_loop * 65536 + p_loop * 512 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_specified_max_inst_size():
    src_shape = [128, 512]
    src_layout = "PF"
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @T.function
    def copy(A_ptr: T.handle) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        B_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.tensor_copy(A_sbuf, B_sbuf, max_inst_size=128)

    @T.function
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf", layout=None)
        B_sbuf = T.alloc_tensor((128, 512), scope="trn.sbuf", layout=None)
        for b_loop in T.serial(0, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(128, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(A_sbuf[p_loop, b_loop * 128 + f_loop], B_sbuf[p_loop, b_loop * 128 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()


@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("copy_to_sbuf", [False, True])
def test_explicit_transpose_prepares_identity_and_psum(columns, copy_to_sbuf):
    """The caller owns the transpose algorithm, including its temporary tensors."""
    layout = TileLayout(S[(128, columns) : (1 @ P, 1 @ F)])
    square = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)])

    @T.function
    def transpose():
        T.device_entry()
        source = T.alloc_tensor((128, columns), "float32", scope="trn.sbuf", layout=layout)
        identity = T.alloc_tensor((128, 128), "float32", scope="trn.sbuf", layout=square)
        accumulator = T.alloc_tensor(
            (128, 128), "float32", scope="trn.psum", layout=square, allocated_addr=[0, 0]
        )
        result = T.alloc_tensor(
            (columns, 128),
            "float32",
            scope="trn.sbuf",
            layout=TileLayout(S[(columns // 128, 128, 128) : (128 @ F, 1 @ P, 1 @ F)]),
        )
        with T.nki.tensorized_instruction():
            for p in T.serial(128, annotations={"nki_dim": "P"}):
                for f in T.serial(128, annotations={"nki_dim": "F"}):
                    T.nki.identity(identity[p, f], 128)
        for block in range(columns // 128):
            T.trn.tile.matmul(
                accumulator, source[:, block * 128 : (block + 1) * 128], identity, transpose_A=True
            )
            if T.constexpr(copy_to_sbuf):
                T.trn.tile.tensor_copy(result[block * 128 : (block + 1) * 128, :], accumulator)

    mod = tvm.IRModule({"main": transpose})
    with target:
        allocated = tvm.tirx.trn.transform.TrnPrivateBufferAlloc()(mod)
        # No hidden identity or PSUM workspace is needed by either instruction.
        _assert_structural_equal(mod, allocated)
        lowered = tvm.tirx.transform.LowerTIRx()(allocated)
    names = []
    tvm_ffi.structural_walk(
        lowered,
        (tvm.ir.Call, lambda n: names.append(n.op.name) if isinstance(n.op, tvm.ir.Op) else None),
    )
    assert "tirx.nki.identity" in names
    assert "tirx.nki.matmul" in names
    assert ("tirx.nki.tensor_copy" in names) == copy_to_sbuf
    assert not any(".tile." in name for name in names)


def test_tensor_copy_rejects_partition_transpose():
    @T.function
    def copy():
        T.device_entry()
        a = T.alloc_tensor(
            (128, 128),
            "float32",
            scope="trn.sbuf",
            layout=TileLayout(S[(128, 128) : (1 @ P, 1 @ F)]),
        )
        b = T.alloc_tensor(
            (128, 128),
            "float32",
            scope="trn.sbuf",
            layout=TileLayout(S[(128, 128) : (1 @ F, 1 @ P)]),
        )
        T.trn.tile.tensor_copy(b, a)

    with target, pytest.raises(RuntimeError, match="identity|transpose"):
        tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": copy}))
