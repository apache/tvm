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

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.trn.axis import F, P
from tvm_tirx.backend.trn.tile.instruction_generator import InstructionGenerator
from tvm_tirx.ir import assert_structural_equal as _assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer
from tvm_tirx.testing import env
from tvm_tirx.tirx import TensorType, Var
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


def test_instruction_generator_compact_global_layout():
    tensor = Var("tensor", TensorType((8, 16), "float32"))
    region = tensor[0:8, 0:16]
    generator = InstructionGenerator((region,), Analyzer())
    layout = generator.split_layout_views[region]
    assert int(layout.apply(3, 5, shape=tensor.shape)["m"]) == 53
    assert tensor.layout is None


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_parameter_stride_contract():
    """Physical index lowering must preserve the host's DLTensor stride checks."""
    layout = TileLayout(S[(2, 2) : (3, 1)])

    @T.function
    def copy(
        A: T.Tensor((2, 2), "float32", layout=layout, data_alignment=1),
        B: T.Tensor((2, 2), "float32"),
    ):
        for i, j in T.grid(2, 2):
            B[i, j] = A[i, j]

    mod = tvm_tirx.IRModule({"main": copy})
    lowered = tvm_tirx.backend.trn.transform.LowerTrainiumLayout()(mod)
    # Execute the lowered scalar accesses and packed wrapper on the host, without
    # requiring a Trainium device or compiling Trainium-specific instructions.
    lib = tvm_tirx.tirx.compile(lowered, target="llvm")
    data = np.arange(6, dtype="float32").reshape(2, 3)[:, :2]
    result = tvm_tirx.runtime.empty((2, 2), "float32")
    lib(data, result)
    np.testing.assert_array_equal(result.numpy(), data)
    with pytest.raises(ValueError, match="strides"):
        lib(np.ascontiguousarray(data), result)


def test_simple_copy():
    src_shape = [128, 512]
    src_layout = T.TileLayout(T.S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.load(A_sbuf, A)

    @T.function
    def expected(A: T.Tensor((128, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((65536,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(A_sbuf[p_loop, f_loop], A_1[p_loop * 512 + f_loop])

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_simple_copy_2():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 4, 128) : (512, 128, 1)])

    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 4, 128) : (4 @ F, 1 @ F, 1 @ P)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.load(A_sbuf, A)

    @T.function
    def expected(A: T.Tensor((128, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((65536,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 512):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 1, annotations={"nki_dim": "F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop], A_1[b_loop * 128 + p_loop])

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(4, 128, 512) : (512 * 128, 512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(4):
            T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.function
    def expected(A: T.Tensor((512, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((262144,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop]
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_in_a_loop_2():
    src_shape = [512, 512]
    src_layout = T.TileLayout(T.S[(128, 2048) : (2048, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        A_sbuf_view = A_sbuf.view(128, 4, 512)
        A_view = A.view(128, 4, 512)
        for i in range(4):
            T.trn.tile.load(A_sbuf_view[:, i, :], A_view[:, i, :])

    @T.function
    def expected(A: T.Tensor((512, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        _A_flat = T.decl_tensor(
            A.data, ty_args=[T.Tensor((262144,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        A_sbuf_view = T.decl_tensor(
            A_sbuf.data, ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf", layout=None)]
        )
        A_view = T.decl_tensor(
            _A_flat.data, ty_args=[T.Tensor((262144,), "float32", layout=None, data=_A_flat.data)]
        )
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf_view[p_loop, i * 512 + f_loop],
                        A_view[p_loop * 2048 + i * 512 + f_loop],
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.tensor_copy(B_sbuf, A_sbuf)

    @T.function
    def expected():
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 256), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 256), "float32", scope="trn.sbuf")])
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
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_different_shape():
    src_shape = [512, 64]
    src_layout = TileLayout(S[(4, 128, 4, 4, 4) : (64 @ F, 1 @ P, 16 @ F, 4 @ F, 1 @ F)])
    dst_shape = [4, 128, 4]
    dst_layout = TileLayout(S[(4, 128, 4) : (4 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        B_sbuf_view = B_sbuf.view(512, 4)
        T.trn.tile.tensor_copy(B_sbuf_view, A_sbuf[:, 0:4])

    @T.function
    def expected():
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 256), "float32", scope="trn.sbuf")])
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 16), "float32", scope="trn.sbuf")])
        B_sbuf_view = T.decl_tensor(
            B_sbuf.data, ty_args=[T.Tensor((128, 16), "float32", scope="trn.sbuf", layout=None)]
        )
        for b_loop in T.serial(0, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 4, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(
                        B_sbuf_view[p_loop, b_loop * 4 + f_loop],
                        A_sbuf[p_loop, b_loop * 64 + f_loop],
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_irregular_shape():
    src_shape = [128, 10000]
    src_layout = TileLayout(S[(128, 10000) : (10000, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(4):
            T.trn.tile.store(A[:, i * 512 : i * 512 + 512], A_sbuf)

    @T.function
    def expected(A: T.Tensor((128, 10000), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((1280000,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.store(A_1[p_loop * 10000 + i * 512 + f_loop], A_sbuf[p_loop, f_loop])

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(32):
            T.trn.tile.load(A_sbuf, A[i, :, :])

    @T.function
    def expected(A: T.Tensor((32, 128, 512), "float32", layout=src_layout)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((2097152,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(32, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, f_loop], A_1[i * 65536 + p_loop * 128 + f_loop])
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_offset():
    src_shape = [256, 512]
    src_layout = TileLayout(S[(256, 512) : (512, 1)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(2):
            T.trn.tile.load(A_sbuf[i * 256 : i * 256 + 256, :], A)

    @T.function
    def expected(A: T.Tensor((256, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((131072,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(2, 2):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf[p_loop, i * 1024 + b_loop * 512 + f_loop],
                        A_1[b_loop * 65536 + p_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_large_dma_copy():
    src_shape = [512, 4096]
    src_layout = T.TileLayout(T.S[(4, 128, 4096) : (4096 * 128, 4096, 1)])
    dst_shape = [512, 4096]
    dst_layout = TileLayout(S[(4, 128, 4096) : (4096 @ F, 1 @ P, 1 @ F)])

    @T.function
    def copy(A: T.Tensor(src_shape, "float32", layout=src_layout)) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(4):
            T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, :], A[i * 128 : i * 128 + 128, :])

    @T.function
    def expected(A: T.Tensor((512, 4096), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((2097152,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 16384), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 4096, annotations={"nki_dim": "F"}):
                    T.nki.load(
                        A_sbuf[p_loop, i * 4096 + f_loop],
                        A_1[i * 524288 + p_loop * 4096 + f_loop],
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_inst_size_limit():
    src_shape = [512, 4096]
    src_layout = dst_layout = TileLayout(S[(4, 128, 4096) : (4096 @ F, 1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout

    @T.function
    def copy(A_ptr: T.handle) -> None:
        T.device_entry()
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)]
        )
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for i in range(4):
            T.trn.tile.tensor_copy(
                A_sbuf[i * 128 : i * 128 + 128, :], B_sbuf[i * 128 : i * 128 + 128, :]
            )

    @T.function
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        B_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 16384), "float32", scope="trn.sbuf")])
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 16384), "float32", scope="trn.sbuf")])
        for i, b_loop in T.grid(4, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(0, 512, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(
                        A_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                        B_sbuf[p_loop, i * 4096 + b_loop * 512 + f_loop],
                    )

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)]
        )
        T.trn.tile.load(A_sbuf[1, 0:2048, 0:1024], A[2048: 4096, 3072:4096])

    @T.function
    def expected(A: T.Tensor((4096, 4096), "float32", layout=A_layout)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((16777216,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 32768), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop * 2048 + f_loop + 16384], A_1[b_loop * 524288 + p_loop * 4096 + f_loop + 12584960])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(A_sbuf_shape, "float32", scope="trn.sbuf", layout=A_sbuf_layout)]
        )
        T.trn.tile.load(A_sbuf[2048: 4096, 3072:4096], A[1, 0:2048, 0:1024])

    @T.function
    def expected(A: T.Tensor((2, 2048, 1024), "float32", layout=A_layout)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((4194304,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 131072), "float32", scope="trn.sbuf")])
        for b_loop in T.serial(0, 8):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 2048, annotations={"nki_dim":"F"}):
                    T.nki.load(A_sbuf[p_loop, b_loop * 4096 + f_loop + 100352], A_1[b_loop * 262144 + p_loop * 2048 + f_loop + 2097152])  # noqa: E501
            # fmt: on

    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for j in range(4):
            for i in range(4):
                T.trn.tile.load(A_sbuf[i * 128 : i * 128 + 128, 0:128*j], A[i * 128 : i * 128 + 128, 0:128*j])  # noqa: E501

    @T.function
    def expected(A: T.Tensor((512, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((262144,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for j, i, b_loop in T.grid(4, 4, 1):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 384, annotations={"nki_dim":"F"}):
                    if f_loop < j * 128:
                        T.nki.load(A_sbuf[p_loop, i * 512 + f_loop], A_1[i * 65536 + p_loop * 512 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
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
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        for j in range(4):
            for i in range(4):
                T.trn.tile.load(A_sbuf[0:128*j, 0:128*i], A[0:128*j, 0:128*i])

    @T.function
    def expected(A: T.Tensor((512, 512), "float32", layout=None)):
        T.func_attr({"global_symbol": "copy"})

        A_1 = T.decl_tensor(
            A.data, ty_args=[T.Tensor((262144,), "float32", layout=None, data=A.data)]
        )
        A_sbuf = T.alloc_tensor(ty_args=[T.Tensor((128, 2048), "float32", scope="trn.sbuf")])
        for j, i, b_loop in T.grid(4, 4, 3):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(0, 128, annotations={"nki_dim":"P"}):
                for f_loop in T.serial(0, 384, annotations={"nki_dim":"F"}):
                    if b_loop - j < 0 and f_loop < i * 128:
                        T.nki.load(A_sbuf[p_loop, b_loop * 512 + f_loop], A_1[b_loop * 65536 + p_loop * 512 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        assert_structural_equal(mod["main"], expected)


def test_copy_with_specified_max_inst_size():
    src_shape = [128, 512]
    src_layout = TileLayout(
        S[
            (
                128,
                512,
            ) : (
                1 @ P,
                1 @ F,
            )
        ]
    )
    dst_shape = src_shape
    dst_layout = src_layout

    # fmt: off
    @T.function
    def copy(A_ptr: T.handle) -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
        )
        T.trn.tile.tensor_copy(A_sbuf, B_sbuf, max_inst_size=128)

    @T.function
    def expected(A_ptr: T.handle):
        T.func_attr({"global_symbol": "copy"})
        A_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf", layout=None)]
        )
        B_sbuf = T.alloc_tensor(
            ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf", layout=None)]
        )
        for b_loop in T.serial(0, 4):
            T.nki.tensorized_instruction()
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(128, annotations={"nki_dim": "F"}):
                    T.nki.tensor_copy(A_sbuf[p_loop, b_loop * 128 + f_loop], B_sbuf[p_loop, b_loop * 128 + f_loop])  # noqa: E501
            # fmt: on
    with target:
        mod = tvm_tirx.IRModule({"main": copy})
        mod = tvm_tirx.backend.trn.transform.LowerTIRx()(mod)
        assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()


@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("copy_to_sbuf", [False, True])
def test_explicit_transpose_prepares_identity_and_psum(columns, copy_to_sbuf):
    """The caller owns the transpose algorithm, including its temporary tensors."""
    layout = TileLayout(S[(128, columns) : (1 @ P, 1 @ F)])
    square = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)])

    @T.function
    def transpose():
        T.device_entry()
        source = T.alloc_tensor(
            ty_args=[T.Tensor((128, columns), "float32", scope="trn.sbuf", layout=layout)]
        )
        identity = T.alloc_tensor(
            ty_args=[T.Tensor((128, 128), "float32", scope="trn.sbuf", layout=square)]
        )
        accumulator = T.alloc_tensor(
            (0, 0), ty_args=[T.Tensor((128, 128), "float32", scope="trn.psum", layout=square)]
        )
        result = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (columns, 128),
                    "float32",
                    scope="trn.sbuf",
                    layout=TileLayout(S[(columns // 128, 128, 128) : (128 @ F, 1 @ P, 1 @ F)]),
                )
            ]
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

    mod = tvm_tirx.IRModule({"main": transpose})
    with target:
        allocated = tvm_tirx.tirx.trn.transform.TrnPrivateTensorAlloc()(mod)
        # No hidden identity or PSUM workspace is needed by either instruction.
        _assert_structural_equal(mod, allocated)
        lowered = tvm_tirx.backend.trn.transform.LowerTIRx()(allocated)
    names = []
    tvm_ffi.structural_walk(
        lowered,
        (
            tvm_tirx.ir.Call,
            lambda n: names.append(n.op.name) if isinstance(n.op, tvm_tirx.ir.Op) else None,
        ),
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
            ty_args=[
                T.Tensor(
                    (128, 128),
                    "float32",
                    scope="trn.sbuf",
                    layout=TileLayout(S[(128, 128) : (1 @ P, 1 @ F)]),
                )
            ]
        )
        b = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (128, 128),
                    "float32",
                    scope="trn.sbuf",
                    layout=TileLayout(S[(128, 128) : (1 @ F, 1 @ P)]),
                )
            ]
        )
        T.trn.tile.tensor_copy(b, a)

    with target, pytest.raises(RuntimeError, match="identity|transpose"):
        tvm_tirx.backend.trn.transform.LowerTIRx()(tvm_tirx.IRModule({"main": copy}))
