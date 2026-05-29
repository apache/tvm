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
from tvm.ir import assert_structural_equal
from tvm.script import tirx as Tx
from tvm.tirx.layout import F, P, S, TileLayout
from tvm.tirx.transform.trn import TrnPrivateBufferAlloc

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def test_copy_transpose():
    src_shape = [512, 512]
    src_layout = TileLayout(S[(128, 2048) : (1 @ P, 1 @ F)])
    dst_shape = [512, 512]
    dst_layout = TileLayout(S[(2048, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def copy() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.copy(B_sbuf, A_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            identity = Tx.alloc_buffer((128, 128), scope="trn.sbuf")
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in Tx.serial(128, annotations={"nki_dim": "F"}):
                        Tx.nki.identity(identity[p_loop, rhs_f_loop], 128)
            A_sbuf = Tx.alloc_buffer((512, 512), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 2048) : (1 @ P, 1@F)]))
            B_sbuf = Tx.alloc_buffer((512, 512), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(2048, 128) : (1@F, 1@P)]))
            Tx.copy(B_sbuf[0:512, 0:512], A_sbuf[0:512, 0:512], workspace={"acc_psum": acc_psum, "identity": identity})  # noqa: E501

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_normal_copy():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.copy(A_sbuf, A)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": copy})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], copy)


def test_unary_with_bias_scale():
    src_shape = [512, 1024]
    src_layout = TileLayout(S[(128, 4096) : (1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout
    bias = Tx.float32(1.0)
    scale = Tx.float32(2.0)

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.exp(C_sbuf, A_sbuf, bias=bias, scale=scale)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})
        with Tx.kernel():
            const_bias = Tx.alloc_buffer((128, 512), scope="trn.sbuf")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(512, annotations={"nki_dim": "F"}):
                        Tx.nki.memset(const_bias[p_loop, f_loop], Tx.float32(1.0))
            A_sbuf = Tx.alloc_buffer((512, 1024), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096) : (1@P, 1@F)]))
            C_sbuf = Tx.alloc_buffer((512, 1024), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096) : (1@P, 1@F)]))
            Tx.exp(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], Tx.float32(1.0), Tx.float32(2.0), workspace={"const_bias": const_bias})  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": unary})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_reduction_two_stage():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(S[(128, 32 * 32 * 4) : (1 @ P, 1 @ F)])
    dst_shape = [128, 4]
    dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def reduction():
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.sum(B_sbuf, A_sbuf, axes=(1, 3))

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "reduction"})
        with Tx.kernel():
            partial_reduce = Tx.alloc_buffer((128, 32), scope="trn.sbuf")
            A_sbuf = Tx.alloc_buffer((128, 32, 4, 32), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 32 * 32 * 4) : (1@P, 1@F)]))
            B_sbuf = Tx.alloc_buffer((128, 4), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4) : (1@P, 1@F)]))
            Tx.sum(B_sbuf[0:128, 0:4], A_sbuf[0:128, 0:32, 0:4, 0:32], [1, 3], False, workspace={"partial_reduce": partial_reduce})  # noqa: E501

    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_gemm():
    A_layout = TileLayout(S[(4, 128, 8, 128) : (1024 @ F, 1 @ F, 1 @ F, 1 @ P)])
    B_layout = TileLayout(S[(8, 128, 2, 128) : (256 @ F, 1 @ P, 128 @ F, 1 @ F)])

    C_layout = TileLayout(S[(4, 128, 2, 128) : (256 @ F, 1 @ F, 128 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def gemm() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = Tx.alloc_buffer((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
            C_sbuf = Tx.alloc_buffer((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                for k in range(2):
                    Tx.gemm(
                        C_sbuf[256 * i : 256 * i + 256, :],
                        A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                        B_sbuf[512 * k : 512 * k + 512, :],
                        C_sbuf[256 * i : 256 * i + 256, :],
                    )
    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "gemm"})
        with Tx.kernel():
            acc_psum = Tx.alloc_buffer((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
            A_sbuf = Tx.alloc_buffer((512, 1024), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(4, 128, 8, 128) : (1024@F, 1@F, 1@F, 1@P)]))  # noqa: E501
            B_sbuf = Tx.alloc_buffer((1024, 256), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(8, 128, 2, 128) : (256@F, 1@P, 128@F, 1@F)]))  # noqa: E501
            C_sbuf = Tx.alloc_buffer((512, 256), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(4, 128, 2, 128) : (256@F, 1@F, 128@F, 1@P)]))  # noqa: E501
            for i, k in Tx.grid(2, 2):
                Tx.gemm(C_sbuf[256 * i:256 * i + 256, 0:256], A_sbuf[256 * i:256 * i + 256, 512 * k:512 * k + 512], B_sbuf[512 * k:512 * k + 512, 0:256], C_sbuf[256 * i:256 * i + 256, 0:256], False, False, Tx.float32(1.0), Tx.float32(0.0), workspace={"acc_psum": acc_psum})  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_binary_reduce_two_stage():
    src1_shape = [512, 1024, 4]
    src1_layout = TileLayout(S[(128, 4096, 4) : (1 @ P, 1 @ F, 4096 @ F)])
    dst1_shape = src1_shape
    dst1_layout = src1_layout
    reduce_dst_shape = [512]
    reduce_dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def tensor_scalar_reduce() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
            B_sbuf = Tx.alloc_buffer(dst1_shape, "float32", scope="trn.sbuf", layout=dst1_layout)
            C_sbuf = Tx.alloc_buffer(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)  # noqa: E501
            Tx.binary_reduce(B_sbuf, C_sbuf, A_sbuf, 1.0, "add", "sum", reduce_axes=(1, 2))

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "tensor_scalar_reduce"})
        with Tx.kernel():
            partial_reduce = Tx.alloc_buffer((128, 4), scope="trn.sbuf")
            A_sbuf = Tx.alloc_buffer((512, 1024, 4), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096, 4) : (1 @ P, 1 @ F, 4096 @ F)]))  # noqa: E501
            B_sbuf = Tx.alloc_buffer((512, 1024, 4), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096, 4) : (1 @ P, 1 @ F, 4096 @ F)]))  # noqa: E501
            C_sbuf = Tx.alloc_buffer((512,), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4) : (1 @ P, 1 @ F)]))
            Tx.binary_reduce(B_sbuf[0:512, 0:1024, 0:4], C_sbuf[0:512], A_sbuf[0:512, 0:1024, 0:4], Tx.float32(1.0), "add", "sum", [1, 2], workspace={"partial_reduce": partial_reduce})  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": tensor_scalar_reduce})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_activation_reduce_two_stage():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(S[(16 * 1024, 128) : (1 @ F, 1 @ P)])
    B_shape = (16, 512, 128)
    B_layout = TileLayout(S[(2, 4, 1024, 128) : (1024 @ F, 2048 @ F, 1 @ F, 1 @ P)])
    C_shape = (1, 128)
    C_layout = TileLayout(S[(1, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def activation_reduce():
        with Tx.kernel():
            A = Tx.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = Tx.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = Tx.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tx.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=(0,1))

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "activation_reduce"})
        with Tx.kernel():
            partial_reduce = Tx.alloc_buffer((128, 8), scope="trn.sbuf")
            const_bias = Tx.alloc_buffer((128, 1024), scope="trn.sbuf")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(1024, annotations={"nki_dim": "F"}):
                        Tx.nki.memset(const_bias[p_loop, f_loop], Tx.float32(0.0))
            A = Tx.alloc_buffer((32, 512, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(16 * 1024, 128) : (1@F, 1@P)]))
            B = Tx.alloc_buffer((16, 512, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(2, 4, 1024, 128) : (1024@F, 2048@F, 1@F, 1@P)]))  # noqa: E501
            C = Tx.alloc_buffer((1, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(1, 128) : (1@F, 1@P)]))
            for i in range(2):
                Tx.unary_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], "sqrt", "sum", None, None, [0, 1], workspace={"const_bias": const_bias, "partial_reduce": partial_reduce})  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_partial_workspace_specify():
    A_shape = (32, 512, 128)
    A_layout = TileLayout(S[(16 * 1024, 128) : (1 @ F, 1 @ P)])
    B_shape = (16, 512, 128)
    B_layout = TileLayout(S[(2, 4, 1024, 128) : (1024 @ F, 2048 @ F, 1 @ F, 1 @ P)])
    C_shape = (1, 128)
    C_layout = TileLayout(S[(1, 128) : (1 @ F, 1 @ P)])

    # fmt: off
    @Tx.prim_func
    def activation_reduce():
        with Tx.kernel():
            partial_reduce = Tx.alloc_buffer((128, 16), scope="trn.sbuf")
            A = Tx.alloc_buffer(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
            B = Tx.alloc_buffer(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
            C = Tx.alloc_buffer(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
            for i in range(2):
                Tx.unary_reduce(B, C, A[i*16:i*16+16], "sqrt", "sum", reduce_axes=(0,1), workspace={"partial_reduce": partial_reduce})  # noqa: E501

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "activation_reduce"})
        with Tx.kernel():
            const_bias = Tx.alloc_buffer((128, 1024), scope="trn.sbuf")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(1024, annotations={"nki_dim": "F"}):
                        Tx.nki.memset(const_bias[p_loop, f_loop], Tx.float32(0.0))
            partial_reduce = Tx.alloc_buffer((128, 16), scope="trn.sbuf")
            A = Tx.alloc_buffer((32, 512, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(16 * 1024, 128) : (1@F, 1@P)]))
            B = Tx.alloc_buffer((16, 512, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(2, 4, 1024, 128) : (1024@F, 2048@F, 1@F, 1@P)]))  # noqa: E501
            C = Tx.alloc_buffer((1, 128), scope="trn.sbuf",
                               layout=Tx.TileLayout(Tx.S[(1, 128) : (1@F, 1@P)]))
            for i in range(2):
                Tx.unary_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], "sqrt", "sum", None, None, [0, 1], workspace={"const_bias": const_bias, "partial_reduce": partial_reduce})  # noqa: E501
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": activation_reduce})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_workspace_reuse():
    src_shape = [512, 1024]
    src_layout = TileLayout(S[(128, 4096) : (1 @ P, 1 @ F)])
    dst_shape = src_shape
    dst_layout = src_layout
    scale = Tx.float32(2.0)

    # fmt: off
    @Tx.prim_func
    def unary() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            C_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.exp(C_sbuf, A_sbuf, bias=0.0, scale=scale, max_inst_size=1024)
            Tx.exp(C_sbuf, C_sbuf)

    @Tx.prim_func
    def expected():
        Tx.func_attr({"global_symbol": "unary"})
        with Tx.kernel():
            const_bias = Tx.alloc_buffer((128, 1024), scope="trn.sbuf")
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(128, annotations={"nki_dim": "P"}):
                    for f_loop in Tx.serial(1024, annotations={"nki_dim": "F"}):
                        Tx.nki.memset(const_bias[p_loop, f_loop], Tx.float32(0.0))
            A_sbuf = Tx.alloc_buffer((512, 1024), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096) : (1 @ P, 1 @ F)]))
            C_sbuf = Tx.alloc_buffer((512, 1024), scope="trn.sbuf",
                                    layout=Tx.TileLayout(Tx.S[(128, 4096) : (1 @ P, 1 @ F)]))
            Tx.exp(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], Tx.float32(0.0), Tx.float32(2.0), workspace={"const_bias": const_bias}, max_inst_size=1024)  # noqa: E501
            Tx.exp(C_sbuf[0:512, 0:1024], C_sbuf[0:512, 0:1024], None, None, workspace={"const_bias": const_bias})  # noqa: E501

    # fmt: on

    with target:
        mod = tvm.IRModule({"main": unary})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], expected)


def test_no_rewrite_with_existing_workspace():
    src_shape = [128, 32, 4, 32]
    src_layout = TileLayout(S[(128, 32 * 32 * 4) : (1 @ P, 1 @ F)])
    dst_shape = [128, 4]
    dst_layout = TileLayout(S[(128, 4) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def reduction():
        with Tx.kernel():
            intermediate_buffer = Tx.alloc_buffer((128, 64), scope="trn.sbuf")
            A_sbuf = Tx.alloc_buffer(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
            B_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
            Tx.sum(B_sbuf, A_sbuf, axes=(1, 3), workspace={"partial_reduce": intermediate_buffer})
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": reduction})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], reduction)


def test_no_rewrite_with_psum_output():
    A_layout = TileLayout(S[(128, 128) : (1 @ F, 1 @ P)])
    B_layout = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)])

    C_layout = TileLayout(S[(128, 128) : (1 @ P, 1 @ F)])

    # fmt: off
    @Tx.prim_func
    def gemm() -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=A_layout)
            B_sbuf = Tx.alloc_buffer((128, 128), "float32", scope="trn.sbuf", layout=B_layout)
            C_psum = Tx.alloc_buffer((128, 128), "float32", scope="trn.psum", layout=C_layout)
            Tx.gemm(C_psum, A_sbuf, B_sbuf, C_psum)
    # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], gemm)


if __name__ == "__main__":
    tvm.testing.main()
