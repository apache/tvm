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
from tvm.script import tirx as T
from tvm.tirx.layout import F, P, S, TileLayout
from tvm.tirx.trn.transform import TrnPrivateBufferAlloc

target = tvm.target.Target("aws/trn1/trn1.2xlarge")


def test_normal_copy():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @T.function
    def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:

        T.device_entry()
        A_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.load(A_sbuf, A)
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
    bias = T.float32(1.0)
    scale = T.float32(2.0)

    # fmt: off
    @T.function
    def unary() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        C_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.activation(C_sbuf, A_sbuf, bias=bias, scale=scale, opcode='exp')

    @T.function
    def expected():
        T.func_attr({"global_symbol": "unary"})
        T.device_entry()
        const_bias = T.alloc_tensor((128, 512), scope="trn.sbuf")
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(512, annotations={"nki_dim": "F"}):
                    T.nki.memset(const_bias[p_loop, f_loop], T.float32(1.0))
        A_sbuf = T.alloc_tensor((512, 1024), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096) : (1@P, 1@F)]))
        C_sbuf = T.alloc_tensor((512, 1024), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096) : (1@P, 1@F)]))
        T.trn.tile.activation(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], scale=T.float32(2.0), bias=T.float32(1.0), const_bias=const_bias, opcode='exp')  # noqa: E501
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
    @T.function
    def reduction():
        T.device_entry()
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.tensorreduce(B_sbuf, A_sbuf, axes=(1, 3), reduce_op='sum')

    @T.function
    def expected():
        T.func_attr({"global_symbol": "reduction"})
        T.device_entry()
        partial_reduce = T.alloc_tensor((128, 32), scope="trn.sbuf")
        A_sbuf = T.alloc_tensor((128, 32, 4, 32), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 32 * 32 * 4) : (1@P, 1@F)]))
        B_sbuf = T.alloc_tensor((128, 4), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4) : (1@P, 1@F)]))
        T.trn.tile.tensorreduce(B_sbuf[0:128, 0:4], A_sbuf[0:128, 0:32, 0:4, 0:32], partial_reduce=partial_reduce, axes=[1, 3], reduce_op='sum')  # noqa: E501

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
    @T.function
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor((512, 1024), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_tensor((1024, 256), "float32", scope="trn.sbuf", layout=B_layout)
        C_sbuf = T.alloc_tensor((512, 256), "float32", scope="trn.sbuf", layout=C_layout)
        for i in range(2):
            for k in range(2):
                T.trn.tile.matmul(
                    C_sbuf[256 * i : 256 * i + 256, :],
                    A_sbuf[256 * i : 256 * i + 256, 512 * k : 512 * k + 512],
                    B_sbuf[512 * k : 512 * k + 512, :],
                    C_sbuf[256 * i : 256 * i + 256, :],
                )

    @T.function
    def expected():
        T.func_attr({"global_symbol": "gemm"})
        T.device_entry()
        acc_psum = T.alloc_tensor((8, 128, 512), scope="trn.psum", allocated_addr=[0, 0])
        A_sbuf = T.alloc_tensor((512, 1024), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(4, 128, 8, 128) : (1024@F, 1@F, 1@F, 1@P)]))  # noqa: E501
        B_sbuf = T.alloc_tensor((1024, 256), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(8, 128, 2, 128) : (256@F, 1@P, 128@F, 1@F)]))  # noqa: E501
        C_sbuf = T.alloc_tensor((512, 256), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(4, 128, 2, 128) : (256@F, 1@F, 128@F, 1@P)]))  # noqa: E501
        for i, k in T.grid(2, 2):
            T.trn.tile.matmul(C_sbuf[256 * i:256 * i + 256, 0:256], A_sbuf[256 * i:256 * i + 256, 512 * k:512 * k + 512], B_sbuf[512 * k:512 * k + 512, 0:256], C_sbuf[256 * i:256 * i + 256, 0:256], False, False, T.float32(1.0), T.float32(0.0), acc_psum=acc_psum)  # noqa: E501
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
    @T.function
    def tensor_scalar_reduce() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(src1_shape, "float32", scope="trn.sbuf", layout=src1_layout)
        B_sbuf = T.alloc_tensor(dst1_shape, "float32", scope="trn.sbuf", layout=dst1_layout)
        C_sbuf = T.alloc_tensor(reduce_dst_shape, "float32", scope="trn.sbuf", layout=reduce_dst_layout)  # noqa: E501
        T.trn.tile.tensorscalar_reduce(
            B_sbuf, C_sbuf, A_sbuf, 1.0, opcode="add", reduce_op="sum", axes=(1, 2)
        )

    @T.function
    def expected():
        T.func_attr({"global_symbol": "tensor_scalar_reduce"})
        T.device_entry()
        partial_reduce = T.alloc_tensor((128, 4), scope="trn.sbuf")
        A_sbuf = T.alloc_tensor((512, 1024, 4), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096, 4) : (1 @ P, 1 @ F, 4096 @ F)]))
        B_sbuf = T.alloc_tensor((512, 1024, 4), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096, 4) : (1 @ P, 1 @ F, 4096 @ F)]))
        C_sbuf = T.alloc_tensor((512,), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4) : (1 @ P, 1 @ F)]))
        T.trn.tile.tensorscalar_reduce(B_sbuf[0:512, 0:1024, 0:4], C_sbuf[0:512], A_sbuf[0:512, 0:1024, 0:4], T.float32(1.0), partial_reduce=partial_reduce, opcode="add", reduce_op="sum", axes=[1, 2])  # noqa: E501
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
    @T.function
    def activation_reduce():
        T.device_entry()
        A = T.alloc_tensor(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
        B = T.alloc_tensor(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
        C = T.alloc_tensor(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
        for i in range(2):
            T.trn.tile.activation_reduce(
                B, C, A[i * 16 : i * 16 + 16], opcode="sqrt", reduce_op="sum", axes=(0, 1)
            )

    @T.function
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        T.device_entry()
        partial_reduce = T.alloc_tensor((128, 8), scope="trn.sbuf")
        const_bias = T.alloc_tensor((128, 1024), scope="trn.sbuf")
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                    T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
        A = T.alloc_tensor((32, 512, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(16 * 1024, 128) : (1@F, 1@P)]))
        B = T.alloc_tensor((16, 512, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(2, 4, 1024, 128) : (1024@F, 2048@F, 1@F, 1@P)]))
        C = T.alloc_tensor((1, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(1, 128) : (1@F, 1@P)]))
        for i in range(2):
            T.trn.tile.activation_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], const_bias=const_bias, partial_reduce=partial_reduce, opcode="sqrt", reduce_op="sum", axes=[0, 1])  # noqa: E501
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
    @T.function
    def activation_reduce():
        T.device_entry()
        partial_reduce = T.alloc_tensor((128, 16), scope="trn.sbuf")
        A = T.alloc_tensor(A_shape, dtype="float32", scope="trn.sbuf", layout=A_layout)
        B = T.alloc_tensor(B_shape, dtype="float32", scope="trn.sbuf", layout=B_layout)
        C = T.alloc_tensor(C_shape, dtype="float32", scope="trn.sbuf", layout=C_layout)
        for i in range(2):
            T.trn.tile.activation_reduce(B, C, A[i*16:i*16+16], partial_reduce=partial_reduce, opcode="sqrt", reduce_op="sum", axes=(0,1))  # noqa: E501

    @T.function
    def expected():
        T.func_attr({"global_symbol": "activation_reduce"})
        T.device_entry()
        const_bias = T.alloc_tensor((128, 1024), scope="trn.sbuf")
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                    T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
        partial_reduce = T.alloc_tensor((128, 16), scope="trn.sbuf")
        A = T.alloc_tensor((32, 512, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(16 * 1024, 128) : (1@F, 1@P)]))
        B = T.alloc_tensor((16, 512, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(2, 4, 1024, 128) : (1024@F, 2048@F, 1@F, 1@P)]))
        C = T.alloc_tensor((1, 128), scope="trn.sbuf",
                           layout=T.TileLayout(T.S[(1, 128) : (1@F, 1@P)]))
        for i in range(2):
            T.trn.tile.activation_reduce(B[0:16, 0:512, 0:128], C[0, 0:128], A[i * 16:i * 16 + 16, 0:512, 0:128], const_bias=const_bias, partial_reduce=partial_reduce, opcode="sqrt", reduce_op="sum", axes=[0, 1])  # noqa: E501
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
    scale = T.float32(2.0)

    # fmt: off
    @T.function
    def unary() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        C_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.activation(
            C_sbuf, A_sbuf, bias=0.0, scale=scale, max_inst_size=1024, opcode="exp"
        )
        T.trn.tile.activation(C_sbuf, C_sbuf, opcode='exp')

    @T.function
    def expected():
        T.func_attr({"global_symbol": "unary"})
        T.device_entry()
        const_bias = T.alloc_tensor((128, 1024), scope="trn.sbuf")
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(128, annotations={"nki_dim": "P"}):
                for f_loop in T.serial(1024, annotations={"nki_dim": "F"}):
                    T.nki.memset(const_bias[p_loop, f_loop], T.float32(0.0))
        A_sbuf = T.alloc_tensor((512, 1024), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096) : (1 @ P, 1 @ F)]))
        C_sbuf = T.alloc_tensor((512, 1024), scope="trn.sbuf",
                                layout=T.TileLayout(T.S[(128, 4096) : (1 @ P, 1 @ F)]))
        T.trn.tile.activation(C_sbuf[0:512, 0:1024], A_sbuf[0:512, 0:1024], scale=T.float32(2.0), bias=T.float32(0.0), max_inst_size=1024, const_bias=const_bias, opcode='exp')  # noqa: E501
        T.trn.tile.activation(
            C_sbuf[0:512, 0:1024], C_sbuf[0:512, 0:1024], const_bias=const_bias, opcode="exp"
        )

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
    @T.function
    def reduction():
        T.device_entry()
        intermediate_buffer = T.alloc_tensor((128, 64), scope="trn.sbuf")
        A_sbuf = T.alloc_tensor(src_shape, "float32", scope="trn.sbuf", layout=src_layout)
        B_sbuf = T.alloc_tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)
        T.trn.tile.tensorreduce(
            B_sbuf, A_sbuf, axes=(1, 3), partial_reduce=intermediate_buffer, reduce_op="sum"
        )
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
    @T.function
    def gemm() -> None:
        T.device_entry()
        A_sbuf = T.alloc_tensor((128, 128), "float32", scope="trn.sbuf", layout=A_layout)
        B_sbuf = T.alloc_tensor((128, 128), "float32", scope="trn.sbuf", layout=B_layout)
        C_psum = T.alloc_tensor((128, 128), "float32", scope="trn.psum", layout=C_layout)
        T.trn.tile.matmul(C_psum, A_sbuf, B_sbuf, C_psum)
        # fmt: on
    with target:
        mod = tvm.IRModule({"main": gemm})
        mod = TrnPrivateBufferAlloc()(mod)
        assert_structural_equal(mod["main"], gemm)


if __name__ == "__main__":
    tvm.testing.main()
