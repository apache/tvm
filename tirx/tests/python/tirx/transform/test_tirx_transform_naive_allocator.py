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
from tvm.backend.trn.axis import F, P
from tvm.backend.trn.transform import TrnNaiveAllocator
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.tirx.layout import S, TileLayout


def test_one_alloc():
    src_shape = [128, 512]
    src_layout = TileLayout(S[(128, 512) : (512, 1)])
    dst_shape = [128, 512]
    dst_layout = TileLayout(S[(128, 512) : (1 @ P, 1 @ F)])

    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:

            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)]
            )
            T.trn.tile.load(A_sbuf, A)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A: T.Tensor(src_shape, 'float32', layout=src_layout)) -> None:
            T.func_attr({"global_symbol": "copy"})

            T.device_entry()
            A_sbuf = T.alloc_tensor((0,), ty_args=[T.Tensor(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout)])  # noqa: E501
            T.trn.tile.load(A_sbuf, A)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


def test_two_alloc():
    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [256, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [512, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            T.trn.tile.tensor_copy(B_sbuf[0:256, :], A_sbuf)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.func_attr({"global_symbol": "copy"})
            T.device_entry()
            A_sbuf = T.alloc_tensor([0], ty_args=[T.Tensor([256, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_sbuf = T.alloc_tensor([2*512*4], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            T.trn.tile.tensor_copy(B_sbuf[0:256, :], A_sbuf)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


def test_existing_alloc():
    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [256, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_sbuf = T.alloc_tensor([1], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            T.trn.tile.tensor_copy(B_sbuf[0:256, :], A_sbuf)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.func_attr({"global_symbol": "copy"})
            T.device_entry()
            A_sbuf = T.alloc_tensor([4*512*4+1], ty_args=[T.Tensor([256, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_sbuf = T.alloc_tensor([1], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            T.trn.tile.tensor_copy(B_sbuf[0:256, :], A_sbuf)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


def test_workspace():
    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [256, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [512, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            C_sbuf = T.alloc_tensor(ty_args=[T.Tensor([128, 1024], "float32", scope="trn.sbuf")])
            T.trn.tile.activation(B_sbuf[0:256, :], A_sbuf, const_bias=C_sbuf)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.func_attr({"global_symbol": "copy"})
            T.device_entry()
            A_sbuf = T.alloc_tensor([0], ty_args=[T.Tensor([256, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_sbuf = T.alloc_tensor([2*512*4], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            C_sbuf = T.alloc_tensor([2*512*4+4*512*4], ty_args=[T.Tensor([128, 1024], "float32", scope="trn.sbuf")])  # noqa: E501
            T.trn.tile.activation(B_sbuf[0:256, :], A_sbuf, const_bias=C_sbuf)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


def test_other_scope_alloc():
    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [256, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [512, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            C_sbuf = T.alloc_tensor(ty_args=[T.Tensor([8, 128, 512], "float32", scope="global")])
            T.trn.tile.activation(B_sbuf[0:256, :], A_sbuf, const_bias=C_sbuf)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.func_attr({"global_symbol": "copy"})
            T.device_entry()
            A_sbuf = T.alloc_tensor([0], ty_args=[T.Tensor([256, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_sbuf = T.alloc_tensor([2*512*4], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            C_sbuf = T.alloc_tensor(ty_args=[T.Tensor([8, 128, 512], "float32", scope="global")])
            T.trn.tile.activation(B_sbuf[0:256, :], A_sbuf, const_bias=C_sbuf)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


def test_tensor_views():
    # fmt: off
    @I.ir_module
    class Before:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.device_entry()
            A_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [256, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_sbuf = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [512, 512],
                        "float32",
                        scope="trn.sbuf",
                        layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]),
                    )
                ],
            )
            B_view = B_sbuf.view(2, 256, 512)
            T.trn.tile.tensor_copy(B_view[0], A_sbuf)

    @I.ir_module
    class Expected:
        @T.function
        def copy(A_ptr: T.handle) -> None:
            T.func_attr({"global_symbol": "copy"})
            T.device_entry()
            A_sbuf = T.alloc_tensor([0], ty_args=[T.Tensor([256, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(2, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_sbuf = T.alloc_tensor([2*512*4], ty_args=[T.Tensor([512, 512], "float32", scope="trn.sbuf", layout=TileLayout(S[(4, 128, 512) : (512 @ F, 1 @ P, 1 @ F)]))])  # noqa: E501
            B_view = B_sbuf.view(2, 256, 512)
            T.trn.tile.tensor_copy(B_view[0], A_sbuf)
        # fmt: on

    After = TrnNaiveAllocator()(Before)
    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
