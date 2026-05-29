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
from tvm.tirx.transform.trn import TrnNaiveAllocator


def test_one_alloc():
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

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        A = Tx.match_buffer(A_ptr, src_shape, "float32", layout=src_layout)
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer(dst_shape, "float32", scope="trn.sbuf", layout=dst_layout, allocated_addr=[0])  # noqa: E501
            Tx.copy(A_sbuf, A)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_two_alloc():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            Tx.copy(B_sbuf[0:256, :], A_sbuf)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])  # noqa: E501
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])  # noqa: E501
            Tx.copy(B_sbuf[0:256, :], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_existing_alloc():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[1])  # noqa: E501
            Tx.copy(B_sbuf[0:256, :], A_sbuf)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[4*512*4+1])  # noqa: E501
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[1])  # noqa: E501
            Tx.copy(B_sbuf[0:256, :], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_workspace():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            C_sbuf = Tx.alloc_buffer([128, 1024], "float32", scope="trn.sbuf")
            Tx.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])  # noqa: E501
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])  # noqa: E501
            C_sbuf = Tx.alloc_buffer([128, 1024], "float32", scope="trn.sbuf", allocated_addr=[2*512*4+4*512*4])  # noqa: E501
            Tx.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_other_scope_alloc():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            C_sbuf = Tx.alloc_buffer([8, 128, 512], "float32", scope="global")
            Tx.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])  # noqa: E501
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])  # noqa: E501
            C_sbuf = Tx.alloc_buffer([8, 128, 512], "float32", scope="global")
            Tx.copy(B_sbuf[0:256, :], A_sbuf, workspace={"C": C_sbuf})
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


def test_buffer_views():
    # fmt: off
    @Tx.prim_func
    def copy(A_ptr: Tx.handle) -> None:
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF")
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF")
            B_view = B_sbuf.view(2, 256, 512)
            Tx.copy(B_view[0], A_sbuf)

    @Tx.prim_func
    def expected(A_ptr: Tx.handle) -> None:
        Tx.func_attr({"global_symbol": "copy"})
        with Tx.kernel():
            A_sbuf = Tx.alloc_buffer([256, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[0])  # noqa: E501
            B_sbuf = Tx.alloc_buffer([512, 512], "float32", scope="trn.sbuf", layout="PF", allocated_addr=[2*512*4])  # noqa: E501
            B_view = B_sbuf.view(2, 256, 512)
            Tx.copy(B_view[0], A_sbuf)
    # fmt: on

    mod = tvm.IRModule({"copy": copy})
    mod = TrnNaiveAllocator()(mod)
    assert_structural_equal(mod["copy"], expected)


if __name__ == "__main__":
    tvm.testing.main()
