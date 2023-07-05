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
"""Test layout and bijective-layout node"""

import pytest
import tvm
import tvm.error
from tvm.topi.utils import get_const_tuple


def test_layout():
    layout = tvm.tir.layout("NCHW16c")
    assert layout is not None
    assert isinstance(layout, tvm.tir.Layout)

    assert layout.factor_of("c") == 16
    assert layout.factor_of("C") == 16
    assert layout.factor_of("N") == -1

    assert layout.index_of("N") == 0
    assert layout.index_of("C") == 1
    assert layout.index_of("H") == 2
    assert layout.index_of("W") == 3
    assert layout.index_of("c") == 4
    assert layout.index_of("O") == -1

    assert "N" in layout
    assert "C" in layout
    assert "H" in layout
    assert "W" in layout
    assert "c" in layout
    assert "O" not in layout

    assert layout[0] == "N"
    assert layout[1] == "C"
    assert layout[2] == "H"
    assert layout[3] == "W"
    assert layout[4] == "c"
    assert layout[-1] == "c"


def test_layout_dtype():
    layout_i32 = tvm.tir.layout("NCHW")
    assert layout_i32.axes[0].var.dtype == "int32"
    assert layout_i32.axes[0].dom.min.dtype == "int32"
    assert layout_i32.axes[0].dom.extent.dtype == "int32"
    assert layout_i32.axes[1].var.dtype == "int32"
    assert layout_i32.axes[1].dom.min.dtype == "int32"
    assert layout_i32.axes[1].dom.extent.dtype == "int32"

    layout_i64 = tvm.tir.layout("NCHW", dtype="int64")
    assert layout_i64.axes[2].var.dtype == "int64"
    assert layout_i64.axes[2].dom.min.dtype == "int64"
    assert layout_i64.axes[2].dom.extent.dtype == "int64"
    assert layout_i64.axes[3].var.dtype == "int64"
    assert layout_i64.axes[3].dom.min.dtype == "int64"
    assert layout_i64.axes[3].dom.extent.dtype == "int64"

    with pytest.raises(TypeError):
        tvm.tir.layout("NCHW", dtype="float32")
    with pytest.raises(TypeError):
        tvm.tir.layout("NCHW", dtype=None)


def test_bilayout_convertible():
    # not convertible
    assert tvm.tir.bijective_layout("NCHW", "ABCD") is None
    assert tvm.tir.bijective_layout("__undef__", "NCHW") is None
    assert tvm.tir.bijective_layout("NCHW", "__undef__") is None
    assert tvm.tir.bijective_layout("__undef__", "__undef__") is None
    assert tvm.tir.bijective_layout("", "NCHW") is None
    assert tvm.tir.bijective_layout("NCHW", "") is None
    assert tvm.tir.bijective_layout("", "") is None
    # convertible
    assert tvm.tir.bijective_layout("NCHW", "NCHW16c") is not None


def test_bilayout_shape():
    bilayout = tvm.tir.bijective_layout("NCHW", "NCHW16c")
    assert isinstance(bilayout, tvm.tir.BijectiveLayout)

    dst_shape = bilayout.forward_shape((1, 32, 7, 7))
    assert get_const_tuple(dst_shape) == (1, 2, 7, 7, 16)

    src_shape = bilayout.backward_shape(dst_shape)
    assert get_const_tuple(src_shape) == (1, 32, 7, 7)


def test_bilayout_index():
    bilayout = tvm.tir.bijective_layout("NCHW", "NCHW16c")

    dst_index = bilayout.forward_index([0, 18, 6, 6])
    assert get_const_tuple(dst_index) == (0, 1, 6, 6, 2)

    src_index = bilayout.backward_index([0, 1, 6, 6, 2])
    assert get_const_tuple(src_index) == (0, 18, 6, 6)


if __name__ == "__main__":
    test_layout()
    test_layout_dtype()
    test_bilayout_convertible()
    test_bilayout_shape()
    test_bilayout_index()
