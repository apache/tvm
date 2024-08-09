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
# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest

import tvm
import tvm.testing
from tvm.script import tir as T


alignment_required = tvm.testing.parameter(1, 32, 64)
offset = tvm.testing.parameter(0, 3, 32, 64)


def test_aligned_dltensor(alignment_required, offset):
    """Alignment of buffer arguments checked during DLTensor unpacking

    TVM allocates buffers that are aligned according to the value of
    `tvm::runtime::kAllocAlignment`.  However, buffers may be
    non-aligned, either when provided by an external source, or when
    the TVM runtime used for compilation and for execution have a
    different value of `tvm::runtime::kAllocAlignment` through the
    `TVM_KALLOC_ALIGNMENT` macro definition.  In addition, while
    `tvm::runtime::kAllocAlignment` is the default alignment for TIR
    buffers, it can be overridden on a per-buffer basis.

    This test varies the alignment required by a buffer argument and
    the alignment provided by an externally-owned array, validating
    that non-aligned buffers may always be converted to TVM, and must
    have their alignment validated when calling a PrimFunc.
    """
    torch = pytest.importorskip("torch")

    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, 16, dtype="int8", align=alignment_required)
        T.evaluate(0)

    built = tvm.build(func)

    torch_tensor = torch.arange(128, dtype=torch.int8)
    torch_view = torch_tensor[offset : offset + 16]
    tvm_array = tvm.nd.from_dlpack(torch_view)

    satisfies_alignment = offset % alignment_required == 0
    if satisfies_alignment:
        built(tvm_array)
    else:
        with pytest.raises(tvm.TVMError):
            built(tvm_array)


contiguity_test_case = tvm.testing.parameter(
    by_dict={
        "entire_first_row": ([4, 16], [0, 0]),
        "entire_second_row": ([4, 16], [1, 0]),
        "left_half_of_first_row": ([4, 32], [0, 0]),
        "right_half_of_first_row": ([4, 32], [0, 16]),
    }
)


def test_contiguous_dltensor(contiguity_test_case):
    """Validate argument buffer is compact when strides are unspecified."""
    torch = pytest.importorskip("torch")

    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, [1, 16], dtype="int8", align=1)
        T.evaluate(0)

    built = tvm.build(func)

    view_backing_shape, view_offset = contiguity_test_case
    torch_tensor = torch.zeros(*view_backing_shape, dtype=torch.int8)
    torch_view = torch_tensor[
        view_offset[0] : view_offset[0] + 1,
        view_offset[1] : view_offset[1] + 16,
    ]
    tvm_array = tvm.nd.from_dlpack(torch_view)

    built(tvm_array)


strided_test_case = tvm.testing.parameter(
    by_dict={
        "entire_buffer": (8, 16),
        "split_in_slowest_changing_dim": (32, 16),
        "split_in_fastest_changing_dim": (8, 64),
    }
)


def test_dynamic_striding_on_external_dltensor(strided_test_case):
    """External buffers may be strided.

    Validity is checked by the TIR unpacking of the DLTensor, based on
    the requirements of the TIR buffer.
    """
    torch = pytest.importorskip("torch")

    @T.prim_func
    def func(a: T.handle):
        stride_i = T.var("int32")
        stride_j = T.var("int32")
        A = T.match_buffer(a, [8, 16], strides=[stride_i, stride_j], dtype="int8", align=1)
        T.evaluate(0)

    built = tvm.build(func)

    torch_tensor = torch.zeros(*strided_test_case, dtype=torch.int8)
    torch_view = torch_tensor[:8, :16]
    tvm_array = tvm.nd.from_dlpack(torch_view)

    built(tvm_array)


def test_static_striding_on_external_dltensor(strided_test_case):
    """External buffers may be strided.

    Import of strided arrays from external sources is legal.  The
    validity for any given PrimFunc is checked by the TIR unpacking of
    the DLTensor, based on the requirements of the TIR buffer.
    """
    torch = pytest.importorskip("torch")

    @T.prim_func
    def func(a: T.handle):
        A = T.match_buffer(a, [8, 16], dtype="int8", align=1)
        T.evaluate(0)

    built = tvm.build(func)

    torch_tensor = torch.zeros(*strided_test_case, dtype=torch.int8)
    torch_view = torch_tensor[:8, :16]
    tvm_array = tvm.nd.from_dlpack(torch_view)

    has_correct_striding = strided_test_case[1] == 16
    if has_correct_striding:
        built(tvm_array)
    else:
        with pytest.raises(tvm.TVMError):
            built(tvm_array)


if __name__ == "__main__":
    tvm.testing.main()
