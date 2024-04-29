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

import numpy as np
import pytest


def test_1d_full_view_of_1d_arr():
    """NDArray::CreateView may return the same array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([1024])
    np_expected = np_input

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_1d_view_of_first_half_of_1d_arr():
    """NDArray::CreateView may return a subset of an array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([512])
    np_expected = np_input[0:512]

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_1d_view_of_first_half_of_1d_arr():
    """Subset returned by NDArray::CreateView may have a byte offset"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([512], relative_byte_offset=512 * 4)
    np_expected = np_input[512:1024]

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_view_larger_than_original_is_invalid():
    """Subset may not be larger than the original array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    with pytest.raises(ValueError, match="the NDArray being viewed only contains 4096 bytes"):
        tvm_input._create_view([2048])


def test_view_entirely_outside_bounds_of_original_is_invalid():
    """The byte_offset may not place a view outside the original array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    with pytest.raises(ValueError, match="would occupy bytes 8192 <= i_byte < 12288"):
        tvm_input._create_view([1024], relative_byte_offset=2048 * 4)


def test_view_partially_outside_bounds_of_original_is_invalid():
    """The byte_offset may not place any elements of a view outside the original array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    with pytest.raises(ValueError, match="would occupy bytes 2048 <= i_byte < 6144"):
        tvm_input._create_view([1024], relative_byte_offset=512 * 4)


def test_subview_first_half_of_first_half():
    """NDArray::CreateView be applied to a view

    The first view is at element offset 0 (byte offset 0).  The second
    view is at element offset 0 (byte offset 0) relative to the first
    view, or element offset 0 (byte offset 0) relative to the original
    array.

    """
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_view = tvm_input._create_view(
        [512],
        relative_byte_offset=0,
    )
    tvm_subview = tvm_view._create_view(
        [256],
        relative_byte_offset=0,
    )
    np_expected = np_input[0:512][0:256]

    np.testing.assert_equal(tvm_subview.numpy(), np_expected)


def test_subview_first_half_of_second_half():
    """NDArray::CreateView be applied to a view

    The first view is at element offset 512 (byte offset 2048).  The
    second view is at element offset 0 (byte offset 0) relative to the
    first view, or element offset 512 (byte offset 2048) relative to
    the original array.

    """
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_view = tvm_input._create_view(
        [512],
        relative_byte_offset=512 * 4,
    )
    tvm_subview = tvm_view._create_view(
        [256],
        relative_byte_offset=0,
    )
    np_expected = np_input[512:1024][0:256]

    np.testing.assert_equal(tvm_subview.numpy(), np_expected)


def test_subview_second_half_of_first_half():
    """NDArray::CreateView be applied to a view

    The first view is at element offset 0 (byte offset 0).  The second
    view is at element offset 256 (byte offset 1024) relative to the
    first view, or element offset 256 (byte offset 1024) relative to
    the original array.

    """
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_view = tvm_input._create_view(
        [512],
        relative_byte_offset=0,
    )
    tvm_subview = tvm_view._create_view(
        [256],
        relative_byte_offset=256 * 4,
    )
    np_expected = np_input[0:512][256:512]

    np.testing.assert_equal(tvm_subview.numpy(), np_expected)


def test_subview_second_half_of_second_half():
    """NDArray::CreateView be applied to a view

    The first view is at element offset 512 (byte offset 2048).  The
    second view is at element offset 256 (byte offset 1024) relative
    to the first view, or element offset 768 (byte offset 3072)
    relative to the original array.

    """
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_view = tvm_input._create_view(
        [512],
        relative_byte_offset=512 * 4,
    )
    tvm_subview = tvm_view._create_view(
        [256],
        relative_byte_offset=256 * 4,
    )
    np_expected = np_input[512:1024][256:512]

    np.testing.assert_equal(tvm_subview.numpy(), np_expected)


def test_subview_must_be_in_range_of_immediate_parent():
    """Bounds-checking is applied relative to the NDArray

    The first view is at location and covers bytes [0,2048).  The
    subview would occupy bytes [2048, 4096), and raises an error as
    this is outside the range of the view.

    """
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_view = tvm_input._create_view(
        [512],
        relative_byte_offset=0,
    )

    with pytest.raises(ValueError, match="would occupy bytes 2048 <= i_byte < 4096"):
        tvm_view._create_view(
            [512],
            relative_byte_offset=512 * 4,
        )


def test_2d_view_into_1d_arr():
    """NDArray::CreateView may change the dimensionality of an array"""
    np_input = np.arange(1024, dtype="int32")
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([32, 32])
    np_expected = np_input.reshape(32, 32)

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_2d_full_view_into_2d_arr():
    """NDArray::CreateView may change the shape of an array"""
    np_input = np.arange(1024, dtype="int32").reshape(32, 32)
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([16, 64])
    np_expected = np_input.reshape(16, 64)

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_2d_view_of_first_half_of_2d_arr():
    """NDArray::CreateView may return a multi-dimensional view"""
    np_input = np.arange(1024, dtype="int32").reshape(32, 32)
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([16, 32])
    np_expected = np_input[0:16, :]

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


def test_2d_view_of_second_half_of_2d_arr():
    """NDArray::CreateView may return a multi-dimensional view with byte offset"""
    np_input = np.arange(1024, dtype="int32").reshape(32, 32)
    tvm_input = tvm.nd.array(np_input)

    tvm_output = tvm_input._create_view([16, 32], relative_byte_offset=32 * 16 * 4)
    np_expected = np_input[16:32, :]

    np.testing.assert_equal(tvm_output.numpy(), np_expected)


if __name__ == "__main__":
    tvm.testing.main()
