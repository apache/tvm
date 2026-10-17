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
"""Runtime tensor views and imported storage."""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing


@pytest.mark.parametrize(
    "input_shape,view_shape,offset",
    [
        ((1024,), (32, 32), 0),
        ((1024,), (512,), 0),
        ((32, 32), (16, 32), 512),
    ],
)
def test_tensor_view(input_shape, view_shape, offset):
    values = np.arange(1024, dtype="int32").reshape(input_shape)
    tensor = tvm_tirx.runtime.tensor(values)
    view = tensor._create_view(view_shape, relative_byte_offset=offset * 4)
    expected = values.ravel()[offset : offset + np.prod(view_shape)].reshape(view_shape)
    np.testing.assert_equal(view.numpy(), expected)


@pytest.mark.parametrize("parent_offset", [0, 512])
@pytest.mark.parametrize("child_offset", [0, 256])
def test_nested_view(parent_offset, child_offset):
    values = np.arange(1024, dtype="int32")
    tensor = tvm_tirx.runtime.tensor(values)
    parent = tensor._create_view((512,), relative_byte_offset=parent_offset * 4)
    child = parent._create_view((256,), relative_byte_offset=child_offset * 4)
    expected = values[parent_offset : parent_offset + 512][child_offset : child_offset + 256]
    np.testing.assert_equal(child.numpy(), expected)


@pytest.mark.parametrize(
    "size,offset,message",
    [
        (2048, 0, "the Tensor being viewed only contains 4096 bytes"),
        (1024, 2048, "would occupy bytes 8192 <= i_byte < 12288"),
        (1024, 512, "would occupy bytes 2048 <= i_byte < 6144"),
    ],
)
def test_view_bounds(size, offset, message):
    tensor = tvm_tirx.runtime.tensor(np.arange(1024, dtype="int32"))
    with pytest.raises(ValueError, match=message):
        tensor._create_view((size,), relative_byte_offset=offset * 4)


def test_subview_must_be_in_range_of_immediate_parent():
    tensor = tvm_tirx.runtime.tensor(np.arange(1024, dtype="int32"))
    parent = tensor._create_view((512,))
    with pytest.raises(ValueError, match="would occupy bytes 2048 <= i_byte < 4096"):
        parent._create_view((512,), relative_byte_offset=512 * 4)


def test_from_dlpack_shape_one():
    torch = pytest.importorskip("torch")
    values = torch.randn(1, 16)
    tensor = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(values))
    np.testing.assert_equal(tensor.numpy(), values.numpy())
    view = tensor._create_view((2, 8))
    np.testing.assert_equal(view.numpy(), values.numpy().reshape(2, 8))


@pytest.mark.parametrize(
    "dtype",
    [
        "float4_e2m1fn",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ],
)
def test_low_precision_numpy_roundtrip(dtype):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    if not hasattr(ml_dtypes, dtype):
        pytest.skip(f"ml_dtypes does not provide {dtype}")
    values = np.random.default_rng(0).random((8, 8)).astype(getattr(ml_dtypes, dtype))
    tensor = tvm_tirx.runtime.tensor(values)
    assert tensor.dtype == dtype
    np.testing.assert_equal(tensor.numpy(), values)


@pytest.mark.parametrize("container", [False, True], ids=["tensor", "tensor-dict"])
def test_tensor_json_roundtrip(container):
    tensor = tvm_tirx.runtime.tensor(np.arange(4, dtype="float32"))
    value = (
        {"key1": tensor, "key2": tvm_tirx.runtime.tensor(np.arange(4, dtype="float32") + 1)}
        if container
        else tensor
    )
    restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(value))
    tvm_tirx.ir.assert_structural_equal(value, restored)
    if container:
        for key in value:
            np.testing.assert_array_equal(value[key].numpy(), restored[key].numpy())
    else:
        np.testing.assert_array_equal(value.numpy(), restored.numpy())


def test_tensor_container_identity():
    tensor = tvm_tirx.runtime.tensor([1, 2, 3])
    values = tvm_ffi.convert([tensor, tensor])
    assert values[0].same_as(tensor)
    assert values[1].same_as(tensor)
    assert isinstance(values[0], tvm_tirx.runtime.Tensor)


if __name__ == "__main__":
    tvm_tirx.testing.main()
