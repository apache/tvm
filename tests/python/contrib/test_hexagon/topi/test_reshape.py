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
"""Test reshape class."""
import numpy as np

import tvm
import tvm.testing
import tvm.topi.hexagon.slice_ops as sl
from tvm import te
from tvm.contrib.hexagon import allocate_hexagon_array

from ..infrastructure import transform_numpy, get_hexagon_target

BATCH_FLATTEN_FP16_TESTS = (
    ([1, 1, 1, 2048], [1, 2048], "nhwc-1024c-2d", "nc-1024-2d", "float16"),
    ([1, 2, 4, 2048], [1, 2 * 4 * 2048], "nhwc-1024c-2d", "nc-1024-2d", "float16"),
    ([1, 8, 8, 1024], [1, 8 * 8 * 1024], "nhwc-1024c-2d", "nc-1024-2d", "float16"),
    ([2, 4, 8, 1024], [2, 4 * 8 * 1024], "nhwc-1024c-2d", "nc-1024-2d", "float16"),
)

BATCH_FLATTEN_UINT8_TESTS = (
    ([1, 1, 1, 2048], [1, 2048], "nhwc-2048c-2d", "nc-2048-2d", "uint8"),
    ([1, 2, 4, 2048], [1, 2 * 4 * 2048], "nhwc-2048c-2d", "nc-2048-2d", "uint8"),
)


def reshape_helper(
    func,
    fcompute,
    fschedule,
    data_type,
    input_shape,
    input_layout,
    output_shape,
    output_layout,
    hexagon_session,
):
    """Reshape helper function."""

    a_tensor = te.placeholder(input_shape, name="a_tensor", dtype=data_type)
    if func == "reshape":
        d_tesnsor = fcompute(a_tensor, output_shape)
    elif func == "batch_flatten":
        d_tesnsor = fcompute(a_tensor)
    else:
        raise RuntimeError(f"Unexpected func'{func}'")
    tir_s = fschedule(
        d_tesnsor,
        a_tensor,
        output_layout,
        input_layout,
    )
    with tvm.transform.PassContext(opt_level=3):
        runtime_module = tvm.build(tir_s.mod, target=get_hexagon_target("v69"), name=func)

    mod = hexagon_session.load_module(runtime_module)

    a_numpy = (np.random.uniform(-10, 10, input_shape)).astype(data_type)
    ref = np.reshape(a_numpy, output_shape)

    input_np_transformed = transform_numpy(a_numpy, "nhwc", input_layout)
    ref_np_transformed = transform_numpy(ref, "nhwc", output_layout)
    input_axis_sep = [4]
    if output_layout in ["nhwc-8h2w32c2w-2d", "nhwc-8h8w32c-2d"]:
        output_axis_sep = [4]
    elif output_layout in ["nc-1024-2d", "nc-2048-2d"]:
        output_axis_sep = [2]
    else:
        raise RuntimeError(f"Unexpected layout '{output_layout}'")

    a_tvm = allocate_hexagon_array(
        hexagon_session.device,
        data=input_np_transformed,
        axis_separators=input_axis_sep,
        mem_scope="global.vtcm",
    )
    output = allocate_hexagon_array(
        hexagon_session.device,
        ref_np_transformed.shape,
        data_type,
        axis_separators=output_axis_sep,
        mem_scope="global.vtcm",
    )

    mod(a_tvm, output)
    np.testing.assert_allclose(output.numpy(), ref_np_transformed, atol=1e-07, rtol=0)


class BaseTestBatchFlatten:
    """Test batch flatten class."""

    (input_shape, output_shape, input_layout, output_layout, data_type,) = tvm.testing.parameters(
        *BATCH_FLATTEN_FP16_TESTS,
        *BATCH_FLATTEN_UINT8_TESTS,
    )


class TestBatchFlatten(BaseTestBatchFlatten):
    """Test batch flatten class."""

    @tvm.testing.requires_hexagon
    def test_batch_flatten(
        self,
        data_type,
        input_shape,
        input_layout,
        output_shape,
        output_layout,
        hexagon_session,
    ):
        """Test batch flatten."""
        reshape_helper(
            "batch_flatten",
            sl.batch_flatten_compute,
            sl.batch_flatten_stir_schedule,
            data_type,
            input_shape,
            input_layout,
            output_shape,
            output_layout,
            hexagon_session,
        )


class BaseTestReshape(BaseTestBatchFlatten):
    """Test reshape base class."""

    reshape_fp16_tests = (
        ([1, 8, 4, 64], [1, 8, 8, 32], "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", "float16"),
        ([1, 16, 8, 128], [1, 16, 16, 64], "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", "float16"),
    )

    reshape_uint8_tests = (
        ([1, 8, 8, 128], [1, 8, 16, 64], "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d", "uint8"),
        ([1, 16, 64, 128], [1, 16, 128, 64], "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d", "uint8"),
    )

    (input_shape, output_shape, input_layout, output_layout, data_type,) = tvm.testing.parameters(
        *BATCH_FLATTEN_FP16_TESTS,
        *BATCH_FLATTEN_UINT8_TESTS,
        *reshape_fp16_tests,
        *reshape_uint8_tests,
    )


class TestReshape(BaseTestReshape):
    """Test reshape class."""

    @tvm.testing.requires_hexagon
    def test_reshape(
        self,
        data_type,
        input_shape,
        input_layout,
        output_shape,
        output_layout,
        hexagon_session,
    ):
        """Test reshape."""
        reshape_helper(
            "reshape",
            sl.reshape_compute,
            sl.reshape_stir_schedule,
            data_type,
            input_shape,
            input_layout,
            output_shape,
            output_layout,
            hexagon_session,
        )


if __name__ == "__main__":
    tvm.testing.main()
