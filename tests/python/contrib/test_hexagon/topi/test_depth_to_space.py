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
# pylint: disable=line-too-long, redefined-outer-name

"""Test depth_to_space slice op for hexagon"""

import numpy as np
import pytest

import tvm
from tvm import te
import tvm.testing
from tvm.topi.hexagon.slice_ops.depth_to_space import d2s_compute, d2s_schedule
from tvm.topi.testing import depth_to_space_python

from ..infrastructure import allocate_hexagon_array, transform_numpy, get_hexagon_target


d2s_fp16_tests = (
    ((1, 8, 8, 256), 2, "CDR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 8, 8, 1024), 4, "CDR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 16, 16, 256), 2, "CDR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 16, 16, 1024), 4, "CDR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 8, 8, 256), 2, "DCR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 8, 8, 1024), 4, "DCR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 16, 16, 256), 2, "DCR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
    ((1, 16, 16, 1024), 4, "DCR", "float16", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d"),
)

d2s_uint8_tests = (
    ((1, 8, 8, 256), 2, "CDR", "uint8", "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d"),
    ((1, 8, 8, 1024), 4, "CDR", "uint8", "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d"),
    ((1, 8, 8, 256), 2, "DCR", "uint8", "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d"),
    ((1, 8, 8, 1024), 4, "DCR", "uint8", "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d"),
)


class TestD2SSlice:
    """Test class that defines the Depth to Space slice test"""

    (input_shape, block_size, mode, dtype, input_layout, output_layout,) = tvm.testing.parameters(
        *d2s_fp16_tests,
        *d2s_uint8_tests,
    )

    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        return np.random.uniform(size=input_shape).astype(dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, input_layout):
        return transform_numpy(input_np, "nhwc", input_layout)

    @tvm.testing.fixture
    def ref_output_np(self, input_np, block_size, mode):
        a_np = np.transpose(input_np, axes=[0, 3, 1, 2])
        ref_np = depth_to_space_python(a_np, block_size, mode=mode)
        ref_np = np.transpose(ref_np, axes=[0, 2, 3, 1])
        return ref_np

    @tvm.testing.fixture
    def transformed_ref_output_np(self, ref_output_np, output_layout):
        return transform_numpy(ref_output_np, "nhwc", output_layout)

    @tvm.testing.requires_hexagon
    def test_d2s_slice(
        self,
        input_shape,
        block_size,
        mode,
        dtype,
        input_layout,
        output_layout,
        hexagon_session,
        working_scope,
        transformed_input_np,
        transformed_ref_output_np,
    ):
        """Top level testing function for depth to space"""
        Input = te.placeholder(input_shape, name="Input", dtype=dtype)

        Output = d2s_compute(Input, block_size, "NHWC", mode)

        tir_s = d2s_schedule(Input, Output, input_layout, output_layout)

        input_data = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=[4],
            mem_scope=working_scope,
        )
        output_data = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_ref_output_np.shape,
            dtype=transformed_ref_output_np.dtype,
            axis_separators=[4],
            mem_scope=working_scope,
        )
        with tvm.transform.PassContext(opt_level=3):
            runtime_module = tvm.build(
                tir_s.mod, [Input, Output], target=get_hexagon_target("v69"), name="depth_to_space"
            )
        mod = hexagon_session.load_module(runtime_module)

        mod(input_data, output_data)
        output_np = output_data.numpy()

        tvm.testing.assert_allclose(
            output_np,
            transformed_ref_output_np,
            1e-3,
            1e-3,
        )


if __name__ == "__main__":
    tvm.testing.main()
