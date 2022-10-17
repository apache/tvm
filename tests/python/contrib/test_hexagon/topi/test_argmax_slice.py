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
""" Tests for Hexagon slice argmax op """
import numpy as np

import tvm
import tvm.testing
from tvm import te
import tvm.topi.hexagon.slice_ops as sl
import tvm.contrib.hexagon
from ..infrastructure import allocate_hexagon_array, transform_numpy, get_hexagon_target


class TestArgMaxSlice:
    """Argmax Slice Op Tests"""

    (
        input_shape,
        input_layout,
        output_layout,
        dtype,
        in_axis,
        in_axis_sep,
        out_axis_sep,
    ) = tvm.testing.parameters(
        ((1, 64, 64, 32), "nhwc-8h2w32c2w-2d", "nhw-32h16w-2d", "float16", [3], [4], [3]),
        ((3, 32, 16, 32), "nhwc-8h2w32c2w-2d", "nhw-32h16w-2d", "float16", [3], [4], [3]),
        ((1, 32, 32, 64), "nhwc-8h2w32c2w-2d", "nhw-32h16w-2d", "float16", [3], [4], [3]),
        ((1, 64, 64, 32), "nhwc-8h8w32c-2d", "nhw-32h16w-2d", "int8", [3], [4], [3]),
        ((3, 32, 16, 32), "nhwc-8h8w32c-2d", "nhw-32h16w-2d", "int8", [3], [4], [3]),
        ((1, 32, 32, 64), "nhwc-8h8w32c-2d", "nhw-32h16w-2d", "int8", [3], [4], [3]),
    )
    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        return np.random.uniform(size=input_shape).astype(dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, input_layout):
        return transform_numpy(input_np, "nhwc", input_layout)

    @tvm.testing.fixture
    def expected_output_np(self, input_np, in_axis):
        ref_np = np.argmax(input_np, *in_axis).astype("int32")
        return ref_np

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, output_layout):
        return transform_numpy(expected_output_np, "nhw", output_layout)

    @tvm.testing.requires_hexagon
    def test_argmax_slice(
        self,
        input_shape,
        dtype,
        input_layout,
        output_layout,
        in_axis,
        transformed_input_np,
        transformed_expected_output_np,
        in_axis_sep,
        out_axis_sep,
        hexagon_session,
        working_scope,
    ):
        """Top level testing function for argmax"""
        argmax_input = te.placeholder(input_shape, name="A", dtype=dtype)
        output = sl.argmax.argmax_compute(argmax_input, in_axis)
        argmax_func = te.create_prim_func([argmax_input, output])
        tir_s = sl.argmax_schedule(argmax_func, input_layout, output_layout)
        input_data = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=in_axis_sep,
            mem_scope=working_scope,
        )
        output_data = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_expected_output_np.shape,
            dtype=transformed_expected_output_np.dtype,
            axis_separators=out_axis_sep,
            mem_scope=working_scope,
        )
        with tvm.transform.PassContext(opt_level=3):
            tir_irm = tvm.lower(tir_s.mod, [argmax_input, output], name="argmax")
            runtime_module = tvm.build(
                tir_irm, [argmax_input, output], target=get_hexagon_target("v69"), name="argmax"
            )
        mod = hexagon_session.load_module(runtime_module)

        mod(input_data, output_data)
        output_np = output_data.numpy()
        tvm.testing.assert_allclose(
            output_np,
            transformed_expected_output_np,
            1e-3,
            1e-3,
        )


if __name__ == "__main__":
    tvm.testing.main()
