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

# pylint: disable=invalid-name

import numpy as np

from tvm import te
import tvm.testing
import tvm.topi.hexagon.slice_ops as sl
from tvm.contrib.hexagon import allocate_hexagon_array

from ...infrastructure import transform_numpy, get_hexagon_target

input_layout = tvm.testing.parameter(
    "nhwc-8h2w32c2w-2d",
)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return np.random.random(input_shape).astype(dtype)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, "nhwc", output_layout)


@tvm.testing.fixture
def transformed_input_np(input_np, input_layout):
    return transform_numpy(input_np, "nhwc", input_layout)


class TestClipSlice:
    input_shape, output_shape, A_min, A_max, output_layout, dtype = tvm.testing.parameters(
        ([1, 8, 4, 32], [1, 8, 4, 32], 0.1, 0.5, "nhwc-8h2w32c2w-2d", "float16")
    )

    @tvm.testing.fixture
    def expected_output_np(self, input_np, A_min, A_max):
        ref_np = np.clip(input_np, A_min, A_max)
        return ref_np

    @tvm.testing.requires_hexagon
    def test_clip_slice(
        self,
        input_shape,
        output_shape,
        input_np,
        input_layout,
        output_layout,
        dtype,
        A_min,
        A_max,
        transformed_input_np,
        transformed_expected_output_np,
        hexagon_session,
    ):
        # establish target and input placeholder
        A = te.placeholder(input_shape, name="A", dtype=dtype)

        # get the compute function and schedule
        M = sl.clip_compute(A, A_min, A_max)

        # Assume layout is nhwc-8h2w32c2w-2d
        tir_schedule = sl.clip_schedule(M, A, output_layout, input_layout)

        # build the function
        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                tir_schedule.mod,
                target=get_hexagon_target("v69"),
                name="clip",
            )

        # allocate input and output nd arrays
        axis_separators = [4]
        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=axis_separators,
            mem_scope="global.vtcm",
        )

        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            dtype=dtype,
            axis_separators=axis_separators,
            mem_scope="global.vtcm",
        )

        # execute
        mod = hexagon_session.load_module(func)
        mod(input_arr, output_arr)

        # convert output nd array to numpy array
        output_np = output_arr.numpy()
        b, h, w, c = output_shape
        reshaped_output_np = np.reshape(output_np, [b, h // 8, w // 4, c // 32, 8, 2, 32, 2])

        # test results
        np.testing.assert_allclose(
            reshaped_output_np, transformed_expected_output_np, rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    tvm.testing.main()
