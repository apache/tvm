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
""" Test for Hexagon slice tanh op """
import numpy as np

import tvm
import tvm.testing
from tvm import te
import tvm.topi.hexagon.slice_ops as sl
import tvm.contrib.hexagon
from tvm.contrib.hexagon import allocate_hexagon_array

from ...infrastructure import transform_numpy, get_hexagon_target

# pylint: disable=invalid-name


class TestTanhSlice:
    """For Testing Tanh fp16 op"""

    input_shape, orig_layout, input_layout, output_layout, axis_sep = tvm.testing.parameters(
        ((1, 8, 4, 32), "nhwc", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 16, 12, 64), "nhwc", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 64, 64, 32), "nhwc", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
    )
    dtype = tvm.testing.parameter("float16")
    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        return np.random.uniform(size=input_shape).astype(dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, orig_layout, input_layout):
        return transform_numpy(input_np, orig_layout, input_layout)

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        ref_np = np.tanh(input_np)
        return ref_np

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, orig_layout, output_layout):
        return transform_numpy(expected_output_np, orig_layout, output_layout)

    @tvm.testing.requires_hexagon
    def test_tanh(
        self,
        input_shape,
        dtype,
        input_layout,
        output_layout,
        transformed_input_np,
        transformed_expected_output_np,
        axis_sep,
        hexagon_session,
        working_scope,
    ):
        """Top Level testing function for tanh fp16 op"""

        A = te.placeholder(input_shape, name="A", dtype=dtype)
        M = sl.tanh_te_compute(A)
        tanhf16_func = te.create_prim_func([A, M])
        tir_s = sl.tanhf16_schedule(tanhf16_func, input_layout, output_layout)
        A_data = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )
        M_data = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_expected_output_np.shape,
            dtype=transformed_expected_output_np.dtype,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )
        with tvm.transform.PassContext(opt_level=3):
            tir_irm = tvm.lower(tir_s.mod, [A, M], name="tanhf16")
            runtime_module = tvm.build(tir_irm, target=get_hexagon_target("v69"), name="tanhf16")
        mod = hexagon_session.load_module(runtime_module)

        mod(A_data, M_data)
        output_np = M_data.numpy()
        tvm.testing.assert_allclose(
            output_np,
            transformed_expected_output_np,
            1e-3,
            1e-3,
        )


if __name__ == "__main__":
    tvm.testing.main()
