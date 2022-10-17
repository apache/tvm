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

""" Tests for Hexagon dequantize """
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.topi.hexagon import qnn
from ..infrastructure import (
    allocate_hexagon_array,
    transform_numpy,
    quantize_np,
    get_hexagon_target,
)


class TestDequantizeSlice2d:
    """
    For testing Dequantize Slice ops
    """

    input_shape, orig_layout, input_layout, output_layout, axis_sep, dtype = tvm.testing.parameters(
        ((1, 16, 64, 128), "nhwc", "nhwc-8h8w32c-2d", "nhwc-4h2w32c2w-2d", [4], "int8"),
        ((1, 16, 64, 128), "nhwc", "nhwc-8h8w32c-2d", "nhwc-4h2w32c2w-2d", [4], "uint8"),
        ((1, 8, 8, 32), "nhwc", "nhwc-8h8w32c-2d", "nhwc-4h2w32c2w-2d", [4], "int8"),
        ((1, 8, 8, 32), "nhwc", "nhwc-8h8w32c-2d", "nhwc-4h2w32c2w-2d", [4], "uint8"),
        ((1, 2048), "nc", "nc-2048c-2d", "nc-512c-2d", [2], "int8"),
        ((1, 2048), "nc", "nc-2048c-2d", "nc-512c-2d", [2], "uint8"),
    )

    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape):
        arr_np = np.random.random(size=input_shape).astype("float32")
        return arr_np

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, orig_layout, input_layout, dtype):
        quant_arr, scale, zero_point = quantize_np(input_np, dtype)
        return [transform_numpy(quant_arr, orig_layout, input_layout), scale, zero_point]

    @tvm.testing.fixture
    def expected_output_np(self, input_np, dtype):
        quant_np, scale, zero_point = quantize_np(input_np, dtype)
        ref_np = (scale * (quant_np.astype("int32") - zero_point)).astype("float32")
        return ref_np

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, orig_layout, output_layout):
        return transform_numpy(expected_output_np, orig_layout, output_layout)

    @tvm.testing.requires_hexagon
    def test_dequant_qnn(
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
        """
        Top level testing function for dequantize
        """

        dequant_input = te.placeholder(input_shape, name="A", dtype=dtype)

        in_data_np, in_scale, in_zero_pt = transformed_input_np

        dequant_output = qnn.dequantize_compute(dequant_input, in_scale, in_zero_pt)

        tir_s = qnn.dequantize_schedule(dequant_input, dequant_output, input_layout, output_layout)

        input_data = allocate_hexagon_array(
            hexagon_session.device,
            data=in_data_np,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )
        output_data = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_expected_output_np.shape,
            dtype=transformed_expected_output_np.dtype,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )
        with tvm.transform.PassContext(opt_level=3):
            tir_irm = tvm.lower(tir_s.mod, [dequant_input, dequant_output], name="dequantize")
            runtime_module = tvm.build(tir_irm, target=get_hexagon_target("v69"), name="dequantize")
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
