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
""" Tests for Hexagon slice cast ops """
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te
import tvm.topi.hexagon.slice_ops as sl
from ..infrastructure import allocate_hexagon_array

# pylint: disable=invalid-name


def transform_numpy(arr_np, layout):
    """
    Layout transformation on numpy arrays
    """
    if layout in ["nhwc-8h2w32c2w-2d"]:
        N, H, W, C = arr_np.shape
        return arr_np.reshape([N, H // 8, 8, W // 4, 2, 2, C // 32, 32]).transpose(
            0, 1, 3, 6, 2, 4, 7, 5
        )
    if layout in ["nhwc-4h2w32c2w-2d"]:
        N, H, W, C = arr_np.shape
        return arr_np.reshape([N, H // 4, 4, W // 4, 2, 2, C // 32, 32]).transpose(
            0, 1, 3, 6, 2, 4, 7, 5
        )
    if layout in ["nc-1024c-2d"]:
        N, C = arr_np.shape
        return arr_np.reshape([N, C // 1024, 1024])
    if layout in ["nc-512c-2d"]:
        N, C = arr_np.shape
        return arr_np.reshape([N, C // 512, 512])
    raise RuntimeError(f"Unexpected layout '{layout}'")


class TestCastF16F32Slice2d:
    """
    For testing Cast F16  to F32 Slice ops
    """

    input_shape, input_layout, output_layout, axis_sep = tvm.testing.parameters(
        ((1, 16, 12, 64), "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 64, 64, 32), "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 16, 12, 64), "nhwc-8h2w32c2w-2d", "nhwc-4h2w32c2w-2d", [4]),
        ((1, 64, 64, 32), "nhwc-8h2w32c2w-2d", "nhwc-4h2w32c2w-2d", [4]),
        ((1, 1024), "nc-1024c-2d", "nc-1024c-2d", [2]),
        ((1, 1024), "nc-1024c-2d", "nc-512c-2d", [2]),
    )
    dtype = tvm.testing.parameter("float16")
    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        return np.random.uniform(size=input_shape).astype(dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, input_layout):
        return transform_numpy(input_np, input_layout)

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        ref_np = input_np.astype("float32")
        return ref_np

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, output_layout):
        return transform_numpy(expected_output_np, output_layout)

    @tvm.testing.requires_hexagon
    def test_cast_fp16_fp32_slice(
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
        Top level testing function for cast fp16 to fp32
        """
        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        M = sl.cast_f16_f32_compute(A)
        cast_func = te.create_prim_func([A, M])
        tir_s = sl.cast_f16_f32_schedule(cast_func, input_layout, output_layout)
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
            tir_irm = tvm.lower(tir_s.mod, [A, M], name="cast_f16_f32")
            runtime_module = tvm.build(tir_irm, target=target, name="cast_f16_f32")
        mod = hexagon_session.load_module(runtime_module)

        mod(A_data, M_data)
        output_np = M_data.numpy()
        tvm.testing.assert_allclose(
            output_np,
            transformed_expected_output_np,
            1e-3,
            1e-3,
        )


class TestCastF32F16Slice2d:
    """
    For testing Cast F32 to F16 Slice ops
    """

    (input_shape, input_layout, output_layout, axis_sep,) = tvm.testing.parameters(
        ((1, 16, 12, 64), "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 64, 64, 32), "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 16, 12, 64), "nhwc-4h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 64, 64, 32), "nhwc-4h2w32c2w-2d", "nhwc-8h2w32c2w-2d", [4]),
        ((1, 1024), "nc-1024c-2d", "nc-1024c-2d", [2]),
        ((1, 1024), "nc-512c-2d", "nc-1024c-2d", [2]),
    )
    dtype = tvm.testing.parameter("float32")
    working_scope = tvm.testing.parameter("global.vtcm")

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        return np.random.uniform(size=input_shape).astype(dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, input_layout):
        return transform_numpy(input_np, input_layout)

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        ref_np = input_np.astype("float16")
        return ref_np

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, output_layout):
        return transform_numpy(expected_output_np, output_layout)

    @tvm.testing.requires_hexagon
    def test_cast_fp32_fp16_slice(
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
        Top level testing function for cast fp32 to fp16
        """

        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        M = sl.cast_f32_f16_compute(A)
        cast_func = te.create_prim_func([A, M])
        tir_s = sl.cast_f32_f16_schedule(cast_func, input_layout, output_layout)
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
            tir_irm = tvm.lower(tir_s.mod, [A, M], name="cast_f32_f16")
            runtime_module = tvm.build(tir_irm, target=target, name="cast_f32_f16")
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
    sys.exit(tvm.testing.main(sys.argv))
