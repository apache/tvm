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

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.hexagon.slice_ops as sl
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.topi import testing

from .infrastructure import allocate_hexagon_array


def n11c_1024c_1d(n, h, w, c):
    return [n, h, w, c // 1024, tvm.te.AXIS_SEPARATOR, c % 1024]


def nc_1024_1d(n, c):
    return [n, c // 1024, tvm.te.AXIS_SEPARATOR, c % 1024]


def transform_numpy(arr_np, layout):
    if layout == "nhwc":
        return arr_np
    elif layout == "n11c-1024c-1d":
        N, H, W, C = arr_np.shape
        return arr_np.reshape([N, H, W, C // 1024, 1024])
    elif layout == "nc-1d":
        N, C = arr_np.shape
        return arr_np.reshape([N, C // 1024, 1024])


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, output_layout)


class BaseTestBatchFlatten:
    (
        input_shape,
        input_layout,
        output_layout,
        input_axis_sep,
        output_axis_sep,
    ) = tvm.testing.parameters(
        ((1, 1, 1, 2048), "n11c-1024c-1d", "nc-1d", [4], [2]),
        ((1, 2, 4, 2048), "n11c-1024c-1d", "nc-1d", [4], [2]),
        ((1, 8, 8, 1024), "n11c-1024c-1d", "nc-1d", [4], [2]),
        ((2, 4, 8, 1024), "n11c-1024c-1d", "nc-1d", [4], [2]),
        ((2, 3, 5, 2048), "n11c-1024c-1d", "nc-1d", [4], [2]),
    )
    data_type = tvm.testing.parameter("float16")


class TestBatchFlatten(BaseTestBatchFlatten):
    @tvm.testing.fixture
    def output_shape(self, input_shape):
        return input_shape[0], input_shape[1] * input_shape[2] * input_shape[3]

    @tvm.testing.requires_hexagon
    def test_batch_flatten(
        self,
        data_type,
        input_shape,
        input_layout,
        input_axis_sep,
        output_shape,
        output_layout,
        output_axis_sep,
        hexagon_session,
    ):
        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        A = te.placeholder(input_shape, name="A", dtype=data_type)
        D = sl.batch_flatten_compute(A)
        tir_s = sl.batch_flatten_stir_schedule(
            D,
            A,
            nc_1024_1d,
            n11c_1024c_1d,
        )
        func_name = "batch_flatten"
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_assert": True}):
            tir_irm = tvm.lower(tir_s.mod, [A, D], name=func_name)
            runtime_module = tvm.build(tir_irm, [A, D], target=target, name=func_name)

        mod = hexagon_session.load_module(runtime_module)

        a_numpy = (np.random.uniform(-1, 1, input_shape)).astype(data_type)
        ref = np.reshape(a_numpy, output_shape)

        input_np_transformed = transform_numpy(a_numpy, input_layout)
        ref_np_transformed = transform_numpy(ref, output_layout)

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


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
