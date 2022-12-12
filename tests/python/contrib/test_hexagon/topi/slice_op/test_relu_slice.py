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

import tvm
import tvm.testing
from tvm.topi.hexagon.slice_ops.relu import relu_compute, relu_stir_schedule
from tvm import te
from tvm.contrib.hexagon import allocate_hexagon_array

from ...infrastructure import transform_numpy, get_hexagon_target


@tvm.testing.fixture
def input_np(in_shape, dtype):
    return np.random.uniform(size=in_shape).astype(dtype)


@tvm.testing.fixture
def ref_output_np(input_np):
    output_np = input_np * (input_np > 0)
    return output_np


@tvm.testing.fixture
def transformed_input_np(input_np, input_layout):
    return transform_numpy(input_np, "nhwc", input_layout)


@tvm.testing.fixture
def transformed_ref_output_np(ref_output_np, output_layout):
    return transform_numpy(ref_output_np, "nhwc", output_layout)


class BaseRelu:
    in_shape = tvm.testing.parameter(
        (1, 8, 4, 32),
        (1, 16, 4, 32),
        (1, 16, 8, 32),
        (1, 16, 8, 64),
        (2, 8, 4, 32),
        (2, 16, 4, 32),
        (2, 16, 8, 32),
        (2, 16, 8, 64),
    )
    dtype = tvm.testing.parameter("float16")
    working_scope = tvm.testing.parameter("global.vtcm")
    input_layout = tvm.testing.parameter("nhwc-8h2w32c2w-2d")
    output_layout = tvm.testing.parameter("nhwc-8h2w32c2w-2d")


class TestReluSlice(BaseRelu):
    @tvm.testing.requires_hexagon
    def test_relu(
        self,
        in_shape,
        dtype,
        input_layout,
        output_layout,
        transformed_input_np,
        transformed_ref_output_np,
        working_scope,
        hexagon_session,
    ):
        InputTensor = te.placeholder(in_shape, name="InputTensor", dtype=dtype)

        OutputTensor = relu_compute(InputTensor)

        tir_s = relu_stir_schedule(InputTensor, OutputTensor, input_layout, output_layout)

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

        func_name = "relu"
        with tvm.transform.PassContext(opt_level=3):
            runtime_module = tvm.build(tir_s.mod, target=get_hexagon_target("v69"), name=func_name)

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
