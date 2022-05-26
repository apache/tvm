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
import scipy
import scipy.signal

import tvm
import tvm.testing
import tvm.topi.hexagon.slice_ops as sl
from tvm import te, topi
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.topi import testing

from .infrastructure import allocate_hexagon_array


def transform_numpy(arr_np):
    N, H, W, C = arr_np.shape
    return arr_np.reshape([N, H // 8, 8, W // 4, 2, 2, C // 32, 32]).transpose(
        0, 1, 3, 6, 2, 4, 7, 5
    )


def transform_2d(arr_np):
    N, H, W, C, h, w1, c, w2 = arr_np.shape
    return arr_np.reshape(N * H * W * C, h * w1 * c * w2)


@tvm.testing.fixture
def input_np(in_shape, dtype):
    return np.random.uniform(size=in_shape).astype(dtype)


@tvm.testing.fixture
def input_np_padded(input_np, in_shape, padded_in_shape):
    pad_height = padded_in_shape[1] - in_shape[1]
    pad_width = padded_in_shape[2] - in_shape[2]
    pad_channel = padded_in_shape[3] - in_shape[3]
    input_padded = np.pad(
        input_np, ((0, 0), (0, pad_height), (0, pad_width), (0, pad_channel)), "constant"
    )
    return input_padded


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


class TestReluSlice(BaseRelu):
    @tvm.testing.fixture
    def padded_in_shape(self, in_shape):
        in_batch, in_height, in_width, in_channel = in_shape
        in_height = ((in_height + 7) // 8) * 8
        in_width = ((in_width + 3) // 4) * 4
        in_channel = ((in_channel + 31) // 32) * 32
        return in_batch, in_height, in_width, in_channel

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        output_np = input_np * (input_np > 0)
        return output_np

    @tvm.testing.requires_hexagon
    def test_relu(
        self,
        in_shape,
        padded_in_shape,
        dtype,
        input_np,
        input_np_padded,
        expected_output_np,
        target,
        working_scope,
        hexagon_session,
    ):
        InputTensor = tvm.te.placeholder(padded_in_shape, name="InputTensor", dtype=dtype)

        OutputTensor = sl.relu_te_compute(InputTensor, dtype)

        def layout_func(n, h, w, c):
            return [n, h // 8, w // 4, c // 32, h % 8, (w % 4) // 2, c % 32, w % 2]

        target_hexagon = tvm.target.hexagon("v69", codegen_options="emit-llvm, emit-asm=1")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)

        reluf16_func = te.create_prim_func([InputTensor, OutputTensor])
        tir_s = sl.reluf16_stir_sched(
            reluf16_func,
            layout_func,
        )

        func_name = "reluf16"
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_assert": True}):
            tir_irm = tvm.lower(tir_s.mod, [InputTensor, OutputTensor], name=func_name)
            runtime_module = tvm.build(
                tir_irm, [InputTensor, OutputTensor], target=target, name=func_name
            )

        input_np_transformed = transform_numpy(input_np_padded)
        input_np_tr_2d = transform_2d(input_np_transformed)
        output_np_transformed = transform_numpy(expected_output_np)
        output_np_tr_2d = transform_2d(output_np_transformed)

        input_arr = tvm.nd.empty(
            input_np_tr_2d.shape,
            input_np_tr_2d.dtype,
            hexagon_session.device,
            mem_scope=working_scope,
        )
        input_arr.copyfrom(input_np_tr_2d)

        output_arr = tvm.nd.empty(
            output_np_tr_2d.shape,
            output_np_tr_2d.dtype,
            hexagon_session.device,
            mem_scope=working_scope,
        )

        mod = hexagon_session.load_module(runtime_module)
        mod(input_arr, output_arr)
        output_np = output_arr.numpy()

        np.testing.assert_allclose(output_np, output_np_tr_2d)
