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
from tvm import te
from tvm import tir
from tvm import topi
from tvm.contrib.hexagon.build import HexagonLauncher

from .infrastructure import allocate_hexagon_array, get_hexagon_target


def sigmoid_compute(Input):
    return topi.sigmoid(Input)


def sigmoid_stir_schedule(Input, Output):
    sigmoid_func = te.create_prim_func([Input, Output])
    sch = tir.Schedule(sigmoid_func, debug_mask="all")
    block = sch.get_block("compute")

    (n,) = sch.get_loops(block)
    sch.vectorize(n)
    return sch


@tvm.testing.fixture
def input_np(in_shape, dtype, min_val, max_val):
    return np.random.uniform(low=min_val, high=max_val, size=in_shape).astype(dtype)


@tvm.testing.fixture
def ref_output_np(input_np):
    output_np = 1 / (1 + np.exp(-input_np))
    return output_np


class BaseSigmoid:
    (in_shape, dtype, min_val, max_val,) = tvm.testing.parameters(
        ((64,), "float16", -8.0, 8.0),
        ((64,), "float16", -6.0, 7.0),
        ((64,), "float16", -10.0, 15.0),
        ((64,), "float16", -10.0, 0.0),
        ((64,), "float16", 0.0, 10.0),
    )


class TestSigmoid(BaseSigmoid):
    @tvm.testing.requires_hexagon
    def test_sigmoid(
        self,
        in_shape,
        dtype,
        input_np,
        ref_output_np,
        hexagon_session,
    ):
        InputTensor = te.placeholder(in_shape, name="InputTensor", dtype=dtype)

        OutputTensor = sigmoid_compute(InputTensor)

        tir_s = sigmoid_stir_schedule(InputTensor, OutputTensor)

        input_data = allocate_hexagon_array(
            hexagon_session.device,
            data=input_np,
        )
        output_data = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=ref_output_np.shape,
            dtype=ref_output_np.dtype,
        )

        func_name = "sigmoid"
        with tvm.transform.PassContext(opt_level=3):
            runtime_module = tvm.build(tir_s.mod, target=get_hexagon_target("v69"), name=func_name)

        assert "hvx_sigmoid" in runtime_module.get_source("asm")
        assert "vmin" in runtime_module.get_source("asm")
        assert "vmax" in runtime_module.get_source("asm")
        mod = hexagon_session.load_module(runtime_module)

        mod(input_data, output_data)
        output_np = output_data.numpy()

        tvm.testing.assert_allclose(
            output_np,
            ref_output_np,
            1e-3,
            1e-3,
        )


if __name__ == "__main__":
    tvm.testing.main()
