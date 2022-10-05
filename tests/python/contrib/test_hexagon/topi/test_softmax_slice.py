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
from tvm import te
from tvm.topi.testing import softmax_python
import tvm.topi.hexagon.slice_ops as sl

from ..infrastructure import allocate_hexagon_array, get_hexagon_target


def transform_numpy(arr_np, layout):

    if layout in ["nc-512c-2d"]:
        N, C = arr_np.shape
        return arr_np.reshape([N, C // 512, 512])
    raise RuntimeError(f"Unexpected layout '{layout}'")


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return (np.random.uniform(size=input_shape)).astype(dtype)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, output_layout)


@tvm.testing.fixture
def transformed_input_np(input_np, input_layout):
    return transform_numpy(input_np, input_layout)


class Basesoftmax2d:

    input_shape, input_layout, output_layout, axis_sep = tvm.testing.parameters(
        ((1, 1024), "nc-512c-2d", "nc-512c-2d", [2])
    )
    dtype = tvm.testing.parameter("float32")
    working_scope = tvm.testing.parameter("global.vtcm")


class TestSoftmax2d(Basesoftmax2d):
    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        if len(input_np.shape) == 2:
            ref_np_2d = softmax_python(input_np)
            return ref_np_2d
        raise RuntimeError(f"Unexpected input shape '{input_np.shape}'")

    @tvm.testing.requires_hexagon
    def test_softmax_f32(
        self,
        dtype,
        input_layout,
        output_layout,
        input_shape,
        input_np,
        transformed_input_np,
        transformed_expected_output_np,
        expected_output_np,
        working_scope,
        axis_sep,
        hexagon_session,
    ):
        target_hexagon = tvm.target.hexagon(
            "v69",
            llvm_options="--disable-loop-unrolling-pass",
        )
        A = te.placeholder(input_shape, name="A", dtype=dtype)

        O = sl.softmax_compute(A)

        if input_layout == "nc-512c-2d":
            tir_s = sl.softmax_stir_schedule(O, A, output_layout, input_layout)
            sch = tir_s.mod
        else:
            raise RuntimeError(f"Unexpected input layout '{input_layout}'")

        with tvm.transform.PassContext(
            opt_level=3,
            config={
                "tir.LoopPartition": {"partition_const_loop": True},
            },
        ):

            func = tvm.build(
                sch,
                [A, O],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="softmax_slice",
            )

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )

        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_expected_output_np.shape,
            dtype=transformed_expected_output_np.dtype,
            axis_separators=axis_sep,
            mem_scope=working_scope,
        )

        mod = hexagon_session.load_module(func)
        mod(input_arr, output_arr)

        n, c = input_np.shape
        output_np = output_arr.numpy().reshape(1, c // 512, 512)

        np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
