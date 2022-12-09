#!/usr/bin/env python3

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

""" Test 2d physical buffers """

import contextlib

import numpy as np
import pytest
import tvm

# Needed to register the link_shared packedfunc.
import tvm.contrib.hexagon
import tvm.testing
from tvm import te
from tvm.contrib.hexagon.pytest_plugin import requires_hexagon_toolchain
from tvm.tir.stmt_functor import post_order_visit
from tvm.contrib.hexagon import allocate_hexagon_array

from .infrastructure import get_hexagon_target

# Disabling invalid name as pylint assumes global variables as constants and
# expects them to be all upper-case. Since these are used as
# tvm.testing.parameters, if they are made upper-case, the functions which take
# them as arguments would also need to be upper-case, and pylint would complain
# there as well
# pylint: disable=invalid-name

schedule_type = tvm.testing.parameter("TE", "TIR")

dtype = tvm.testing.parameter("int8")
batch_size = tvm.testing.parameter(
    16,
    2,
)
input_channels = tvm.testing.parameter(
    32,
)
input_image_shape = tvm.testing.parameter(
    by_dict={
        "8x8": (8, 8),
        "32x32": (32, 32),
    }
)

input_layout = tvm.testing.parameter(
    "nhwc",
    "nchw-8h8w32c-1d",
)
output_layout = tvm.testing.parameter(
    "nhwc",
    "nchw-8h8w32c-1d",
)
working_layout, working_scope = tvm.testing.parameters(
    ("nhwc", "global"),
    ("nhwc", "global.vtcm"),
    ("nchw-8h8w32c-1d", "global"),
    ("nchw-8h8w32c-1d", "global.vtcm"),
    # 2-d memory may only occur in vtcm memory
    ("nchw-8h8w32c-2d", "global.vtcm"),
)

# pylint: enable=invalid-name


@tvm.testing.fixture
def target_host():
    """Return tvm target.Target with host attached"""
    return get_hexagon_target("v68")


# Disabling redefined-outer-name for the whole file as there isn't any easy
# solution yet to refactor tvm.testing.fixture fixtures that avoid redefining
# outer variable names
# pylint: disable=redefined-outer-name


@tvm.testing.fixture
def input_shape(batch_size, input_channels, input_image_shape):
    return [batch_size, *input_image_shape, input_channels]


def transform_shape(shape, layout):
    if layout == "nhwc":
        return shape
    if layout in ["nchw-8h8w32c-1d", "nchw-8h8w32c-2d"]:
        batch, height, width, channel = shape
        return [batch, (channel + 31) // 32, (height + 7) // 8, (width + 7) // 8, 8, 8, 32]
    raise RuntimeError(f"Unexpected layout '{layout}'")


def transform_numpy(arr_np, layout):
    if layout == "nhwc":
        return arr_np
    if layout in ["nchw-8h8w32c-1d", "nchw-8h8w32c-2d"]:
        batch, height, width, channel = arr_np.shape
        return arr_np.reshape([batch, height // 8, 8, width // 8, 8, channel // 32, 32]).transpose(
            0, 5, 1, 3, 2, 4, 6
        )
    raise RuntimeError(f"Unexpected layout '{layout}'")


@tvm.testing.fixture
def transformed_input_shape(input_shape, input_layout):
    return transform_shape(input_shape, input_layout)


@tvm.testing.fixture
def transformed_output_shape(output_shape, output_layout):
    return transform_shape(output_shape, output_layout)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return (100 * np.random.uniform(size=input_shape)).astype(dtype)


@tvm.testing.fixture
def transformed_input_np(input_np, input_layout):
    return transform_numpy(input_np, input_layout)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, output_layout)


def layout_transform_1d(batch, height, width, channel):
    return [
        batch,
        channel // 32,
        height // 8,
        width // 8,
        height % 8,
        width % 8,
        channel % 32,
    ]


def layout_transform_2d(batch, height, width, channel):
    return [
        batch,
        channel // 32,
        height // 8,
        width // 8,
        te.AXIS_SEPARATOR,
        height % 8,
        width % 8,
        channel % 32,
    ]


def extract_buffers(stmt):
    buffers = []

    def visitor(node):
        if isinstance(node, (tvm.tir.BufferLoad, tvm.tir.BufferStore, tvm.tir.BufferRealize)):
            buffers.append(node.buffer)

    post_order_visit(stmt, visitor)
    return buffers


class TestElementWise:
    """TestElementWise"""

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        return 2 * input_np

    @tvm.testing.fixture
    def output_shape(self, input_shape):
        return input_shape

    @tvm.testing.fixture
    def schedule_args(
        self,
        schedule_type,
        input_shape,
        dtype,
        input_layout,
        output_layout,
        working_layout,
        working_scope,
    ):
        """Create and return the schedule and input args after applying layout transform"""
        if schedule_type == "TE":

            return self._te_schedule_args(
                input_shape, dtype, input_layout, output_layout, working_layout, working_scope
            )
        elif schedule_type == "TIR":
            return self._tir_schedule_args(
                input_shape, dtype, input_layout, output_layout, working_layout, working_scope
            )

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def _te_tensors(self, input_shape, dtype):
        input_tensor = te.placeholder(input_shape, dtype, name="Input")
        output_tensor = te.compute(
            shape=input_tensor.shape,
            fcompute=lambda *indices: (2 * input_tensor[indices]).astype(dtype),
            name="Output",
        )
        return input_tensor, output_tensor

    def _te_schedule_args(
        self,
        input_shape,
        dtype,
        input_layout,
        output_layout,
        working_layout,
        working_scope,
    ):
        input_tensor, output_tensor = self._te_tensors(input_shape, dtype)

        schedule = te.create_schedule(output_tensor.op)

        write_cache = schedule.cache_write(output_tensor, working_scope)
        read_cache = schedule.cache_read(input_tensor, working_scope, [write_cache])

        def apply_transform(tensor, layout):
            if layout == "nhwc":
                return None
            if layout == "nchw-8h8w32c-1d":
                return schedule[tensor].transform_layout(layout_transform_1d)
            if layout == "nchw-8h8w32c-2d":
                return schedule[tensor].transform_layout(layout_transform_2d)
            raise RuntimeError(f"Unexpected layout '{layout}'")

        apply_transform(input_tensor, input_layout)
        compute_loopnest = apply_transform(output_tensor, output_layout) or output_tensor.op.axis
        schedule[write_cache].compute_at(schedule[output_tensor], compute_loopnest[0])

        apply_transform(read_cache, working_layout)
        apply_transform(write_cache, working_layout)

        return [schedule, [input_tensor, output_tensor]]

    def _tir_schedule_args(
        self, input_shape, dtype, input_layout, output_layout, working_layout, working_scope
    ):
        tensors = self._te_tensors(input_shape, dtype)

        sch = tvm.tir.Schedule(te.create_prim_func(tensors))

        cache_read_block = sch.cache_read("Output", 0, working_scope)
        cache_write_block = sch.cache_write("Output", 0, working_scope)

        def apply_transform(block, buffer_name, layout):
            if layout == "nhwc":
                pass
            elif layout == "nchw-8h8w32c-1d":
                sch.transform_layout(block, buffer_name, layout_transform_1d)
            elif layout == "nchw-8h8w32c-2d":
                sch.transform_layout(block, buffer_name, layout_transform_2d)
            else:
                raise RuntimeError(f"Unexpected layout '{layout}'")

        apply_transform(cache_read_block, ("read", 0), input_layout)
        apply_transform(cache_read_block, ("write", 0), working_layout)
        apply_transform(cache_write_block, ("read", 0), working_layout)
        apply_transform(cache_write_block, ("write", 0), output_layout)

        return [sch.mod]

    @tvm.testing.fixture
    def ir_module(self, schedule_args):
        # If the two buffers are accessed with the same indices, CSE
        # will replace them with a Let binding.  Since this makes it
        # harder to test what the transformed indices are, disabling
        # the CSE pass for this test.
        with tvm.transform.PassContext(disabled_pass=["tir.CommonSubexprElimTIR"]):
            return tvm.lower(*schedule_args)

    @tvm.testing.fixture
    def uses_unsupported_physical_dimensions(  # pylint: disable=invalid-name
        self, target_host, input_layout, working_layout, output_layout
    ):
        uses_2d_memory = "nchw-8h8w32c-2d" in [input_layout, working_layout, output_layout]
        can_handle_2d_memory = target_host.kind.name == "hexagon"

        return uses_2d_memory and not can_handle_2d_memory

    def test_param_shapes(self, ir_module, transformed_input_shape, transformed_output_shape):
        func = ir_module["main"]
        primfunc_input_shape, primfunc_output_shape = [
            list(func.buffer_map[param].shape) for param in func.params
        ]
        assert primfunc_input_shape == transformed_input_shape
        assert primfunc_output_shape == transformed_output_shape

    def test_cache_shape(self, ir_module, input_layout, working_layout, output_layout):
        """Test function to check expected_physical_dimensions for cached buffers"""
        func = ir_module["main"]
        for buffer in extract_buffers(func.body):
            buffer_layout = {
                "Input": input_layout,
                "Input.global": working_layout,
                "Output.global": working_layout,
                "Input.global.vtcm": working_layout,
                "Output.global.vtcm": working_layout,
                "Output": output_layout,
            }[buffer.name.replace("_", ".")]

            expected_physical_dimensions = {
                "nhwc": 1,
                "nchw-8h8w32c-1d": 1,
                "nchw-8h8w32c-2d": 2,
            }[buffer_layout]

            assert len(buffer.shape) == expected_physical_dimensions

    def test_lower(self, schedule_args):
        return tvm.lower(*schedule_args)

    @requires_hexagon_toolchain
    def test_build(self, schedule_args, target_host, input_layout, working_layout, output_layout):
        """Testing build success/failure

        * On Hexagon targets, build must succeed for both 1-d and 2-d memory.
        * On non-Hexagon targets, build must succeed 1-d memory.
        * On non-Hexagon targets, build must fail and report an error for 2-d memory.
        """
        # contextlib.nullcontext wasn't added until python3.7, and the
        # CI currently runs on python3.6.  Therefore, using ExitStack
        # to manage an optional context instead.
        stack = contextlib.ExitStack()

        with stack:
            is_hexagon = target_host.kind.name == "hexagon"
            uses_2d_memory = "nchw-8h8w32c-2d" in [input_layout, working_layout, output_layout]
            if uses_2d_memory and not is_hexagon:
                stack.enter_context(pytest.raises(tvm.TVMError))

            tvm.build(*schedule_args, target=target_host)

    @tvm.testing.fixture
    def runtime_module(self, schedule_args, target_host):
        if target_host.kind.name != "hexagon":
            pytest.skip("Only running on hexagon")

        return tvm.build(*schedule_args, target=target_host)

    @tvm.testing.requires_hexagon
    def test_execute(
        self,
        runtime_module,
        transformed_input_np,
        transformed_expected_output_np,
        input_layout,
        output_layout,
        hexagon_session,
    ):
        """Test execution of computes with 2d physical buffers"""
        if input_layout == "nchw-8h8w32c-2d":
            input_axis_separators = [4]
        else:
            input_axis_separators = []

        if output_layout == "nchw-8h8w32c-2d":
            output_axis_separators = [4]
        else:
            output_axis_separators = []

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=input_axis_separators,
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=np.zeros_like(transformed_expected_output_np),
            axis_separators=output_axis_separators,
        )

        mod = hexagon_session.load_module(runtime_module)

        mod(input_arr, output_arr)
        output_np = output_arr.numpy()

        np.testing.assert_array_equal(output_np, transformed_expected_output_np)


if __name__ == "__main__":
    tvm.testing.main()
