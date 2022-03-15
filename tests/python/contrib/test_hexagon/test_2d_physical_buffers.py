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

import contextlib
import sys
import tempfile
import pathlib

import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.tir.stmt_functor import post_order_visit

from .conftest import requires_hexagon_toolchain

# Needed to register the link_shared packedfunc.
import tvm.contrib.hexagon.hexagon


dtype = tvm.testing.parameter("int8")
batch_size = tvm.testing.parameter(16)
input_channels = tvm.testing.parameter(32)
output_channels = tvm.testing.parameter(32)
input_image_shape = tvm.testing.parameter((64, 64))
filter_size = tvm.testing.parameter((5, 5))

input_layout = tvm.testing.parameter(
    "nhwc",
    "nchw-8h8w32c",
    "nchw-8h8w32c-flat",
)
working_layout = tvm.testing.parameter(
    "nhwc",
    "nchw-8h8w32c",
    "nchw-8h8w32c-flat",
)
output_layout = tvm.testing.parameter(
    "nhwc",
    "nchw-8h8w32c",
    "nchw-8h8w32c-flat",
)
working_scope = tvm.testing.parameter(
    "global",
    "global.vtcm",
)


@tvm.testing.fixture
def target_host(target):
    target = tvm.target.Target(target)

    if target.kind.name == "hexagon":
        # Shouldn't have to modify the target here, current
        # workaround.  In the future, should move the parameter
        # handling from tvm.target to target_kind.cc.
        target = tvm.target.hexagon("v68", link_params=True)
        host = target
    else:
        host = None
    return tvm.target.Target(target, host=host)


@tvm.testing.fixture
def input_shape(batch_size, input_channels, input_image_shape):
    return [batch_size, *input_image_shape, input_channels]


def transform_shape(shape, layout):
    if layout == "nhwc":
        return shape
    elif layout in ["nchw-8h8w32c", "nchw-8h8w32c-flat"]:
        N, H, W, C = shape
        return [N, (C + 31) // 32, (H + 7) // 8, (W + 7) // 8, 8, 8, 32]
    else:
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


def layout_transform_1d(n, h, w, c):
    return [
        n,
        c // 32,
        h // 8,
        w // 8,
        h % 8,
        w % 8,
        c % 32,
    ]


def layout_transform_2d(n, h, w, c):
    return [
        n,
        c // 32,
        h // 8,
        w // 8,
        te.AXIS_SEPARATOR,
        h % 8,
        w % 8,
        c % 32,
    ]


def extract_buffers(stmt):
    buffers = []

    def visitor(node):
        if isinstance(node, (tvm.tir.BufferLoad, tvm.tir.BufferStore, tvm.tir.BufferRealize)):
            buffers.append(node.buffer)

    post_order_visit(stmt, visitor)
    return buffers


class TestElementWise:
    @tvm.testing.fixture
    def output_np(self, input_np):
        return 2 * input_np

    @tvm.testing.fixture
    def output_shape(self, input_shape):
        return input_shape

    @tvm.testing.fixture
    def schedule_args(
        self,
        input_shape,
        dtype,
        input_layout,
        output_layout,
        working_layout,
        working_scope,
    ):
        InputTensor = te.placeholder(input_shape, dtype, name="Input")
        OutputTensor = te.compute(
            shape=InputTensor.shape,
            fcompute=lambda *indices: 2 * InputTensor[indices],
            name="Output",
        )
        schedule = te.create_schedule(OutputTensor.op)

        WriteCache = schedule.cache_write(OutputTensor, working_scope)
        ReadCache = schedule.cache_read(InputTensor, working_scope, [WriteCache])

        def apply_transform(tensor, layout):
            if layout == "nhwc":
                pass
            elif layout == "nchw-8h8w32c":
                return schedule[tensor].transform_layout(layout_transform_2d)
            elif layout == "nchw-8h8w32c-flat":
                return schedule[tensor].transform_layout(layout_transform_1d)
            else:
                raise RuntimeError(f"Unexpected layout '{layout}'")

        apply_transform(InputTensor, input_layout)
        compute_loopnest = apply_transform(OutputTensor, output_layout) or OutputTensor.op.axis
        schedule[WriteCache].compute_at(schedule[OutputTensor], compute_loopnest[0])

        apply_transform(ReadCache, working_layout)
        apply_transform(WriteCache, working_layout)

        return [schedule, [InputTensor, OutputTensor]]

    @tvm.testing.fixture
    def ir_module(self, schedule_args):
        # If the two buffers are accessed with the same indices, CSE
        # will replace them with a Let binding.  Since this makes it
        # harder to test what the transformed indices are, disabling
        # the CSE pass for this test.
        with tvm.transform.PassContext(disabled_pass=["tir.CommonSubexprElimTIR"]):
            return tvm.lower(*schedule_args)

    @tvm.testing.fixture
    def uses_unsupported_physical_dimensions(
        self, target_host, input_layout, working_layout, output_layout
    ):
        uses_2d_memory = "nchw-8h8w32c" in [input_layout, working_layout, output_layout]
        can_handle_2d_memory = target_host.kind.name == "hexagon"

        return uses_2d_memory and not can_handle_2d_memory

    def test_param_shapes(self, ir_module, transformed_input_shape, transformed_output_shape):
        func = ir_module["main"]
        primfunc_input_shape, primfunc_output_shape = [
            list(func.preflattened_buffer_map[param].shape) for param in func.params
        ]
        assert primfunc_input_shape == transformed_input_shape
        assert primfunc_output_shape == transformed_output_shape

    def test_cache_shape(self, ir_module, input_layout, working_layout, output_layout):
        func = ir_module["main"]
        for buffer in extract_buffers(func.body):
            buffer_layout = {
                "Input": input_layout,
                "Input.global": working_layout,
                "Output.global": working_layout,
                "Input.global.vtcm": working_layout,
                "Output.global.vtcm": working_layout,
                "Output": output_layout,
            }[buffer.name]

            expected_physical_dimensions = {
                "nhwc": 1,
                "nchw-8h8w32c": 2,
                "nchw-8h8w32c-flat": 1,
            }[buffer_layout]

            assert len(buffer.shape) == expected_physical_dimensions

    def test_lower(self, schedule_args):
        return tvm.lower(*schedule_args)

    @requires_hexagon_toolchain
    def test_build(self, schedule_args, target_host, input_layout, working_layout, output_layout):
        # contextlib.nullcontext wasn't added until python3.7, and the
        # CI currently runs on python3.6.  Therefore, using ExitStack
        # to manage an optional context instead.
        stack = contextlib.ExitStack()

        with stack:
            is_hexagon = target_host.kind.name == "hexagon"
            uses_2d_memory = "nchw-8h8w32c" in [input_layout, working_layout, output_layout]
            if uses_2d_memory and not is_hexagon:
                stack.enter_context(pytest.raises(tvm.TVMError))

            tvm.build(*schedule_args, target=target_host)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
