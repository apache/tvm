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

"""Arm(R) Ethos(TM)-N tests for complex network topologies."""

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from tvm.relay.op.contrib.ethosn import Available, ethosn_available

from . import infrastructure as tei


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_split_add_concat(dtype):
    """Test a model with split, add and contatenate."""

    def get_model(input_shape, dtype, var_names):
        """Return a model"""

        a = relay.var(next(var_names), shape=input_shape, dtype=dtype)
        split_scale = relay.const(0.25, "float32")
        split_zp = relay.const(100, "int32")
        add_scale = relay.const(0.75, "float32")
        add_zp = relay.const(120, "int32")
        axis = 2

        split = relay.split(a, indices_or_sections=4, axis=axis)
        b = relay.qnn.op.add(
            split[0],
            split[1],
            lhs_scale=split_scale,
            lhs_zero_point=split_zp,
            rhs_scale=split_scale,
            rhs_zero_point=split_zp,
            output_scale=add_scale,
            output_zero_point=add_zp,
        )
        conc = relay.qnn.op.concatenate(
            [b, split[2], split[3]],
            input_scales=(add_scale, split_scale, split_scale),
            input_zero_points=(add_zp, split_zp, split_zp),
            output_scale=add_scale,
            output_zero_point=add_zp,
            axis=axis,
        )
        return conc

    np.random.seed(0)
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 16, 16, 4), dtype=dtype
            )
        ),
    }

    outputs = []
    for npu in [False, True]:
        model = get_model(inputs["a"].shape, dtype, iter(inputs))
        mod = tei.make_module(model, [])

        expected_host_ops = 0
        npu_partitions = 1

        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                expected_host_ops=expected_host_ops,
                npu_partitions=npu_partitions,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    if outputs:
        tei.verify(outputs, dtype, 2)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_multiple_command_streams(dtype):
    """Check that multiple Ethos-N partitions are correctly handled.

    If there's more than one Ethos-N graph partition, more than one command
    stream will be created. This should be handled correctly by both the
    Ethos-N codegen and Ethos-N runtime module. This test checks against a
    simple graph which creates two Ethos-N partitions and checks the result
    against an 'all-CPU' run through TVM.
    """

    def get_model(dtype):
        """
        max_pool2d
             |
            abs
             |
        max_pool2d
        """
        x = relay.var("x", shape=(1, 4, 4, 4), dtype=dtype)
        out = relay.nn.max_pool2d(x, (2, 2), (2, 2), layout="NHWC")  # supported
        out = relay.op.abs(out)  # not supported
        out = relay.nn.max_pool2d(out, (2, 2), (2, 2), layout="NHWC")  # supported
        return out

    np.random.seed(0)
    inputs = {
        "x": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 4, 4, 4), dtype=dtype
            )
        )
    }
    model = get_model(dtype)
    mod = tei.make_module(model, {})

    # Mock inference is only supported when the whole graph is offloaded to the NPU
    if ethosn_available() == Available.SW_ONLY:
        tei.build(mod, {}, npu=True, expected_host_ops=1, npu_partitions=2)
    else:
        tei.build_and_run(mod, inputs, 1, {}, npu=True, expected_host_ops=1, npu_partitions=2)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_order(dtype):
    """Test the output order."""

    def get_model(input_shape, dtype, var_names):
        """Return a model"""

        min_value = np.iinfo(dtype).min
        max_value = np.iinfo(dtype).max
        a = relay.var(next(var_names), shape=input_shape, dtype=dtype)

        op_z = relay.op.clip(a, min_value, max_value)
        op_b = relay.op.clip(op_z, min_value, min_value + 15)
        op_c = relay.op.clip(op_z, min_value + 16, min_value + 31)
        op_d = relay.op.clip(op_z, min_value + 32, min_value + 47)
        op_e = relay.op.clip(op_z, min_value + 48, min_value + 63)
        op_f = relay.op.clip(op_z, min_value + 64, min_value + 79)
        op_g = relay.op.clip(op_z, min_value + 80, min_value + 95)
        op_h = relay.op.clip(op_z, min_value + 96, min_value + 111)
        op_i = relay.op.clip(op_z, min_value + 112, max_value)
        return relay.Tuple((op_d, op_c, op_e, op_f, op_i, op_b, op_h, op_g))

    np.random.seed(0)
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 16, 16, 4), dtype=dtype
            )
        ),
    }

    outputs = []
    for npu in [False, True]:
        model = get_model(inputs["a"].shape, dtype, iter(inputs))
        mod = tei.make_module(model, [])
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                8,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_order_different_sizes(dtype):
    """
    Test the output order when there are multiple outputs of different sizes.
    """

    np.random.seed(0)
    input_name = "a"
    input_shape = (1, 8, 8, 4)
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    def get_model():
        var = relay.var(input_name, shape=input_shape, dtype=dtype)
        clip = relay.op.clip(var, dtype_min, dtype_max)
        max_pool = relay.nn.max_pool2d(clip, (2, 2), (2, 2), ceil_mode=True, layout="NHWC")
        mean = relay.op.cast(clip, "int32")
        mean = relay.mean(mean, axis=[1, 2], keepdims=True)
        mean = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(0.0784314, "float32"),
            input_zero_point=relay.const(dtype_min + 128, "int32"),
            output_scale=relay.const(0.0784314, "float32"),
            output_zero_point=relay.const(dtype_min + 128, "int32"),
            out_dtype=dtype,
        )

        return relay.Tuple((mean, max_pool, clip))

    inputs = {
        input_name: tvm.nd.array(
            np.random.randint(dtype_min, dtype_max + 1, size=input_shape, dtype=dtype)
        ),
    }

    outputs = []
    for npu in [False, True]:
        model = get_model()
        mod = tei.make_module(model, [])
        outputs.append(
            tei.build_and_run(mod, inputs, 3, {}, npu=npu, expected_host_ops=0, npu_partitions=1)
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,splits,axis",
    [
        ((1, 16, 16, 32), (2, 7, 10), 2),
    ],
)
def test_split_with_asym_concats(dtype, shape, splits, axis):
    """Test a model with split and contatenates."""
    np.random.seed(0)

    def get_model(shape, dtype, splits, axis):
        a = relay.var("a", shape=shape, dtype=dtype)
        split = relay.op.split(a, indices_or_sections=splits, axis=axis)
        zeroi = relay.const(1, "int32")
        zerof = relay.const(0.5, "float32")
        con1 = relay.qnn.op.concatenate(
            [split[0], split[1]],
            input_scales=[zerof] * 2,
            input_zero_points=[zeroi] * 2,
            output_scale=zerof,
            output_zero_point=zeroi,
            axis=axis,
        )
        con2 = relay.qnn.op.concatenate(
            [split[2], split[3]],
            input_scales=[zerof] * 2,
            input_zero_points=[zeroi] * 2,
            output_scale=zerof,
            output_zero_point=zeroi,
            axis=axis,
        )
        return relay.Tuple((con2, con1))

    outputs = []
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
        )
    }
    for npu in [False, True]:
        model = get_model(shape, dtype, splits, axis)
        mod = tei.make_module(model, {})

        expected_host_ops = 0
        npu_partitions = 1

        # Mock inference is only supported when the whole graph is offloaded to the NPU
        if ethosn_available() == Available.SW_ONLY:
            tei.build(
                mod,
                {},
                npu=npu,
                expected_host_ops=expected_host_ops,
                npu_partitions=npu_partitions,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        else:
            outputs.append(
                tei.build_and_run(
                    mod,
                    inputs,
                    2,
                    {},
                    npu=npu,
                    expected_host_ops=expected_host_ops,
                    npu_partitions=npu_partitions,
                    additional_config_args={"inline_non_compute_intensive_partitions": False},
                )
            )

    if outputs:
        tei.verify(outputs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_tuple_propagation(dtype):
    """This tests the case where the output tuple must be inferred
    as having dummy tensor information."""

    def get_model(dtype):
        a = relay.var("a", shape=(1, 4, 4, 16), dtype=dtype)
        split = relay.op.split(a, indices_or_sections=4, axis=2)
        return relay.Tuple((split[0], split[1], split[2], split[3]))

    np.random.seed(0)
    outputs = []
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 4, 4, 16), dtype=dtype
            )
        )
    }
    for npu in [False, True]:
        model = get_model(dtype)
        mod = tei.make_module(model, {})
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                4,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_input_tuples(dtype):
    """Test a model with a tuple as input."""

    def get_model(shapes, dtype, axis):
        tup = []
        for i, shape in enumerate(shapes):
            a = relay.var("in" + str(i), shape=shape, dtype=dtype)
            tup.append(a)

        zeroi = relay.const(1, "int32")
        zerof = relay.const(0.5, "float32")
        con = relay.qnn.op.concatenate(
            tup,
            input_scales=[zerof] * len(shapes),
            input_zero_points=[zeroi] * len(shapes),
            output_scale=zerof,
            output_zero_point=zeroi,
            axis=axis,
        )

        return con

    np.random.seed(0)
    inputs = {
        "in0": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 4), dtype=dtype
            )
        ),
        "in1": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 6), dtype=dtype
            )
        ),
    }
    outputs = []
    for npu in [False, True]:
        model = get_model([(1, 4), (1, 6)], dtype, 1)
        if not npu:
            mod = tei.make_module(model, {})
        else:
            mod = tei.make_ethosn_partition(model)
        lib = tei.build(
            mod,
            {},
            npu=False,
            additional_config_args={"inline_non_compute_intensive_partitions": False},
        )
        outputs.append(tei.run(lib, inputs, 1, npu=npu))

    tei.verify(outputs, dtype, 0)


@requires_ethosn
def test_inline_non_compute_intensive_operations():
    """Tests the case when a subgraph is unpartitioned."""
    np.random.seed(0)
    dtype = "int8"
    shape = (1, 2, 2, 4)

    inp = relay.var("x", shape=shape, dtype=dtype)
    reshape = relay.reshape(inp, newshape=(1, 1, 4, 4))

    inputs = {
        "x": tvm.nd.array(
            np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
        ),
    }
    outputs = []

    for npu in [False, True]:
        mod = tei.make_module(reshape, {})
        outputs.append(
            tei.build_and_run(mod, inputs, 1, {}, npu=npu, expected_host_ops=1, npu_partitions=0)
        )

    tei.verify(outputs, dtype, 0)
