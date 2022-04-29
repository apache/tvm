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
from tvm.relay.op.contrib.ethosn import Available
from . import infrastructure as tei


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_split_add_concat(dtype):
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
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

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
    outputs = []
    inputs = {
        "x": tvm.nd.array(
            np.random.randint(
                np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=(1, 4, 4, 4), dtype=dtype
            )
        )
    }
    model = get_model(dtype)
    mod = tei.make_module(model, {})
    outputs.append(
        tei.build_and_run(mod, inputs, 1, {}, npu=True, expected_host_ops=1, npu_partitions=2)
    )


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_output_order(dtype):
    def get_model(input_shape, dtype, var_names):
        """Return a model"""

        min = np.iinfo(dtype).min
        max = np.iinfo(dtype).max
        a = relay.var(next(var_names), shape=input_shape, dtype=dtype)

        z = relay.op.clip(a, min, max)
        b = relay.op.clip(z, min, min + 15)
        c = relay.op.clip(z, min + 16, min + 31)
        d = relay.op.clip(z, min + 32, min + 47)
        e = relay.op.clip(z, min + 48, min + 63)
        f = relay.op.clip(z, min + 64, min + 79)
        g = relay.op.clip(z, min + 80, min + 95)
        h = relay.op.clip(z, min + 96, min + 111)
        i = relay.op.clip(z, min + 112, max)
        return relay.Tuple((d, c, e, f, i, b, h, g))

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
        outputs.append(tei.build_and_run(mod, inputs, 8, {}, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_split_with_asym_concats(dtype):
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

    trials = [
        ((1, 16, 16, 32), (2, 7, 10), 2),
    ]

    np.random.seed(0)
    for shape, splits, axis in trials:
        outputs = []
        inputs = {
            "a": tvm.nd.array(
                np.random.randint(
                    np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype
                )
            )
        }
        for npu in [False, True]:
            model = get_model(shape, dtype, splits, axis)
            mod = tei.make_module(model, {})
            outputs.append(tei.build_and_run(mod, inputs, 2, {}, npu=npu))

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
        outputs.append(tei.build_and_run(mod, inputs, 4, {}, npu=npu))

    tei.verify(outputs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_input_tuples(dtype):
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
        lib = tei.build(mod, {}, npu=False)
        outputs.append(tei.run(lib, inputs, 1, npu=npu))

    tei.verify(outputs, dtype, 0)
