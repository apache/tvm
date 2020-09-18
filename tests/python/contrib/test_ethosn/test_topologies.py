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
"""Ethos-N tests for complex network topologies."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available, Available
from . import infrastructure as tei


def test_split_add_concat():
    if not ethosn_available():
        return

    def get_model(input_shape, var_names):
        """Return a model"""

        a = relay.var(next(var_names), shape=input_shape, dtype="uint8")
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

    inputs = {
        "a": tvm.nd.array(np.random.randint(0, high=255, size=(1, 16, 16, 4), dtype="uint8")),
    }

    outputs = []
    for npu in [False, True]:
        model = get_model(inputs["a"].shape, iter(inputs))
        mod = tei.make_module(model, [])
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

    tei.verify(outputs, 2)


def test_multiple_command_streams():
    """Check that multiple Ethos-N partitions are correctly handled.

    If there's more than one Ethos-N graph partition, more than one command
    stream will be created. This should be handled correctly by both the
    Ethos-N codegen and Ethos-N runtime module. This test checks against a
    simple graph which creates two Ethos-N partitions and checks the result
    against an 'all-CPU' run through TVM.
    """
    if ethosn_available() != Available.SW_AND_HW:
        return

    def get_model():
        """
        max_pool2d
             |
            abs
             |
        max_pool2d
        """
        x = relay.var("x", shape=(1, 4, 4, 4), dtype="uint8")
        out = relay.nn.max_pool2d(x, (2, 2), (2, 2), layout="NHWC")  # supported
        out = relay.op.abs(out)  # not supported
        out = relay.nn.max_pool2d(out, (2, 2), (2, 2), layout="NHWC")  # supported
        return out

    np.random.seed(0)
    outputs = []
    inputs = {"x": tvm.nd.array(np.random.randint(0, high=256, size=(1, 4, 4, 4), dtype="uint8"))}
    for npu in [False, True]:
        model = get_model()
        mod = tei.make_module(model, {})
        outputs.append(
            tei.build_and_run(mod, inputs, 1, {}, npu=npu, expected_host_ops=1, npu_partitions=2)
        )

    tei.verify(outputs, 0)


def test_output_order():
    if not ethosn_available():
        return

    def get_model(input_shape, var_names):
        """Return a model"""

        a = relay.var(next(var_names), shape=input_shape, dtype="uint8")

        z = relay.op.clip(a, 0, 255)
        b = relay.op.clip(z, 0, 15)
        c = relay.op.clip(z, 16, 31)
        d = relay.op.clip(z, 32, 47)
        e = relay.op.clip(z, 48, 63)
        f = relay.op.clip(z, 64, 79)
        g = relay.op.clip(z, 80, 95)
        h = relay.op.clip(z, 96, 111)
        i = relay.op.clip(z, 112, 127)
        return relay.Tuple((d, c, e, f, i, b, h, g))

    inputs = {
        "a": tvm.nd.array(np.random.randint(0, high=255, size=(1, 16, 16, 4), dtype="uint8")),
    }

    outputs = []
    for npu in [False, True]:
        model = get_model(inputs["a"].shape, iter(inputs))
        mod = tei.make_module(model, [])
        outputs.append(tei.build_and_run(mod, inputs, 8, {}, npu=npu))

    tei.verify(outputs, 1)


def test_split_with_asym_concats():
    if not ethosn_available():
        return

    def get_model(shape, splits, axis):
        a = relay.var("a", shape=shape, dtype="uint8")
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
        inputs = {"a": tvm.nd.array(np.random.randint(0, high=256, size=shape, dtype="uint8"))}
        for npu in [False, True]:
            model = get_model(shape, splits, axis)
            mod = tei.make_module(model, {})
            outputs.append(tei.build_and_run(mod, inputs, 2, {}, npu=npu))

        tei.verify(outputs, 0)


def test_output_tuple_propagation():
    """This tests the case where the output tuple must be inferred
    as having dummy tensor information."""
    if not ethosn_available():
        return

    def get_model():
        a = relay.var("a", shape=(1, 4, 4, 16), dtype="uint8")
        split = relay.op.split(a, indices_or_sections=4, axis=2)
        return relay.Tuple((split[0], split[1], split[2], split[3]))

    np.random.seed(0)
    outputs = []
    inputs = {"a": tvm.nd.array(np.random.randint(0, high=256, size=(1, 4, 4, 16), dtype="uint8"))}
    for npu in [False, True]:
        model = get_model()
        mod = tei.make_module(model, {})
        outputs.append(tei.build_and_run(mod, inputs, 4, {}, npu=npu))

    tei.verify(outputs, 0)


def test_input_tuples():
    if not ethosn_available():
        return

    def get_model(shapes, axis):
        tup = []
        for i, shape in enumerate(shapes):
            a = relay.var("in" + str(i), shape=shape, dtype="uint8")
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
        "in0": tvm.nd.array(np.random.randint(0, high=256, size=(1, 4), dtype="uint8")),
        "in1": tvm.nd.array(np.random.randint(0, high=256, size=(1, 6), dtype="uint8")),
    }
    outputs = []
    for npu in [False, True]:
        model = get_model([(1, 4), (1, 6)], 1)
        if not npu:
            mod = tei.make_module(model, {})
        else:
            mod = tei.make_ethosn_partition(model)
        lib = tei.build(mod, {}, npu=False)
        outputs.append(tei.run(lib, inputs, 1, npu=npu))

    tei.verify(outputs, 0)
