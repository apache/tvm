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

"""Integration tests for Multiply."""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


def _get_model(
    shape,
    constant_shape,
    input_zp,
    input_sc,
    input2_zp,
    input2_sc,
    output_zp,
    output_sc,
    dtype,
    reverse_inputs=False,
    constant_data=None,
):
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    x = relay.var("x", shape=shape, dtype=dtype)
    if constant_data:
        y_data = np.array(constant_data, dtype=dtype).reshape(constant_shape)
    else:
        y_data = np.random.randint(data_min, data_max + 1, size=constant_shape, dtype=dtype)
    y = relay.const(y_data, dtype=dtype)

    out = relay.qnn.op.mul(
        y if reverse_inputs else x,
        x if reverse_inputs else y,
        relay.const(input_sc, "float32"),
        relay.const(input_zp, "int32"),
        relay.const(input2_sc, "float32"),
        relay.const(input2_zp, "int32"),
        relay.const(output_sc, "float32"),
        relay.const(output_zp, "int32"),
    )
    params = {"y": y_data}
    return out, params


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,constant_shape",
    [((1, 4, 4, 8), (1, 1, 1, 8)), ((1, 16, 12, 4), (4,))],
)
@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_multiply_to_depthwise(dtype, shape, constant_shape, reverse_inputs):
    """Compare Multiply -> Depthwise conversion output with TVM."""

    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )

    model, params = _get_model(
        shape,
        constant_shape,
        input_zp,
        input_sc,
        input2_zp,
        input2_sc,
        output_zp,
        output_sc,
        dtype,
        reverse_inputs,
    )
    inputs = {"x": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype))}
    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,constant_shape", [((1, 4, 5, 8), (1, 1, 1, 1)), ((1, 3, 7, 10), None)]
)
@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_multiply_to_reinterpret_quantize(shape, constant_shape, reverse_inputs):
    """Compare Multiply -> Reinterpret Quantize conversion output with TVM."""
    np.random.seed(0)

    dtype = "uint8"
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    # Multiply can only be offloaded as a reinterpret quantize operation if
    # it is an identity option. We must choose the quantization and constant
    # data carefully to make sure that this is the case.
    input_zp = 0
    input_sc = 0.007814894430339336
    input2_zp = 0
    input2_sc = 0.5
    output_zp = 0
    output_sc = 0.9963990449905396
    constant_data = 255

    model, params = _get_model(
        shape,
        constant_shape,
        input_zp,
        input_sc,
        input2_zp,
        input2_sc,
        output_zp,
        output_sc,
        dtype,
        reverse_inputs,
        constant_data,
    )
    inputs = {"x": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype))}
    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, params)
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                params,
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
def test_multiply_multiple_inputs_unsupported():
    """Check multiply operator with two inputs is not offloaded."""

    np.random.seed(0)

    shape = (1, 4, 5, 6)
    dtype = "int8"

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )

    x = relay.var("x", shape=shape, dtype=dtype)
    y = relay.var("y", shape=shape, dtype=dtype)
    model = relay.qnn.op.mul(
        x,
        y,
        relay.const(input_sc, "float32"),
        relay.const(input_zp, "int32"),
        relay.const(input2_sc, "float32"),
        relay.const(input2_zp, "int32"),
        relay.const(output_sc, "float32"),
        relay.const(output_zp, "int32"),
    )

    expected_host_ops = 1
    npu_partitions = 0
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        tei.build(
            mod,
            {},
            npu=npu,
            expected_host_ops=expected_host_ops,
            npu_partitions=npu_partitions,
        )


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,constant_shape",
    [
        ("int16", (1, 4, 5, 6), (1, 1, 1, 6)),
        ("int8", (1, 1, 3), (1, 1, 1, 3)),
        ("int8", (1, 2, 4, 8), (1, 2, 4, 8)),
    ],
)
def test_multiply_unsupported(dtype, shape, constant_shape):
    """Check multiply operator with unsupported attributes is not offloaded."""

    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[-1]
    )

    model, params = _get_model(
        shape,
        constant_shape,
        input_zp,
        input_sc,
        input2_zp,
        input2_sc,
        output_zp,
        output_sc,
        dtype,
        reverse_inputs=False,
        constant_data=False,
    )

    expected_host_ops = 1
    npu_partitions = 0
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        tei.build(
            mod,
            params,
            npu=npu,
            expected_host_ops=expected_host_ops,
            npu_partitions=npu_partitions,
        )
