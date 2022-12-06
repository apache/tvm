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

"""Test that constants aren't duplicated for Arm(R) Ethos(TM)-N"""

import numpy as np
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model():
    """Return a model and any parameters it may have"""
    shape = (1, 4, 4, 4)
    kernel_h = 3
    kernel_w = 3
    out_channels = 8

    a = relay.var("a", shape=shape, dtype="uint8")
    add_const_value = tvm.nd.array(np.random.randint(0, high=10, size=shape, dtype="uint8"))
    add_const = relay.const(add_const_value, "uint8")
    a = relay.add(a, add_const)
    weight_shape = (kernel_h, kernel_w, shape[3], out_channels)
    weights_array = tvm.nd.array(
        np.random.randint(low=0, high=255, size=weight_shape, dtype="uint8")
    )
    weights = relay.const(weights_array, "uint8")
    conv = relay.qnn.op.conv2d(
        a,
        weights,
        input_zero_point=relay.const(0, "int32"),
        kernel_zero_point=relay.const(0, "int32"),
        input_scale=relay.const(0.3, "float32"),
        kernel_scale=relay.const(0.4, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout="HWIO",
        dilation=(1, 1),
        strides=(1, 1),
        groups=1,
        channels=out_channels,
        padding=(0, 0, 0, 0),
        out_dtype="int32",
    )
    b = tvm.nd.array(np.random.randint(0, high=10, size=(out_channels,), dtype="int32"))
    biasc = relay.const(b, "int32")
    bias = relay.nn.bias_add(conv, biasc, axis=3)
    req = relay.qnn.op.requantize(
        bias,
        relay.const(0.3 * 0.4, "float32"),  # input zero scale
        relay.const(0, "int32"),  # input zero point
        relay.const(0.4, "float32"),  # output zero scale
        relay.const(0, "int32"),  # output zero point
        out_dtype="uint8",
    )
    params = {"w": weights_array, "b": b}
    return req, params


@requires_ethosn
def test_constant_duplication():
    """Test that constants are not duplicated."""

    np.random.seed(0)
    model, params = _get_model()
    mod = tei.make_module(model, params)
    res = tei.build(mod, params, npu=True, expected_host_ops=1)
    for key, value in res.params.items():
        assert key == "p0"
        assert value.numpy().size == 64
