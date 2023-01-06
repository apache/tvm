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
import pytest

pytest.importorskip("ethosu.vela")
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_te
from tvm.relay.backend.contrib.ethosu.tir.scheduler import OperatorCompute
import tvm.relay.backend.contrib.ethosu.op as ethosu_ops


def test_ethosu_conv2d():
    ifm = relay.var("ifm", shape=(1, 10, 20, 30), dtype="uint8")
    weight = relay.var("weight", shape=(40, 3, 3, 30), dtype="uint8")
    scale_bias = relay.var("scale_bias", shape=(40, 10), dtype="uint8")
    lut = relay.var("lut", shape=(), dtype="uint8")
    conv = ethosu_ops.ethosu_conv2d(
        ifm,
        weight,
        scale_bias,
        lut,
        ifm_scale=0.5,
        ifm_zero_point=10,
        weight_zero_point=12,
        ofm_scale=0.25,
        ofm_zero_point=14,
        ofm_channels=40,
        padding=(1, 1, 1, 1),
        kernel_shape=(3, 3),
        strides=(1, 1),
        dilation=(1, 1),
    )
    expr = relay.Function(relay.analysis.free_vars(conv), conv)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    lowered = lower_to_te(mod["main"])
    assert len(lowered.outputs) == 1
    assert len(lowered.inputs) == 4
    conv2d_compute = OperatorCompute.from_output(lowered.outputs[0])
    assert conv2d_compute.op.name == "ethosu_conv2d"
    input_shapes = set()
    for inp in lowered.inputs:
        input_shapes.add(tuple([x.value for x in inp.shape]))
    assert input_shapes == {(40, 10), (1, 10, 20, 30), (40, 3, 3, 30), ()}


if __name__ == "__main__":
    tvm.testing.main()
