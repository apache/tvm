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
import os
from pathlib import Path
import re

import numpy as np
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor

import paddle
import paddle.nn as nn

PADDLE_TEST_DATA_ROOT_PATH = Path(Path("~").expanduser(), ".tvm_test_data", "paddle")
PADDLE_TEST_DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)

def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))

def get_paddle_model(func, input_spec):
    global PADDLE_TEST_DATA_ROOT_PATH
    model_path = Path(PADDLE_TEST_DATA_ROOT_PATH, "model")

    paddle.jit.save(func, str(model_path), input_spec=input_spec)
    baseline_model = paddle.jit.load(str(model_path))

    os.remove(str(model_path) + '.pdmodel')
    return baseline_model

def verify_model(func, input_data, rtol=1e-5, atol=1e-5):
    if not (isinstance(input_data, list) or isinstance(input_data, tuple)):
        input_data = [input_data]

    input_spec = []
    input_names = []
    input_shape_dict = {}
    for idx, data in enumerate(input_data):
        input_name = "input{}".format(idx)
        input_spec.append(paddle.static.InputSpec(dtype=data.dtype, shape=data.shape, name=input_name))
        input_names.append(input_name)
        input_shape_dict[input_name] = data.shape

    baseline_model = get_paddle_model(func, input_spec)
    with paddle.no_grad():
        baseline_outputs = baseline_model(*[input.clone() for input in input_data])

    # get paddle outputs
    if isinstance(baseline_outputs, tuple):
        baseline_outputs = tuple(out.numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.numpy(),)

    mod, params = relay.frontend.from_paddle(baseline_model, input_shape_dict)
    for arg in mod["main"].params[: len(input_names)]:
        assert arg.name_hint in input_names
    compiled_input = dict(zip(input_names, [inp.clone().numpy() for inp in input_data]))
    
    with tvm.transform.PassContext(opt_level=3):
        for target, dev in tvm.testing.enabled_targets():
            lib = relay.build(mod, target=target, params=params)
            gmod = graph_executor.GraphModule(lib["default"](dev))
            for name, inp in compiled_input.items():
                gmod.set_input(name, inp)
            gmod.run()

            for i, baseline_output in enumerate(baseline_outputs):
                compiled_output = gmod.get_output(i).numpy()

                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(baseline_output, compiled_output, rtol=rtol, atol=atol)

@tvm.testing.uses_gpu
def test_forward_add():
    paddle.set_grad_enabled(False)
    input_shape = [10]

    @paddle.jit.to_static
    def add(inputs):
        return paddle.add(inputs, inputs)
    
    @paddle.jit.to_static
    def add2(inputs):
        return inputs + 1

    @paddle.jit.to_static
    def add3(inputs):
        ones = paddle.ones([10], dtype="float32")
        return inputs + ones

    class add4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, input1, input2):
            return input1 + input2

    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(add, input_data)
    verify_model(add2, input_data)
    verify_model(add3, input_data)
    input_data = paddle.rand([2, 3], dtype="float32")
    input_data2 = paddle.rand([2, 3], dtype="float32")
    model = add4()
    verify_model(model, [input_data, input_data2])

if __name__ == "__main__":
    test_forward_add()
