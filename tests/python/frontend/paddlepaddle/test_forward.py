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
import shutil

import numpy as np
from paddle.fluid.layers.nn import pad
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

    shutil.rmtree(str(PADDLE_TEST_DATA_ROOT_PATH))
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
def test_forward_add_subtract():
    input_shape = [10]

    @paddle.jit.to_static
    def add_subtract(inputs):
        return paddle.subtract(paddle.add(inputs, inputs), inputs)
    
    @paddle.jit.to_static
    def add_subtract2(inputs):
        return inputs + 1 - 2

    @paddle.jit.to_static
    def add_subtract3(inputs1, inputs2):
        ones = paddle.ones([10], dtype="float32")
        return inputs1 + ones - inputs2

    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(add_subtract, input_data)
    verify_model(add_subtract2, input_data)
    input_data2 = paddle.rand(input_shape, dtype="float32")
    verify_model(add_subtract3, [input_data, input_data2])

@tvm.testing.uses_gpu
def test_forward_multiply():
    input_shape = [10]

    @paddle.jit.to_static
    def multiply1(inputs):
        return inputs * inputs

    @paddle.jit.to_static
    def multiply2(inputs):
        return inputs * 1.0

    @paddle.jit.to_static
    def multiply3(inputs):
        ones = paddle.ones([10], dtype="float32")
        return inputs * ones

    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(multiply1, input_data=input_data)
    verify_model(multiply2, input_data=input_data)
    verify_model(multiply3, input_data=input_data)

@tvm.testing.uses_gpu
def test_forward_matmul():

    class MatMul1(nn.Layer):
        def forward(self, input1, input2):
            return paddle.matmul(input1, input2)

    # matrix x vector
    input_data1 = paddle.randn((3, 4), dtype="float32")
    input_data2 = paddle.randn((4,), dtype="float32")
    verify_model(MatMul1(), input_data=[input_data1, input_data2])

    # matrix x matrix
    input_data1 = paddle.randn((5, 4), dtype="float32")
    input_data2 = paddle.randn((4, 5), dtype="float32")
    verify_model(MatMul1(), input_data=[input_data1, input_data2])

    # batched matrix x batched matrix
    input_data1 = paddle.randn((10, 3, 4), dtype="float32")
    input_data2 = paddle.randn((10, 4, 5), dtype="float32")
    verify_model(MatMul1(), input_data=[input_data1, input_data2])

    # batched matrix x broadcasted matrix
    input_data1 = paddle.randn((10, 3, 4), dtype="float32")
    input_data2 = paddle.randn((4, 5), dtype="float32")
    verify_model(MatMul1(), input_data=[input_data1, input_data2])

@tvm.testing.uses_gpu
def test_forward_conv():
    conv2d_input_shape = [1, 3, 10, 10]

    class Conv2D1(nn.Layer):
        def __init__(self):
            super(Conv2D1, self).__init__()
            self.conv = nn.Conv2D(3, 6, 7, bias_attr=True)
            self.softmax = nn.Softmax()
        
        @paddle.jit.to_static
        def forward(self, inputs):
            return self.softmax(self.conv(inputs))

    class Conv2D2(nn.Layer):
        def __init__(self):
            super(Conv2D2, self).__init__()
            self.conv = nn.Conv2D(3, 6, 7, groups=3, bias_attr=False)
            self.softmax = nn.Softmax()

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.softmax(self.conv(inputs))

    conv2d_input_data = paddle.rand(conv2d_input_shape, dtype="float32")
    verify_model(Conv2D1(), input_data=conv2d_input_data)
    verify_model(Conv2D2(), input_data=conv2d_input_data)


if __name__ == "__main__":
    test_forward_add_subtract()
    test_forward_multiply()
    test_forward_matmul()
    test_forward_conv()
