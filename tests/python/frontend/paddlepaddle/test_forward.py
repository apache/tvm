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
from pathlib import Path
import shutil

import numpy as np

import paddle
from paddle.framework import dtype
import paddle.nn as nn

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor


PADDLE_TEST_DATA_ROOT_PATH = Path(Path("~").expanduser(), ".tvm_test_data", "paddle")
PADDLE_TEST_DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)


def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Paddle Output shapes {} and TVM shapes {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))
    if tru.dtype != est.dtype:
        msg = "Paddle Output dtype {} and TVM dtype {} don't match"
        raise AssertionError(msg.format(tru.dtype, est.dtype))


def get_paddle_model(func, input_spec):
    global PADDLE_TEST_DATA_ROOT_PATH
    model_path = Path(PADDLE_TEST_DATA_ROOT_PATH, "model")
    paddle.jit.save(func, str(model_path), input_spec=input_spec)
    baseline_model = paddle.jit.load(str(model_path))

    shutil.rmtree(str(PADDLE_TEST_DATA_ROOT_PATH))
    return baseline_model


def get_tvm_output_with_vm(mod, params, target, device, input_data):
    """Generic function to execute and get tvm output with vm executor"""

    ex = relay.create_executor("vm", mod=mod, device=device, target=target)
    params.update(input_data)
    result = ex.evaluate()(**params)
    if isinstance(result, tvm.runtime.NDArray):
        return [
            result.numpy(),
        ]
    return [r.numpy() for r in result]


def get_tvm_output(mod, params, target, device, input_data, compiled_names, num):
    """Generic function to execute and get tvm output"""

    lib = relay.build(mod, target=target, params=params)
    gmod = graph_executor.GraphModule(lib["default"](device))
    for name in compiled_names:
        gmod.set_input(name, input_data[name])
    gmod.run()
    outputs = []
    for i in range(num):
        outputs.append(gmod.get_output(i).numpy())
    return outputs


def verify_model(func, input_data, rtol=1e-5, atol=1e-5, input_shape=None):
    if not (isinstance(input_data, (tuple, list))):
        input_data = [input_data]

    input_spec = []
    input_names = []
    input_shape_dict = {}
    compiled_input = {}
    for idx, data in enumerate(input_data):
        input_name = "input{}".format(idx)
        if input_shape:
            shape = input_shape[idx]
        else:
            shape = data.shape
        input_shape_dict[input_name] = shape
        input_spec.append(paddle.static.InputSpec(dtype=data.dtype, shape=shape, name=input_name))
        input_names.append(input_name)
        if isinstance(data, np.ndarray):
            compiled_input[input_name] = data
        else:
            compiled_input[input_name] = data.numpy()

    baseline_model = get_paddle_model(func, input_spec)
    baseline_outputs = baseline_model(*[input[:] for input in input_data])

    # get paddle outputs
    if isinstance(baseline_outputs, (tuple, list)):
        baseline_outputs = tuple(out.numpy() for out in baseline_outputs)
    else:
        baseline_outputs = (baseline_outputs.numpy(),)

    mod, params = relay.frontend.from_paddle(baseline_model, input_shape_dict)
    parms_num = min(len(input_names), len(mod["main"].params))
    compiled_names = []
    for arg in mod["main"].params:
        assert arg.name_hint in input_names or arg.name_hint in params
        if arg.name_hint in input_names:
            compiled_names.append(arg.name_hint)

    with tvm.transform.PassContext(opt_level=3):
        for target, dev in tvm.testing.enabled_targets():
            if input_shape:
                tvm_output = get_tvm_output_with_vm(mod, params, target, dev, compiled_input)
            else:
                tvm_output = get_tvm_output(
                    mod, params, target, dev, compiled_input, compiled_names, len(baseline_outputs)
                )

            for baseline_output, compiled_output in zip(baseline_outputs, tvm_output):
                assert_shapes_match(baseline_output, compiled_output)
                tvm.testing.assert_allclose(baseline_output, compiled_output, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_forward_math():
    class MathOp(nn.Layer):
        def __init__(self, op_name):
            super(MathOp, self).__init__()
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, op_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.func(inputs)

    input_data = paddle.rand([1, 2, 5, 5], dtype="float32")
    op_list = [
        "abs",
        "acos",
        "asin",
        "atan",
        "ceil",
        "cos",
        "cosh",
        "erf",
        "exp",
        "floor",
        "log",
        "log2",
        "log10",
        "log1p",
        "numel",
        "relu",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "rsqrt",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ]
    for op_name in op_list:
        verify_model(MathOp(op_name), input_data)


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
def test_forward_addmm():
    @paddle.jit.to_static
    def addmm(input, x, y, alpha=1, beta=1):
        return paddle.addmm(input, x, y, alpha, beta)

    input_shape = [10, 10]
    x_shape = [10, 3]
    y_shape = [3, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    x_data = paddle.rand(x_shape, dtype="float32")
    y_data = paddle.rand(y_shape, dtype="float32")
    verify_model(addmm, input_data=[input_data, x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_argmax():
    input_shape = [1, 3, 10, 10]

    class ArgMax(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.argmax(inputs)

    class ArgMax1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmax(axis=1)

    class ArgMax2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmax(axis=1, keepdim=False)

    class ArgMax3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmax(axis=2, keepdim=True)

    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(ArgMax(), input_data=input_data)
    verify_model(ArgMax1(), input_data=input_data)
    verify_model(ArgMax2(), input_data=input_data)
    verify_model(ArgMax3(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_argmin():
    input_shape = [1, 3, 10, 10]

    class ArgMin(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.argmin(inputs)

    class ArgMin1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmin(axis=1)

    class ArgMin2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmin(axis=1, keepdim=False)

    class ArgMin3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmin(axis=2, keepdim=True)

    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(ArgMin(), input_data=input_data)
    verify_model(ArgMin1(), input_data=input_data)
    verify_model(ArgMin2(), input_data=input_data)
    verify_model(ArgMin3(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_argsort():
    @paddle.jit.to_static
    def argsort(inputs):
        return paddle.argsort(inputs)

    @paddle.jit.to_static
    def argsort2(inputs):
        return paddle.argsort(inputs, axis=0, descending=True)

    input_shape = [2, 3, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(argsort, input_data)
    input_data2 = np.random.randint(100, size=input_shape)
    verify_model(argsort2, input_data2)


@tvm.testing.uses_gpu
def test_forward_assign():
    class Assign(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.assign(inputs)

    input_shape = [2, 3]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(Assign(), [input_data])
    input_data2 = np.random.randint(100, size=input_shape)
    verify_model(Assign(), [input_data2], input_shape=[[-1, -1]])


@tvm.testing.uses_gpu
def test_forward_batch_norm():
    class BatchNorm1D(nn.Layer):
        def __init__(self):
            super(BatchNorm1D, self).__init__()
            self.batch_norm = nn.BatchNorm1D(2)

        @paddle.jit.to_static
        def forward(self, input_data):
            return self.batch_norm(input_data)

    class BatchNorm2D(nn.Layer):
        def __init__(self):
            super(BatchNorm2D, self).__init__()
            self.batch_norm = nn.BatchNorm2D(2)

        @paddle.jit.to_static
        def forward(self, input_data):
            return self.batch_norm(input_data)

    class BatchNorm3D(nn.Layer):
        def __init__(self):
            super(BatchNorm3D, self).__init__()
            self.batch_norm = nn.BatchNorm3D(2)

        @paddle.jit.to_static
        def forward(self, input_data):
            return self.batch_norm(input_data)

    input_data = paddle.rand((2, 2, 3), dtype="float32")
    verify_model(BatchNorm1D(), input_data=input_data)
    input_data = paddle.rand((2, 2, 2, 3), dtype="float32")
    verify_model(BatchNorm2D(), input_data=input_data)
    input_data = paddle.rand((2, 2, 2, 2, 3), dtype="float32")
    verify_model(BatchNorm3D(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_cast():
    @paddle.jit.to_static
    def cast1(inputs, dtype="uint8"):
        return paddle.cast(inputs, dtype)

    @paddle.jit.to_static
    def cast2(inputs, dtype="int64"):
        return inputs.cast(dtype)

    input_shape = [2, 3]
    input_data = paddle.rand(input_shape, dtype="float32") * 100
    verify_model(cast1, [input_data])
    verify_model(cast2, [input_data])


@tvm.testing.uses_gpu
def test_forward_clip():
    @paddle.jit.to_static
    def clip(inputs):
        return paddle.clip(inputs, min=3, max=5)

    @paddle.jit.to_static
    def clip2(inputs, max_value):
        return paddle.clip(inputs, max=max_value)

    @paddle.jit.to_static
    def clip3(inputs, min_value):
        return paddle.clip(inputs, min=min_value)

    @paddle.jit.to_static
    def clip4(inputs, min_value, max_value):
        return paddle.clip(inputs, min=min_value, max=max_value)

    verify_model(clip, paddle.to_tensor([[1, 2], [4, 6]], dtype="int32"))
    x = np.array([[1.2, 3.5], [4.5, 6.4]])
    x1 = paddle.to_tensor(x, dtype="float32")
    min_value = paddle.to_tensor(np.array([2.1]), dtype="float32")
    max_value = paddle.to_tensor(np.array([4.5]), dtype="float32")
    verify_model(clip2, [x1, max_value])
    verify_model(clip3, [x1, min_value])
    verify_model(clip4, [x1, min_value, max_value])


@tvm.testing.uses_gpu
def test_forward_concat_unsqueeze():
    @paddle.jit.to_static
    def concat_unsqueeze1(inputs):
        return paddle.concat([inputs[:, 0].unsqueeze(1), inputs[:, 1].unsqueeze(1)], axis=1)

    @paddle.jit.to_static
    def concat_unsqueeze2(inputs):
        a = (inputs[:, :, 0] + 2) * 7
        b = (inputs[:, :, 1] + 3) * 11
        c = (inputs[:, :, 2] + 5) * 13
        return paddle.concat([paddle.unsqueeze(t, axis=2) for t in [a, b, c]], axis=2)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(concat_unsqueeze1, input_data=input_data)
    verify_model(concat_unsqueeze2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_crop():
    @paddle.jit.to_static
    def crop1(inputs):
        return paddle.crop(inputs, shape=[2, 2])

    @paddle.jit.to_static
    def crop2(inputs, shape):
        return paddle.crop(inputs, shape=shape, offsets=[0, 1])

    @paddle.jit.to_static
    def crop3(inputs):
        offsets = paddle.to_tensor(np.array([1, 0]).astype("int32"))
        return paddle.crop(inputs, shape=[3, 3], offsets=offsets)

    @paddle.jit.to_static
    def crop4(inputs, shape, offsets):
        return paddle.crop(inputs, shape=shape, offsets=offsets)

    input_shape = [10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(crop1, input_data=[input_data])
    shape = paddle.to_tensor(np.array([3, 3], "int32"))
    verify_model(crop2, [input_data, shape], input_shape=[[-1, -1], [2]])
    verify_model(crop3, input_data=[input_data])
    offsets = paddle.to_tensor(np.array([1, 1]).astype("int32"))
    verify_model(crop4, input_data=[input_data, shape, offsets], input_shape=[[-1, -1], [2], [2]])


@tvm.testing.uses_gpu
def test_forward_cumsum():
    class Cumsum1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cumsum(inputs)

    class Cumsum2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cumsum(inputs, axis=0)

    class Cumsum3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cumsum(inputs, axis=1)

    input_data = paddle.randint(0, 100, (10, 10), dtype=paddle.int32)
    verify_model(Cumsum1(), input_data)
    verify_model(Cumsum1(), [input_data.astype(paddle.int64)])
    verify_model(Cumsum2(), input_data)
    verify_model(Cumsum3(), input_data)


@tvm.testing.uses_gpu
def test_forward_conv():
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

    class Conv2D3(nn.Layer):
        def __init__(self):
            super(Conv2D3, self).__init__()
            self.conv = nn.Conv2D(3, 6, 7, groups=3, bias_attr=False, padding="SAME")

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.conv(inputs)

    class Conv2D4(nn.Layer):
        def __init__(self):
            super(Conv2D4, self).__init__()
            self.conv = nn.Conv2D(
                3, 6, 7, groups=3, bias_attr=False, padding=[1, 2, 0, 1], stride=2, dilation=2
            )

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.conv(inputs)

    conv2d_input_shape = [1, 3, 112, 112]
    conv2d_input_data = paddle.rand(conv2d_input_shape, dtype="float32")
    verify_model(Conv2D1(), input_data=conv2d_input_data)
    verify_model(Conv2D2(), input_data=conv2d_input_data)
    verify_model(Conv2D3(), input_data=conv2d_input_data)
    verify_model(Conv2D4(), input_data=conv2d_input_data)
    verify_model(Conv2D1(), conv2d_input_data, input_shape=[[-1, 3, 112, 112]])


@tvm.testing.uses_gpu
def test_forward_dist():
    @paddle.jit.to_static
    def dist(x, y):
        return paddle.dist(x, y, p=2)

    @paddle.jit.to_static
    def dist2(x, y):
        return paddle.dist(x, y, p=20)

    @paddle.jit.to_static
    def dist3(x, y):
        return paddle.dist(x, y, p=float("-inf"))

    @paddle.jit.to_static
    def dist4(x, y):
        return paddle.dist(x, y, p=float("inf"))

    x_shape = [10, 3]
    y_shape = [10, 1]
    x_data = paddle.rand(x_shape, dtype="float32")
    y_data = paddle.rand(y_shape, dtype="float32")
    verify_model(dist, input_data=[x_data, y_data])
    verify_model(dist2, input_data=[x_data, y_data])
    verify_model(dist3, input_data=[x_data, y_data])
    verify_model(dist4, input_data=[x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_dot():
    @paddle.jit.to_static
    def dot(x, y):
        return paddle.dot(x, y)

    x_shape = [10, 3]
    y_shape = [10, 3]
    x_data = paddle.rand(x_shape, dtype="float32")
    y_data = paddle.rand(y_shape, dtype="float32")
    verify_model(dot, input_data=[x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_dropout():
    @paddle.jit.to_static
    def dropout(inputs):
        return nn.functional.dropout(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(dropout, input_data=input_data[0, 0])
    verify_model(dropout, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_expand():
    @paddle.jit.to_static
    def expand1(inputs):
        return paddle.expand(inputs, shape=[2, 3])

    @paddle.jit.to_static
    def expand2(inputs, shape):
        return paddle.expand(inputs, shape=shape)

    x_shape = [3]
    x_data = paddle.rand(x_shape, dtype="float32")
    verify_model(expand1, input_data=[x_data])
    shape = paddle.to_tensor(np.array([2, 3]).astype("int32"))
    verify_model(expand2, [x_data, shape], input_shape=[[3], [2]])


@tvm.testing.uses_gpu
def test_forward_expand_as():
    @paddle.jit.to_static
    def expand_as(x, y):
        z = paddle.expand_as(x, y)
        z += y
        return z

    data_x = paddle.to_tensor([1, 2, 3], dtype="int32")
    data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
    verify_model(expand_as, [data_x, data_y])


@tvm.testing.uses_gpu
def test_forward_ones():
    @paddle.jit.to_static
    def ones1(inputs):
        ones = paddle.ones([1, 3, 10, 10])
        out = inputs + ones
        return out

    @paddle.jit.to_static
    def ones2(inputs):
        shape = paddle.to_tensor([1, 3, 10, 10], dtype="int32")
        ones = paddle.ones(shape)
        out = inputs + ones
        return out

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(ones1, input_data=input_data)
    verify_model(ones2, input_data=input_data)


def test_forward_elemwise():
    class ElemwiseOp(nn.Layer):
        def __init__(self, op_name):
            super(ElemwiseOp, self).__init__()
            self.op_name_ = op_name
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, op_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, input1, input2):
            y = self.func(input1, input2)
            if "equal" in self.op_name_ or "than" in self.op_name_:
                y = paddle.cast(y, "int32")
            return y

    op_list = [
        "floor_divide",
        "floor_mod",
        "maximum",
        "minimum",
        "equal",
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "not_equal",
    ]
    input_shape = [10, 10]
    input_shape_2 = [
        10,
    ]
    x_data = paddle.rand(input_shape, dtype="float32")
    y_data = paddle.rand(input_shape_2, dtype="float32")
    x_data_2 = paddle.randint(1, 100, input_shape_2, dtype="int32")
    y_data_2 = paddle.randint(1, 100, input_shape, dtype="int32")
    for op_name in op_list:
        if op_name not in ["floor_divide"]:
            verify_model(ElemwiseOp(op_name), [x_data, y_data])
        verify_model(ElemwiseOp(op_name), [x_data_2, y_data_2])


@tvm.testing.uses_gpu
def test_forward_gelu():
    @paddle.jit.to_static
    def gelu(inputs):
        return nn.functional.gelu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(gelu, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_activation():
    class Activation(nn.Layer):
        def __init__(self, op_name):
            super(Activation, self).__init__()
            self.op_name_ = op_name
            for candidate in (paddle.nn.functional, paddle):
                self.func = getattr(candidate, op_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.func(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.normal(shape=input_shape) * 10.0
    input_data_2 = paddle.normal(shape=input_shape).astype("float64") * 10.0
    op_list = [
        "elu",
        "hardshrink",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "log_sigmoid",
        "log_softmax",
        "selu",
        "sigmoid",
        "softsign",
    ]
    for op_name in op_list:
        verify_model(Activation(op_name), input_data=input_data)
        verify_model(Activation(op_name), input_data=input_data_2)


@tvm.testing.uses_gpu
def test_forward_isfinite():
    @paddle.jit.to_static
    def isfinite(inputs):
        return paddle.cast(paddle.isfinite(inputs), "int32")

    input_shape = [5, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(isfinite, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_isinf():
    @paddle.jit.to_static
    def isinf(inputs):
        return paddle.cast(paddle.isinf(inputs), "int32")

    input_shape = [5, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(isinf, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_isnan():
    @paddle.jit.to_static
    def isnan(inputs):
        return paddle.cast(paddle.isnan(inputs), "int32")

    input_shape = [5, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(isnan, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_layer_norm():
    @paddle.jit.to_static
    def layer_norm(inputs, weight, bias):
        return nn.functional.layer_norm(inputs, inputs.shape[-1], weight=weight, bias=bias)

    class LayerNorm(nn.Layer):
        def __init__(self):
            super(LayerNorm, self).__init__()
            data_shape = [10]
            self.layer_norm = nn.LayerNorm(data_shape)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.layer_norm(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    weight = paddle.rand([10], dtype="float32")
    bias = paddle.rand([10], dtype="float32")
    verify_model(layer_norm, input_data=[input_data, weight, bias])
    verify_model(LayerNorm(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    @paddle.jit.to_static
    def leaky_relu(inputs):
        return nn.functional.leaky_relu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(leaky_relu, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_logical_op():
    class LogicalOp(nn.Layer):
        def __init__(self, op_name, out=False):
            super(LogicalOp, self).__init__()
            self.out = out
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, op_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, x, y):
            if self.out:
                out = paddle.to_tensor([True, True, True])
                z = self.func(x, y, out=out)
            else:
                z = self.func(x, y)
            return paddle.cast(z, "int32")

    class LogicalOp_not(LogicalOp):
        @paddle.jit.to_static
        def forward(self, x):
            if self.out:
                out = paddle.to_tensor([True, True, True])
                z = self.func(x, out=out)
            else:
                z = self.func(x)
            return paddle.cast(z, "int32")

    op_list = [
        "logical_or",
        "logical_xor",
        "logical_and",
    ]
    x = paddle.to_tensor([True])
    y = paddle.to_tensor([True, False, True, False])
    for op_name in op_list:
        verify_model(LogicalOp(op_name, False), [x, y])
        verify_model(LogicalOp(op_name, True), [x, y])
    verify_model(LogicalOp_not("logical_not", False), [y])
    verify_model(LogicalOp_not("logical_not", True), [y])


@tvm.testing.uses_gpu
def test_forward_multiply():
    @paddle.jit.to_static
    def multiply1(inputs):
        return inputs * inputs

    @paddle.jit.to_static
    def multiply2(inputs):
        return inputs * 1.0 / 2.0

    @paddle.jit.to_static
    def multiply3(inputs, inputs2):
        ones = paddle.ones([10], dtype="float32")
        return inputs * ones / inputs2

    input_shape = [10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(multiply1, input_data=input_data)
    verify_model(multiply2, input_data=input_data)
    input_data2 = paddle.rand(input_shape, dtype="float32")
    verify_model(multiply3, input_data=[input_data, input_data2])


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
def test_forward_meshgrid():
    @paddle.jit.to_static
    def t(x, y, z):
        return paddle.meshgrid(x, y, z)

    x = paddle.randint(low=0, high=100, shape=[2])
    y = paddle.randint(low=0, high=100, shape=[3])
    z = paddle.randint(low=0, high=100, shape=[5])
    verify_model(t, [x, y, z])


def test_forward_mm():
    class Mm(nn.Layer):
        def forward(self, input1, input2):
            return paddle.mm(input1, input2)

    # matrix x vector
    input_data1 = paddle.randn((3, 4), dtype="float32")
    input_data2 = paddle.randn((4,), dtype="float32")
    verify_model(Mm(), input_data=[input_data1, input_data2])

    # matrix x matrix
    input_data1 = paddle.randn((5, 4), dtype="float32")
    input_data2 = paddle.randn((4, 5), dtype="float32")
    verify_model(Mm(), input_data=[input_data1, input_data2])

    # batched matrix x batched matrix
    input_data1 = paddle.randn((10, 3, 4), dtype="float32")
    input_data2 = paddle.randn((10, 4, 5), dtype="float32")
    verify_model(Mm(), input_data=[input_data1, input_data2])

    # batched matrix x broadcasted matrix
    input_data1 = paddle.randn((10, 3, 4), dtype="float32")
    input_data2 = paddle.randn((4, 5), dtype="float32")
    verify_model(Mm(), input_data=[input_data1, input_data2])


@tvm.testing.uses_gpu
def test_forward_mv():
    class Mv(nn.Layer):
        def forward(self, input1, input2):
            return paddle.mv(input1, input2)

    # matrix x vector
    input_data1 = paddle.randn((3, 4), dtype="float32")
    input_data2 = paddle.randn((4,), dtype="float32")
    verify_model(Mv(), input_data=[input_data1, input_data2])


@tvm.testing.uses_gpu
def test_forward_pool2d():
    @paddle.jit.to_static
    def pool2d1(inputs):
        return nn.functional.avg_pool2d(inputs, kernel_size=2, stride=2, padding=0)

    @paddle.jit.to_static
    def pool2d2(inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, output_size=[3, 3])

    @paddle.jit.to_static
    def pool2d3(inputs):
        output = nn.functional.max_pool2d(inputs, kernel_size=2, stride=2, padding=0)
        return output

    @paddle.jit.to_static
    def pool2d4(inputs):
        output, max_indices = nn.functional.max_pool2d(
            inputs, kernel_size=2, stride=2, padding=0, return_mask=True
        )
        return output

    input_data = paddle.uniform(shape=[1, 2, 32, 32], dtype="float32", min=-1, max=1)
    verify_model(pool2d1, input_data, input_shape=[[-1, 2, 32, 32]])
    verify_model(pool2d2, input_data=input_data)
    input_data1 = paddle.uniform(shape=[1, 2, 1, 50], dtype="float32", min=-1, max=1)
    verify_model(pool2d3, input_data=input_data1)


@tvm.testing.uses_gpu
def test_forward_rank():
    class Rank(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            rank = paddle.rank(inputs)
            rank = paddle.unsqueeze(rank, axis=0)
            output = inputs + rank
            return output

    input_shape = [1, 2, 1, 3, 1]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(Rank(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reshape():
    @paddle.jit.to_static
    def reshape1(inputs, new_shape):
        return paddle.reshape(inputs, new_shape)

    @paddle.jit.to_static
    def reshape2(inputs):
        return inputs.reshape([-1])

    @paddle.jit.to_static
    def reshape3(inputs):
        data_shape = inputs.shape
        return inputs.reshape([data_shape[1], data_shape[2], data_shape[0]])

    @paddle.jit.to_static
    def reshape4(inputs, x):
        new_shape = paddle.shape(x)
        return paddle.reshape(inputs, [new_shape[2], 2, -1])

    input_shape = [2, 1, 10, 1, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    input_data2 = paddle.randn([2, 1, 10, 10])
    new_shape = paddle.shape(input_data2)
    verify_model(reshape1, [input_data, new_shape], input_shape=[[2, 1, 10, 1, 10], [4]])
    verify_model(reshape2, input_data=input_data)
    verify_model(reshape3, input_data=paddle.randn((2, 3, 4)))
    verify_model(reshape4, input_data=[input_data, input_data2])


@tvm.testing.uses_gpu
def test_forward_scale():
    @paddle.jit.to_static
    def scale1(inputs):
        return paddle.scale(inputs, scale=2.0, bias=1.0)

    @paddle.jit.to_static
    def scale2(inputs):
        return paddle.scale(inputs, scale=3, bias=2.1, act="gelu")

    input_data = paddle.randn(shape=[2, 3], dtype="float32")
    verify_model(
        scale1,
        input_data=[
            input_data,
        ],
    )
    verify_model(scale2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_scatter():
    @paddle.jit.to_static
    def scatter(x, index, updates):
        return paddle.scatter(x, index, updates, overwrite=True)

    @paddle.jit.to_static
    def scatter2(x, index, updates):
        return paddle.scatter(x, index, updates, overwrite=False)

    x = paddle.rand([10, 8, 5], dtype="float32")
    index = paddle.to_tensor(
        [
            2,
            1,
            0,
            6,
        ]
    )
    updates = paddle.rand([4, 8, 5], dtype="float32")
    verify_model(scatter, [x, index, updates], input_shape=[[-1, 8, 5], [4], [4, 8, 5]])
    verify_model(scatter2, [x, index, updates])


def test_forward_scatter_nd():
    @paddle.jit.to_static
    def scatter_nd(index, updates):
        shape = [3, 5, 9, 10]
        return paddle.scatter_nd(index, updates, shape)

    @paddle.jit.to_static
    def scatter_nd_add(x, index, updates):
        return paddle.scatter_nd_add(x, index, updates)

    index_data = np.array([[1, 1], [0, 1], [1, 3]]).astype(np.int64)
    index = paddle.to_tensor(index_data)
    updates = paddle.rand(shape=[3, 9, 10], dtype="float32")
    verify_model(scatter_nd, [index, updates])
    x = paddle.rand(shape=[3, 5, 4, 9, 10], dtype="float32")
    updates = paddle.rand(shape=[3, 2, 9, 10], dtype="float32")
    index = paddle.randint(0, 3, shape=[3, 2, 3])
    verify_model(scatter_nd_add, [x, index, updates])


@tvm.testing.uses_gpu
def test_forward_slice():
    @paddle.jit.to_static
    def slice1(inputs, end):
        return inputs[:, :, :, :end]

    @paddle.jit.to_static
    def slice2(inputs):
        return inputs[0, :, :-3, :]

    @paddle.jit.to_static
    def slice3(inputs):
        return inputs[0::2, 0::2] + inputs[1::2, 1::2]

    @paddle.jit.to_static
    def slice4(inputs):
        x0 = paddle.to_tensor([2]) - paddle.to_tensor([1])
        x1 = paddle.to_tensor([3]) + paddle.to_tensor([1])
        return inputs[:, x0:, 1:x1, :]

    @paddle.jit.to_static
    def slice5(inputs):
        x0 = paddle.to_tensor([3])
        return inputs[:, 1::1, 2::x0, 4:10]

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    end = paddle.to_tensor(np.array([3]))
    verify_model(slice1, [input_data, end], input_shape=[[1, 3, 10, 10], [1]])
    verify_model(slice2, input_data=input_data)
    verify_model(slice3, input_data=paddle.randn((4, 4)))
    verify_model(slice4, input_data=input_data)
    verify_model(slice5, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_sort():
    @paddle.jit.to_static
    def sort(inputs):
        return paddle.sort(inputs)

    @paddle.jit.to_static
    def sort2(inputs):
        return paddle.sort(inputs, axis=0, descending=True)

    input_shape = [2, 3, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(sort, input_data)
    input_data2 = np.random.randint(100, size=input_shape)
    verify_model(sort2, input_data2)


@tvm.testing.uses_gpu
def test_forward_subtract():
    class Subtract(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, y):
            return paddle.subtract(x, y)

    input_data1 = paddle.to_tensor([2, np.nan, 5], dtype="float32")
    input_data2 = paddle.to_tensor([1, 4, np.nan], dtype="float32")
    verify_model(Subtract(), input_data=[input_data1, input_data2])

    input_data1 = paddle.randint(0, 10, (3, 4), dtype="int32")
    input_data2 = paddle.randint(0, 10, (4,), dtype="int32")
    verify_model(Subtract(), input_data=[input_data1, input_data2])

    input_data1 = paddle.randint(0, 10, (10, 3, 4), dtype="int64")
    input_data2 = paddle.randint(0, 10, (3, 4), dtype="int64")
    verify_model(Subtract(), input_data=[input_data1, input_data2])


@tvm.testing.uses_gpu
@tvm.testing.uses_gpu
def test_forward_while():
    class While(nn.Layer):
        def __init__(self):
            super(While, self).__init__()

        def forward(self, x):
            s = paddle.shape(x)
            i = paddle.slice(s, axes=[0], starts=[0], ends=[1])
            y = paddle.to_tensor(np.array([5]).astype("int32"))
            while i < y:
                i *= np.array([3], dtype="int32")
            return i

    input_data1 = paddle.rand([1, 3, 224, 224], dtype="float32")
    verify_model(While(), input_data=[input_data1], input_shape=[[-1, 3, -1, -1]])


if __name__ == "__main__":
    #test_forward_add_subtract()
    #test_forward_addmm()
    #test_forward_argmax()
    #test_forward_argmin()
    #test_forward_argsort()
    #test_forward_assign()
    #test_forward_batch_norm()
    #test_forward_cast()
    #test_forward_clip()
    #test_forward_concat_unsqueeze()
    #test_forward_conv()
    #test_forward_crop()
    #test_forward_cumsum()
    #test_forward_dist()
    #test_forward_dot()
    #test_forward_dropout()
    #test_forward_elemwise()
    #test_forward_expand()
    #test_forward_expand_as()
    #test_forward_ones()
    #test_forward_gelu()
    #test_forward_math()
    #test_forward_activation()
    #test_forward_isinf()
    #test_forward_layer_norm()
    #test_forward_leaky_relu()
    #test_forward_logical_op()
    #test_forward_lstm()
    #test_forward_gru()
    #test_forward_matmul()
    #test_forward_meshgrid()
    #test_forward_mm()
    #test_forward_mv()
    #test_forward_multiply()
    #test_forward_pool2d()
    test_forward_rank()
    test_forward_reshape()
    test_forward_scale()
    test_forward_scatter()
    test_forward_scatter_nd()
    test_forward_slice()
    test_forward_sort()
    test_forward_subtract()
    test_forward_math()
