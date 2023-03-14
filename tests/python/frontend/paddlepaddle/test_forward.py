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
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor
import pytest

import paddle

paddle.disable_signal_handler()
import paddle.nn as nn

PADDLE_TEST_DATA_ROOT_PATH = Path(Path("~").expanduser(), ".tvm_test_data", "paddle")
PADDLE_TEST_DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)
cached_program = list()


def assert_shapes_match(tru, est):
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))


def get_paddle_model(func, input_spec):
    global PADDLE_TEST_DATA_ROOT_PATH
    global cached_program
    model_path = Path(PADDLE_TEST_DATA_ROOT_PATH, "model")

    paddle.jit.save(func, str(model_path), input_spec=input_spec)
    baseline_model = paddle.jit.load(str(model_path))
    if len(cached_program) >= 4:
        cached_program = list()
    cached_program.append(baseline_model._get_program_holder())

    shutil.rmtree(str(PADDLE_TEST_DATA_ROOT_PATH))
    return baseline_model


def verify_model(func, input_data, use_vm=False, rtol=1e-5, atol=1e-5):
    if not (isinstance(input_data, (tuple, list))):
        input_data = [input_data]

    input_spec = []
    input_names = []
    input_shape_dict = {}
    compiled_input = {}
    for idx, data in enumerate(input_data):
        input_name = "input{}".format(idx)
        input_spec.append(
            paddle.static.InputSpec(dtype=data.dtype, shape=data.shape, name=input_name)
        )
        input_names.append(input_name)
        input_shape_dict[input_name] = data.shape
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
    compiled_names = []
    for arg in mod["main"].params:
        assert arg.name_hint in input_names or arg.name_hint in params
        if arg.name_hint in input_names:
            compiled_names.append(arg.name_hint)

    if use_vm:
        tvm_vm_input = []
        for idx, data in enumerate(input_data):
            if isinstance(data, np.ndarray):
                tvm_vm_input.append(data)
            else:
                tvm_vm_input.append(data.numpy())
        for target, dev in tvm.testing.enabled_targets():
            result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
                *tvm_vm_input, **params
            )
            tvm_vm_output = []
            if isinstance(result, tvm.runtime.NDArray):
                tvm_vm_output = result.numpy()
            else:
                tvm_vm_output = [r.numpy() for r in result]
            if not isinstance(tvm_vm_output, list):
                tvm_vm_output = [tvm_vm_output]

            for i, baseline_output in enumerate(baseline_outputs):
                assert_shapes_match(baseline_output, tvm_vm_output[i])
                tvm.testing.assert_allclose(baseline_output, tvm_vm_output[i], rtol=rtol, atol=atol)
    else:
        with tvm.transform.PassContext(opt_level=3):
            for target, dev in tvm.testing.enabled_targets():
                lib = relay.build(mod, target=target, params=params)
                gmod = graph_executor.GraphModule(lib["default"](dev))
                for name in compiled_names:
                    gmod.set_input(name, compiled_input[name])
                gmod.run()

                for i, baseline_output in enumerate(baseline_outputs):
                    compiled_output = gmod.get_output(i).numpy()

                    assert_shapes_match(baseline_output, compiled_output)
                    tvm.testing.assert_allclose(
                        baseline_output, compiled_output, rtol=rtol, atol=atol
                    )


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
    class Addmm(nn.Layer):
        def __init__(self, alpha=1.0, beta=1.0):
            super(Addmm, self).__init__()
            self.alpha = alpha
            self.beta = beta

        @paddle.jit.to_static
        def forward(self, inputs, x, y):
            return paddle.addmm(inputs, x, y, self.alpha, self.beta)

    input_shapes = [[10, 10], [1, 1], [7, 1]]
    x_shapes = [[10, 3], [5, 6], [7, 7]]
    y_shapes = [[3, 10], [6, 2], [7, 3]]
    input_shapes = [[10, 10]]
    x_shapes = [[10, 3]]
    y_shapes = [[3, 10]]

    for i in range(len(input_shapes)):
        input_data = paddle.rand(input_shapes[i], dtype="float32")
        x_data = paddle.rand(x_shapes[i], dtype="float32")
        y_data = paddle.rand(y_shapes[i], dtype="float32")
        verify_model(Addmm(), input_data=[input_data, x_data, y_data])
        verify_model(Addmm(0.5, 0.3), input_data=[input_data, x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_arg_max_min():
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
            return inputs.argmax(axis=1, keepdim=False)

    class ArgMin3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return inputs.argmin(axis=2, keepdim=True)

    input_shapes = [[256], [5, 28], [10, 5, 4], [1, 3, 8, 8]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(ArgMax(), input_data=input_data)
        verify_model(ArgMin(), input_data=input_data)
    for input_shape in input_shapes[1:]:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(ArgMax1(), input_data=input_data)
        verify_model(ArgMax2(), input_data=input_data)
        verify_model(ArgMin1(), input_data=input_data)
        verify_model(ArgMin2(), input_data=input_data)
    for input_shape in input_shapes[2:]:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(ArgMax3(), input_data=input_data)
        verify_model(ArgMin3(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_argsort():
    class ArgSort1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.argsort(inputs)

    class ArgSort2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.argsort(inputs, axis=0, descending=True)

    class ArgSort3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.argsort(inputs, axis=-1, descending=True)

    input_shapes = [[256], [10, 20], [10, 5, 3], [1, 3, 5, 5]]
    for input_shape in input_shapes:
        # Avoid duplicate elements in the array which will bring
        # different results with different sort algorithms
        np.random.seed(13)
        np_data = np.random.choice(range(-5000, 5000), np.prod(input_shape), replace=False)
        input_data = paddle.to_tensor(np_data.reshape(input_shape).astype("int64"))
        verify_model(ArgSort1(), [input_data])
        verify_model(ArgSort2(), [input_data])
        verify_model(ArgSort3(), [input_data])


@tvm.testing.uses_gpu
def test_forward_assign():
    @paddle.jit.to_static
    def assign(inputs):
        return paddle.assign(inputs)

    @paddle.jit.to_static
    def assign_value(inputs):
        x = paddle.to_tensor(np.array([3]).astype("float32"))
        return inputs + x

    input_shape = [2, 3]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(
        assign,
        [
            input_data,
        ],
    )
    input_data2 = np.random.randint(100, size=input_shape)
    verify_model(
        assign,
        [
            input_data2,
        ],
    )
    verify_model(assign_value, [input_data])


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
def test_forward_bmm():
    class Bmm(nn.Layer):
        def __init__(self):
            super(Bmm, self).__init__()

        @paddle.jit.to_static
        def forward(self, x, y):
            return paddle.bmm(x, y)

    x_shapes = [[10, 3, 4], [5, 6, 2], [1, 7, 7]]
    y_shapes = [[10, 4, 5], [5, 2, 7], [1, 7, 3]]
    for i in range(len(x_shapes)):
        x_data = paddle.rand(x_shapes[i], dtype="float32")
        y_data = paddle.rand(y_shapes[i], dtype="float32")
        verify_model(Bmm(), input_data=[x_data, y_data])


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
    verify_model(
        cast1,
        [
            input_data,
        ],
    )
    verify_model(
        cast2,
        [
            input_data,
        ],
    )


@tvm.testing.uses_gpu
def test_forward_check_tensor():
    class IsFinite(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cast(paddle.isfinite(inputs), "int32")

    class IsNan(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cast(paddle.isnan(inputs), "int32")

    class IsInf(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.cast(paddle.isinf(inputs), "int32")

    input_shapes = [[32], [8, 32], [2, 5, 20], [2, 3, 8, 8], [2, 2, 3, 6, 6]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(IsFinite(), input_data=input_data)
        verify_model(IsNan(), input_data=input_data)
        verify_model(IsInf(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_clip():
    class Clip1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.clip(inputs, min=0.3, max=0.55)

    class Clip2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs, max_value):
            return paddle.clip(inputs, max=max_value)

    class Clip3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs, min_value):
            return paddle.clip(inputs, min=min_value)

    class Clip4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs, min_value, max_value):
            return paddle.clip(inputs, min=min_value, max=max_value)

    input_data = paddle.rand((2, 2, 2, 3), dtype="float32")
    max_value = paddle.to_tensor([0.55])
    min_value = paddle.to_tensor([0.3])
    verify_model(Clip1(), input_data)
    verify_model(Clip2(), [input_data, max_value])
    verify_model(Clip3(), [input_data, min_value])
    verify_model(Clip4(), [input_data, min_value, max_value])


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
def test_forward_cumsum():
    @paddle.jit.to_static
    def cusum1(inputs):
        return paddle.cumsum(inputs)

    @paddle.jit.to_static
    def cusum2(inputs):
        return paddle.cumsum(inputs, axis=0)

    @paddle.jit.to_static
    def cusum3(inputs):
        return paddle.cumsum(inputs, axis=1)

    input_data = paddle.randint(0, 100, (10, 10), dtype=paddle.int32)
    verify_model(cusum1, [input_data])
    verify_model(cusum1, [input_data.astype(paddle.int64)])
    verify_model(
        cusum2,
        [
            input_data,
        ],
    )
    verify_model(
        cusum3,
        [
            input_data,
        ],
    )


@tvm.testing.uses_gpu
def test_forward_conv():
    class Conv2D1(nn.Layer):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
            super(Conv2D1, self).__init__()
            self.conv = nn.Conv2D(
                3,
                6,
                3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
            )
            self.softmax = nn.Softmax()

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.softmax(self.conv(inputs))

    input_shapes = [[1, 3, 10, 10], [1, 3, 12, 12]]

    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Conv2D1(), input_data=input_data)
        verify_model(Conv2D1(stride=2, padding="VALID", dilation=3), input_data=input_data)
        verify_model(Conv2D1(stride=2, padding="SAME", dilation=3), input_data=input_data)
        verify_model(
            Conv2D1(stride=2, padding=3, dilation=3, padding_mode="replicate"),
            input_data=input_data,
        )
        verify_model(Conv2D1(stride=2, padding="SAME", dilation=2, groups=3), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_conv_transpose():
    class Conv2DTranspose(nn.Layer):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
            super(Conv2DTranspose, self).__init__()
            self.conv = nn.Conv2DTranspose(
                6,
                3,
                3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            self.softmax = nn.Softmax()

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.softmax(self.conv(inputs))

    input_shapes = [[1, 6, 10, 10], [2, 6, 8, 8]]

    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Conv2DTranspose(), input_data=input_data)
        verify_model(Conv2DTranspose(stride=2, padding="VALID"), input_data=input_data)
        verify_model(Conv2DTranspose(stride=2, padding="SAME", dilation=1), input_data=input_data)
        verify_model(Conv2DTranspose(stride=2, padding=3), input_data=input_data)
        verify_model(Conv2DTranspose(stride=3, padding="SAME", groups=1), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_conv3d():
    class Conv3D(nn.Layer):
        def __init__(self, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
            super(Conv3D, self).__init__()
            self.conv = nn.Conv3D(
                3,
                6,
                3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                padding_mode=padding_mode,
            )
            self.softmax = nn.Softmax()

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.softmax(self.conv(inputs))

    input_shapes = [[1, 3, 10, 10, 10], [1, 3, 12, 12, 12]]

    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Conv3D(), input_data=input_data)
        verify_model(Conv3D(stride=2, padding="VALID", dilation=3), input_data=input_data)
        verify_model(Conv3D(stride=2, padding="SAME", dilation=3), input_data=input_data)
        verify_model(
            Conv3D(stride=2, padding=(3, 3, 4, 4, 2, 2), dilation=3),
            input_data=input_data,
        )
        verify_model(
            Conv3D(stride=2, padding=3, dilation=3, padding_mode="reflect"),
            input_data=input_data,
        )
        verify_model(
            Conv3D(stride=2, padding=3, dilation=3, padding_mode="replicate"),
            input_data=input_data,
        )
        verify_model(Conv3D(stride=2, padding="SAME", dilation=2, groups=3), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_dot():
    class Dot(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, y):
            return paddle.dot(x, y)

    input_shapes = [[128], [8, 24]]
    for input_shape in input_shapes:
        x_data = paddle.rand(input_shape, dtype="float32")
        y_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Dot(), input_data=[x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_dropout():
    @paddle.jit.to_static
    def dropout(inputs):
        return nn.functional.dropout(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(dropout, input_data=input_data[0, 0])
    verify_model(dropout, input_data=input_data)


def test_forward_elemwise():
    class ElemwiseAPI(nn.Layer):
        def __init__(self, api_name):
            super(ElemwiseAPI, self).__init__()
            self.api_name_ = api_name
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, api_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, input1, input2):
            y = self.func(input1, input2)
            if "equal" in self.api_name_ or "than" in self.api_name_:
                # for compare operation, cast boolean result to int32
                y = paddle.cast(y, "int32")
            return y

    api_list = [
        "equal",
        "floor_divide",
        "greater_equal",
        "greater_than",
        "less_equal",
        "less_than",
        "maximum",
        "minimum",
        "pow",
    ]
    x_shapes = [[128], [8, 20], [4, 20, 3], [2, 3, 8, 8], [2, 3, 3, 9, 9]]
    y_shapes = [[1], [8, 20], [4, 1, 1], [2, 3, 8, 8], [2, 3, 3, 9, 1]]
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        x_data = paddle.randint(1, 10, x_shape, dtype="int32")
        y_data = paddle.randint(1, 10, y_shape, dtype="int32")
        for api_name in api_list:
            if api_name == "pow":
                # only support float for pow
                x_data = x_data.astype("float32")
                y_data = y_data.astype("float32")
            verify_model(ElemwiseAPI(api_name), [x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_expand():
    @paddle.jit.to_static
    def expand1(inputs):
        return paddle.expand(inputs, shape=[2, 128])

    @paddle.jit.to_static
    def expand2(inputs):
        return paddle.expand(inputs, shape=[2, 1, 4, 16])

    @paddle.jit.to_static
    def expand3(inputs):
        return paddle.expand(inputs, shape=[2, 1, 3, 7, 7])

    @paddle.jit.to_static
    def expand4(inputs):
        shape = paddle.to_tensor(np.array([2, 128]).astype("int32"))
        return paddle.expand(inputs, shape=shape)

    @paddle.jit.to_static
    def expand5(inputs):
        shape = paddle.to_tensor(np.array([2, 1, 4, 16]).astype("int32"))
        return paddle.expand(inputs, shape=shape)

    @paddle.jit.to_static
    def expand6(inputs):
        shape = paddle.to_tensor(np.array([2, 1, 3, 7, 7]).astype("int32"))
        return paddle.expand(inputs, shape=shape)

    data = paddle.rand([128], dtype="float32")
    verify_model(expand1, input_data=[data])
    verify_model(expand4, input_data=[data])
    data = paddle.rand([4, 16], dtype="float32")
    verify_model(expand2, input_data=[data])
    verify_model(expand5, input_data=[data])
    data = paddle.rand([1, 3, 7, 7], dtype="float32")
    verify_model(expand3, input_data=[data])
    verify_model(expand6, input_data=[data])


@tvm.testing.uses_gpu
def test_forward_expand_as():
    class ExpandAs(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, y):
            z = paddle.expand_as(x, y)
            z += y
            return z

    x_shapes = [[1], [8, 128], [8, 1, 1], [2, 3, 229, 229], [2, 3, 3, 224, 1]]
    y_shapes = [[128], [8, 128], [8, 200, 300], [2, 3, 229, 229], [2, 3, 3, 224, 224]]
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        x_data = paddle.rand(x_shape, dtype="float32")
        y_data = paddle.rand(y_shape, dtype="float32")
        verify_model(ExpandAs(), [x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_fill_zeros_like():
    class FilZeroLike(nn.Layer):
        def __init__(self, dtype=None):
            super(FilZeroLike, self).__init__()
            self.dtype = dtype

        @paddle.jit.to_static
        def forward(self, x):
            return paddle.zeros_like(x, dtype=self.dtype)

    input_shape = [2, 3, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(FilZeroLike("float32"), input_data=input_data)
    verify_model(FilZeroLike("int32"), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_flatten():
    class Flatten(nn.Layer):
        def __init__(self, start_axis=0, stop_axis=-1):
            super(Flatten, self).__init__()
            self.start_axis = start_axis
            self.stop_axis = stop_axis

        @paddle.jit.to_static
        def forward(self, x):
            return paddle.flatten(x, start_axis=self.start_axis, stop_axis=self.stop_axis)

    input_data = paddle.rand([2, 3, 4, 5, 2], dtype="float32")
    verify_model(Flatten(), input_data=input_data)
    verify_model(Flatten(2), input_data=input_data)
    verify_model(Flatten(2, -2), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_flip():
    class Flip(nn.Layer):
        def __init__(self, axis):
            super(Flip, self).__init__()
            self.axis = axis

        @paddle.jit.to_static
        def forward(self, x):
            return paddle.flip(x, axis=self.axis)

    input_data = paddle.rand([2, 3, 4], dtype="float32")
    verify_model(Flip(0), input_data)
    verify_model(Flip(-1), input_data)
    verify_model(Flip([0, 1]), input_data)


@tvm.testing.uses_gpu
def test_forward_gather():
    class Gather(nn.Layer):
        def __init__(self, axis=None):
            super(Gather, self).__init__()
            self.axis = axis

        @paddle.jit.to_static
        def forward(self, x, index):
            return paddle.gather(x, index, axis=self.axis)

    x_shapes = [[20, 10], [10, 10, 8]]
    index = paddle.to_tensor(np.array([1, 3, 5]).astype("int64"))
    for x_shape in x_shapes:
        x_data = paddle.rand(x_shape, dtype="float32")
        verify_model(Gather(), [x_data, index])
        verify_model(Gather(axis=0), [x_data, index])
        verify_model(Gather(axis=1), [x_data, index])


@tvm.testing.uses_gpu
def test_forward_gather_nd():
    class GatherNd(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, index):
            return paddle.gather_nd(x, index)

    x_shapes = [[20], [8, 8], [4, 5, 6], [3, 4, 3, 5]]
    y_shapes = [[2, 1], [2], [1, 2, 3], [3]]
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        x_data = paddle.rand(x_shape, dtype="float32")
        y_data = paddle.randint(low=0, high=3, shape=y_shape, dtype="int64")
        verify_model(GatherNd(), [x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_group_norm():
    class GroupNorm(nn.Layer):
        def __init__(self, channels, groups):
            super(GroupNorm, self).__init__()
            self.group_norm = paddle.nn.GroupNorm(num_channels=channels, num_groups=groups)

        def forward(self, inputs):
            return self.group_norm(inputs)

    input_shapes = [[1, 4, 6, 6], [2, 2, 4, 7], [2, 8, 1, 1]]
    for input_shape in input_shapes:
        num_channels = input_shape[1]
        input_data = paddle.uniform(input_shape)
        verify_model(GroupNorm(num_channels, 1), input_data, rtol=1e-4, atol=1e-4)
        verify_model(GroupNorm(num_channels, 2), input_data, rtol=1e-4, atol=1e-4)


@tvm.testing.uses_gpu
def test_forward_grid_sampler():
    class GridSampler(nn.Layer):
        def __init__(self, mode="bilinear", padding_mode="zeros", align_corners=True):
            super(GridSampler, self).__init__()
            self.mode = mode
            self.padding_mode = padding_mode
            self.align_corners = align_corners

        def forward(self, x, grid):
            return paddle.nn.functional.grid_sample(
                x,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )

    x_2D = paddle.rand(shape=[4, 4, 8, 8], dtype="float32")
    grid_2D = paddle.rand(shape=[4, 8, 8, 2], dtype="float32")
    verify_model(GridSampler(mode="nearest"), input_data=[x_2D, grid_2D])
    verify_model(GridSampler(padding_mode="reflection"), input_data=[x_2D, grid_2D])
    verify_model(GridSampler(padding_mode="border"), input_data=[x_2D, grid_2D])
    verify_model(GridSampler(align_corners=False), input_data=[x_2D, grid_2D])

    x_3D = paddle.rand(shape=[4, 4, 4, 4, 4], dtype="float32")
    grid_3D = paddle.rand(shape=[4, 8, 8, 8, 3], dtype="float32")
    verify_model(GridSampler(mode="nearest"), input_data=[x_3D, grid_3D])
    verify_model(GridSampler(padding_mode="reflection"), input_data=[x_3D, grid_3D])
    verify_model(GridSampler(padding_mode="border"), input_data=[x_3D, grid_3D])
    verify_model(GridSampler(align_corners=False), input_data=[x_3D, grid_3D])


@tvm.testing.uses_gpu
def test_forward_scatter():
    class Scatter(nn.Layer):
        def __init__(self, overwrite=True):
            super(Scatter, self).__init__()
            self.overwrite = overwrite

        @paddle.jit.to_static
        def forward(self, x, index, updates):
            return paddle.scatter(x, index, updates, overwrite=self.overwrite)

    x_shapes = [[10], [4, 5], [6, 4, 5], [4, 5, 6, 4]]
    index_shapes = [[10], [4], [6], [4]]
    for x_shape, index_shape in zip(x_shapes, index_shapes):
        x_data = paddle.rand(x_shape, dtype="float32")
        updates = paddle.rand(x_shape, dtype="float32") + 1.0
        index = paddle.randint(low=0, high=3, shape=index_shape)
        verify_model(Scatter(), [x_data, index, updates])
        verify_model(Scatter(False), [x_data, index, updates])


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
def test_forward_shape_full():
    @paddle.jit.to_static
    def full1(inputs):
        return paddle.full(paddle.shape(inputs), 3.14)

    @paddle.jit.to_static
    def full2(inputs):
        return paddle.full(paddle.shape(inputs), 1.0, dtype=inputs.dtype)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(full1, input_data=[input_data])
    verify_model(full2, input_data=[input_data])


@tvm.testing.uses_gpu
def test_forward_split():
    class Split(nn.Layer):
        def __init__(
            self, axis=None, num_or_sections=None, axis_is_tensor=False, num_is_tensor=False
        ):
            super(Split, self).__init__()
            self.axis = axis
            self.num_or_sections = num_or_sections
            self.axis_is_tensor = axis_is_tensor
            self.num_is_tensor = num_is_tensor

        @paddle.jit.to_static
        def forward(self, inputs):
            axis = self.axis
            if self.axis_is_tensor:
                axis = paddle.to_tensor(axis, dtype="int32")
            num_or_sections = self.num_or_sections
            if self.num_is_tensor:
                new_num_or_sections = []
                for i in num_or_sections:
                    if isinstance(i, list):
                        i = paddle.to_tensor(i, dtype="int32")
                    new_num_or_sections.append(i)
                num_or_sections = new_num_or_sections
            return paddle.split(inputs, num_or_sections=num_or_sections, axis=axis)

    input_shape = [3, 6, 2]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(Split(axis=1, num_or_sections=3), input_data=input_data)
    verify_model(
        Split(axis=[1], num_or_sections=[2, 3, 1], axis_is_tensor=True), input_data=input_data
    )
    verify_model(
        Split(axis=1, num_or_sections=[2, -1, [3]], num_is_tensor=True), input_data=input_data
    )


@tvm.testing.uses_gpu
def test_forward_squeeze():
    class Squeeze(nn.Layer):
        def __init__(self, axis=None):
            super(Squeeze, self).__init__()
            self.axis = axis

        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.squeeze(inputs, axis=self.axis)

    input_shapes = [[1, 1, 3, 1, 5], [5, 1, 6]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Squeeze(axis=None), input_data=input_data)
        verify_model(Squeeze(axis=1), input_data=input_data)
    input_data = paddle.rand([1], dtype="float32")
    verify_model(Squeeze(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_ones_like():
    @paddle.jit.to_static
    def ones_like1(inputs):
        return paddle.ones_like(inputs)

    @paddle.jit.to_static
    def ones_like2(inputs):
        return paddle.ones_like(inputs, dtype="int32")

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(ones_like1, input_data=input_data)
    verify_model(ones_like2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_gelu():
    @paddle.jit.to_static
    def gelu(inputs):
        return nn.functional.gelu(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(gelu, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_hard_sigmoid():
    @paddle.jit.to_static
    def hard_sigmoid(inputs):
        return nn.functional.hardsigmoid(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(hard_sigmoid, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_hard_swish():
    @paddle.jit.to_static
    def hard_swish(inputs):
        return nn.functional.hardswish(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(hard_swish, input_data=input_data)


def test_forward_instance_norm():
    class InstanceNorm(nn.Layer):
        def __init__(self, num_features, epsilon=1e-05):
            super(InstanceNorm, self).__init__()
            self.instance_norm = paddle.nn.InstanceNorm2D(
                num_features=num_features, epsilon=epsilon
            )

        def forward(self, inputs):
            return self.instance_norm(inputs)

    input_shapes = [[2, 2, 2, 3], [1, 3, 5, 5]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(InstanceNorm(input_shape[1]), input_data)
        verify_model(InstanceNorm(input_shape[1], 1e-03), input_data)


@tvm.testing.uses_gpu
def test_forward_interpolate():
    class Interpolate(nn.Layer):
        def __init__(
            self,
            mode="nearest",
            align_corners=False,
            align_mode=0,
            data_format="NCHW",
            use_scale=False,
            use_list=False,
            use_const=False,
            use_scaler=False,
        ):
            super(Interpolate, self).__init__()
            self.mode = mode
            self.align_corners = align_corners
            self.align_mode = align_mode
            self.data_format = data_format
            self.use_scale = use_scale
            self.use_list = use_list
            self.use_const = use_const
            self.use_scaler = use_scaler

        @paddle.jit.to_static
        def forward(self, x):
            size = np.array([15, 19]).astype("int32")
            scale = np.array([2.0, 1.0]).astype("float32")
            if not self.use_list and not self.use_const:
                size = paddle.to_tensor(size)
                scale = paddle.to_tensor(scale)
            elif not self.use_const:
                size0 = paddle.to_tensor(size[0:1])
                size = [size0, int(size[1])]
            elif not self.use_scaler:
                size = size.tolist()
                scale = scale.tolist()
            else:
                size = list(size)
                h, w = paddle.rand(size).shape  # add decrease_axis
                size = [h, w]
            if not self.use_scale:
                return paddle.nn.functional.interpolate(
                    x,
                    size=size,
                    mode=self.mode,
                    align_corners=self.align_corners,
                    align_mode=self.align_mode,
                    data_format=self.data_format,
                )
            else:
                return paddle.nn.functional.interpolate(
                    x,
                    scale_factor=scale,
                    mode=self.mode,
                    align_corners=self.align_corners,
                    align_mode=self.align_mode,
                    data_format=self.data_format,
                )

    input_data = paddle.rand([1, 2, 8, 12]).astype("float32")
    verify_model(Interpolate(), input_data)
    verify_model(Interpolate(use_list=True), input_data)
    verify_model(Interpolate(use_scale=True, use_const=True), input_data)
    verify_model(Interpolate(use_const=True, use_scaler=True), input_data)
    verify_model(Interpolate("bilinear", use_scale=True), input_data)
    verify_model(Interpolate("bilinear", use_scale=True, align_corners=True), input_data)
    verify_model(
        Interpolate(
            "bilinear",
            use_scale=True,
            align_corners=True,
            align_mode=1,
            data_format="NHWC",
            use_const=True,
        ),
        input_data,
    )
    verify_model(
        Interpolate("bicubic", use_scale=True, align_corners=True, align_mode=1), input_data
    )


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
def test_forward_logical_api():
    class LogicalAPI(nn.Layer):
        def __init__(self, api_name):
            super(LogicalAPI, self).__init__()
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, api_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, x, y):
            z = self.func(x, y)
            return paddle.cast(z, "int32")

    x_shapes = [[128], [8, 20], [4, 20, 3], [2, 3, 8, 8], [2, 3, 3, 9, 9]]
    y_shapes = [[1], [8, 20], [4, 1, 1], [2, 3, 8, 8], [2, 3, 3, 9, 1]]
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        x_data = paddle.randint(0, 2, x_shape).astype("bool")
        y_data = paddle.randint(0, 2, y_shape).astype("bool")
        verify_model(LogicalAPI("logical_and"), [x_data, y_data])
        verify_model(LogicalAPI("logical_or"), [x_data, y_data])
        verify_model(LogicalAPI("logical_xor"), [x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_logical_not():
    class LogicalNot(nn.Layer):
        def __init__(self):
            super(LogicalNot, self).__init__()

        @paddle.jit.to_static
        def forward(self, x):
            return paddle.logical_not(x).astype("int32")

    input_shapes = [[128], [8, 20], [4, 20, 3], [2, 3, 8, 8], [2, 3, 3, 9, 9]]
    for input_shape in input_shapes:
        input_data = paddle.randint(-2, 2, input_shape).astype("bool")
        verify_model(LogicalNot(), input_data)


@tvm.testing.uses_gpu
def test_forward_look_up():
    @paddle.jit.to_static
    def look_up(inputs, weight):
        return nn.functional.embedding(inputs, weight)

    class LookUp(nn.Layer):
        def __init__(self):
            super(LookUp, self).__init__()
            self.embedding = paddle.nn.Embedding(10, 4, sparse=True)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.embedding(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.randint(0, 10, input_shape, dtype="int32")
    weight = paddle.rand([10, 4], dtype="float32")
    verify_model(look_up, input_data=[input_data, weight])
    verify_model(LookUp(), input_data=input_data)


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
def test_forward_pool2d():
    class Pool2D1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.avg_pool2d(inputs, kernel_size=2, stride=2, padding=0)

    class Pool2D2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.adaptive_avg_pool2d(inputs, output_size=[3, 3])

    class Pool2D3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.avg_pool2d(
                inputs,
                kernel_size=3,
                stride=1,
                padding=[1, 1],
                exclusive=False,
                divisor_override=2.5,
            )

    input_shapes = [[1, 2, 8, 8], [1, 3, 10, 10]]
    for input_shape in input_shapes:
        input_data = paddle.uniform(shape=input_shape, dtype="float32", min=-1, max=1)
        verify_model(Pool2D1(), input_data=input_data)
        verify_model(Pool2D2(), input_data=input_data)
        verify_model(Pool2D3(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_pad1d():
    class Pad1D(nn.Layer):
        def __init__(self, padding=0, mode="constant", value=0.0, data_format="NCL"):
            super(Pad1D, self).__init__()
            self.pad1d = paddle.nn.Pad1D(padding, mode=mode, value=value, data_format=data_format)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.pad1d(inputs)

    input_shapes = [[1, 2, 5], [2, 5, 9]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Pad1D(padding=2), input_data=input_data)
        verify_model(Pad1D(padding=[1, 2], data_format="NLC"), input_data=input_data)
        verify_model(Pad1D(padding=[0, 2], value=0.3), input_data=input_data)
        verify_model(Pad1D(padding=[2, 2], mode="reflect"), input_data=input_data)
        verify_model(Pad1D(padding=3, mode="replicate"), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_pad2d():
    class Pad2D(nn.Layer):
        def __init__(self, padding=0, mode="constant", value=0.0, data_format="NCHW"):
            super(Pad2D, self).__init__()
            self.pad2d = paddle.nn.Pad2D(padding, mode=mode, value=value, data_format=data_format)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.pad2d(inputs)

    input_shapes = [[1, 2, 5, 5], [2, 2, 5, 9]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Pad2D(padding=2), input_data=input_data)
        verify_model(Pad2D(padding=[1, 2, 0, 2], data_format="NHWC"), input_data=input_data)
        verify_model(Pad2D(padding=[1, 2, 0, 2], value=0.3), input_data=input_data)
        verify_model(Pad2D(padding=[1, 2, 0, 2], mode="reflect"), input_data=input_data)
        verify_model(Pad2D(padding=3, mode="replicate"), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_pad3d():
    class Pad3D(nn.Layer):
        def __init__(self, padding=0, mode="constant", value=0.0, data_format="NCDHW"):
            super(Pad3D, self).__init__()
            self.pad3d = paddle.nn.Pad3D(padding, mode=mode, value=value, data_format=data_format)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.pad3d(inputs)

    input_shapes = [[1, 2, 2, 5, 5], [1, 2, 2, 5, 9]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        verify_model(Pad3D(padding=2), input_data=input_data)
        verify_model(Pad3D(padding=[1, 2, 0, 2, 1, 1], data_format="NDHWC"), input_data=input_data)
        verify_model(Pad3D(padding=[1, 2, 0, 2, 1, 1], value=0.3), input_data=input_data)
        verify_model(Pad3D(padding=[1, 2, 0, 2, 1, 1], mode="reflect"), input_data=input_data)
        verify_model(Pad3D(padding=3, mode="replicate"), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_transpose():
    class Transpose(nn.Layer):
        def __init__(self, perm):
            super(Transpose, self).__init__()
            self.perm = perm

        @paddle.jit.to_static
        def forward(self, inputs):
            inputs = inputs * 2
            return paddle.transpose(inputs, perm=self.perm)

    input_data = paddle.rand([1, 3, 5, 4, 3], dtype="float32")
    verify_model(Transpose([0, 1, 2, 3, 4]), input_data=input_data)
    verify_model(Transpose([4, 3, 2, 0, 1]), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reduce():
    class Reduce(nn.Layer):
        def __init__(self, op_name, axis=None, keepdim=False):
            super(Reduce, self).__init__()
            self.op_name = op_name
            self.axis = axis
            self.keepdim = keepdim

        @paddle.jit.to_static
        def forward(self, inputs):
            result = getattr(paddle, self.op_name)(inputs, axis=self.axis, keepdim=self.keepdim)
            result = result.astype("float32")
            return result

    input_shapes = [[1, 2, 2, 5, 5], [2, 3, 4], [4, 20], [2, 3, 30, 30]]
    for input_shape in input_shapes:
        input_data = paddle.uniform(min=-3, max=3, shape=input_shape, dtype="float32")
        verify_model(Reduce("all"), input_data=input_data.astype("bool"))
        verify_model(Reduce("any", 1), input_data=input_data.astype("bool"))
        verify_model(Reduce("max", 0, True), input_data=input_data)
        verify_model(Reduce("min", 1, True), input_data=input_data)
        verify_model(Reduce("prod", 0), input_data=input_data)
        verify_model(Reduce("sum", 0, True), input_data=input_data)
        verify_model(Reduce("mean", -1, True), input_data=input_data)
        # logsumexp only supports tensor with rank less than 5
        if len(input_shape) < 5:
            verify_model(Reduce("logsumexp", -1, True), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_reshape():
    @paddle.jit.to_static
    def reshape1(inputs, x):
        new_shape = paddle.shape(x)
        return paddle.reshape(inputs, new_shape)

    @paddle.jit.to_static
    def reshape2(inputs):
        return inputs.reshape([-1])

    @paddle.jit.to_static
    def reshape3(inputs):
        data_shape = inputs.shape
        return inputs.reshape([data_shape[0] * data_shape[1], data_shape[2]])

    @paddle.jit.to_static
    def reshape4(inputs, x):
        new_shape = paddle.shape(x)
        return paddle.reshape(inputs, [new_shape[2], 2, -1])

    input_shape = [2, 1, 10, 1, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    input_data2 = paddle.randn([2, 1, 10, 10])
    verify_model(reshape1, input_data=[input_data, input_data2])
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
def test_forward_slice():
    @paddle.jit.to_static
    def slice1(inputs):
        return inputs[:, :, :, :3]

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
        b, c, h, w = inputs  # add decrease_axis
        return h

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(
        slice1,
        input_data=[
            input_data,
        ],
    )
    verify_model(slice2, input_data=input_data)
    verify_model(slice3, input_data=paddle.randn((4, 4)))
    verify_model(slice4, input_data=input_data)
    # verify_model(slice5, input_data=paddle.randn((4,)))


@tvm.testing.uses_gpu
def test_forward_unique():
    class Unique(nn.Layer):
        def __init__(
            self,
            return_index=False,
            return_inverse=False,
            return_counts=False,
            axis=None,
            dtype="int64",
        ):
            super(Unique, self).__init__()
            self.return_index = return_index
            self.return_inverse = return_inverse
            self.return_counts = return_counts
            self.axis = None
            self.dtype = dtype

        @paddle.jit.to_static
        def forward(self, inputs):
            result = paddle.unique(
                inputs,
                return_inverse=self.return_inverse,
                return_counts=self.return_counts,
                axis=self.axis,
                dtype=self.dtype,
            )
            return result

    input_shape = [2, 3, 5]
    input_data = paddle.rand(input_shape)
    verify_model(Unique(), input_data=input_data)
    verify_model(Unique(return_index=True), input_data=input_data)
    verify_model(Unique(return_index=True, return_inverse=True), input_data=input_data)
    verify_model(
        Unique(return_index=True, return_inverse=True, return_counts=True), input_data=input_data
    )


@tvm.testing.uses_gpu
def run_math_api(func):
    api_name = func.__name__.split("_")[-1]
    print("func_name:", api_name)

    class MathAPI(nn.Layer):
        def __init__(self, api_name):
            super(MathAPI, self).__init__()
            for candidate in (paddle, paddle.nn.functional):
                self.func = getattr(candidate, api_name, None)
                if self.func:
                    break

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.func(inputs)

    input_shapes = [[128], [2, 100], [10, 2, 5], [7, 3, 4, 1]]
    for input_shape in input_shapes:
        input_data = paddle.rand(input_shape, dtype="float32")
        if api_name in ["log", "log2", "log10", "reciprocal", "sqrt", "rsqrt"]:
            # avoid illegal input, all elements should be positive
            input_data = paddle.uniform(input_shape, min=0.01, max=0.99)
        verify_model(MathAPI(api_name), input_data=input_data)


@run_math_api
def test_forward_abs():
    pass


@run_math_api
def test_forward_acos():
    pass


@run_math_api
def test_forward_abs():
    pass


@run_math_api
def test_forward_atan():
    pass


@run_math_api
def test_forward_ceil():
    pass


@run_math_api
def test_forward_cos():
    pass


@run_math_api
def test_forward_cosh():
    pass


@run_math_api
def test_forward_elu():
    pass


@run_math_api
def test_forward_erf():
    pass


@run_math_api
def test_forward_exp():
    pass


@run_math_api
def test_forward_floor():
    pass


@run_math_api
def test_forward_hardshrink():
    pass


@run_math_api
def test_forward_hardtanh():
    pass


@run_math_api
def test_forward_log_sigmoid():
    pass


@run_math_api
def test_forward_log_softmax():
    pass


@run_math_api
def test_forward_log():
    pass


@run_math_api
def test_forward_log2():
    pass


@run_math_api
def test_forward_log10():
    pass


@run_math_api
def test_forward_log1p():
    pass


@run_math_api
def test_forward_reciprocal():
    pass


@run_math_api
def test_forward_relu():
    pass


@run_math_api
def test_forward_round():
    pass


@run_math_api
def test_forward_rsqrt():
    pass


@run_math_api
def test_forward_selu():
    pass


@run_math_api
def test_forward_sigmoid():
    pass


@run_math_api
def test_forward_sign():
    pass


@run_math_api
def test_forward_sin():
    pass


@run_math_api
def test_forward_softplus():
    pass


@run_math_api
def test_forward_sqrt():
    pass


@run_math_api
def test_forward_square():
    pass


@run_math_api
def test_forward_sin():
    pass


@run_math_api
def test_forward_softsign():
    pass


@run_math_api
def test_forward_sqrt():
    pass


@run_math_api
def test_forward_square():
    pass


@run_math_api
def test_forward_swish():
    pass


@run_math_api
def test_forward_tan():
    pass


@run_math_api
def test_forward_tanh():
    pass


@tvm.testing.uses_gpu
def test_forward_meshgrid():
    @paddle.jit.to_static
    def t(x, y, z):
        return paddle.meshgrid(x, y, z)

    x = paddle.randint(low=0, high=100, shape=[2])
    y = paddle.randint(low=0, high=100, shape=[3])
    z = paddle.randint(low=0, high=100, shape=[5])
    verify_model(t, [x, y, z])


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
def test_forward_pixel_shuffle():
    class PixelShuffle(nn.Layer):
        def __init__(self, upscale_factor):
            super(PixelShuffle, self).__init__()
            self.pixel_shuffle = paddle.nn.PixelShuffle(upscale_factor)

        @paddle.jit.to_static
        def forward(self, x):
            return self.pixel_shuffle(x)

    input_shapes = [[1, 4, 3, 3], [2, 8, 2, 5]]
    for input_shape in input_shapes:
        x = paddle.rand(input_shape, dtype="float32")
        verify_model(PixelShuffle(2), x)


@tvm.testing.uses_gpu
def test_forward_prelu():
    class PRelu(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, w):
            return paddle.nn.functional.prelu(x, w)

    x = paddle.normal(shape=[4, 3, 5, 5])
    w = paddle.to_tensor(
        np.array(
            [
                0.25,
            ]
        ).astype("float32")
    )
    verify_model(PRelu(), [x, w])
    w2 = paddle.to_tensor(np.array([0.25, 0.5, 0.8]).astype("float32"))
    verify_model(PRelu(), [x, w2])


@tvm.testing.uses_gpu
def test_forward_arange():
    @paddle.jit.to_static
    def arange(inputs):
        return paddle.arange(paddle.shape(inputs)[0], 9, 2.0)

    @paddle.jit.to_static
    def arange1(inputs):
        return inputs + paddle.arange(0, 10.0, 8, dtype="float32")

    input_shape = [2, 2]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(arange, input_data)
    verify_model(arange1, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_rnn():
    class RNN(nn.Layer):
        def __init__(self, api_name, input_size, hidden_size, num_layers, direction="forward"):
            super(RNN, self).__init__()
            rnn_func = getattr(paddle.nn, api_name, None)
            self.rnn = rnn_func(input_size, hidden_size, num_layers, direction=direction)

        @paddle.jit.to_static
        def forward(self, inputs, prev_h):
            y, h = self.rnn(inputs, prev_h)
            return y

    input_size, hidden_size, num_layers = 8, 16, 2
    input_shape = [4, 5, 8]
    input_data = paddle.rand(input_shape, dtype="float32")

    for api_name in ("SimpleRNN", "GRU"):
        prev_h = paddle.rand([4, 4, 16], dtype="float32")
        verify_model(
            RNN(api_name, input_size, hidden_size, num_layers, direction="bidirectional"),
            input_data=[input_data, prev_h],
        )
        prev_h = paddle.rand([2, 4, 16], dtype="float32")
        verify_model(
            RNN(api_name, input_size, hidden_size, num_layers), input_data=[input_data, prev_h]
        )


@tvm.testing.uses_gpu
def test_forward_topk():
    @paddle.jit.to_static
    def topk1(inputs):
        return paddle.topk(inputs, k=1)

    @paddle.jit.to_static
    def topk2(inputs):
        k = paddle.to_tensor([1], dtype=paddle.int32)
        return paddle.topk(inputs, k=k)

    @paddle.jit.to_static
    def topk3(inputs):
        return paddle.topk(inputs, k=1, largest=False)

    @paddle.jit.to_static
    def topk4(inputs):
        return paddle.topk(inputs, k=2, sorted=True)

    @paddle.jit.to_static
    def topk5(inputs):
        return paddle.topk(inputs, k=2, sorted=False)

    @paddle.jit.to_static
    def topk6(inputs):
        return paddle.topk(inputs, k=1, axis=0)

    # paddle.fluid.layers.topk
    @paddle.jit.to_static
    def topk7(inputs):
        return paddle.fluid.layers.topk(inputs, k=1)

    @paddle.jit.to_static
    def topk8(inputs):
        return paddle.fluid.layers.topk(inputs, k=2)

    input_data = paddle.to_tensor([[1, 4, 5, 7], [3, 6, 2, 5]], dtype=paddle.int32)
    input_data_fp32 = paddle.to_tensor([[1, 4, 5, 7], [3, 6, 2, 5]], dtype=paddle.float32)
    verify_model(topk1, input_data=input_data)
    # verify_model(topk2, input_data=input_data)
    verify_model(topk3, input_data=input_data)
    verify_model(topk4, input_data=input_data)
    verify_model(topk5, input_data=input_data)
    verify_model(topk6, input_data=input_data)
    verify_model(topk7, input_data=input_data_fp32)
    verify_model(topk8, input_data=input_data_fp32)


@tvm.testing.uses_gpu
def test_forward_one_hot_v2():
    @paddle.jit.to_static
    def one_hot_v2_1(inputs):
        return nn.functional.one_hot(inputs, num_classes=4)

    input_data = paddle.to_tensor([1, 1, 3, 0], dtype=paddle.int32)
    verify_model(one_hot_v2_1, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_norm():
    @paddle.jit.to_static
    def norm_1(inputs):
        return paddle.fluid.layers.l2_normalize(inputs, -1, 1e-12)

    def norm_2(inputs):
        return paddle.fluid.layers.l2_normalize(inputs, 1, 1e-12)

    input_data = paddle.to_tensor(
        [[[1, 2], [3, 1], [4, 5]], [[3, 1], [3, 5], [2, 4]]], dtype=paddle.float32
    )
    verify_model(norm_1, input_data=input_data)
    verify_model(norm_2, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_where_index():
    @paddle.jit.to_static
    def where_index_1(inputs):
        return paddle.nonzero(inputs)

    input_data = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    verify_model(where_index_1, input_data=input_data, use_vm=True)


@tvm.testing.uses_gpu
def test_forward_take_along_axis():
    @paddle.jit.to_static
    def take_along_axis_1(inputs, index):
        return paddle.take_along_axis(inputs, index, 0)

    input_data = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = paddle.to_tensor([[0]])
    verify_model(take_along_axis_1, input_data=[input_data, index])


@tvm.testing.uses_gpu
def test_forward_stack():
    class Stack1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, input0, input1, input2):
            return paddle.stack([input0, input1, input2], axis=-1)

    class Stack2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, input0, input1, input2):
            return paddle.stack([input0, input1, input2], axis=1)

    class Stack3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, input0, input1, input2):
            return paddle.stack([input0, input1, input2], axis=2)

    input_shapes = [[2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        input_data_0 = paddle.randn(shape=input_shape, dtype="float32")
        input_data_1 = paddle.randn(shape=input_shape, dtype="float32")
        input_data_2 = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(Stack1(), [input_data_0, input_data_1, input_data_2])
        verify_model(Stack2(), [input_data_0, input_data_1, input_data_2])
        verify_model(Stack3(), [input_data_0, input_data_1, input_data_2])


@tvm.testing.uses_gpu
def test_forward_unstack():
    class UnStack1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.unstack(inputs, axis=-1)

    class UnStack2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.unstack(inputs, axis=1)

    class UnStack3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.unstack(inputs, axis=0)

    input_shapes = [[2, 3], [5, 10, 11], [3, 4, 5, 6], [1, 3, 4, 1, 1]]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(UnStack1(), input_data)
        verify_model(UnStack2(), input_data)
        verify_model(UnStack3(), input_data)


@tvm.testing.uses_gpu
def test_forward_silu():
    class Silu(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.silu(inputs)

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(Silu(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_softshrink():
    @paddle.jit.to_static
    def Softshrink1(input):
        return nn.functional.softshrink(input, threshold=0.0)

    @paddle.jit.to_static
    def Softshrink2(input):
        return nn.functional.softshrink(input, threshold=0.5)

    @paddle.jit.to_static
    def Softshrink3(input):
        return nn.functional.softshrink(input, threshold=1.0)

    x = paddle.to_tensor([-0.9, -0.2, 0.1, 0.8])
    verify_model(Softshrink2, x)

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(Softshrink1, input_data=input_data)
        verify_model(Softshrink2, input_data=input_data)
        verify_model(Softshrink3, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_where():
    @paddle.jit.to_static
    def where1(x, y):
        return paddle.where(x > 1, x, y)

    @paddle.jit.to_static
    def where2(x, y):
        return paddle.where(x > y, x, y)

    x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
    y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
    verify_model(where1, [x, y])

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        x = paddle.randn(shape=input_shape, dtype="float32")
        y = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(where1, [x, y])
        verify_model(where2, [x, y])


@tvm.testing.uses_gpu
def test_forward_tile():
    class Tile1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.tile(inputs, repeat_times=[10])

    class Tile2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.tile(inputs, repeat_times=[2, 3])

    class Tile3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.tile(inputs, repeat_times=[1, 2, 3])

    class Tile4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.tile(inputs, repeat_times=[2, 3, 4, 1, 5])

    class Tile5(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            reps = paddle.to_tensor([3, 2])
            reps = paddle.cast(reps, "int32")
            return paddle.tile(inputs, repeat_times=reps)

    class Tile6(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            rep_0 = paddle.to_tensor([3])
            rep_1 = paddle.to_tensor([2])
            rep_0 = paddle.cast(rep_0, "int32")
            rep_1 = paddle.cast(rep_1, "int32")
            return paddle.tile(inputs, repeat_times=[rep_0, rep_1])

    input_shapes = [
        [10],
        [2, 3],
        [3, 4, 5],
        [5, 3, 1, 4],
        [1, 3, 1, 6, 7],
    ]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(Tile1(), input_data=input_data)
        verify_model(Tile2(), input_data=input_data)
        verify_model(Tile3(), input_data=input_data)
        verify_model(Tile4(), input_data=input_data)
        verify_model(Tile5(), input_data=input_data)
        verify_model(Tile6(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_mish():
    class Mish(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.mish(inputs)

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    if paddle.version.full_version >= "2.4.2":
        for input_shape in input_shapes:
            input_data = paddle.randn(shape=input_shape, dtype="float32")
            verify_model(Mish(), input_data=input_data)
            input_data += 20.0
            verify_model(Mish(), input_data=input_data)

        input_data = paddle.to_tensor([-5.0, 0.0, 5.0, 23.1, 20.0])
        verify_model(Mish(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_thresholded_relu():
    class ThresholdedRelu1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.thresholded_relu(inputs)

    class ThresholdedRelu2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return nn.functional.thresholded_relu(inputs, threshold=0.5)

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        verify_model(ThresholdedRelu1(), input_data=input_data)
        verify_model(ThresholdedRelu2(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_index_select():
    class IndexSelect1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, index):
            return paddle.index_select(x, index, axis=0)

    class IndexSelect2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, index):
            return paddle.index_select(x, index, axis=-1)

    input_shapes = [[10], [2, 3], [5, 10, 11], [3, 4, 5, 6]]
    for input_shape in input_shapes:
        input_data = paddle.randn(shape=input_shape, dtype="float32")
        index = paddle.to_tensor([0, 1, 1], dtype="int32")
        verify_model(IndexSelect1(), input_data=[input_data, index])
        verify_model(IndexSelect2(), input_data=[input_data, index])


@tvm.testing.uses_gpu
def test_forward_eye():
    class Eye1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.eye(3, 5, dtype="int32"), paddle.eye(3, 5, dtype="float32"), inputs

    class Eye2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.eye(5, 3, dtype="int64"), paddle.eye(5, 3, dtype="float64"), inputs

    class Eye3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.eye(0, 3, dtype="int64"), paddle.eye(0, 0, dtype="float64"), inputs

    class Eye4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.eye(4, None, dtype="int64"), paddle.eye(4, None, dtype="float64"), inputs

    x = paddle.to_tensor([1], dtype="float32")
    verify_model(Eye1(), input_data=[x])
    verify_model(Eye2(), input_data=[x])
    verify_model(Eye3(), input_data=[x])
    verify_model(Eye4(), input_data=[x])


@tvm.testing.uses_gpu
def test_forward_linspace():
    class Linspace1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            out1 = paddle.linspace(0.5, 7, 1, "int32")
            out2 = paddle.linspace(1.3, 7.1, 5, "float32")
            out3 = paddle.linspace(1, 1000000000, 10, "int64")
            out4 = paddle.linspace(1, 7.1, 5, "float64")
            return out1, out2, out3, out4, inputs

    class Linspace2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            start = paddle.to_tensor([-2.5])
            stop = paddle.to_tensor([31.6])
            num = paddle.to_tensor([13])
            start = paddle.cast(start, "float32")
            stop = paddle.cast(stop, "float32")
            num = paddle.cast(num, "int32")
            out1 = paddle.linspace(start, stop, num, "int32")
            out2 = paddle.linspace(start, stop, num, "float32")
            out3 = paddle.linspace(start, stop, num, "int64")
            out4 = paddle.linspace(start, stop, num, "float64")
            return out1, out2, out3, out4, inputs

    class Linspace3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, start, stop, num):
            out1 = paddle.linspace(start, stop, num, "int32")
            out2 = paddle.linspace(start, stop, num, "float32")
            out3 = paddle.linspace(start, stop, num, "int64")
            out4 = paddle.linspace(start, stop, num, "float32")
            return out1

    start = paddle.to_tensor([1.3])
    stop = paddle.to_tensor([5.1])
    num = paddle.to_tensor([3])
    start = paddle.cast(start, "float32")
    stop = paddle.cast(stop, "float32")
    num = paddle.cast(num, "int32")
    x = paddle.to_tensor([1], dtype="float32")
    verify_model(Linspace1(), input_data=[x])
    verify_model(Linspace2(), input_data=[x])
    verify_model(Linspace3(), input_data=[start, stop, num], use_vm=True)
    num = paddle.to_tensor([1])
    num = paddle.cast(num, "int32")
    verify_model(Linspace3(), input_data=[start, stop, num], use_vm=True)


@tvm.testing.uses_gpu
def test_forward_dist():
    class Dist(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, y):
            l0_norm = paddle.dist(x, y, 0)
            l2_norm = paddle.dist(x, y, 2)
            float_norm = paddle.dist(x, y, 1.3)
            inf_norm = paddle.dist(x, y, float("inf"))
            ninf_norm = paddle.dist(x, y, float("-inf"))
            return l0_norm, l2_norm, float_norm, inf_norm, ninf_norm

    x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
    y = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
    w = paddle.to_tensor([[1, 2]], dtype="float32")
    v = paddle.to_tensor([[2.1]], dtype="float32")
    verify_model(Dist(), input_data=[x, y])
    verify_model(Dist(), input_data=[x, w])
    verify_model(Dist(), input_data=[w, v])
    verify_model(Dist(), input_data=[y, v])


if __name__ == "__main__":
    tvm.testing.main()
