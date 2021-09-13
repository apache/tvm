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
import paddle.nn as nn

import tvm
from tvm.contrib.sparse import array
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor


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
    for arg in mod["main"].params[:parms_num]:
        assert arg.name_hint in input_names
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
def test_forward_unary_op():
    class UnaryOp(nn.Layer):
        def __init__(self, op_name):
            super(UnaryOp, self).__init__()
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
        "exp",
        "floor",
        "log",
        "log2",
        "log10",
        "log1p",
        "numel",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sin",
        "square",
        "tan",
        "tanh",
    ]
    for op_name in op_list:
        verify_model(UnaryOp(op_name), input_data)


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
def test_forward_arange():
    @paddle.jit.to_static
    def arange(inputs):
        return paddle.arange(paddle.shape(inputs)[0], 9, 2.0)

    @paddle.jit.to_static
    def arange2(inputs):
        return paddle.arange(0, 10.0, paddle.shape(inputs)[1], dtype="float32")

    input_shape = [2, 2]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(arange, input_data)
    verify_model(arange2, input_data=input_data)


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
def test_forward_assign():
    @paddle.jit.to_static
    def assign(inputs):
        return paddle.assign(inputs)

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
    def crop2(inputs):
        shape = paddle.to_tensor(np.array([3, 3]).astype("int32"))
        return paddle.crop(inputs, shape=shape, offsets=[0, 1])

    @paddle.jit.to_static
    def crop3(inputs):
        offsets = paddle.to_tensor(np.array([1, 0]).astype("int32"))
        return paddle.crop(inputs, shape=[3, 3], offsets=offsets)

    @paddle.jit.to_static
    def crop4(inputs):
        shape = paddle.to_tensor(np.array([3, 3]).astype("int32"))
        offsets = paddle.to_tensor(np.array([1, 1]).astype("int32"))
        return paddle.crop(inputs, shape=shape, offsets=offsets)

    input_shape = [10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(crop1, input_data=[input_data])
    verify_model(crop2, input_data=[input_data])
    verify_model(crop3, input_data=[input_data])
    verify_model(crop4, input_data=[input_data])


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


@tvm.testing.uses_gpu
def test_forward_conv_transpose():
    # Note we do not test with groups  > 1 because that is not supported
    # in tvm for conv transpose operations

    class Conv2DTranspose1(nn.Layer):
        def __init__(self):
            super(Conv2DTranspose1, self).__init__()
            self.conv_transpose = nn.Conv2DTranspose(3, 5, 3)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.conv_transpose(inputs)

    class Conv2DTranspose2(nn.Layer):
        def __init__(self):
            super(Conv2DTranspose2, self).__init__()
            self.conv_transpose = nn.Conv2DTranspose(
                3,
                5,
                3,
                stride=2,
                padding=[[0, 0], [0, 0], [1, 2], [3, 4]],
                output_padding=1,
                bias_attr=True,
            )

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.conv_transpose(inputs)

    class Conv2DTranspose3(nn.Layer):
        def __init__(self):
            super(Conv2DTranspose3, self).__init__()
            self.conv_transpose = nn.Conv2DTranspose(
                3, 5, 3, stride=3, padding="VALID", output_padding=2, bias_attr=True
            )

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.conv_transpose(inputs)

    # Conv 2D Transpose Tests
    conv2d_transpose_input_shape = [1, 3, 128, 256]
    conv2d_transpose_input_data = paddle.rand(conv2d_transpose_input_shape, dtype="float32")
    verify_model(Conv2DTranspose1(), input_data=conv2d_transpose_input_data)
    verify_model(Conv2DTranspose2(), input_data=conv2d_transpose_input_data)
    verify_model(Conv2DTranspose3(), input_data=conv2d_transpose_input_data)


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
    def expand2(inputs):
        shape = paddle.to_tensor(np.array([2, 3]).astype("int32"))
        return paddle.expand(inputs, shape=shape)

    x_shape = [3]
    x_data = paddle.rand(x_shape, dtype="float32")
    verify_model(expand1, input_data=[x_data])
    verify_model(expand2, input_data=[x_data])


@tvm.testing.uses_gpu
def test_forward_flatten():
    @paddle.jit.to_static
    def flatten1(inputs):
        return paddle.flatten(inputs, start_axis=1, stop_axis=2)

    def flatten2(inputs):
        return paddle.flatten(inputs, start_axis=0, stop_axis=-1)

    x_shape = [3, 100, 100, 4]
    x_data = paddle.rand(x_shape, dtype="float32")
    verify_model(flatten1, input_data=[x_data])
    verify_model(flatten2, input_data=[x_data])


@tvm.testing.uses_gpu
def test_forward_shape_full():
    @paddle.jit.to_static
    def full1(inputs):
        return paddle.full(paddle.shape(inputs), 3.14)

    @paddle.jit.to_static
    def full2(inputs):
        return paddle.full(paddle.shape(inputs), 1.0, dtype=inputs.dtype)

    @paddle.jit.to_static
    def shape1(inputs):
        return paddle.shape(inputs)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(shape1, input_data=[input_data])
    verify_model(full1, input_data=[input_data])
    verify_model(full2, input_data=[input_data])


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
def test_forward_gather_assign_value():
    @paddle.jit.to_static
    def gather1(x):
        index = paddle.to_tensor(np.array([1, 3, 5, 7, 9]).astype("int64"))
        return paddle.gather(x, index, axis=None)

    @paddle.jit.to_static
    def gather2(x):
        index = paddle.to_tensor(np.array([1, 3, 5, 7, 9]).astype("int64"))
        return paddle.gather(x, index, axis=1)

    x_shape = [30, 40]
    x_data = paddle.rand(x_shape, dtype="float32")
    verify_model(gather1, input_data=[x_data])
    verify_model(gather2, input_data=[x_data])


@tvm.testing.uses_gpu
def test_forward_gather_nd():
    @paddle.jit.to_static
    def gather_nd1(x):
        index = paddle.to_tensor(np.array([[0, 1]]).astype("int64"))
        return paddle.gather_nd(x, index)

    @paddle.jit.to_static
    def gather_nd2(x):
        index = paddle.to_tensor(np.array([[0, 1], [1, 2]]).astype("int32"))
        return paddle.gather_nd(x, index)

    x_shape = [30, 40, 20]
    x_data = paddle.rand(x_shape, dtype="float32")
    verify_model(gather_nd1, input_data=[x_data])
    verify_model(gather_nd2, input_data=[x_data])


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


@tvm.testing.uses_gpu
def test_forward_index_select():
    @paddle.jit.to_static
    def index_select1(x, index):
        return paddle.index_select(x, index)

    @paddle.jit.to_static
    def index_select2(x, index):
        return paddle.index_select(x, index, axis=1)

    input_shape = [3, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    index = paddle.to_tensor(np.array([0, 1, 1]).astype("int32"))
    verify_model(index_select1, input_data=[input_data, index])
    verify_model(index_select2, input_data=[input_data, index])


@tvm.testing.uses_gpu
def test_forward_isinf():
    @paddle.jit.to_static
    def isinf(inputs):
        return paddle.cast(paddle.isinf(inputs), "int32")

    input_shape = [5, 5]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(isinf, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_interpolate():
    class TestBilinear(nn.Layer):
        def __init__(self):
            super(TestBilinear, self).__init__()
            self.conv = nn.Conv2D(3, 5, 3, stride=2)

        def forward(self, x):
            shape = paddle.shape(x)[2:]
            y = self.conv(x)
            return nn.functional.interpolate(y, size=shape, mode="nearest")

    def bilinear_interp1(inputs):
        return nn.functional.interpolate(inputs, size=[12, 12], mode="bilinear")

    @paddle.jit.to_static
    def bilinear_interp2(inputs):
        return nn.functional.interpolate(
            inputs, scale_factor=[2.0, 1.0], mode="bilinear", align_corners=True, align_mode=1
        )

    @paddle.jit.to_static
    def bilinear_interp3(inputs):
        return nn.functional.interpolate(inputs, scale_factor=[1.0, 2.0], mode="bicubic")

    @paddle.jit.to_static
    def bilinear_interp4(inputs):
        return nn.functional.interpolate(
            inputs, scale_factor=3.0, mode="bicubic", align_corners=True, align_mode=0
        )

    input_shape = [2, 3, 6, 12]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(TestBilinear(), input_data=input_data)
    verify_model(bilinear_interp1, input_data=input_data)
    verify_model(bilinear_interp2, input_data=input_data)
    verify_model(bilinear_interp3, input_data=input_data)
    verify_model(bilinear_interp4, input_data=input_data)


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

    op_list = [
        "logical_or",
        "logical_and",
    ]
    x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
    y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
    x = paddle.to_tensor(x_data)
    y = paddle.to_tensor(y_data)
    for op_name in op_list:
        verify_model(LogicalOp(op_name, False), [x, y])
        verify_model(LogicalOp(op_name, True), [x, y])


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
def test_forward_lstm():
    lstm_input_shape = [25, 1, 288]

    class LSTM1(nn.Layer):
        def __init__(self):
            super(LSTM1, self).__init__()
            self.lstm = nn.LSTM(288, 48, 2, direction="bidirect", time_major=True)

        @paddle.jit.to_static
        def forward(self, inputs, prev_h, prev_c):
            y, (h, c) = self.lstm(inputs, (prev_h, prev_c))
            return y

    lstm_input_data = paddle.rand(lstm_input_shape, dtype="float32")
    prev_h = paddle.rand([4, 1, 48], dtype="float32")
    prev_c = paddle.rand([4, 1, 48], dtype="float32")
    verify_model(LSTM1(), input_data=[lstm_input_data, prev_h, prev_c])


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
def test_forward_nonzero():
    class Nonzero(nn.Layer):
        def __init__(self, as_tuple=False):
            super().__init__()
            self.as_tuple = as_tuple

        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.nonzero(inputs, self.as_tuple)

    x1 = paddle.to_tensor([[1.0, 0.0, 0.0, 2.0], [0.0, 2.0, 0.0, 1.1], [0.0, 0.0, 3.0, 0.0]])
    verify_model(Nonzero(), x1, input_shape=[[3, 4]])
    verify_model(Nonzero(True), x1, input_shape=[[3, 4]])
    x2 = paddle.to_tensor([0, 1, 0, 3])
    verify_model(
        Nonzero(),
        x2,
        input_shape=[
            [
                4,
            ]
        ],
    )


def test_forward_norm():
    class Norm1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float("inf"), axis=None, keepdim=False)

    class Norm2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float("-inf"), axis=None, keepdim=False)

    class Norm3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float("-inf"), axis=None, keepdim=True)

    class Norm4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float("inf"), axis=[1, 2], keepdim=False)

    class Norm5(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float("inf"), axis=-1, keepdim=True)

    class Norm6(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float(0.5), axis=1, keepdim=True)

    class Norm7(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float(1), axis=None, keepdim=False)

    class Norm8(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float(2.0), axis=1, keepdim=False)

    class Norm9(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float(-0.5), axis=[1, 2], keepdim=False)

    class Norm10(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.norm(inputs, p=float(-2), axis=(1), keepdim=False)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(Norm1(), input_data=input_data)
    verify_model(Norm2(), input_data=input_data)
    verify_model(Norm3(), input_data=input_data)
    verify_model(Norm4(), input_data=input_data)
    verify_model(Norm5(), input_data=input_data)
    verify_model(Norm6(), input_data=input_data)
    verify_model(Norm7(), input_data=input_data)
    verify_model(Norm8(), input_data=input_data)
    verify_model(Norm9(), input_data=input_data)
    verify_model(Norm10(), input_data=input_data)


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
        return nn.functional.max_pool2d(
            inputs, kernel_size=2, stride=2, padding=0, return_mask=True
        )

    input_data = paddle.uniform(shape=[1, 2, 32, 32], dtype="float32", min=-1, max=1)
    verify_model(pool2d1, input_data=input_data)
    verify_model(pool2d2, input_data=input_data)
    # need op max_pool2d_with_index
    # verify_model(pool2d3, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_pad():
    class Pad1(nn.Layer):
        def __init__(self):
            super(Pad1, self).__init__()
            self.pad = nn.Pad3D(padding=[1, 2, 3, 4, 5, 6], mode="replicate", value=0.5)

        @paddle.jit.to_static
        def forward(self, inputs):
            return self.pad(inputs)

    @paddle.jit.to_static
    def pad2(inputs):
        return paddle.nn.functional.pad(
            inputs, [1, 3, 1, 4, 1, 0], mode="constant", value=2.2, data_format="NDHWC"
        )

    @paddle.jit.to_static
    def pad3(inputs):
        return paddle.nn.functional.pad(
            inputs, [2, 3, 1, 0], mode="reflect", value=2.0, data_format="NCHW"
        )

    @paddle.jit.to_static
    def pad4(inputs):
        return paddle.nn.functional.pad(
            inputs, [2, 1], mode="replicate", value=2.0, data_format="NLC"
        )

    input_data = paddle.rand([2, 3, 6, 7, 8], dtype="float32")
    verify_model(Pad1(), input_data=input_data)
    verify_model(pad2, input_data=input_data)
    input_data = paddle.rand([2, 4, 3, 5], dtype="float32")
    verify_model(pad3, input_data=input_data)
    input_data = paddle.rand([2, 4, 5], dtype="float32")
    verify_model(pad4, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_pow():
    class Pow(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x):
            output = paddle.pow(x, 2)
            return output

    class Pow1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x):
            output = paddle.pow(x, 2.5)
            return output

    class Pow2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, x, y):
            output = paddle.pow(x, y)
            return output

    x_data = paddle.to_tensor([1, 2, 3], dtype="float32")
    y_data = paddle.to_tensor([2], dtype="float32")
    verify_model(Pow(), input_data=[x_data])
    verify_model(Pow1(), input_data=[x_data])
    verify_model(Pow2(), input_data=[x_data, y_data])


@tvm.testing.uses_gpu
def test_forward_reduce_op():
    class ReduceOp_Bool(nn.Layer):
        def __init__(self, op_name):
            super(ReduceOp_Bool, self).__init__()
            self.func = getattr(paddle, op_name, None)

        @paddle.jit.to_static
        def forward(self, inputs):
            inputs = paddle.cast(inputs, "bool")
            output = self.func(inputs)
            output = paddle.cast(output, "int32")
            return output

    class ReduceOp_Bool1(ReduceOp_Bool):
        @paddle.jit.to_static
        def forward(self, inputs):
            inputs = paddle.cast(inputs, "bool")
            output = self.func(inputs, axis=0)
            output = paddle.cast(output, "int32")
            return output

    class ReduceOp_Bool2(ReduceOp_Bool):
        @paddle.jit.to_static
        def forward(self, inputs):
            inputs = paddle.cast(inputs, "bool")
            output = self.func(inputs, axis=[0, 1, -1], keepdim=True)
            output = paddle.cast(output, "int32")
            return output

    class ReduceOp_Math(nn.Layer):
        def __init__(self, op_name):
            super(ReduceOp_Math, self).__init__()
            self.func = getattr(paddle, op_name, None)

        @paddle.jit.to_static
        def forward(self, inputs):
            output = self.func(inputs)
            return output

    class ReduceOp_Math1(ReduceOp_Math):
        @paddle.jit.to_static
        def forward(self, inputs):
            output = self.func(inputs, axis=0)
            return output

    class ReduceOp_Math2(ReduceOp_Math):
        @paddle.jit.to_static
        def forward(self, inputs):
            output = self.func(inputs, axis=[0, 1], keepdim=True)
            return output

    input_data = paddle.randn([1, 2, 3])
    op_list_bool = [
        "all",
        "any",
    ]
    for op_name in op_list_bool:
        verify_model(ReduceOp_Bool(op_name), input_data)
        verify_model(ReduceOp_Bool1(op_name), input_data)
        verify_model(ReduceOp_Bool2(op_name), input_data)
    input_data1 = paddle.rand([2, 4, 5], dtype="float32")
    op_list_math = [
        "max",
        "min",
        "prod",
        "sum",
        "mean",
        "logsumexp",
    ]
    for op_name in op_list_math:
        verify_model(ReduceOp_Math(op_name), input_data1)
        verify_model(ReduceOp_Math1(op_name), input_data1)
        verify_model(ReduceOp_Math2(op_name), input_data1)


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
        return inputs.reshape([data_shape[1], data_shape[2], data_shape[0]])

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

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(slice1, input_data=input_data)
    verify_model(slice2, input_data=input_data)
    # need op "strided_slice"
    # verify_model(slice3, input_data=paddle.randn((4, 4)))
    verify_model(slice4, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_split():
    @paddle.jit.to_static
    def split(inputs):
        return paddle.split(inputs, 2, axis=paddle.to_tensor([0], "int32"))

    @paddle.jit.to_static
    def split2(inputs):
        return paddle.split(inputs, [1, 2, -1], axis=1)

    @paddle.jit.to_static
    def split3(inputs):
        return paddle.split(inputs, [paddle.to_tensor([2]), 1, paddle.to_tensor(3)], axis=0)

    input_shape = [6, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(split, input_data=input_data)
    verify_model(split2, input_data=input_data)
    verify_model(split3, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_squeeze2():
    @paddle.jit.to_static
    def squeeze(inputs):
        return paddle.squeeze(inputs)

    @paddle.jit.to_static
    def squeeze2(inputs):
        return paddle.squeeze(inputs, axis=0)

    @paddle.jit.to_static
    def squeeze3(inputs):
        return paddle.squeeze(inputs, axis=[0, -1])

    input_shape = [1, 2, 1, 3, 1]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(squeeze, input_data=input_data)
    verify_model(squeeze2, input_data=input_data)
    verify_model(squeeze3, input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_topk():
    class Topk1(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3)

    class Topk2(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3, axis=-2)

    class Topk3(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3, axis=3)

    class Topk4(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3, largest=True)

    class Topk5(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3, largest=False)

    class Topk6(nn.Layer):
        @paddle.jit.to_static
        def forward(self, inputs):
            return paddle.topk(inputs, k=3, sorted=True)

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(Topk1(), input_data=input_data)
    verify_model(Topk2(), input_data=input_data)
    verify_model(Topk3(), input_data=input_data)
    verify_model(Topk4(), input_data=input_data)
    verify_model(Topk5(), input_data=input_data)
    verify_model(Topk6(), input_data=input_data)


@tvm.testing.uses_gpu
def test_forward_stack():
    @paddle.jit.to_static
    def stack(input1, input2, input3):
        return paddle.stack([input1, input2, input3])

    @paddle.jit.to_static
    def stack2(input1, input2, input3):
        return paddle.stack([input1, input2, input3], axis=-1)

    input_shape = [2, 3]
    input_data = paddle.rand(input_shape, dtype="float32")
    input_data2 = paddle.rand(input_shape, dtype="float32")
    input_data3 = paddle.rand(input_shape, dtype="float32")
    verify_model(stack, input_data=[input_data, input_data2, input_data3])
    verify_model(stack2, input_data=[input_data, input_data2, input_data3])


@tvm.testing.uses_gpu
def test_forward_tile():
    @paddle.jit.to_static
    def tile(inputs):
        return paddle.tile(inputs, [3, 2])

    @paddle.jit.to_static
    def tile2(inputs, inputs2):
        inputs2 = paddle.shape(inputs2)
        inputs2 = paddle.cast(inputs2, "int32")
        return paddle.tile(inputs, inputs2)

    @paddle.jit.to_static
    def tile3(inputs, inputs2):
        inputs2 = paddle.shape(inputs2)[0]
        inputs2 = paddle.cast(inputs2, "int32")
        return paddle.tile(inputs, [inputs2, 3])

    input_shape = [2, 2]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(
        tile,
        input_data=[
            input_data,
        ],
    )
    input_data2 = paddle.rand([1, 2], dtype="float32")
    verify_model(tile2, input_data=[input_data, input_data2])
    verify_model(tile3, input_data=[input_data, input_data2])


@tvm.testing.uses_gpu
def test_forward_zeros():
    @paddle.jit.to_static
    def zeros1(inputs):
        zeros = paddle.zeros([1, 3, 10, 10])
        out = inputs + zeros
        return out

    @paddle.jit.to_static
    def zeros2(inputs):
        shape = paddle.to_tensor([1, 3, 10, 10], dtype="int32")
        zeros = paddle.zeros(shape)
        out = inputs + zeros
        return out

    input_shape = [1, 3, 10, 10]
    input_data = paddle.rand(input_shape, dtype="float32")
    verify_model(zeros1, input_data=input_data)
    verify_model(zeros2, input_data=input_data)


if __name__ == "__main__":
    test_forward_add_subtract()
    test_forward_addmm()
    test_forward_arange()
    test_forward_argmax()
    test_forward_argmin()
    test_forward_assign()
    test_forward_batch_norm()
    test_forward_cast()
    test_forward_concat_unsqueeze()
    test_forward_conv()
    test_forward_crop()
    test_forward_cumsum()
    test_forward_dot()
    test_forward_dropout()
    test_forward_elemwise()
    test_forward_expand()
    test_forward_flatten()
    test_forward_shape_full()
    test_forward_ones()
    test_forward_ones_like()
    test_forward_gather_assign_value()
    test_forward_gather_nd()
    test_forward_gelu()
    test_forward_hard_sigmoid()
    test_forward_hard_swish()
    test_forward_index_select()
    test_forward_interpolate()
    test_forward_isinf()
    test_forward_layer_norm()
    test_forward_leaky_relu()
    test_forward_logical_op()
    test_forward_look_up()
    test_forward_lstm()
    test_forward_matmul()
    test_forward_multiply()
    test_forward_nonzero()
    test_forward_norm()
    test_forward_pool2d()
    test_forward_pad()
    test_forward_pow()
    test_forward_reduce_op()
    test_forward_reshape()
    test_forward_scale()
    test_forward_slice()
    test_forward_split()
    test_forward_squeeze2()
    test_forward_topk()
    test_forward_tile()
    test_forward_conv_transpose()
    test_forward_unary_op()
    test_forward_zeros()
