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
import platform

import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
from tvm.relay.testing.temp_op_attr import TempOpAttr

# We use llvm target for testing functionality. `llvm` points to an older Intel
# generation machine, that legalizes to a simple lowering. Therefore, the
# legalization is overwritten such that it can be skipped and we use the
# QNNCanonicalizeOps lowering for the testing.
def legalize_qnn_conv2d(attrs, inputs, types):
    return None


def get_ref_func(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    padding,
    strides,
    dilation,
    data_layout,
    kernel_layout,
    out_dtype,
    groups,
    channels=None,
):
    if isinstance(input_zero_point, (int, float)):
        input_zero_point = relay.const(input_zero_point, "int32")
    if isinstance(kernel_zero_point, (int, float)):
        kernel_zero_point = relay.const(kernel_zero_point, "int32")
    else:
        # Kernel zero point expression requires manual broadcasting for some layouts.
        if kernel_layout == "OIHW":
            kernel_zero_point = relay.reshape(kernel_zero_point, [-1, 1, 1, 1])
        elif kernel_layout == "HWOI":
            kernel_zero_point = relay.reshape(kernel_zero_point, [1, 1, -1, 1])

    casted_data = relay.op.cast(data, "int32")
    casted_kernel = relay.op.cast(kernel, "int32")
    shifted_data = relay.op.subtract(casted_data, input_zero_point)
    shifted_kernel = relay.op.subtract(casted_kernel, kernel_zero_point)
    func = relay.op.nn.conv2d(
        shifted_data,
        shifted_kernel,
        padding=padding,
        strides=strides,
        dilation=dilation,
        groups=groups,
        channels=channels,
        kernel_size=kernel_size,
        out_dtype=out_dtype,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    func = relay.Function(relay.analysis.free_vars(func), func)
    return func


def get_qnn_func(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    padding,
    strides,
    dilation,
    data_layout,
    kernel_layout,
    out_dtype,
    channels,
    groups,
):
    if isinstance(input_zero_point, (int, float)):
        input_zero_point = relay.const(input_zero_point, "int32")
    if isinstance(kernel_zero_point, (int, float)):
        kernel_zero_point = relay.const(kernel_zero_point, "int32")

    func = relay.qnn.conv2d(
        data,
        kernel,
        input_zero_point=input_zero_point,
        kernel_zero_point=kernel_zero_point,
        input_scale=relay.const(input_scale, "float32"),
        kernel_scale=relay.const(kernel_scale, "float32"),
        kernel_size=kernel_size,
        strides=strides,
        dilation=dilation,
        padding=padding,
        out_dtype=out_dtype,
        groups=groups,
        channels=channels,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(mod)
    return mod


def get_funcs(
    data_shape,
    data_dtype,
    kernel_shape,
    kernel_dtype,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    padding,
    strides,
    dilation,
    data_layout,
    kernel_layout,
    out_dtype,
    groups=1,
    channels=None,
):
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)

    ref_func = get_ref_func(
        data,
        kernel,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        kernel_size,
        padding,
        strides,
        dilation,
        data_layout,
        kernel_layout,
        out_dtype,
        groups,
        channels,
    )
    ref_func = run_infer_type(ref_func)
    ref_func = tvm.IRModule.from_expr(ref_func)
    qnn_func = get_qnn_func(
        data,
        kernel,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        kernel_size,
        padding,
        strides,
        dilation,
        data_layout,
        kernel_layout,
        out_dtype,
        channels,
        groups,
    )

    return (ref_func, qnn_func)


def verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype):
    def get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype):
        # Keeping inputs multiple of 4 because of a bug in Average Pool2d
        # https://discuss.tvm.apache.org/t/pool2d-gives-bad-output-for-integer-inputs/3377
        low = -128
        high = 127
        if data_dtype == "uint8":
            low = 0
            high = 255
        golden_data = np.random.randint(low=low, high=high, size=data_shape).astype(data_dtype)
        low = -128
        high = 127
        if kernel_dtype == "uint8":
            low = 0
            high = 255
        golden_weight = np.random.randint(low=low, high=high, size=kernel_shape).astype(
            kernel_dtype
        )
        return (golden_data, golden_weight)

    def get_output(func, golden_inputs):
        with tvm.transform.PassContext(opt_level=2):
            golden_data, golden_weight = golden_inputs
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).numpy()
            return res

    golden_inputs = get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype)
    golden_output = get_output(ref_func, golden_inputs)
    qnn_output = get_output(qnn_func, golden_inputs)
    np.testing.assert_equal(qnn_output, golden_output)


def test_no_zero_point():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 1, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 1, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=0,
            kernel_zero_point=0,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # int8 input
        data_shape = (2, 1, 2, 4)
        data_dtype = "int8"
        kernel_shape = (3, 1, 2, 2)
        kernel_dtype = "int8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=0,
            kernel_zero_point=0,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_kernel_zero_point():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=0,
            kernel_zero_point=1,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # int8 input
        data_shape = (2, 1, 2, 4)
        data_dtype = "int8"
        kernel_shape = (3, 1, 2, 2)
        kernel_dtype = "int8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=0,
            kernel_zero_point=5,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_input_zero_point():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=0,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # int8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "int8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "int8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=0,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_both_zero_point():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # int8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "int8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "int8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_dynamic_zero_point():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input with non static zero points.
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        input_zero_point = relay.op.multiply(
            relay.const(2, dtype="int32"), relay.const(2, dtype="int32")
        )
        kernel_zero_point = relay.const(np.random.randint(10, size=[3]), "int32")
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # int8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "int8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "int8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_layout():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 2, 4, 4)  # NHWC
        data_dtype = "uint8"
        kernel_shape = (2, 2, 4, 3)  # HWIO
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # NHWC and HWOI layout. Used in depthwise conv.
        data_shape = (2, 2, 4, 3)  # NHWC
        data_dtype = "uint8"
        kernel_shape = (2, 2, 3, 1)  # HWOI
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            groups=3,
            data_layout="NHWC",
            kernel_layout="HWOI",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_padding():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (1, 4, 2, 2)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=8,
            kernel_zero_point=5,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # Try different layout
        data_shape = (2, 2, 4, 4)  # NHWC
        data_dtype = "uint8"
        kernel_shape = (2, 2, 4, 3)  # HWIO
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=8,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # Try asymmetric padding
        data_shape = (2, 2, 4, 4)  # NHWC
        data_dtype = "uint8"
        kernel_shape = (2, 2, 4, 3)  # HWIO
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=8,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(1, 1, 2, 2),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_dilation():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # Non-zero kernel point - fall back to simpler lowering.
        data_shape = (2, 4, 4, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(2, 2),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # Zero kernel point
        data_shape = (2, 4, 4, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=0,
            kernel_zero_point=0,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(2, 2),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_const_folding():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 2, 2)
        kernel_dtype = "uint8"

        golden_weight = np.random.randint(low=0, high=255, size=kernel_shape).astype(kernel_dtype)
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        kernel = relay.const(golden_weight)
        qnn_func = get_qnn_func(
            data,
            kernel,
            input_zero_point=8,
            kernel_zero_point=3,
            kernel_size=(2, 2),
            input_scale=1.0,
            kernel_scale=1.0,
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
            channels=kernel_shape[0],
            groups=1,
        )
        folded_mod = transform.FoldConstant()(qnn_func)
        folded_func = folded_mod["main"]
        assert "reshape" not in folded_func.astext()


def test_kernel_size_1x1():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 1, 1)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        assert "avg_pool2d" not in qnn_func.astext()
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_kernel_size_1x1_strides_2():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 4, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 4, 1, 1)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=5,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=(2, 2),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        assert "avg_pool2d" not in qnn_func.astext()
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Fails due to encountering none type in autotvm. See https://github.com/apache/tvm/issues/16538",
)
def test_tflite_large_irregular():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (1, 1024, 1, 1)
        data_dtype = "uint8"
        kernel_shape = (1001, 1024, 1, 1)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=127,
            kernel_zero_point=127,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(1, 1),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        golden_data = np.full(data_shape, 127).astype("uint8")
        golden_weight = np.full(kernel_shape, 127).astype("uint8")

        with tvm.transform.PassContext(opt_level=2):
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(qnn_func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            qnn_output = mod.get_output(0).numpy()
        golden_output = np.full((1, 1001, 1, 1), 0).astype("uint8")
        np.testing.assert_equal(qnn_output, golden_output)


def test_tflite_output_multiplier_greater_than_one():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (2, 1, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 1, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_scale=1.0,
            kernel_scale=1.0,
            input_zero_point=128,
            kernel_zero_point=128,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(2, 2),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        golden_data = 128 + np.array((1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4)).reshape(
            data_shape
        ).astype("uint8")
        golden_weight = 128 + np.array((1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1)).reshape(
            kernel_shape
        )
        golden_weight = golden_weight.astype("uint8")

        with tvm.transform.PassContext(opt_level=2):
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(qnn_func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            qnn_output = mod.get_output(0).numpy()
        golden_output = np.array((17, 17, 0, 0, 2, 2, 16, 36, 2, 2, 0, 0)).reshape(2, 3, 1, 2)
        np.testing.assert_equal(qnn_output, golden_output)


def test_tflite_anistropic_strides():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input
        data_shape = (1, 1, 3, 6)
        data_dtype = "uint8"
        kernel_shape = (1, 1, 2, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=127,
            kernel_zero_point=127,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(2, 2),
            padding=(0, 0),
            strides=(1, 3),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )
        golden_data = np.array(
            (
                133,
                131,
                129,
                125,
                123,
                121,
                135,
                133,
                131,
                123,
                121,
                119,
                137,
                135,
                133,
                121,
                119,
                117,
            )
        ).reshape(data_shape)
        golden_data = golden_data.astype("uint8")
        golden_weight = np.array((129, 131, 133, 135)).reshape(kernel_shape)
        golden_weight = golden_weight.astype("uint8")

        with tvm.transform.PassContext(opt_level=2):
            params = {"kernel": golden_weight}
            graph, lib, params = relay.build(qnn_func, "llvm", params=params)
            mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            qnn_output = mod.get_output(0).numpy()
        golden_output = np.array((124, -92, 164, -132)).reshape(1, 1, 2, 2)
        np.testing.assert_equal(qnn_output, golden_output)


def test_broadcast_layout():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # Test broadcast support for NHWC layout.
        data_shape = (1, 229, 229, 3)  # NHWC
        data_dtype = "uint8"
        kernel_shape = (7, 7, 3, 64)  # HWIO
        kernel_dtype = "int8"
        _, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=8,
            kernel_zero_point=3,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(7, 7),
            padding=(1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        func = qnn_func["main"].body
        bias = relay.var("bias", shape=(64,), dtype="int32")
        bias2 = relay.var("bias2", shape=(1, 225, 225, 1), dtype="int32")

        # Check broadcast support on both lhs and rhs
        func = relay.add(func, bias2)
        func = relay.add(bias2, func)
        func = relay.add(bias, func)
        func = relay.add(func, bias)
        func = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(func)
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(
                mod, "llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512"
            )


def test_depthwise_depth_multiplier():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):

        # uint8 input, NCHW and OIHW
        # Depthwise multiplier = 1
        data_shape = (2, 4, 16, 16)
        data_dtype = "uint8"
        kernel_shape = (4, 1, 3, 3)
        kernel_dtype = "uint8"
        input_zero_point = relay.op.multiply(
            relay.const(2, dtype="int32"), relay.const(2, dtype="int32")
        )
        kernel_zero_point = relay.const(np.random.randint(10, size=[4]), "int32")
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(3, 3),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
            groups=4,
            channels=4,
        )

        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # Depthwise multiplier = 2
        data_shape = (10, 4, 16, 16)
        data_dtype = "uint8"
        kernel_shape = (4, 2, 3, 3)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(3, 3),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
            groups=4,
            channels=8,
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # uint8 input, NHWC and HWOI
        # Depthwise multiplier = 1
        data_shape = (2, 16, 16, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 3, 4, 1)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(3, 3),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWOI",
            out_dtype="int32",
            groups=4,
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

        # Depthwise multiplier = 2
        data_shape = (2, 16, 16, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 3, 4, 2)
        kernel_dtype = "uint8"
        ref_func, qnn_func = get_funcs(
            data_shape=data_shape,
            data_dtype=data_dtype,
            kernel_shape=kernel_shape,
            kernel_dtype=kernel_dtype,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            input_scale=1.0,
            kernel_scale=1.0,
            kernel_size=(3, 3),
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWOI",
            out_dtype="int32",
            groups=4,
            channels=8,
        )
        verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_per_channel_kernel_scale():
    with TempOpAttr("qnn.conv2d", "FTVMQnnLegalize", legalize_qnn_conv2d):
        data_shape = (2, 1, 2, 4)
        data_dtype = "uint8"
        kernel_shape = (3, 1, 2, 2)
        kernel_dtype = "uint8"
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)
        kernel_scales = [2, 2, 2]
        kernel_scales = relay.const(np.array(kernel_scales).astype("float32"))
        func = relay.qnn.conv2d(
            data,
            kernel,
            input_zero_point=relay.const(0, "int32"),
            kernel_zero_point=relay.const(0, "int32"),
            input_scale=relay.const(2.0, "float32"),
            kernel_scale=kernel_scales,
            kernel_size=(2, 2),
            channels=kernel_shape[0],
            padding=(0, 0),
            strides=(1, 1),
            dilation=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_dtype="int32",
        )

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)


if __name__ == "__main__":
    test_no_zero_point()
    test_input_zero_point()
    test_kernel_zero_point()
    test_both_zero_point()
    test_layout()
    test_padding()
    test_dilation()
    test_const_folding()
    test_kernel_size_1x1()
    test_kernel_size_1x1_strides_2()
    test_tflite_large_irregular()
    test_broadcast_layout()
    test_tflite_output_multiplier_greater_than_one()
    test_tflite_anistropic_strides()
    test_depthwise_depth_multiplier()
    test_per_channel_kernel_scale()
