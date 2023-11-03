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

import numpy as np
import tvm
from tvm import relay, te
from tvm.contrib import graph_executor
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.relay.testing.temp_op_attr import TempOpAttr


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
    casted_data = relay.op.cast(data, "int32")
    casted_kernel = relay.op.cast(kernel, "int32")
    shifted_data = relay.op.subtract(casted_data, relay.const(input_zero_point, "int32"))
    shifted_kernel = relay.op.subtract(casted_kernel, relay.const(kernel_zero_point, "int32"))
    func = relay.op.nn.conv2d_transpose(
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
    func = relay.qnn.conv2d_transpose(
        data,
        kernel,
        input_zero_point=relay.const(input_zero_point, "int32"),
        kernel_zero_point=relay.const(kernel_zero_point, "int32"),
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
            libs = relay.build(func, "llvm", params=params)
            mod = graph_executor.create(libs.graph_json, libs.lib, device=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**libs.params)
            mod.run()
            res = mod.get_output(0).numpy()
            return res

    golden_inputs = get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype)
    golden_output = get_output(ref_func, golden_inputs)
    qnn_output = get_output(qnn_func, golden_inputs)
    np.testing.assert_equal(qnn_output, golden_output)


def test_no_zero_point():
    # uint8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (1, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = "int8"
    kernel_shape = (1, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_kernel_zero_point():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = "int8"
    kernel_shape = (1, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_input_zero_point():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = "int8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_both_zero_point():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = "int8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_different_dtype():
    # uint8 input and int8 weight
    data_shape = (2, 4, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
        channels=kernel_shape[1],
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # int8 input and uint8 weight
    data_shape = (2, 4, 2, 4)
    data_dtype = "int8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
        channels=kernel_shape[1],
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_layout():
    # uint8 input
    data_shape = (2, 2, 4, 4)  # NHWC
    data_dtype = "uint8"
    kernel_shape = (2, 2, 3, 4)  # HWOI
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
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    data_shape = (2, 2, 4, 3)  # NHWC
    data_dtype = "uint8"
    kernel_shape = (2, 2, 1, 3)  # HWOI
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
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_padding():
    # uint8 input
    data_shape = (1, 4, 2, 2)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # Try different layout
    data_shape = (2, 2, 4, 4)  # NHWC
    data_dtype = "uint8"
    kernel_shape = (2, 2, 3, 4)  # HWOI
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
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)

    # Try asymmetric padding
    data_shape = (2, 8, 6, 4)  # NHWC
    data_dtype = "uint8"
    kernel_shape = (2, 2, 3, 4)  # HWOI
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
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_const_folding():
    data_shape = (2, 4, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (4, 3, 2, 2)
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
        kernel_layout="IOHW",
        out_dtype="int32",
        channels=kernel_shape[1],
        groups=1,
    )
    folded_mod = transform.FoldConstant()(qnn_func)
    folded_func = folded_mod["main"]
    assert "reshape" not in folded_func.astext()


def test_broadcast_layout():
    # Test broadcast support for NHWC layout.
    data_shape = (1, 229, 229, 3)  # NHWC
    data_dtype = "uint8"
    kernel_shape = (7, 7, 64, 3)  # HWOI
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
        kernel_layout="HWOI",
        out_dtype="int32",
    )
    func = qnn_func["main"].body
    bias = relay.var("bias", shape=(64,), dtype="int32")
    bias2 = relay.var("bias2", shape=(1, 233, 233, 64), dtype="int32")

    # Check broadcast support on both lhs and rhs
    func = relay.add(func, bias2)
    func = relay.add(bias2, func)
    func = relay.add(bias, func)
    func = relay.add(func, bias)
    func = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(opt_level=3):
        libs = relay.build(mod, "llvm -mtriple=x86_64-linux-gnu -mcpu=skylake-avx512")


def test_non_scalar_input_scale_zp():
    data_shape = (2, 1, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (1, 3, 2, 2)
    kernel_dtype = "uint8"
    ref_func, qnn_func = get_funcs(
        data_shape=data_shape,
        data_dtype=data_dtype,
        kernel_shape=kernel_shape,
        kernel_dtype=kernel_dtype,
        input_zero_point=[0],
        kernel_zero_point=0,
        input_scale=[1.0],
        kernel_scale=1.0,
        kernel_size=(2, 2),
        padding=(0, 0),
        strides=(1, 1),
        dilation=(1, 1),
        data_layout="NCHW",
        kernel_layout="IOHW",
        out_dtype="int32",
    )
    verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype)


def test_per_channel_kernel_scale():
    data_shape = (2, 1, 2, 4)
    data_dtype = "uint8"
    kernel_shape = (1, 3, 2, 2)
    kernel_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)
    kernel_scales = [2, 2, 2]
    kernel_scales = relay.const(np.array(kernel_scales).astype("float32"))
    func = relay.qnn.conv2d_transpose(
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
        kernel_layout="IOHW",
        out_dtype="int32",
    )

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(mod)


if __name__ == "__main__":
    test_no_zero_point()
    test_input_zero_point()
    test_kernel_zero_point()
    test_both_zero_point()
    test_different_dtype()
    test_layout()
    test_padding()
    test_const_folding()
    test_broadcast_layout()
    test_per_channel_kernel_scale()
