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
"""Test arm mprofile dsp."""
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_nhwc, kernel_size, num_filter, strides, padding, dilation",
    [
        ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
        # bug https://github.com/apache/tvm/issues/9226
        ((1, 49, 10, 1), (10, 4), 64, (2, 2), (4, 1, 5, 1), 1),
        # from Visual Wake Word model
        ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1), 1),
        # from Image Classification model (one of the MLPerfTiny models)
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0, 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, 0, 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, 0, 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 0, 1, 1), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (1, 1, 2, 2), 1),
        ((1, 16, 16, 8), (5, 5), 16, 2, (3, 3, 2, 2), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 1, 2, 3), 1),
    ],
)
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_conv2d(data_shape_nhwc, kernel_size, num_filter, strides, padding, dilation, dtype):
    """Test a subgraph with a single conv2d operator."""
    ishape = data_shape_nhwc
    wshape = (*kernel_size, data_shape_nhwc[-1], num_filter)

    weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(weight_data)
    out0 = relay.op.nn.conv2d(
        input0,
        weight0,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=(dilation, dilation),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="int32",
        out_layout="NHWC",
    )
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, dtype))
    weight1 = relay.const(np.moveaxis(weight_data, 2, -1))
    out1 = relay.op.nn.conv2d(
        input1,
        weight1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation=(dilation, dilation),
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype="int32",
        out_layout="NHWC",
    )
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_nwc, kernel_size, num_filter, strides, padding",
    [
        ((1, 32, 12), 3, 16, 1, 0),
        ((3, 12, 10), 4, 24, 1, 0),
        ((1, 7, 7), 3, 5, 1, 0),
        ((1, 10, 2), 4, 4, 2, (1, 1)),
        ((1, 20, 2), 4, 4, 2, (0, 1)),
        ((1, 16, 4), 1, 12, 1, (1, 0)),
        ((1, 24, 16), 1, 32, 3, (2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_conv1d(data_shape_nwc, kernel_size, num_filter, strides, padding, dtype):
    """Test a subgraph with a single conv1d operator."""
    ishape = data_shape_nwc
    wshape = (kernel_size, data_shape_nwc[-1], num_filter)

    weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(weight_data)
    out0 = relay.op.nn.conv1d(
        input0,
        weight0,
        strides=strides,
        padding=padding,
        data_layout="NWC",
        kernel_layout="WIO",
        out_dtype="int32",
        out_layout="NWC",
    )
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, dtype))
    weight1 = relay.const(np.moveaxis(weight_data, 1, -1))
    out1 = relay.op.nn.conv1d(
        input1,
        weight1,
        strides=strides,
        padding=padding,
        data_layout="NWC",
        kernel_layout="WOI",
        out_dtype="int32",
        out_layout="NWC",
    )
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "dim_m, dim_k, dim_n",
    [
        (1, 32, 64),
        (3, 12, 10),
    ],
)
def test_dense(dim_m, dim_k, dim_n):
    """Test a subgraph with a single dense operator."""
    ishape = (dim_m, dim_k)
    wshape = (dim_n, dim_k)

    input0 = relay.var("input", relay.TensorType(ishape, "int8"))
    dense_f = relay.op.nn.batch_flatten(input0)
    weight0 = relay.const(np.random.randint(low=-10, high=10, size=wshape, dtype="int8"))
    out = relay.op.nn.dense(dense_f, weight0, out_dtype="int32")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int8")}
    output_list = generate_ref_data(mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_nhwc, pool_size, strides, padding",
    [
        ((1, 32, 32, 1), (3, 3), 1, 0),
        ((1, 32, 20, 4), (3, 3), (2, 2), 0),
    ],
)
def test_maxpool_2d(data_shape_nhwc, pool_size, strides, padding):
    """Test a subgraph with a single maxpool_2d operator."""

    ishape = data_shape_nhwc

    input0 = relay.var("input", relay.TensorType(ishape, "int8"))
    out = relay.op.nn.max_pool2d(input0, pool_size, layout="NHWC", strides=strides, padding=padding)

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int8")}
    output_list = generate_ref_data(mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_nwc, pool_size, strides, padding",
    [
        ((1, 32, 1), 3, 1, 0),
        ((1, 20, 4), 3, 2, 0),
    ],
)
def test_maxpool_1d(data_shape_nwc, pool_size, strides, padding):
    """Test a subgraph with a single maxpool_1d operator."""
    ishape = data_shape_nwc

    input0 = relay.var("input", relay.TensorType(ishape, "int8"))
    out = relay.op.nn.max_pool1d(input0, pool_size, layout="NWC", strides=strides, padding=padding)

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int8")}
    output_list = generate_ref_data(mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_nchw, pool_size, strides, padding",
    [
        ((1, 1, 32, 32), (3, 3), 1, 0),
        ((1, 4, 32, 20), (3, 3), (2, 2), 0),
    ],
)
def test_avgpool_2d(data_shape_nchw, pool_size, strides, padding):
    """Test a subgraph with a single avgpool_2d operator."""

    ishape = data_shape_nchw

    input0 = relay.var("input", relay.TensorType(ishape, "int32"))
    out0 = relay.nn.avg_pool2d(input0, pool_size, layout="NCHW", strides=strides, padding=padding)
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, "int16"))
    out1 = relay.op.nn.avg_pool2d(
        input1, pool_size, layout="NCHW", strides=strides, padding=padding
    )
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    input_data = np.random.randint(low=-128, high=127, size=ishape, dtype="int32")
    inputs = {"input": input_data}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(
            module=mod, inputs={"input": input_data.astype(dtype="int16")}, outputs=output_list
        ),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize(
    "data_shape_ncw, pool_size, strides, padding",
    [
        ((1, 1, 32), 3, 1, 0),
        ((1, 4, 20), 3, 2, 2),
    ],
)
def test_avgpool_1d(data_shape_ncw, pool_size, strides, padding):
    """Test a subgraph with a single avgpool_1d operator."""

    ishape = data_shape_ncw

    input0 = relay.var("input", relay.TensorType(ishape, "int32"))
    out0 = relay.op.nn.avg_pool1d(input0, pool_size, layout="NCW", strides=strides, padding=padding)
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, "int16"))
    out1 = relay.op.nn.avg_pool1d(input1, pool_size, layout="NCW", strides=strides, padding=padding)
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    input_data = np.random.randint(low=-10, high=10, size=ishape, dtype="int32")
    inputs = {"input": input_data}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(
            module=mod, inputs={"input": input_data.astype(dtype="int16")}, outputs=output_list
        ),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-mcpu": "cortex-m7",
        },
    )


if __name__ == "__main__":
    tvm.testing.main()
