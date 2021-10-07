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
import sys
import numpy as np
import pytest
import tvm
from tvm import relay
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


@tvm.testing.requires_corstone300
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_conv2d(dtype):
    """Test a subgraph with a single conv2d operator."""
    ishape = (1, 32, 32, 1)
    wshape = (3, 3, 1, 12)

    weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(weight_data)
    out0 = relay.op.nn.conv2d(input0, weight0, kernel_size=(3, 3),
        data_layout="NHWC", kernel_layout="HWIO",
        out_dtype="int32", out_layout="NHWC")
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, dtype))
    weight1 = relay.const(np.moveaxis(weight_data, 2, -1))
    out1 = relay.op.nn.conv2d(input1, weight1, kernel_size=(3, 3),
        data_layout="NHWC", kernel_layout="HWOI",
        out_dtype="int32", out_layout="NHWC")
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
        }
    )


@tvm.testing.requires_corstone300
@pytest.mark.parametrize("dtype", ["int8", "int16"])
def test_conv1d(dtype):
    """Test a subgraph with a single conv1d operator."""
    ishape = (1, 32, 32)
    wshape = (32, 32, 32)

    weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(weight_data)
    out0 = relay.op.nn.conv1d(input0, weight0,
        data_layout="NWC", kernel_layout="WIO",
        out_dtype="int32", out_layout="NWC")
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, dtype))
    weight1 = relay.const(np.moveaxis(weight_data, 1, -1))
    out1 = relay.op.nn.conv1d(input1, weight1,
        data_layout="NWC", kernel_layout="WOI",
        out_dtype="int32", out_layout="NWC")
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
        }
    )


@tvm.testing.requires_corstone300
def test_dense():
    """Test a subgraph with a single dense operator."""
    ishape = (1, 32)
    wshape = (64, 32)

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
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
        }
    )


@tvm.testing.requires_corstone300
def test_maxpool_2d():
    """Test a subgraph with a single maxpool_2d operator."""

    ishape = (1, 32, 32, 1)

    input0 = relay.var("input", relay.TensorType(ishape, "int8"))
    out = relay.op.nn.max_pool2d(input0, (3, 3), layout="NHWC")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int8")}
    output_list = generate_ref_data(mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
        },
    )


@tvm.testing.requires_corstone300
def test_maxpool_1d():
    """Test a subgraph with a single maxpool_1d operator."""
    ishape = (1, 32, 32)

    input0 = relay.var("input", relay.TensorType(ishape, "int8"))
    out = relay.op.nn.max_pool1d(input0, (3,), layout="NWC", strides=2)

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int8")}
    output_list = generate_ref_data(mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
        },
    )


@tvm.testing.requires_corstone300
def test_avgpool_2d():
    """Test a subgraph with a single avgpool_2d operator."""

    ishape = (1, 1, 64, 64)

    input0 = relay.var("input", relay.TensorType(ishape, "int32"))
    out0 = relay.nn.avg_pool2d(input0, pool_size=(3, 3), layout="NCHW")
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, "int16"))
    out1 = relay.op.nn.avg_pool2d(input1, pool_size=(3, 3), layout="NCHW")
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    input_data = np.random.randint(low=-128, high=127, size=ishape, dtype="int32")
    inputs = {"input": input_data}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs={"input": input_data.astype(dtype="int16")}, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
            "-mcpu": "cortex-m7",
        },
    )


@tvm.testing.requires_corstone300
def test_avgpool_1d():
    """Test a subgraph with a single avgpool_1d operator."""

    ishape = (1, 32, 32)

    input0 = relay.var("input", relay.TensorType(ishape, "int32"))
    out0 = relay.op.nn.avg_pool1d(input0, (3,), layout="NCW", strides=2)
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, "int16"))
    out1 = relay.op.nn.avg_pool1d(input1, (3,), layout="NCW", strides=2)
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    input_data = np.random.randint(low=-10, high=10, size=ishape, dtype="int32")
    inputs = {"input": input_data}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs={"input": input_data.astype(dtype="int16")}, outputs=output_list),
        runner=AOT_CORSTONE300_RUNNER,
        interface_api='c',
        use_unpacked_api=True,
        target_opts={
            "-keys": "arm_cpu",
            "-march": "armv7e-m",
            "-mcpu": "cortex-m7",
        },
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
