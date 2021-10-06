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
    wshape = (3, 3, 12, 1)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(np.random.randint(low=-10, high=10, size=wshape, dtype=dtype))
    out = relay.op.nn.conv2d(input0, weight0, kernel_size=(3, 3),
        data_layout="NHWC", kernel_layout="HWOI",
        out_dtype="int32", out_layout="NCHW")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
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
@pytest.mark.parametrize("dtype", ["int8"])
def test_conv1d(dtype):
    """Test a subgraph with a single conv1d operator."""
    ishape = (1, 32, 32)
    wshape = (32, 32, 32)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    weight0 = relay.const(np.random.randint(low=-10, high=10, size=wshape, dtype=dtype))
    out = relay.op.nn.conv1d(input0, weight0,
        data_layout="NCW", kernel_layout="WOI",
        out_dtype="int32", out_layout="NCW")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
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
@pytest.mark.parametrize("dtype", ["int8"])
def test_dense(dtype):
    """Test a subgraph with a single dense operator."""
    ishape = (1, 32)
    wshape = (64, 32)

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    dense_f = relay.op.nn.batch_flatten(input0)
    weight0 = relay.const(np.random.randint(low=-10, high=10, size=wshape, dtype=dtype))
    out = relay.op.nn.dense(dense_f, weight0, out_dtype="int32")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
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
@pytest.mark.parametrize("dtype,ishape", [("int8", (1, 32, 32, 1))])
def test_maxpool_2d(dtype, ishape):
    """Test a subgraph with a single maxpool_2d operator."""

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    out = relay.op.nn.max_pool2d(input0, (3, 3), layout="NHWC")

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
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
@pytest.mark.parametrize("dtype,ishape", [("int8", (1, 32, 32))])
def test_maxpool_1d(dtype, ishape):
    """Test a subgraph with a single maxpool_1d operator."""

    input0 = relay.var("input", relay.TensorType(ishape, dtype))
    out = relay.op.nn.max_pool1d(input0, (3,), layout="NWC", strides=2)

    mod = tvm.IRModule.from_expr(relay.Function([input0], out))
    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype=dtype)}
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

    ishape = (1, 32, 32, 1)

    input0 = relay.var("input", relay.TensorType(ishape, "int32"))
    ap2d0 = relay.op.nn.avg_pool2d(input0, (3, 3), layout="NCHW", strides=(2, 2))
    out0 = relay.op.transpose(ap2d0, (0, 2, 3, 1))
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input", relay.TensorType(ishape, "int16"))
    ap2d1 = relay.op.nn.avg_pool2d(input1, (3, 3), layout="NCHW", strides=(2, 2))
    out1 = relay.op.transpose(ap2d1, (0, 2, 3, 1))
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input": np.random.randint(low=-128, high=127, size=ishape, dtype="int16")}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
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
# @pytest.mark.parametrize("dtype,ishape", [("int32", ])
def test_avgpool_1d():
    """Test a subgraph with a single avgpool_1d operator."""

    ishape = (1, 32, 32)

    input0 = relay.var("input0", relay.TensorType(ishape, "int32"))
    ap1d0 = relay.op.nn.avg_pool1d(input0, (3,), layout="NCW", strides=2)
    out0 = relay.op.transpose(ap1d0, (0, 2, 1))
    ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

    input1 = relay.var("input0", relay.TensorType(ishape, "int16"))
    ap1d1 = relay.op.nn.avg_pool1d(input1, (3,), layout="NCW", strides=2)
    out1 = relay.op.transpose(ap1d1, (0, 2, 1))
    mod = tvm.IRModule.from_expr(relay.Function([input1], out1))

    inputs = {"input0": np.random.randint(low=-128, high=127, size=ishape, dtype="int16")}
    output_list = generate_ref_data(ref_mod, inputs)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
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




###########################################################################
###########################################################################
###########################################################################
x=r"""
def _generate_params(relay_mod):
    import numpy as np

    id = 1
    params = {}
    for _, function in relay_mod.functions.items():
        for param in function.params[1:]:
            name = f'_param_{id}'
            dtype = param.type_annotation.dtype
            shape = param.type_annotation.shape
            low = -3 if len(shape) == 1 else -30
            high = 3 if len(shape) == 1 else 30
            generate_func = np.random.randint if 'int' in dtype else np.random.uniform
            params[name] = generate_func(low, high, size=[int(x) for x in shape], dtype=dtype)
            id += 1
    return params



def complex_test_relay_mod(direct_simd: bool = True):
    import numpy as np
    import tvm
    import tvm.relay as relay

    def rand_data(shape, range, dtype):
        return np.random.randint(*range, size=shape, dtype=dtype)

    def conv1d_kernel_shape(w: int, o: int, i: int):
        return (w, o, i) if direct_simd else (w, i, o)

    def conv2d_kernel_shape(h: int, w: int, o: int, i: int):
        return (h, w, o, i) if direct_simd else (h, w, i, o)

    conv1d_kernel_layout, conv2d_kernel_layout = ("WOI", "HWOI") if direct_simd else ("WIO", "HWIO")
    in_C = 1
    in_N = 1
    conv2d_1_C = 12
    conv2d_2_C = 12
    conv1d_1_C = 12
    conv1d_2_C = 24

    input = relay.var("input", relay.TensorType((in_N, 28, 28, in_C), "int8"))

    # conv2d int16
    conv2d_1_f = relay.cast(input, dtype="int16")
    conv2d_1_w = relay.const(rand_data(conv2d_kernel_shape(3, 3, conv2d_1_C, in_C), dtype="int16", range=(-10, 10)))
    conv2d_1 = relay.op.nn.conv2d(conv2d_1_f, conv2d_1_w, kernel_size=(3, 3), data_layout="NHWC",
                                  kernel_layout=conv2d_kernel_layout,
                                  out_dtype="int32", channels=conv2d_1_C, out_layout="NCHW")
    # avgpool_2d
    ap2d_f = relay.cast(conv2d_1, dtype="int16")
    ap2d = relay.op.nn.avg_pool2d(ap2d_f, (3, 3), layout="NCHW", strides=(2, 2))
    ap2d = relay.op.transpose(ap2d, (0, 2, 3, 1))
    # conv2d int8
    conv2d_2_f = relay.cast(ap2d, dtype="int8")
    conv2d_2_w = relay.const(rand_data(conv2d_kernel_shape(3, 3, conv2d_2_C, conv2d_1_C), dtype="int8", range=(-10, 10)))
    conv2d_2 = relay.op.nn.conv2d(conv2d_2_f, conv2d_2_w, kernel_size=(3, 3), data_layout="NHWC",
                                  kernel_layout=conv2d_kernel_layout,
                                  out_dtype="int32", channels=conv2d_2_C)
    # maxpool_2d
    mp2d_f = relay.cast(conv2d_2, dtype="int8")
    mp2d = relay.op.nn.max_pool2d(mp2d_f, (3, 3), layout="NHWC")
    # conv1d int8
    conv1d_1_f = relay.cast(relay.reshape(mp2d, (in_N, -1, conv2d_2_C)), dtype="int8")
    conv1d_1_w = relay.const(rand_data(conv1d_kernel_shape(3, conv1d_1_C, conv2d_2_C), dtype="int8", range=(-10, 10)))
    conv1d_1 = relay.op.nn.conv1d(conv1d_1_f, conv1d_1_w, kernel_size=3, data_layout="NWC",
                                  kernel_layout=conv1d_kernel_layout,
                                  out_dtype="int32",
                                  channels=conv1d_1_C)
    # maxpool_1d
    mp1d_f = relay.cast(conv1d_1, dtype="int8")
    mp1d = relay.op.nn.max_pool1d(mp1d_f, (3,), layout="NWC", strides=2)

    # conv1d int16
    conv1d_2_f = relay.cast(mp1d, dtype="int16")
    conv1d_2_w = relay.const(rand_data(conv1d_kernel_shape(3, conv1d_2_C, conv1d_1_C), dtype="int16", range=(-10, 10)))
    conv1d_2 = relay.op.nn.conv1d(conv1d_2_f, conv1d_2_w, kernel_size=3, data_layout="NWC",
                                  kernel_layout=conv1d_kernel_layout,
                                  out_dtype="int32",
                                  channels=conv1d_2_C, out_layout="NCW")

    # avgpool_1d
    ap1d_f = relay.cast(conv1d_2, dtype="int16")
    ap1d = relay.op.nn.avg_pool1d(ap1d_f, (3,), layout="NCW", strides=2)
    ap1d = relay.op.transpose(ap1d, (0, 2, 1))

    # dense
    dense_f = relay.op.nn.batch_flatten(ap1d)
    dense_w = relay.const(rand_data((10, 13 * conv1d_2_C), dtype="int16", range=(-10, 10)))
    dense = relay.op.nn.dense(dense_f, dense_w, units=10, out_dtype="int32")

    relay_mod = tvm.IRModule.from_expr(relay.Function([input], dense))

    input = ('input', (1, 28, 28, 1), 'int8')
    output = ('output', (1, 10), 'int32')
    return (relay_mod, _generate_params(relay_mod), input, output)


def get_data(input_shape, input_dtype='int8'):
    import numpy as np

    dataset = []
    for i in range(10):
        data = np.random.randint(low=-128, high=127, size=input_shape, dtype=input_dtype)
        dataset.append((str(i), data))

    return dataset

"""