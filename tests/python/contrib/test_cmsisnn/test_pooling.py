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

"""CMSIS-NN integration tests: Pooling"""
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from tvm.testing.aot import (
    generate_ref_data,
    AOTTestModel,
    compile_and_run,
)
from tvm.micro.testing.aot_test_utils import AOT_USMP_CORSTONE300_RUNNER
from .utils import (
    make_module,
    get_range_for_dtype_str,
    get_same_padding,
    make_qnn_relu,
    assert_partitioned_function,
    assert_no_external_function,
    create_test_runner,
)


def make_model(
    pool_op,
    shape=(1, 28, 28, 12),
    pool_size=(3, 3),
    strides=(2, 2),
    padding="VALID",
    dtype="int8",
    scale=1,
    zero_point=-33,
    relu_type="RELU",
    layout="NHWC",
    input_op=None,
):
    """Return a model and any parameters it may have,
    all parameters are defaulted to known good values
    """
    if input_op:
        op = input_op
    else:
        op = relay.var("input", shape=shape, dtype=dtype)
    pad_ = (0, 0, 0, 0)
    if padding == "SAME":
        dilation = (1, 1)
        pad_ = get_same_padding((shape[1], shape[2]), pool_size, dilation, strides)
        op = relay.nn.pad(
            op,
            pad_width=[(0, 0), (pad_[0], pad_[2]), (pad_[1], pad_[3]), (0, 0)],
            pad_value=zero_point,
            pad_mode="constant",
        )
    if pool_op.__name__ == relay.nn.avg_pool2d.__name__:
        op = relay.cast(op, "int32")
    op = pool_op(
        op, pool_size=pool_size, strides=strides, padding=pad_, ceil_mode=True, layout=layout
    )
    if pool_op.__name__ == relay.nn.avg_pool2d.__name__:
        op = relay.cast(op, dtype)
    op = make_qnn_relu(op, relu_type, scale, zero_point, dtype)
    return op


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("dtype", ["int16", "int8"])
@pytest.mark.parametrize("in_shape", [(1, 28, 28, 12), (1, 64, 100, 4)])
@pytest.mark.parametrize(
    "pool_size, strides, padding", [((3, 3), (2, 2), "SAME"), ((2, 2), (1, 1), "VALID")]
)
@pytest.mark.parametrize("relu_type", ["NONE", "RELU"])
@pytest.mark.parametrize("pool_type", [relay.nn.max_pool2d, relay.nn.avg_pool2d])
@pytest.mark.parametrize("zero_point, scale", [(-34, 0.0256)])
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
def test_ops(
    dtype,
    in_shape,
    pool_size,
    strides,
    padding,
    relu_type,
    pool_type,
    zero_point,
    scale,
    compiler_cpu,
    cpu_flags,
):
    """Tests QNN pooling op for int8 and int16 pooling"""
    interface_api = "c"
    use_unpacked_api = True

    model = make_model(
        pool_op=pool_type,
        shape=in_shape,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        dtype=dtype,
        scale=scale,
        zero_point=zero_point,
        relu_type=relu_type,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    in_min, in_max = get_range_for_dtype_str(dtype)
    np.random.seed(0)
    inputs = {
        "input": np.random.randint(in_min, high=in_max, size=in_shape, dtype=dtype),
    }
    output_list = generate_ref_data(orig_mod["main"], inputs)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=None,
            output_tolerance=1,
        ),
        create_test_runner(compiler_cpu, cpu_flags),
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize(
    "pool_size, strides, padding", [((3, 3), (2, 2), "SAME"), ((2, 2), (1, 1), "VALID")]
)
@pytest.mark.parametrize("relu_type", ["NONE", "RELU"])
def test_int8_pool_with_float32_input(
    pool_size,
    strides,
    padding,
    relu_type,
):
    """Tests QNN maxpool partitions with float32 input"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    in_shape = (1, 28, 28, 12)
    zero_point, scale = (-34, 0.0256)

    input_ = relay.var("input", shape=in_shape, dtype="float32")
    op = relay.op.add(input_, input_)
    op = relay.qnn.op.quantize(op, relay.const(scale), relay.const(zero_point), -1, "int8")

    model = make_model(
        pool_op=relay.nn.max_pool2d,
        shape=in_shape,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        scale=scale,
        zero_point=zero_point,
        relu_type=relu_type,
        input_op=op,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    np.random.seed(0)
    inputs = {"input": np.random.uniform(0, 1, in_shape).astype("float32")}
    output_list = generate_ref_data(orig_mod["main"], inputs)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=None,
            output_tolerance=1,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.nn.avg_pool2d, relay.nn.max_pool2d])
def test_invalid_datatype(op):
    """Checks CMSIS-NN partitioning for non int8 dtype"""
    model = make_model(pool_op=op, dtype="int64")
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)
    assert_no_external_function(cmsisnn_mod)


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.nn.avg_pool2d, relay.nn.max_pool2d])
def test_invalid_batch_size(op):
    """Checks CMSIS-NN partitioning when batch size is not 1"""
    model = make_model(
        pool_op=op,
        shape=(2, 28, 28, 12),
    )

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)
    assert_no_external_function(cmsisnn_mod)


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("op", [relay.nn.avg_pool2d, relay.nn.max_pool2d])
def test_invalid_layout(op):
    """Checks CMSIS-NN partitioning when layout is not NHWC"""
    model = make_model(pool_op=op, layout="NCHW")

    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)
    assert_no_external_function(cmsisnn_mod)


if __name__ == "__main__":
    tvm.testing.main()
