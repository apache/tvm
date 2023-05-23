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

"""CMSIS-NN integration tests: Reshape removal"""
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from tvm.testing.aot import (
    get_dtype_range,
    generate_ref_data,
    AOTTestModel,
    compile_models,
    run_and_check,
)
from tvm.micro.testing.aot_test_utils import AOT_USMP_CORSTONE300_RUNNER
from .utils import (
    make_module,
    get_same_padding,
    make_qnn_relu,
    assert_partitioned_function,
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
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
def test_reshape_removal(padding):
    """Tests reshape is removed from the network"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    in_shape = (1, 28, 28, 12)
    pool_size = (3, 3)
    strides = (2, 2)
    relu_type = "NONE"
    zero_point, scale = (-34, 0.0256)

    max_pool = make_model(
        pool_op=relay.nn.max_pool2d,
        shape=in_shape,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        scale=scale,
        zero_point=zero_point,
        relu_type=relu_type,
    )
    new_shape = (1, 28, 28, 3) if padding == "VALID" else (1, 30, 30, 3)
    reshape = relay.reshape(max_pool, newshape=new_shape)

    model = make_model(
        pool_op=relay.nn.avg_pool2d,
        shape=new_shape,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        scale=scale,
        zero_point=zero_point,
        relu_type=relu_type,
        input_op=reshape,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # generate reference output
    rng = np.random.default_rng(12345)
    in_min, in_max = get_dtype_range("int8")
    inputs = {"input": rng.integers(in_min, high=in_max, size=in_shape, dtype="int8")}
    output_list = generate_ref_data(orig_mod["main"], inputs, params=None)

    # validate presence of depthwise convolution
    compiled_models = compile_models(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=None,
            output_tolerance=1,
        ),
        interface_api,
        use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    main_mod = None
    for target, mod in compiled_models[0].executor_factory.lowered_ir_mods.items():
        if target.kind.name == "c":
            main_mod = mod

    # when padding="SAME", extra padding is introduced which causes Reshape to be fused with the
    # Pad. RemoveReshapes pass cannot remove a fused Reshape. Whereas padding="VALID" doesn't need
    # an extra Pad layer. In this case, the pass removes the Reshape from the graph.
    reshapes_present = any(["reshape" in gv.name_hint for gv in main_mod.get_global_vars()])
    check_reshapes = reshapes_present if padding == "SAME" else not reshapes_present
    expected_reshapes = "a" if padding == "SAME" else "No"
    assert check_reshapes, "Expeting {} reshape layer(s).".format(expected_reshapes)

    # validate the output
    run_and_check(
        models=compiled_models,
        runner=test_runner,
        interface_api=interface_api,
    )


if __name__ == "__main__":
    tvm.testing.main()
