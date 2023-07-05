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

"""CMSIS-NN integration tests: debug_last_error"""

import re
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from tvm.testing.aot import (
    get_dtype_range,
    generate_ref_data,
    AOTTestModel,
    compile_and_run,
)
from .utils import (
    make_module,
    get_same_padding,
    make_qnn_relu,
    assert_partitioned_function,
    create_test_runner,
)


def make_model(
    pool_op,
    shape,
    pool_size,
    strides,
    padding,
    dtype,
    scale,
    zero_point,
    relu_type,
    layout,
    input_op,
):
    """Create a Relay Function / network model"""
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
@pytest.mark.parametrize("debug_last_error", [True, False])
def test_last_error(debug_last_error):
    """Tests debug_last_error"""
    dtype = "int16"
    in_shape = (1, 28, 28, 12)
    pool_size = (3, 3)
    strides = (2, 2)
    padding = "SAME"
    relu_type = "NONE"
    pool_type = relay.nn.avg_pool2d
    zero_point = -34
    scale = 0.0256
    compiler_cpu = "cortex-m55"
    cpu_flags = "+nomve"
    layout = "NHWC"
    input_op = None

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
        layout=layout,
        input_op=input_op,
    )
    orig_mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    in_min, in_max = get_dtype_range(dtype)
    inputs = {
        "input": np.random.randint(in_min, high=in_max, size=in_shape, dtype=dtype),
    }
    output_list = generate_ref_data(orig_mod["main"], inputs)

    def checker(base_path: str) -> bool:
        def read_file(path):
            with open(path) as f:
                return f.read()

        test = read_file(base_path + "/build/test.c")
        test_check = "TVMGetLastError" in test

        default_lib2 = read_file(base_path + "/codegen/host/src/default_lib2.c")
        regex = (
            r"(?s)arm_avgpool_s16(.*?)"
            r'ARM_CMSIS_NN_ARG_ERROR: TVMAPISetLastError\("ARM_CMSIS_NN_ARG_ERROR(.*?)'
            r'ARM_CMSIS_NN_NO_IMPL_ERROR: TVMAPISetLastError\("ARM_CMSIS_NN_NO_IMPL_ERROR'
        )
        default_lib2_check = re.search(regex, default_lib2) is not None

        if debug_last_error:
            return test_check and default_lib2_check
        else:
            return not (test_check or default_lib2_check)

    result = compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=None,
            output_tolerance=1,
        ),
        create_test_runner(compiler_cpu, cpu_flags, debug_last_error=debug_last_error),
        interface_api,
        use_unpacked_api,
        debug_last_error=debug_last_error,
        checker=checker,
    )
    assert result
