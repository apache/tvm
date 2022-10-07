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
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER
from tvm.topi.utils import change_constant_shape


class BasicConv2dTests:
    @tvm.testing.requires_corstone300
    def test_conv2d(
        self,
        data_shape,
        kernel_size,
        kernel_layout,
        num_filter,
        strides,
        padding,
        dilation,
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single conv2d operator."""
        ishape = data_shape
        wshape = (*kernel_size, data_shape[-1], num_filter)

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
        weight1 = change_constant_shape(weight0, "HWIO", kernel_layout)

        out1 = relay.op.nn.conv2d(
            input1,
            weight1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=(dilation, dilation),
            data_layout="NHWC",
            kernel_layout=kernel_layout,
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
            schedule_name=schedule_name,
        )


class TestConv2d_DSP_HWOI(BasicConv2dTests):
    """This test is for conv2d_nhwc_dsp.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        # TODO(mehrdadh): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        # ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        # ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
        # from Keyword Spotting model from MLPerfTiny models
        # TODO(mehrdad): Fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 49, 10, 1), (10, 4), 64, (2, 2), (4, 1, 5, 1), 1),
        # from Visual Wake Word model from MLPerfTiny models
        # TODO(mehrdadh): fails due to https://github.com/apache/tvm/issues/11216
        # ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1), 1),
        # from Image Classification model from MLPerfTiny models
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0, 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, (0, 4, 4, 0), 1),
        ((4, 16, 16, 8), (5, 5), 8, 2, 0, 1),
        ((4, 16, 16, 8), (5, 5), 16, 2, 0, 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 0, 1, 1), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (1, 1, 2, 2), 1),
        ((1, 16, 16, 8), (5, 5), 16, 2, (3, 3, 2, 2), 1),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 1, 2, 3), 1),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("HWOI")
    schedule_name = tvm.testing.parameter("conv2d_nhwc_dsp.arm_cpu")


class TestConv2d_HWIO(BasicConv2dTests):
    """This test is for conv2d_nhwc_spatial_pack.arm_cpu schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((1, 32, 32, 1), (3, 3), 12, 1, 0, 1),
        ((1, 32, 10, 3), (3, 3), 16, 1, 0, 1),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0, 1),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0), 2),
        ((1, 32, 32, 16), (3, 3), 16, 1, (1, 1, 2, 2), 2),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    kernel_layout = tvm.testing.parameter("HWIO")
    schedule_name = tvm.testing.parameter("conv2d_nhwc_spatial_pack.arm_cpu")


class TestConv2d_Tensordot(BasicConv2dTests):
    data_shape, kernel_size, num_filter, strides, padding = tvm.testing.parameters(
        # Disabled because these kernels are not an integral number of words
        # ((1, 32, 32, 1), (3, 3), 12, 1, 0),
        # ((1, 32, 10, 3), (3, 3), 16, 1, 0),
        # ((1, 96, 96, 3), (3, 3), 8, (2, 2), (0, 0, 1, 1)),
        ((4, 16, 16, 8), (5, 5), 8, 2, (0, 3, 3, 0)),
        ((4, 16, 16, 8), (5, 5), 16, 2, (0, 3, 3, 0)),
        ((4, 16, 16, 8), (5, 5), 8, 2, 0),
        ((4, 16, 16, 8), (5, 5), 16, 2, 0),
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0),
        ((1, 16, 16, 32), (1, 1), 64, (2, 2), 0),
        ((1, 49, 10, 1), (10, 4), 64, (2, 1), (4, 1, 5, 1)),
        ((1, 32, 32, 16), (3, 3), 16, 1, (0, 2, 2, 0)),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0),
        ((1, 49, 10, 1), (10, 4), 64, (2, 2), (4, 1, 5, 1)),
        ((1, 16, 16, 8), (3, 3), 16, 2, (0, 0, 1, 1)),
        ((1, 16, 16, 8), (3, 3), 16, 2, (1, 1, 2, 2)),
        ((1, 16, 16, 8), (5, 5), 16, 2, (3, 3, 2, 2)),
        ((1, 32, 32, 16), (3, 3), 16, 1, 0),
        ((1, 16, 16, 32), (1, 1), 64, 1, 0),
    )
    dilation = tvm.testing.parameter(1)
    dtype = tvm.testing.parameter("int8", "int16", "int32")
    kernel_layout = tvm.testing.parameter("OHWI")
    schedule_name = tvm.testing.parameter("conv2d_nhwc_ohwi_dsp.arm_cpu")


if __name__ == "__main__":
    tvm.testing.main()
