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
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
)


class BasicConv1dTests:
    @tvm.testing.requires_corstone300
    def test_conv1d(
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
        """Test a subgraph with a single conv1d_nwc operator."""
        ishape = data_shape
        wshape = (kernel_size, data_shape[-1], num_filter)
        weight_data = np.random.randint(low=-10, high=10, size=wshape, dtype=dtype)

        input0 = relay.var("input", relay.TensorType(ishape, dtype))
        weight0 = relay.const(weight_data)
        out0 = relay.op.nn.conv1d(
            input0,
            weight0,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            data_layout="NWC",
            kernel_layout="WIO",
            out_dtype="int32",
            out_layout="NWC",
        )
        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))

        if kernel_layout == "WOI":
            weight1 = relay.const(np.moveaxis(weight_data, 1, -1))
        else:
            weight1 = relay.const(weight_data)

        out1 = relay.op.nn.conv1d(
            input1,
            weight1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            data_layout="NWC",
            kernel_layout=kernel_layout,
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
            schedule_name=schedule_name,
        )


class TestConv1d_dsp(BasicConv1dTests):
    """This test is for conv1d_dsp schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((4, 32, 16), 3, 12, 1, 0, 1),
        ((4, 16, 32), 3, 12, 1, 0, 1),
        ((4, 32, 16), 3, 12, 1, 0, 1),
        ((1, 32, 12), 3, 16, 1, 0, 1),
        # TODO: The following 4 tests fail due to https://github.com/apache/tvm/issues/11466
        # ((3, 12, 10), 4, 24, 1, 0, 1),
        # ((1, 7, 7), 3, 5, 1, 0, 1),
        # ((1, 10, 2), 4, 4, 2, (1, 1), 1),
        # ((1, 20, 2), 4, 4, 2, (0, 1), 1),
        ((1, 16, 4), 1, 12, 1, (1, 0), 1),
        ((1, 24, 16), 1, 32, 3, (2, 2), 1),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    data_layout = tvm.testing.parameter("NWC")
    kernel_layout = tvm.testing.parameter("WOI")
    schedule_name = tvm.testing.parameter("conv1d_dsp")


class TestConv1d_nwc(BasicConv1dTests):
    """This test is for conv1d_nwc.generic schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((4, 32, 16), 3, 12, 1, 0, 1),
        ((4, 16, 32), 3, 12, 1, 0, 1),
        ((4, 32, 16), 3, 12, 1, 0, 1),
        ((1, 32, 12), 3, 16, 1, 0, 1),
        ((3, 12, 10), 4, 24, 1, 0, 1),
        ((1, 7, 7), 3, 5, 1, 0, 1),
        ((1, 10, 2), 4, 4, 2, (1, 1), 1),
        ((1, 20, 2), 4, 4, 2, (0, 1), 1),
        ((1, 16, 4), 1, 12, 1, (1, 0), 1),
        ((1, 24, 16), 1, 32, 3, (2, 2), 1),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    data_layout = tvm.testing.parameter("NWC")
    kernel_layout = tvm.testing.parameter("WIO")
    schedule_name = tvm.testing.parameter("conv1d_nwc.generic")


if __name__ == "__main__":
    tvm.testing.main()
