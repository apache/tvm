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
        num_filter,
        strides,
        padding,
        dilation,
        dtype,
        schedule_name,
    ):
        """Test a subgraph with a single conv1d_ncw operator."""
        ishape = data_shape
        wshape = (num_filter, data_shape[1], kernel_size)

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
            data_layout="NCW",
            kernel_layout="OIW",
            out_dtype="int32",
            out_layout="NCW",
        )
        ref_mod = tvm.IRModule.from_expr(relay.Function([input0], out0))

        input1 = relay.var("input", relay.TensorType(ishape, dtype))
        weight1 = relay.const(weight_data)

        out1 = relay.op.nn.conv1d(
            input1,
            weight1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            data_layout="NCW",
            kernel_layout="OIW",
            out_dtype="int32",
            out_layout="NCW",
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


class TestConv1d_ncw(BasicConv1dTests):
    """This test is for conv1d_ncw.generic schedule."""

    data_shape, kernel_size, num_filter, strides, padding, dilation = tvm.testing.parameters(
        ((4, 32, 16), 3, 12, 1, 0, 1),
        ((4, 16, 32), 3, 12, 1, 0, 1),
        ((1, 12, 32), 3, 16, 1, 0, 1),
        ((3, 10, 12), 4, 24, 1, 0, 1),
        ((1, 7, 7), 3, 5, 1, 0, 1),
        ((1, 2, 10), 4, 4, 2, (1, 1), 1),
        ((1, 2, 20), 4, 4, 2, (0, 1), 1),
        ((1, 4, 16), 1, 12, 1, (1, 0), 1),
        ((1, 16, 24), 1, 32, 3, (2, 2), 1),
    )
    dtype = tvm.testing.parameter("int8", "int16")
    data_layout = tvm.testing.parameter("NCW")
    schedule_name = tvm.testing.parameter("conv1d_ncw.generic")


if __name__ == "__main__":
    tvm.testing.main()
