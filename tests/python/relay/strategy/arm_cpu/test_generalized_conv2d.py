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
from numpy.random import randint
import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER
from tvm.topi.utils import change_ndarray_layout


class GeneralizedConv2dTests:
    @tvm.testing.requires_corstone300
    def test_conv2d(
        self,
        data_shape,
        kernel_size,
        num_filter,
        in_dtype,
        strides,
        padding,
        groups,
        dilation,
        data_layout,
        kernel_layout,
        out_layout,
        schedule_name,
    ):
        """Test a subgraph with a single conv2d operator."""

        ref_input_data = randint(low=-128, high=127, size=data_shape, dtype=in_dtype)
        ref_input_var = relay.var("input", relay.TensorType(data_shape, in_dtype)) # NHWC layout
        kernel_shape = (*kernel_size, data_shape[-1], num_filter) # HWIO layout
        ref_kernel_data = randint(low=-10, high=10, size=kernel_shape, dtype=in_dtype)

        ref_relay_op = relay.op.nn.conv2d(
            ref_input_var,
            relay.const(ref_kernel_data),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=(dilation, dilation),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
            out_layout="NHWC",
        )
        ref_module = tvm.IRModule.from_expr(relay.Function([ref_input_var], ref_relay_op))
        ref_outputs = generate_ref_data(ref_module, {"input": ref_input_data})

        # Reshape output dictionary to match out_layout
        assert len(ref_outputs) == 1
        axis_order = ["NHWC".index(c) for c in out_layout]
        output_tensor_name, output_tensor = next(iter(ref_outputs.items()))
        ref_outputs[output_tensor_name] = change_ndarray_layout(output_tensor, "NHWC", out_layout)

        test_input_data = change_ndarray_layout(ref_input_data, "NHWC", data_layout)
        test_input_var = relay.var("input", relay.TensorType(test_input_data.shape, in_dtype))
        test_kernel_data = change_ndarray_layout(ref_kernel_data, "HWIO", kernel_layout)

        test_relay_op = relay.op.nn.conv2d(
            test_input_var,
            relay.const(test_kernel_data),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=(dilation, dilation),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            out_dtype="int32",
            out_layout=out_layout,
        )
        test_function = relay.Function([test_input_var], test_relay_op)
        test_model = AOTTestModel(
            module=tvm.IRModule.from_expr(test_function),
            inputs={"input": test_input_data},
            outputs=ref_outputs
        )

        compile_and_run(
            test_model,
            runner=AOT_CORSTONE300_RUNNER,
            interface_api="c",
            use_unpacked_api=True,
            target_opts={
                "-keys": "arm_cpu",
                "-mcpu": "cortex-m7",
            },
            schedule_name=schedule_name,
        )