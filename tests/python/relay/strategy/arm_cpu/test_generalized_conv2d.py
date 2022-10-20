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
"""Helper class for testing variations of 2D convolution. Should be used by subclassing
`GeneralizedConv2dTests`, and then setting the arguments using tvm.testing.parameter(s)."""

import numpy as np

import tvm
import tvm.testing
from tvm import relay
from tvm.testing.aot import AOTTestModel, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER


def _change_ndarray_layout(arr, src_layout, dst_layout):
    """Makes a copy of an ndarray, reshaping it to a new data layout.

    Parameter
    ---------
    arr : numpy.ndarray
        The ndarray to be reformatted.

    src_layout : str
        The current layout of the Relay constant. Must be alphabetic (e.g. NHWC
        or OIHW, but not NCHW2c).

    dst_layout : str
        The desired layout of new the Relay constant. Must be alphabetic (e.g. NHWC
        or OIHW, but not NCHW2c).

    Returns
    -------
    dst_shape : numpy.ndarray
        A copy of the ndarray with the new layout.
    """
    assert src_layout.isalpha() and dst_layout.isalpha()
    axis_order = [src_layout.index(c) for c in dst_layout]
    return np.transpose(arr, axis_order)


class GeneralizedConv2dTests:
    """Superclass which can be used to test regular, depthwise, or grouped conv2D. Cannot be used
    for 5D data formats (NCHWc and such) as written, but could be extended. Might also be worth
    abstracting some of this logic into an even more general class that could be used for other
    operators.

    Note that data_shape should always be a tuple of length four indicating the data shape in NHWC
    format (it will later be reshaped according to the given data_layout), and kernel_size should be
    a length two tuple giving the height and width of the kernel.

    This test (and other base Conv2dTests classes) are not run by Pytest, as their names do not
    start with `Test`."""

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

        ref_input_data = np.random.randint(low=-128, high=127, size=data_shape, dtype=in_dtype)
        ref_input_var = relay.var("input", relay.TensorType(data_shape, in_dtype))  # NHWC layout
        kernel_shape = (*kernel_size, data_shape[-1] // groups, num_filter)  # HWIO layout
        ref_kernel_data = np.random.randint(low=-10, high=10, size=kernel_shape, dtype=in_dtype)

        """Our x86 depthwise implementation only supports HWOI with NHWC, so we need to change our
        kernel layout to work around this. We can't just change the whole thing to HWIO or
        something else, as then group conv2d would not work. Eventually, we should switch to using
        TensorFlow to create the reference output so we can ensure our implementation is right.
        See https://github.com/apache/tvm/issues/13137 for details."""

        ref_relay_op = relay.op.nn.conv2d(
            ref_input_var,
            relay.const(_change_ndarray_layout(ref_kernel_data, "HWIO", self.ref_kernel_layout)),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            dilation=(dilation, dilation),
            data_layout="NHWC",
            kernel_layout=self.ref_kernel_layout,
            out_dtype="int32",
            out_layout="NHWC",
        )
        ref_module = tvm.IRModule.from_expr(relay.Function([ref_input_var], ref_relay_op))
        ref_outputs = generate_ref_data(ref_module, {"input": ref_input_data})

        # Reshape output dictionary to match out_layout
        assert len(ref_outputs) == 1
        output_tensor_name, output_tensor = next(iter(ref_outputs.items()))
        ref_outputs[output_tensor_name] = _change_ndarray_layout(output_tensor, "NHWC", out_layout)

        test_input_data = _change_ndarray_layout(ref_input_data, "NHWC", data_layout)
        test_input_var = relay.var("input", relay.TensorType(test_input_data.shape, in_dtype))
        test_kernel_data = _change_ndarray_layout(ref_kernel_data, "HWIO", kernel_layout)

        test_relay_op = relay.op.nn.conv2d(
            test_input_var,
            relay.const(test_kernel_data),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
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
            outputs=ref_outputs,
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
