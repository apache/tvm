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
"""Test legalize pass"""
import numpy as np
import tvm

from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.qnn.op import register_qnn_legalize
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = relay.Module.from_expr(expr)
    seq = transform.Sequential(passes)
    with transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_qnn_legalize():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype='int8')
        y = relay.qnn.op.requantize(x,
                                    input_scale=1,
                                    input_zero_point=0,
                                    output_scale=1,
                                    output_zero_point=0,
                                    out_dtype='int8')
        y = relay.Function([x], y)
        return y

    @register_qnn_legalize("qnn.requantize", level=100)
    def legalize_qnn_requantize(attrs, inputs, types):
        data = inputs[0]
        data = relay.add(relay.const(0, 'int8'), data)
        y = relay.qnn.op.requantize(data,
                                    input_scale=1,
                                    input_zero_point=0,
                                    output_scale=1,
                                    output_zero_point=0,
                                    out_dtype='int8')
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype='int8')
        y = relay.add(relay.const(0, 'int8'), x)
        z = relay.qnn.op.requantize(y,
                                    input_scale=1,
                                    input_zero_point=0,
                                    output_scale=1,
                                    output_zero_point=0,
                                    out_dtype='int8')
        z = relay.Function([x], z)
        return z

    a = before()

    # Check that Relay Legalize does not change the graph.
    a = run_opt_pass(a, relay.transform.Legalize())
    b = run_opt_pass(before(), transform.InferType())
    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

    # Check that QNN Legalize modifies the graph.
    a = run_opt_pass(a, relay.qnn.transform.Legalize())
    b = run_opt_pass(expected(), transform.InferType())
    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_qnn_legalize_qnn_conv2d():

    def verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape, kernel_dtype):
        def get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype):
            low = -128
            high = 127
            if data_dtype == "uint8":
                low = 0
                high = 255
            golden_data = np.random.random_integers(low=low, high=high,
                    size=data_shape).astype(data_dtype)
            low = -128
            high = 127
            if kernel_dtype == "uint8":
                low = 0
                high = 255
            golden_weight = np.random.random_integers(low=low, high=high,
                    size=kernel_shape).astype(kernel_dtype)
            return (golden_data, golden_weight)

        def get_output(func, golden_inputs):
            with relay.build_config(opt_level=3):
                golden_data, golden_weight = golden_inputs
                params = {'kernel': golden_weight}
                graph, lib, params = relay.build(func, "llvm", params=params)
                mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
                mod.set_input("data", golden_data)
                # mod.set_input("kernel", golden_weight)
                mod.set_input(**params)
                mod.run()
                res = mod.get_output(0).asnumpy()
                return res
        golden_inputs = get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype)
        golden_output = get_output(ref_func, golden_inputs)
        qnn_output = get_output(qnn_func, golden_inputs)
        np.testing.assert_equal(qnn_output, golden_output)

    data_shape = (1, 64, 256, 256)
    kernel_shape = (128, 64, 3, 3)
    for dtype in ['uint8', 'int8']:
        data_dtype =  kernel_dtype = dtype
        data = relay.var("data", shape=data_shape,
                dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape,
                dtype=kernel_dtype)
        func = relay.qnn.op.conv2d(
                data, kernel,
                input_zero_point=1,
                kernel_zero_point=1,
                kernel_size=(3, 3),
                strides=(1, 1),
                dilation=(1, 1),
                out_dtype='int32',
                data_layout='NCHW',
                kernel_layout='OIHW')

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = relay.Module.from_expr(mod)
        ref_mod = relay.qnn.transform.QnnToRelay()(mod)

        with tvm.target.create('llvm'):
            qnn_mod = relay.qnn.transform.Legalize()(mod)
            qnn_mod = relay.qnn.transform.QnnToRelay()(qnn_mod)

        verify(ref_mod, qnn_mod, data_shape, data_dtype, kernel_shape, kernel_dtype)

if __name__ == "__main__":
    test_qnn_legalize()
    test_qnn_legalize_qnn_conv2d()
