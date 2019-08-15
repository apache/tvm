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
from tvm.relay.op import register_legalize
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = relay.Module.from_expr(expr)
    seq = transform.Sequential(passes)
    with transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_legalize():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    @register_legalize("nn.conv2d", level=100)
    def legalize_conv2d(attrs, inputs, types):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, relay.multiply(weight, relay.const(2.0, "float32")),
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    a = before()
    a = run_opt_pass(a, transform.Legalize())
    b = run_opt_pass(expected(), transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_legalize_none():
    """Test doing nothing by returning 'None' """
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    @register_legalize("nn.global_max_pool2d", level=101)
    def legalize_conv2d(attrs, inputs, types):
        called[0] = True
        return None

    a = before()
    a = run_opt_pass(a, transform.Legalize())

    b = before()
    b = run_opt_pass(b, transform.InferType())
    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)
    assert(called[0])

def test_legalize_multi_input():
    """Test directly replacing an operator with a new one"""
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

    @register_legalize("concatenate", level=100)
    def legalize_concatenate(attrs, inputs, types):
        # Check that the correct multi-input case is handled.
        assert len(inputs) == 1
        assert isinstance(inputs[0], tvm.relay.expr.Tuple)
        assert len(types) == 2
        assert isinstance(types[0], tvm.relay.ty.TupleType)
        assert isinstance(types[1], tvm.relay.ty.TensorType)
        return None

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

    a = before()
    a = run_opt_pass(a, transform.Legalize())
    b = run_opt_pass(expected(), transform.InferType())

    assert analysis.alpha_equal(a, b), "Actual = \n" + str(a)

def test_legalize_arm_layout_functional():
    """Test if the legalized conversion yields same result as original"""
    def get_output(func, data_val, parameters):
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(func, target='llvm', params=parameters)
        m = graph_runtime.create(graph, lib, tvm.cpu())
        m.set_input("data", data_val)
        m.set_input(**params)
        m.run()
        out = m.get_output(0, tvm.nd.empty((1, 224, 224, 32), 'float32')).asnumpy()
        return out

    def before():
        n, ic, ih, iw, oc, kh, kw = 1, 16, 224, 224, 32, 3, 3
        data = relay.var("data", relay.TensorType((n, ih, iw, ic), 'float32'))
        kernel = relay.var("kernel", relay.TensorType((kh, kw, ic, oc), 'float32'))
        y = relay.nn.conv2d(data, kernel,
                            kernel_size=(kh, kw),
                            channels=oc,
                            padding=(1, 1),
                            dilation=(1, 1),
                            data_layout='NHWC',
                            kernel_layout='HWIO',
                            out_dtype='float32')
        func = relay.Function([data, kernel], y)
        return func

    @register_legalize("nn.conv2d", level=101)
    def legalize_conv2d(attrs, inputs, types):
        from topi.arm_cpu.conv2d import _conv2d_legalize
        return _conv2d_legalize(attrs, inputs, types)

    a = before()
    b = run_opt_pass(a, transform.Legalize())
    assert b.astext().count('transpose') == 3

    wdata = np.random.rand(3, 3, 16, 32) * 10
    parameters = {"kernel": tvm.nd.array(wdata.astype('float32'))}
    data_val = np.random.rand(1, 224, 224, 16).astype('float32')
    ref_out = get_output(a, data_val, parameters)
    legalized_out = get_output(b, data_val, parameters)
    np.testing.assert_allclose(ref_out, legalized_out, rtol=0.01)


if __name__ == "__main__":
    test_legalize()
    test_legalize_none()
    test_legalize_multi_input()
    test_legalize_arm_layout_functional()
