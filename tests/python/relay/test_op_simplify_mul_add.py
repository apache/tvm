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
# pylint: disable=unused-wildcard-import, invalid-name

"""
Test x86 relay transform - Eliminate const_mul and const_add following a conv optimization
"""
import numpy as np
import tvm
from tvm.runtime import ndarray as nd
from tvm import relay, testing
from tvm.relay.transform.SimplifyMulAdd import simplify_mul_add
from tvm.topi.utils import get_const_tuple
from tvm.relay.backend import Executor, Runtime


def get_test_module_relay_exprs():
    """
    Creates relay expressions that can be used both by
    test module and expected output module
    """

    act_shape = (1, 3, 224, 224)
    data_in = np.random.rand(*get_const_tuple(act_shape))
    data_in_float32 = np.full(data_in.shape, data_in, dtype="float32")
    kernel_shape = (64, 3, 7, 7)

    scalar_multiplier_shape = (64, 1, 1)
    bias = np.random.rand(get_const_tuple(kernel_shape)[0])
    relay_act = relay.var("q1", shape=act_shape, dtype="float32")
    relay_mul_factor = np.random.rand(*get_const_tuple(scalar_multiplier_shape))
    relay_mul_factor = relay.Constant(
        nd.array(np.full(relay_mul_factor.shape, relay_mul_factor, dtype="float32"))
    )
    relay_add_term = np.random.rand(*get_const_tuple(scalar_multiplier_shape))
    relay_add_term = relay.Constant(
        nd.array(np.full(relay_add_term.shape, relay_add_term, dtype="float32"))
    )

    weights = np.random.rand(*get_const_tuple(kernel_shape))
    relay_weights = relay.Constant(nd.array(np.full(weights.shape, weights, dtype="float32")))
    relay_bias = relay.Constant(nd.array(np.full(bias.shape, bias, dtype="float32")))
    return (relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias, data_in_float32)


def get_test_module_graph(relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias):
    """Creates a test relay graph with the specified relay expressions"""
    v1 = relay.nn.conv2d(
        relay_act,
        relay_weights,
        padding=[3, 3, 3, 3],
        channels=64,
        kernel_size=[7, 7],
    )
    v2 = relay.nn.bias_add(v1, relay_bias)
    v3 = relay.multiply(v2, relay_mul_factor)
    graph = relay.add(v3, relay_add_term)
    return graph


def get_test_module(relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias):
    """Creates a test relay module and returns it."""
    graph = get_test_module_graph(
        relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias
    )

    func = relay.Function(relay.analysis.free_vars(graph), graph)
    mod = tvm.IRModule.from_expr(func)
    return mod


def get_expected_output_module_graph(
    relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias
):
    """Creates the relay graph for expected output"""
    new_mul_factor = relay.reshape(relay_mul_factor, [64, 1, 1, 1])
    v1 = relay.multiply(relay_weights, new_mul_factor)
    v2 = relay.nn.conv2d(relay_act, v1, padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7])
    new_bias_factor = relay.reshape(relay_bias, [64, 1, 1])
    v3 = relay.multiply(new_bias_factor, relay_mul_factor)
    v4 = relay.add(v3, relay_add_term)
    new_bias = relay.reshape(v4, 64)
    return relay.nn.bias_add(v2, new_bias)


def get_expected_output_module(
    relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias
):
    """Returns manually created expected output module."""
    graph = get_expected_output_module_graph(
        relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias
    )

    out_func = relay.Function(relay.analysis.free_vars(graph), graph)
    return tvm.IRModule.from_expr(out_func)


def build_module(relay_mod, target):
    """builds a relay module for a specified target"""
    params = {}
    lowered = tvm.relay.build(
        relay_mod,
        tvm.target.Target(target, host=target),
        runtime=Runtime("cpp"),
        executor=Executor("graph"),
        params=params,
    )
    return lowered


def run_module(mod, inputs):
    """invokes run function of specified module with inputs provided"""
    mod.set_input(**inputs)
    mod.run()
    output = mod.get_output(0).numpy()
    return output


def get_test_modules():
    """generates test, expected modules and their inputs"""
    (
        relay_act,
        relay_mul_factor,
        relay_add_term,
        relay_weights,
        relay_bias,
        data_in_float32,
    ) = get_test_module_relay_exprs()
    mod = get_test_module(relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias)
    exp_relay_mod = get_expected_output_module(
        relay_act, relay_mul_factor, relay_add_term, relay_weights, relay_bias
    )

    return mod, exp_relay_mod, {"q1": data_in_float32}


# @tvm.testing.requires_x86
def test_simplify_mul_add():
    """A positive test case"""

    (mod, exp_relay_mod, inputs) = get_test_modules()

    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.relay.transform.InferType()(mod)
        x86_lowered = build_module(mod, tvm.target.Target("llvm", host="llvm"))

    with tvm.transform.PassContext(opt_level=3):
        mod = simplify_mul_add(mod)
        mod = tvm.relay.transform.InferType()(mod)
        exp_relay_mod = tvm.relay.transform.InferType()(exp_relay_mod)
        assert tvm.ir.structural_equal(mod["main"], exp_relay_mod["main"], map_free_vars=True)
        mod = tvm.relay.transform.FoldConstant()(mod)
        x86_lowered_opt = build_module(mod, tvm.target.Target("llvm", host="llvm"))

    # Run unoptimized llvm module
    x86_mod = tvm.contrib.graph_executor.GraphModule(x86_lowered["default"](tvm.cpu(0)))
    expected_output = run_module(x86_mod, inputs)

    # Run optimized llvm module
    x86_mod_opt = tvm.contrib.graph_executor.GraphModule(x86_lowered_opt["default"](tvm.cpu(0)))
    actual_output = run_module(x86_mod_opt, inputs)

    tvm.testing.assert_allclose(actual_output, expected_output, rtol=0.00001)


if __name__ == "__main__":
    testing.main()
