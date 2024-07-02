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
Test hexagon relay transform - qnn.concat optimization
"""
import numpy as np
import tvm
from tvm.runtime import ndarray as nd
from tvm import relay, testing
from tvm.contrib.hexagon.transform import simplify_conv_pat
from tvm.topi.utils import get_const_tuple
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET
from .infrastructure import build_module, run_module


def get_test_module_relay_exprs(isConstScalarMultiplier=True):
    """
    Creates relay expressions that can be used both by
    test module and expected output module
    """

    act_shape = (1, 32, 32, 3)
    data_in = np.random.rand(*get_const_tuple(act_shape))
    data_in_float32 = np.full(data_in.shape, data_in, dtype="float32")
    kernel_shape = (16, 3, 3, 3)
    weights = np.random.rand(*get_const_tuple(kernel_shape))

    bias = np.random.rand(get_const_tuple(kernel_shape)[0])
    relay_act = relay.var("q1", shape=act_shape, dtype="float32")
    if isConstScalarMultiplier:
        relay_mul_factor = relay.const(0.00392151, dtype="float32")
    else:
        relay_mul_factor = np.random.rand(*get_const_tuple(act_shape))
        relay_mul_factor = relay.Constant(
            nd.array(np.full(relay_mul_factor.shape, relay_mul_factor, dtype="float32"))
        )
    relay_sub_term = relay.const(0.5, dtype="float32")
    relay_weights = relay.Constant(nd.array(np.full(weights.shape, weights, dtype="float32")))
    relay_bias = relay.Constant(nd.array(np.full(bias.shape, bias, dtype="float32")))
    return (relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias, data_in_float32)


def get_test_module_graph(relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias):
    """Creates a test relay graph with the specified relay expressions"""
    v1 = relay.multiply(relay_act, relay_mul_factor)
    v2 = relay.subtract(v1, relay_sub_term)
    v3 = relay.transpose(v2, axes=[0, 3, 1, 2])
    weights_type_info = tvm.relay.transform.InferTypeLocal(relay_weights)
    v4 = relay.nn.conv2d(
        v3,
        relay_weights,
        padding=[1, 1, 1, 1],
        channels=weights_type_info.shape[0],
        kernel_size=[3, 3],
    )
    graph = relay.nn.bias_add(v4, relay_bias)
    return graph


def get_test_module(relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias):
    """Creates a test relay module and returns it."""
    graph = get_test_module_graph(
        relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias
    )

    func = relay.Function(relay.analysis.free_vars(graph), graph)
    mod = tvm.IRModule.from_expr(func)
    return mod


def get_expected_output_module_graph(
    relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias
):
    """Creates the relay graph for expected output"""
    v1 = relay.transpose(relay_act, axes=[0, 3, 1, 2])
    v2 = relay.multiply(relay_mul_factor, relay_weights)
    weights_type_info = tvm.relay.transform.InferTypeLocal(relay_weights)
    v3 = relay.nn.conv2d(
        v1, v2, padding=[1, 1, 1, 1], channels=weights_type_info.shape[0], kernel_size=[3, 3]
    )
    type_info = tvm.relay.transform.InferTypeLocal(v1)
    relay_zero_act = relay.Constant(
        nd.array(np.zeros(get_const_tuple(type_info.shape), dtype="float32"))
    )
    v4 = relay.subtract(relay_zero_act, relay_sub_term)
    v5 = relay.nn.bias_add(v3, relay_bias)
    v6 = relay.nn.conv2d(
        v4,
        relay_weights,
        padding=[1, 1, 1, 1],
        channels=weights_type_info.shape[0],
        kernel_size=[3, 3],
    )
    return relay.add(v5, v6)


def get_expected_output_module(
    relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias
):
    """Returns manually created expected output module."""
    graph = get_expected_output_module_graph(
        relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias
    )

    out_func = relay.Function(relay.analysis.free_vars(graph), graph)
    return tvm.IRModule.from_expr(out_func)


def get_test_modules():
    """generates test, expected modules and their inputs"""
    (
        relay_act,
        relay_mul_factor,
        relay_sub_term,
        relay_weights,
        relay_bias,
        data_in_float32,
    ) = get_test_module_relay_exprs()
    mod = get_test_module(relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias)
    exp_relay_mod = get_expected_output_module(
        relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias
    )

    return mod, exp_relay_mod, {"q1": data_in_float32}


@tvm.testing.requires_hexagon
def test_simplify_conv_pat(hexagon_session: Session):
    """A positive test case"""

    (mod, exp_relay_mod, inputs) = get_test_modules()

    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.relay.transform.InferType()(mod)
        hexagon_lowered = build_module(
            mod, tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET)
        )

    with tvm.transform.PassContext(opt_level=3):
        mod = simplify_conv_pat(mod)
        mod = tvm.relay.transform.InferType()(mod)
        exp_relay_mod = tvm.relay.transform.InferType()(exp_relay_mod)
        tvm.ir.assert_structural_equal(mod["main"], exp_relay_mod["main"], map_free_vars=True)
        mod = tvm.relay.transform.FoldConstant()(mod)
        hexagon_lowered_opt = build_module(
            mod, tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET)
        )

    # Run unoptimized llvm module
    hexagon_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    expected_output = run_module(hexagon_mod, inputs)

    # Run optimized llvm module
    hexagon_mod_opt = hexagon_session.get_executor_from_factory(hexagon_lowered_opt)
    actual_output = run_module(hexagon_mod_opt, inputs)

    tvm.testing.assert_allclose(actual_output, expected_output, rtol=0.00001)


def get_negative_test_module():
    """generates a negative test module with non-const multiplier"""
    (
        relay_act,
        relay_mul_factor,
        relay_sub_term,
        relay_weights,
        relay_bias,
        _,
    ) = get_test_module_relay_exprs(False)
    mod = get_test_module(relay_act, relay_mul_factor, relay_sub_term, relay_weights, relay_bias)

    return mod


def test_negative():
    """A negative test case"""
    orig_mod = get_negative_test_module()
    with tvm.transform.PassContext(opt_level=3):
        orig_mod = tvm.relay.transform.InferType()(orig_mod)
        opt_mod = simplify_conv_pat(orig_mod)
        opt_mod = tvm.relay.transform.InferType()(opt_mod)
        tvm.ir.assert_structural_equal(orig_mod["main"], opt_mod["main"], map_free_vars=True)


if __name__ == "__main__":
    testing.main()
