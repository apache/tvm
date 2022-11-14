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

"""
Tests for the 'InlineNonComputeIntensivePartitions' pass.
"""

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from tvm.relay.op.contrib.ethosn import InlineNonComputeIntensivePartitions

from . import infrastructure as tei


def _assert_structural_equal(a, b):
    """Check structural equality of two Relay expressions."""
    reason = (
        "Actual and expected relay functions are not equal. "
        "InlineNonComputeIntensiveSubgraphs is not correctly "
        "transforming the input graph."
    )
    assert tvm.ir.structural_equal(a, b, map_free_vars=True), reason


@requires_ethosn
def test_single_reshape():
    """Check that a single reshape is inlined correctly."""

    def get_reshape():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        return relay.reshape(x, newshape=(2, 2, 4))

    def before():
        reshape = get_reshape()
        return tei.make_ethosn_partition(reshape)

    def expected():
        reshape = get_reshape()
        mod = tvm.IRModule.from_expr(reshape)
        return relay.transform.InferType()(mod)

    mod = before()
    mod = InlineNonComputeIntensivePartitions()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod, expected_mod)


@requires_ethosn
def test_multiple_non_compute_intensive_ops():
    """
    Check that a partitioned function is correctly inlined
    when it contains multiple non-compute intensive operations.
    """

    def get_graph():
        x = relay.var("x", shape=(2, 2, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 2, 2, 4))
        x = relay.clip(x, 0.0, 1.0)
        x = relay.reshape(x, newshape=(2, 2, 4))
        return relay.clip(x, 0.0, 1.0)

    def before():
        func = get_graph()
        return tei.make_ethosn_partition(func)

    def expected():
        func = get_graph()
        mod = tvm.IRModule.from_expr(func)
        return relay.transform.InferType()(mod)

    mod = before()
    mod = InlineNonComputeIntensivePartitions()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod, expected_mod)


@requires_ethosn
def test_compute_intensive_ops():
    """
    Check that a partitioned function that is considered
    compute intensive is not inlined.
    """

    def before():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.nn.max_pool2d(x, layout="NHWC")
        x = relay.reshape(x, newshape=(2, 2, 4))
        return tei.make_ethosn_partition(x)

    mod = before()
    transformed_mod = InlineNonComputeIntensivePartitions()(mod)
    for global_var in mod.get_global_vars():
        _assert_structural_equal(mod[global_var], transformed_mod[global_var])


@requires_ethosn
def test_multiple_partitioned_functions():
    """
    Tests the pass on a number of partitioned functions.
    """

    def before():
        composite_func_name = "ethos-n_0"
        inp = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")

        # partitioned func 1 (non compute intensive)
        x = relay.reshape(inp, newshape=(1, 2, 2, 4))
        partitioned_func_1 = tei.make_ethosn_partition(x)[composite_func_name]
        gv_1 = relay.GlobalVar("ethos-n_0")

        # partitioned func 2 (compute intensive)
        x = relay.nn.max_pool2d(inp, layout="NHWC")
        partitioned_func_2 = tei.make_ethosn_partition(x)[composite_func_name]
        gv_2 = relay.GlobalVar("ethos-n_1")

        # partitioned func 3 (non compute intensive)
        x = relay.clip(inp, 0.0, 1.0)
        partitioned_func_3 = tei.make_ethosn_partition(x)[composite_func_name]
        gv_3 = relay.GlobalVar("ethos-n_2")

        mod = tvm.IRModule({})
        mod[gv_1] = partitioned_func_1
        mod[gv_2] = partitioned_func_2
        mod[gv_3] = partitioned_func_3
        main_expr = relay.Call(gv_1, [inp])
        main_expr = relay.Call(gv_2, [main_expr])
        main_expr = relay.Call(gv_3, [main_expr])
        mod["main"] = relay.Function([inp], main_expr)
        return relay.transform.InferType()(mod)

    def expected():
        composite_func_name = "ethos-n_0"
        inp = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")

        # partitioned func 2 (compute intensive)
        x = relay.nn.max_pool2d(inp, layout="NHWC")
        partitioned_func_2 = tei.make_ethosn_partition(x)[composite_func_name]
        gv_2 = relay.GlobalVar("ethos-n_1")

        mod = tvm.IRModule({})
        mod[gv_2] = partitioned_func_2
        main_expr = relay.reshape(inp, newshape=(1, 2, 2, 4))
        main_expr = relay.Call(gv_2, [main_expr])
        main_expr = relay.clip(main_expr, 0.0, 1.0)
        mod["main"] = relay.Function([inp], main_expr)
        return relay.transform.InferType()(mod)

    mod = before()
    mod = InlineNonComputeIntensivePartitions()(mod)
    expected_mod = expected()
    for global_var in mod.get_global_vars():
        _assert_structural_equal(mod[global_var.name_hint], expected_mod[global_var.name_hint])
