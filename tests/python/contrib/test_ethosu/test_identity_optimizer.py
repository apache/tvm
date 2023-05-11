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
Test the identity optimizer pass that removes redundant identity
operations from the microNPU codegen.
"""
import pytest

pytest.importorskip("ethosu.vela")

import tensorflow as tf

import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.relay.backend.contrib.ethosu.codegen import relay_to_tir
from tvm.relay.backend.contrib.ethosu.codegen import IdentityOptimizer

from . import infra


def _optimize(func, optimize=True):
    """Create IRModule and run identity optimizer pass."""
    func = func.with_attr("Compiler", "ethos-u")
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    if optimize:
        mod = IdentityOptimizer()(mod)
    entry = mod["main"]
    return entry if isinstance(func, relay.Function) else entry.body


def _assert_structural_equal(a, b):
    """Check structural equality of two Relay expressions."""
    reason = (
        "Actual and expected relay functions are not equal. "
        "IdentityOptimizer is not correctly removing redundant "
        "identity operations."
    )
    assert tvm.ir.structural_equal(a, b), reason


def test_simple_reshape_identity_removal():
    """Check identity is removed when there is a reshape in
    the graph and a compute operation follows."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = infra.make_ethosu_conv2d(x, 4, 4, (1, 1), (0, 0), (1, 1), (1, 1))
        x = relay.reshape(x, newshape=(1, 4, 4, 1))
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        x = infra.make_ethosu_unary_elementwise(x, 1, "ABS")
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_simple_strided_slice_identity_removal():
    """Check identity is removed when there is a strided slice
    in the graph and a compute operation follows."""

    def get_graph(get_expected=False):
        dtype = "int8"

        x = relay.var("x", shape=(1, 2, 2, 4), dtype=dtype)
        x = infra.make_ethosu_pooling(x, "MAX", (1, 1), 4, dtype, (1, 1), (0, 0))
        x = relay.strided_slice(x, begin=[0, 0, 0, 0], end=[1, 2, 2, 2])
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        x = infra.make_ethosu_pooling(x, "MAX", (1, 1), 2, dtype, (1, 1), (0, 0))
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_no_identity():
    """Check the graph is not affected when there is no identity in the graph."""

    def get_graph():
        dtype = "int8"

        x = relay.var("x", shape=(1, 2, 2, 4), dtype=dtype)
        x = infra.make_ethosu_conv2d(x, 4, 4, (1, 1), (0, 0), (1, 1), (1, 1))
        x = infra.make_ethosu_pooling(x, "MAX", (1, 1), 4, dtype, (1, 1), (0, 0))
        x = infra.make_ethosu_depthwise_conv2d(x, 4, (1, 1), (0, 0), (1, 1), (1, 1))
        x = infra.make_ethosu_unary_elementwise(x, 4, "ABS")
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_reshape_last():
    """Check that an identity as a leaf of the graph is not removed."""

    def get_graph():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = infra.make_ethosu_conv2d(x, 4, 4, (1, 1), (0, 0), (1, 1), (1, 1))
        x = relay.reshape(x, newshape=(1, 4, 4, 1))
        x = infra.make_ethosu_identity(x)
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_requantize_identity_no_removal():
    """Check that an identity that actually performs a requantize isn't removed."""

    def get_graph():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 1, 4, 4))
        x = infra.make_ethosu_identity(
            x, ifm_scale=0.5, ifm_zero_point=1, ofm_scale=0.3, ofm_zero_point=2
        )
        x = infra.make_ethosu_unary_elementwise(x, 4, "ABS")
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_activation_identity_no_removal():
    """Check thst an identity with an activation isn't removed."""

    def get_graph():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 1, 4, 4))
        x = infra.make_ethosu_identity(x, activation="LUT")
        x = infra.make_ethosu_unary_elementwise(x, 4, "ABS")
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_multiple_output_identity():
    """Check that an identity is removed when it has multiple outputs."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        y = infra.make_ethosu_unary_elementwise(x, 4, "ABS")
        z = infra.make_ethosu_unary_elementwise(x, 4, "ABS")
        out = relay.concatenate((y, z), axis=0)
        return relay.Function(relay.analysis.free_vars(x), out)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_many_output_identity():
    """Check an identity with many outputs. It cannot be removed due
    to having a strided slice as output."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 1, 4, 4))
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        outputs = []
        for _ in range(4):
            outputs.append(infra.make_ethosu_unary_elementwise(x, 4, "ABS"))
        ss = relay.strided_slice(x, begin=(0, 0, 0, 0), end=(1, 1, 4, 4))
        identity_2 = infra.make_ethosu_identity(ss)
        outputs.append(identity_2)
        out = relay.concatenate(outputs, axis=0)
        return relay.Function(relay.analysis.free_vars(out), out)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_identity_before_concatenate_no_removal():
    """Check that an identity isn't removed when the operator
    following it is a concatenate operation."""

    def get_graph():
        x = relay.var("x", shape=(1, 1, 4, 4), dtype="int8")
        y = relay.var("y", shape=(1, 2, 2, 4), dtype="int8")
        z = relay.var("z", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 2, 2, 4))
        y = relay.strided_slice(y, begin=(0, 0, 0, 0), end=(1, 2, 2, 4))
        x = infra.make_ethosu_identity(x)
        y = infra.make_ethosu_identity(y)
        out = relay.concatenate([x, y, z], axis=0)
        return relay.Function(relay.analysis.free_vars(out), out)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_identity_removal_with_multiple_transform_ops():
    """Check that only an identity directly parent to a compute
    operation is removed."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.strided_slice(x, begin=[0, 0, 0, 0], end=[1, 2, 2, 2])
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        x = relay.reshape(x, newshape=(1, 1, 1, 8))
        if not get_expected:
            x = infra.make_ethosu_identity(x)
        x = infra.make_ethosu_unary_elementwise(x, 8, "ABS")
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_identity_removal_on_binary_elementwise():
    """Check identities before binary elementwise are removed correctly."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        y = relay.var("y", shape=(1, 2, 2, 4), dtype="int8")
        if not get_expected:
            x = infra.make_ethosu_identity(x)
            y = infra.make_ethosu_identity(y)
        z = infra.make_ethosu_binary_elementwise(x, y, 4, 4, "ADD", "int8")
        return relay.Function(relay.analysis.free_vars(z), z)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_identity_single_removal_on_binary_elementwise():
    """Check that identity on the second input of the binary elementwise
    operation is removed while the other input has no identity."""

    def get_graph(get_expected=False):
        x = relay.var("x", shape=(1, 4, 1, 4), dtype="int8")
        y = relay.var("y", shape=(1, 2, 2, 4), dtype="int8")
        y = relay.reshape(y, newshape=(1, 4, 1, 4))
        if not get_expected:
            y = infra.make_ethosu_identity(y)
        z = infra.make_ethosu_binary_elementwise(x, y, 4, 4, "ADD", "int8")
        return relay.Function(relay.analysis.free_vars(z), z)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(get_expected=True), optimize=False)
    _assert_structural_equal(actual, expected)


def test_multiple_transform_ops_with_reduction_in_dimensionality():
    """Removal of an identity operation between two transform operations is usually okay.
    However, if the dimensionality of the input is reduced by the second transformation
    operation, it can lead to an output mismatch. Checking that the pass doesn't remove
    an identity given this case."""

    def get_graph():
        x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
        x = relay.strided_slice(x, begin=(0, 0, 0, 0), end=(1, 2, 2, 2))
        x = infra.make_ethosu_identity(x)
        x = relay.reshape(x, newshape=(1, 2, 4))
        x = infra.make_ethosu_identity(x)
        return relay.Function(relay.analysis.free_vars(x), x)

    actual = _optimize(get_graph())
    expected = _optimize(get_graph(), optimize=False)
    _assert_structural_equal(actual, expected)


def test_identity_optimizer_runs_in_compilation_pipeline():
    """Checks that the identity optimization pass is run as part of the NPU compilation pipeline."""

    def get_graph():
        x = relay.var("x", shape=(1, 4, 4, 4), dtype="int8")
        x = relay.reshape(x, newshape=(1, 1, 16, 4))
        x = relay.nn.max_pool2d(x, layout="NHWC")
        func = relay.Function(relay.analysis.free_vars(x), x)
        return tvm.IRModule.from_expr(func)

    mod = get_graph()
    mod = partition_for_ethosu(mod)
    mod = relay_to_tir(mod)

    external_gv_name = mod["main"].body.op.name_hint
    prim_func = mod[external_gv_name]

    # Check for hints in the TIR prim func that the identity optimization pass
    # has ran. There should not be an identity in the prim func.
    assert prim_func.body.value.args[0] == "ethosu_pooling"


def test_same_output():
    """Check that the output remains the same when the identity
    optimizer pass removes some identities inserted during legalization."""
    ifm_shapes = [(1, 1, 25, 8), (1, 5, 5, 8)]

    @tf.function
    def model(x, y):
        x = tf.reshape(x, (1, 5, 5, 8))
        z = tf.add(x, y)
        z = tf.reshape(z, (1, 1, 25, 8))
        return z

    infra.compare_tvm_with_tflite(model, ifm_shapes, "ethos-u55-256", enable_cascader=False)


def test_multi_output_identity_has_same_output():
    """Check that the output remains the same with an identity with
    multiple outputs."""
    ifm_shape = (1, 1, 64, 16)

    @tf.function
    def model(x):
        x = tf.reshape(x, (1, 8, 8, 16))
        outputs = []
        for _ in range(4):
            outputs.append(tf.nn.max_pool2d(x, 1, 1, "VALID"))
        outputs.append(tf.reshape(x, (1, 8, 8, 16)))
        y = tf.concat(outputs, axis=0)
        return y

    infra.compare_tvm_with_tflite(model, [ifm_shape], "ethos-u55-256", enable_cascader=False)


def test_multiple_transform_ops_same_output():
    """Check case of identity removal between transform ops and
    then without, making sure they have the same output."""
    ifm_shape = (1, 2, 2, 4)

    @tf.function
    def model(x):
        x = tf.reshape(x, (1, 1, 4, 4))
        x = tf.slice(x, (0, 0, 0, 0), (1, 1, 4, 3))
        x = tf.reshape(x, (12,))
        return x

    infra.compare_tvm_with_tflite(model, [ifm_shape], "ethos-u55-256", enable_cascader=False)
