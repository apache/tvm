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
"""Test the pass that removes unnecssary identity operation if the identity
uses LUT and the preceding operator is LUT capable and doesn't already have a LUT.
"""
import pytest

pytest.importorskip("ethosu.vela")

import tensorflow as tf
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.codegen import LUTsOptimizer
from tvm.relay.backend.contrib.ethosu.codegen import relay_to_tir
from tvm.relay.op.contrib.ethosu import partition_for_ethosu

from . import infra


def test_merge_lut_into_conv():
    """If an operator that has a LUT attribute is followed by an identity operator
    with LUT, we can merge the two operataors."""

    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    lut1 = relay.const([i for i in range(256)], dtype="int8")
    lut2 = relay.const([i for i in reversed(range(256))], dtype="int8")

    def before():
        conv1 = infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
        id1 = infra.make_ethosu_identity(conv1, lut=lut1, activation="TANH")
        conv2 = infra.make_ethosu_conv2d(id1, 4, 7, (2, 2), (1, 1), (1, 1), (1, 1))
        id2 = infra.make_ethosu_identity(conv2, lut=lut2, activation="SIGMOID")

        func = relay.Function(relay.analysis.free_vars(id2), id2)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        return mod

    def after():
        conv1 = infra.make_ethosu_conv2d(
            ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1), lut=lut1, activation="TANH"
        )
        conv2 = infra.make_ethosu_conv2d(
            conv1, 4, 7, (2, 2), (1, 1), (1, 1), (1, 1), lut=lut2, activation="SIGMOID"
        )

        func = relay.Function(relay.analysis.free_vars(conv2), conv2)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        return mod

    mod = LUTsOptimizer()(before())
    mod = relay.transform.InferType()(mod)

    assert tvm.ir.structural_equal(mod, after())


def test_merge_lut_into_binary_elementwise():
    """If an binary elementwise operator is followed by an identity operator
    with LUT, we can merge the two operataors."""

    shape = (1, 8, 8, 4)
    dtype = "int8"
    ifm = relay.var("x", shape=shape, dtype=dtype)
    ifm2 = relay.var("x", shape=shape, dtype=dtype)
    lut1 = relay.const([i for i in range(256)], dtype=dtype)
    lut2 = relay.const([i for i in reversed(range(256))], dtype=dtype)

    def before():
        sub = infra.make_ethosu_binary_elementwise(ifm, ifm2, shape[-1], shape[-1], "SUB", dtype)
        id1 = infra.make_ethosu_identity(sub, lut=lut1, activation="TANH")
        add = infra.make_ethosu_binary_elementwise(id1, ifm2, shape[-1], shape[-1], "ADD", dtype)
        id2 = infra.make_ethosu_identity(add, lut=lut2, activation="SIGMOID")

        func = relay.Function(relay.analysis.free_vars(id2), id2)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        return mod

    def after():
        sub = infra.make_ethosu_binary_elementwise(
            ifm, ifm2, shape[-1], shape[-1], "SUB", dtype, lut=lut1, activation="TANH"
        )
        add = infra.make_ethosu_binary_elementwise(
            sub, ifm2, shape[-1], shape[-1], "ADD", dtype, lut=lut2, activation="SIGMOID"
        )

        func = relay.Function(relay.analysis.free_vars(add), add)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        return mod

    mod = LUTsOptimizer()(before())
    mod = relay.transform.InferType()(mod)

    assert tvm.ir.structural_equal(mod, after())


def test_multiple_luts():
    """Test that when an operation already has a LUT, we don't overwrite that LUT"""

    ifm = relay.var("x", shape=(1, 8, 8, 4), dtype="int8")
    lut1 = relay.const([i for i in range(256)], dtype="int8")
    lut2 = relay.const([i for i in reversed(range(256))], dtype="int8")

    def before():
        conv1 = infra.make_ethosu_conv2d(ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1))
        id1 = infra.make_ethosu_identity(conv1, lut=lut1, activation="TANH")
        id2 = infra.make_ethosu_identity(id1, lut=lut2, activation="TANH")

        func = relay.Function(relay.analysis.free_vars(id2), id2)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        return mod

    def after():
        conv1 = infra.make_ethosu_conv2d(
            ifm, 4, 4, (3, 3), (1, 1), (1, 1), (1, 1), lut=lut1, activation="TANH"
        )
        id2 = infra.make_ethosu_identity(conv1, lut=lut2, activation="TANH")

        func = relay.Function(relay.analysis.free_vars(id2), id2)
        func = func.with_attr("Compiler", "ethos-u")
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        return mod

    mod = LUTsOptimizer()(before())
    mod = relay.transform.InferType()(mod)

    assert tvm.ir.structural_equal(mod, after())


def test_lut_optimizer_runs_in_compilation_pipeline():
    """Test that the LUT optimization pass runs as part of the NPU compilation pipeline."""
    ifm_shape = (1, 4, 4, 4)

    @tf.function
    def get_graph(x):
        weight1 = tf.constant(np.random.uniform(size=(1, 1, 4, 4)), dtype=tf.float32)
        op = tf.nn.conv2d(x, weight1, (1, 1), "VALID")
        op = tf.nn.tanh(op)
        weight2 = tf.constant(np.random.uniform(size=(1, 1, 4, 1)), dtype=tf.float32)
        op = tf.nn.depthwise_conv2d(op, weight2, (1, 1, 1, 1), "VALID")
        return tf.nn.tanh(op)

    mod, _ = infra.get_tflite_graph(get_graph, [ifm_shape])
    mod = partition_for_ethosu(mod)
    mod = relay_to_tir(mod)

    external_gv_name = mod["main"].body.op.name_hint
    prim_func = mod[external_gv_name]

    # Check for hints in the TIR prim func that the LUT optimization pass has ran.
    # If the module was optimized, there should be no identity operations.
    def check_identity(stmt):
        if isinstance(stmt, tvm.tir.expr.Call):
            assert stmt.args[0] != "ethosu_identity"

    tvm.tir.stmt_functor.post_order_visit(prim_func.body, check_identity)
