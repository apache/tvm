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
"""Unit test for annotating device placement."""
import os
import sys
import numpy as np
import pytest

import tvm
import tvm.relay.testing
import tvm.relay.transform as transform
from tvm import relay
from tvm import runtime
from tvm.contrib import utils
from tvm import relay, tir, autotvm
from tvm.relay import transform
from tvm.relay.expr import Call, TupleGetItem, Var, Constant, Tuple
from tvm.ir import Op

#        a  b  a  b
#         \/    \/
#         add  add
#          \   /
#           \ /
#           mul
#           /  \
#       c  / c  |
#        \/   \/
#        mul  mul
#         \   /
#          \ /
#          add


def get_expected_model(cpu_ctx, dev_ctx):
    a = relay.var("a", shape=(2, 3))
    b = relay.var("b", shape=(2, 3))
    c = relay.var("c", shape=(2, 3))
    add1 = relay.add(a, b)
    add2 = relay.add(a, b)
    mul1 = relay.annotation.on_device(relay.multiply(add1, add2), dev_ctx)
    mul2 = relay.annotation.on_device(relay.multiply(mul1, c), dev_ctx)
    mul3 = relay.annotation.on_device(relay.multiply(mul1, c), dev_ctx)
    add3 = relay.add(mul2, mul3)
    func = relay.Function([a, b, c], add3)

    mod = tvm.IRModule()
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    return mod


def get_annotated_model(cpu_ctx, dev_ctx):
    a = relay.var("a", shape=(2, 3))
    b = relay.var("b", shape=(2, 3))
    c = relay.var("c", shape=(2, 3))
    add1 = relay.add(a, b)
    add2 = relay.add(a, b)
    mul1 = relay.multiply(add1, add2)
    mul2 = relay.multiply(mul1, c)
    mul3 = relay.multiply(mul1, c)
    add3 = relay.add(mul2, mul3)
    func = relay.Function([a, b, c], add3)

    mod = tvm.IRModule()
    mod["main"] = func

    def get_placement(expr):
        """This method is called for each Call node in the graph. Return the targeted
        compiler for each Op or "default"
        """
        target_ops = ["multiply"]
        placement = -1
        if isinstance(expr, Call):
            if isinstance(expr.op, Op):
                if expr.op.name in target_ops:
                    placement = dev_ctx.device_type
        return placement

    mod = relay.transform.AnnotateDevicePlacement(get_placement)(mod)
    return mod


def test_device_placement():
    ctx1 = tvm.context("cpu")
    ctx2 = tvm.context("llvm")
    mod = get_annotated_model(ctx1, ctx2)
    expected_mod = get_expected_model(ctx1, ctx2)
    assert tvm.ir.structural_equal(mod["main"], expected_mod["main"], map_free_vars=True)
