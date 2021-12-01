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
# pylint: disable=unused-wildcard-import
import numpy as np
import pytest

import tvm
import pytest
from tests.python.relay.test_dataflow_pattern import (
    K_BROADCAST,
    K_ELEMWISE,
    K_INJECTIVE,
    K_OUT_EWISE_FUSABLE,
)
from tvm import relay
from tvm.ir.module import IRModule
from tvm.relay import transform
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.testing import run_opt_pass
from tvm.relay.transform.transform import FuseWithPattern

DEPTH_OP = 256


def test_fuse_max():
    """Test the constraint of number of nodes in op fusion."""

    def before(n):
        x = relay.var("x", shape=(10, 20))
        y = x
        for i in range(n):
            y = relay.exp(y)
        return relay.Function([x], y)

    n = 300
    z = before(n)
    fuse_pattern = relay.transform.FuseWithPattern()
    zz = run_opt_pass(z, fuse_pattern)
    zzz = run_opt_pass(z, fuse_pattern)

    print(zz)
    print("-------------------")
    print(zzz)
    # assert tvm.ir.structural_equal(zz, zzz)


def test_fuse_max_configs():
    """Test the constraint of number of nodes in op fusion."""

    def before(n):
        x = relay.var("x", shape=(10, 20))
        y = x
        for i in range(n):
            y = relay.exp(y)
        return relay.Function([x], y)

    n = 300
    z = before(n)
    fuse_pattern = relay.transform.FuseWithPattern()
    zz = run_opt_pass(z, fuse_pattern)
    zzz = run_opt_pass(z, fuse_pattern)

    print(zz)
    print("-------------------")
    print(zzz)
    # assert tvm.ir.structural_equal(zz, zzz)


def three_paths():
    x = relay.var("x", shape=(1, 16, 64, 64))
    w1 = relay.var("w1", shape=(16, 16, 3, 3))
    b1 = relay.var("b", shape=(3,))
    w2 = relay.var("w2", shape=(16, 16, 3, 3))
    w3 = relay.var("w3", shape=(16, 16, 3, 3))
    w4 = relay.var("w4", shape=(16, 16, 3, 3))
    b2 = relay.var("b", shape=(3,))

    y0 = relay.nn.conv2d(x, w1, kernel_size=(3, 3), padding=(1, 1, 1, 1), channels=16)
    y1 = relay.multiply(y0, relay.const(1, "float32"))
    y2 = relay.add(relay.const(2, "float32"), y1)
    y = relay.nn.relu(y2)
    # second path
    z1 = relay.nn.conv2d(y, w2, kernel_size=(3, 3), padding=(1, 1, 1, 1), channels=16)
    z2 = relay.multiply(z1, relay.const(1, "float32"))
    z3 = relay.add(relay.const(2, "float32"), z2)
    z4 = relay.nn.relu(z3)
    # third path
    c1 = relay.nn.conv2d(z4, w3, kernel_size=(3, 3), padding=(1, 1, 1, 1), channels=16)
    c2 = relay.multiply(c1, relay.const(1, "float32"))
    c3 = relay.add(relay.const(2, "float32"), c2)
    c4 = relay.nn.relu(c3)
    return relay.Function(relay.analysis.free_vars(c4), c4)


def make_pattern_vertical_fuse():
    x = wildcard()
    y = wildcard()
    z = wildcard()
    w = wildcard()
    conv_node = wildcard().has_attr({"TOpPattern": K_OUT_EWISE_FUSABLE})(x, y)
    bc = wildcard().has_attr({"TOpPattern": K_BROADCAST})(conv_node, z)
    for i in range(1, DEPTH_OP):
        bc = bc | wildcard().has_attr({"TOpPattern": K_BROADCAST})(w, bc)
    r = wildcard().has_attr({"TOpPattern": K_ELEMWISE})(bc)
    return r


def test_fuse_pattern():

    vertical_fuse_patten = [("make_pattern_vertical_fuse", make_pattern_vertical_fuse())]

    input_func = three_paths()

    mod = IRModule.from_expr(input_func)

    seq = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.MergeComposite(vertical_fuse_patten),
            relay.transform.PartitionGraph(),
        ]
    )
    mod = seq(mod)

    mod_r = IRModule.from_expr(input_func)

    seq = tvm.transform.Sequential([relay.transform.InferType(), relay.transform.FuseOps()])
    mod_r = seq(mod_r)

    fuse_patter = run_opt_pass(input_func, relay.transform.FuseWithPattern(), import_prelude=False)

    print("-----------Pattern Partitioner--------")
    print(mod)

    print("-----------Legacy Fuse Ops--------")
    print(mod_r)

    print("----- FuseWithPattern ------------")
    print(fuse_patter)


if __name__ == "__main__":
    pytest.main([__file__])
