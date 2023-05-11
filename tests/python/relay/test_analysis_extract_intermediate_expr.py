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
"""Test function extraction"""
import pytest
import tvm
from tvm import relay


def get_conv_net():
    """This gets the net for:
          conv2d
          /  |
         /   |
    conv2d   |
        \    |
         \   |
        elemwise add
             |
             |
             |
           split
             |
             |
             |
        elemwise add
    """
    dshape = (1, 1, 5, 1)
    x = relay.var("x", shape=dshape)
    y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
    x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)

    z = relay.add(y, x1)

    tuple_out = relay.op.split(z, indices_or_sections=1, axis=0)

    tuple_0_add = relay.add(tuple_out[0], relay.const(1, dtype="float32"))

    return tvm.IRModule.from_expr(tuple_0_add)


def get_conv2d():
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
    y = relay.nn.conv2d(
        x,
        weight1,
        channels=32,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    return tvm.IRModule.from_expr(y)


def test_extract():
    dshape = (1, 1, 5, 1)

    def before():
        return get_conv_net()

    def expected_0():
        x = relay.var("x", shape=dshape)
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        return tvm.IRModule.from_expr(y)

    def expected_1():
        x = relay.var("x", shape=dshape)
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        return tvm.IRModule.from_expr(x1)

    def expected_2():
        x = relay.var("x", shape=dshape)
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        z = relay.add(y, x1)
        return tvm.IRModule.from_expr(z)

    def expected_3():
        x = relay.var("x", shape=dshape)
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        z = relay.add(y, x1)
        tuple_out = relay.op.split(z, indices_or_sections=1, axis=0)
        return tvm.IRModule.from_expr(tuple_out.astuple())

    def expected_4():
        # check tuple node
        x = relay.var("x", shape=dshape)
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)
        z = relay.add(y, x1)
        tuple_out = relay.op.split(z, indices_or_sections=1, axis=0)
        return tvm.IRModule.from_expr(tuple_out[0])

    assert tvm.ir.structural_equal(
        relay.analysis.extract_intermdeiate_expr(before(), 0), expected_0()
    )
    assert tvm.ir.structural_equal(
        relay.analysis.extract_intermdeiate_expr(before(), 1), expected_1()
    )
    assert tvm.ir.structural_equal(
        relay.analysis.extract_intermdeiate_expr(before(), 2), expected_2()
    )
    assert tvm.ir.structural_equal(
        (relay.analysis.extract_intermdeiate_expr(before(), 3)), expected_3()
    )
    assert tvm.ir.structural_equal(
        relay.analysis.extract_intermdeiate_expr(before(), 4), expected_4()
    )
    assert tvm.ir.structural_equal(relay.analysis.extract_intermdeiate_expr(before(), 5), before())


if __name__ == "__main__":
    tvm.testing.main()
