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
from tvm.relay.testing.resnet import get_workload
from tvm.relay.testing import run_opt_pass


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
    """
    dshape = (1, 1, 5, 1)
    x = relay.var("x", shape=dshape)
    y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=1)
    x1 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(3, 3), padding=(1, 1), channels=1)

    z = relay.add(y, x1)

    return tvm.IRModule.from_expr(z)


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


def test_extract_identity():
    mod = get_conv2d()
    op_freqs = relay.analysis.list_op_freqs(mod)
    assert len(op_freqs) == 1
    assert op_freqs["nn.conv2d"] == 1


def test_extract_conv_net():
    mod = get_conv_net()
    op_freqs = relay.analysis.list_op_freqs(mod)
    assert len(op_freqs) == 2
    assert op_freqs["add"] == 1
    assert op_freqs["nn.conv2d"] == 2


def test_extract_fused():
    mod = get_conv_net()
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FuseOps(3)(mod)

    op_freqs = relay.analysis.list_op_freqs(mod)
    assert len(op_freqs) == 2
    assert op_freqs["add"] == 1
    assert op_freqs["nn.conv2d"] == 2


def test_extract_resnet():
    mod, _params = get_workload()
    expected_op_freqs = {
        "nn.batch_norm": 19,
        "nn.conv2d": 21,
        "nn.relu": 18,
        "nn.max_pool2d": 1,
        "add": 8,
        "nn.global_avg_pool2d": 1,
        "nn.batch_flatten": 1,
        "nn.dense": 1,
        "nn.bias_add": 1,
        "nn.softmax": 1,
    }
    op_freqs = relay.analysis.list_op_freqs(mod)
    assert len(op_freqs) == len(expected_op_freqs)
    assert all([op_freqs[op] == expected_op_freqs[op] for op in expected_op_freqs])


if __name__ == "__main__":
    tvm.testing.main()
