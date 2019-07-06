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
"""Test alter conv2d layout pass"""
import tvm
import nnvm

from tvm import relay
from tvm import autotvm
from tvm.relay import transform
from tvm.relay.analysis import alpha_equal


def test_alter_layout_conv2d():
    """Additional layout transformations should occour on the graph.
    """

    def convnet():
        """Alternating layout of simple convnet (from image super-resolution).
        """
        bias1 = relay.var('bias1', shape=(64,))
        bias2 = relay.var('bias2', shape=(64,))
        bias3 = relay.var('bias3', shape=(64,))
        bias4 = relay.var('bias4', shape=(64,))
        weight1 = relay.var('weight1', shape=(64, 1, 5, 5))
        weight2 = relay.var('weight2', shape=(64, 64, 3, 3))
        weight3 = relay.var('weight3', shape=(64, 64, 3, 3))
        weight4 = relay.var('weight4', shape=(64, 64, 3, 3))
        data = relay.var("x", shape=(1, 1, 224, 224))
        n00 = relay.nn.conv2d(data, weight1, padding=[2, 2], kernel_size=[5, 5])
        n01 = relay.expand_dims(bias1, axis=1, num_newaxis=2)
        n02 = relay.add(n00, n01)
        n03 = relay.nn.relu(n02)
        n04 = relay.nn.conv2d(n03, weight2, padding=[1, 1], kernel_size=[3, 3])
        n05 = relay.expand_dims(bias2, axis=1, num_newaxis=2)
        n06 = relay.add(n04, n05)
        n07 = relay.nn.relu(n06)
        n08 = relay.nn.conv2d(n07, weight3, padding=[1, 1], kernel_size=[3, 3])
        n09 = relay.expand_dims(bias3, axis=1, num_newaxis=2)
        n10 = relay.add(n08, n09)
        n11 = relay.nn.relu(n10)
        n12 = relay.nn.conv2d(n11, weight4, padding=[1, 1], kernel_size=[3, 3])
        n13 = relay.expand_dims(bias4, axis=1, num_newaxis=2)
        n14 = relay.add(n12, n13)
        n15 = relay.reshape(n14, newshape=[1, 1, 3, 3, 224, 224])
        n16 = relay.transpose(n15, axes=[0, 1, 4, 2, 5, 3])
        net = relay.reshape(n16, newshape=[1, 1, 672, 672])
        args = relay.analysis.free_vars(net)
        return relay.Function(args, net)

    # orig net
    N = convnet()

    # trigger a test
    # for each known alter_conv2d
    targets=['cuda',
             'opencl -device=mali',
             'opencl -device=intel_graphics',
             'llvm -device=arm_cpu',
             'llvm -device=core-avx-ii']

    for tgt in targets:
        with tvm.target.create(tgt) as target:
            with autotvm.tophub.context(target):
                mod = relay.Module.from_expr(N)
                mod = transform.AlterOpLayout()(mod)
                O = mod["main"]

                # graph should differ
                assert not relay.analysis.alpha_equal(N, O)

if __name__ == "__main__":
    np.random.seed(42)
    test_alter_layout_conv2d()
