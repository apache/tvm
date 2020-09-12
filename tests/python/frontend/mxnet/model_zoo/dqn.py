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
The mxnet symbol of Nature DQN

Reference:
Mnih, Volodymyr, et al.
"Human-level control through deep reinforcement learning."
Nature 518.7540 (2015): 529.
"""

import mxnet as mx


def get_symbol(num_action=18):
    data = mx.sym.Variable(name="data")
    net = mx.sym.Convolution(data, kernel=(8, 8), stride=(4, 4), num_filter=32, name="conv1")
    net = mx.sym.Activation(net, act_type="relu", name="relu1")
    net = mx.sym.Convolution(net, kernel=(4, 4), stride=(2, 2), num_filter=64, name="conv2")
    net = mx.sym.Activation(net, act_type="relu", name="relu2")
    net = mx.sym.Convolution(net, kernel=(3, 3), stride=(1, 1), num_filter=64, name="conv3")
    net = mx.sym.Activation(net, act_type="relu", name="relu3")
    net = mx.sym.FullyConnected(net, num_hidden=512, name="fc4")
    net = mx.sym.Activation(net, act_type="relu", name="relu4")
    net = mx.sym.FullyConnected(net, num_hidden=num_action, name="fc5", flatten=False)

    return net
