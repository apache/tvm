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
Net of Nature DQN
Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
Nature 518.7540 (2015): 529.
"""

from tvm import relay
from . import layers
from .init import create_workload

def get_net(batch_size, num_actions=18, image_shape=(4, 84, 84), dtype="float32"):
    """get symbol of nature dqn"""
    data_shape = (batch_size,) + image_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)
    conv1 = layers.conv2d(data, kernel_size=(8, 8), strides=(4, 4), padding=(0, 0),
                     channels=32, name='conv1')
    relu1 = relay.nn.relu(conv1)
    conv2 = layers.conv2d(relu1, kernel_size=(4, 4), strides=(2, 2), padding=(0, 0),
                     channels=64, name='conv2')
    relu2 = relay.nn.relu(conv2)
    conv3 = layers.conv2d(relu2, kernel_size=(3, 3), strides=(1, 1), padding=(0, 0),
                     channels=64, name='conv3')
    relu3 = relay.nn.relu(conv3)
    bf1 = relay.nn.batch_flatten(relu3)
    dense1_weight = relay.var("dense1_weight")
    dense1 = relay.nn.dense(bf1, dense1_weight, units=512)
    relu4 = relay.nn.relu(dense1)
    dense2_weight = relay.var("dense2_weight")
    dense2 = relay.nn.dense(relu4, dense2_weight, units=num_actions)

    args = relay.ir_pass.free_vars(dense2)
    return relay.Function(args, dense2)


def get_workload(batch_size, num_actions=18, image_shape=(4, 84, 84), dtype="float32"):
    """Get benchmark workload for a Deep Q Network
    Parameters
    ----------
    batch_size : int
        The batch size used in the model
    num_actions : int, optional
        Number of actions
    image_shape : tuple, optional
        The input image shape
    dtype : str, optional
        The data type
    Returns
    -------
    net : nnvm.symbol
        The computational graph
    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, num_actions=num_actions, image_shape=image_shape, dtype=dtype)
    return create_workload(net)
