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


def get_net(batch_size, num_actions=18, image_shape=(4, 84, 84), dtype="float32", layout="NCHW"):
    """get symbol of nature dqn"""
    data_shape = (batch_size,) + image_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)

    bias_axis = layout.index("C")

    conv1_bias = relay.var("conv1_bias")
    conv1 = layers.conv2d(
        data,
        kernel_size=(8, 8),
        strides=(4, 4),
        padding=(0, 0),
        channels=32,
        name="conv1",
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
    )
    conv1 = relay.nn.bias_add(conv1, conv1_bias, bias_axis)
    relu1 = relay.nn.relu(conv1)

    conv2_bias = relay.var("conv2_bias")
    conv2 = layers.conv2d(
        relu1,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding=(0, 0),
        channels=64,
        name="conv2",
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
    )
    conv2 = relay.nn.bias_add(conv2, conv2_bias, bias_axis)
    relu2 = relay.nn.relu(conv2)

    conv3_bias = relay.var("conv3_bias")
    conv3 = layers.conv2d(
        relu2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(0, 0),
        channels=64,
        name="conv3",
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
    )
    conv3 = relay.nn.bias_add(conv3, conv3_bias, bias_axis)
    relu3 = relay.nn.relu(conv3)

    bf1 = relay.nn.batch_flatten(relu3)
    dense1 = layers.dense_add_bias(bf1, units=512, name="dense1")
    relu4 = relay.nn.relu(dense1)
    dense2 = layers.dense_add_bias(relu4, units=num_actions, name="dense2")

    args = relay.analysis.free_vars(dense2)
    return relay.Function(args, dense2)


def get_workload(
    batch_size, num_actions=18, image_shape=(4, 84, 84), dtype="float32", layout="NCHW"
):
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
    mod : tvm.IRModule
        The relay module that contains a DQN network.
    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(
        batch_size, num_actions=num_actions, image_shape=image_shape, dtype=dtype, layout=layout
    )
    return create_workload(net)
