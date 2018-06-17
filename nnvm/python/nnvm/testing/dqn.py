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
Symbol of Nature DQN

Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
Nature 518.7540 (2015): 529.
"""

from .. import symbol as sym
from . utils import create_workload

def get_symbol(num_actions=18):
    """get symbol of nature dqn"""
    data = sym.Variable(name='data')
    net = sym.conv2d(data, kernel_size=(8, 8), strides=(4, 4), padding=(0, 0),
                     channels=32, name='conv1')
    net = sym.relu(net, name='relu1')
    net = sym.conv2d(net, kernel_size=(4, 4), strides=(2, 2), padding=(0, 0),
                     channels=64, name='conv2')
    net = sym.relu(net, name='relu2')
    net = sym.conv2d(net, kernel_size=(3, 3), strides=(1, 1), padding=(0, 0),
                     channels=64, name='conv3')
    net = sym.relu(net, name='relu3')
    net = sym.flatten(net, name='flatten')
    net = sym.dense(net, units=512, name='fc4')
    net = sym.relu(net, name='relu4')
    net = sym.dense(net, units=num_actions, name='fc5')

    return net


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
    net = get_symbol(num_actions=num_actions)
    return create_workload(net, batch_size, image_shape, dtype)
