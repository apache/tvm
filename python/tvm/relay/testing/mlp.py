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
a simple multilayer perceptron
"""
from __future__ import absolute_import
from tvm import relay
from .init import create_workload


def get_net(batch_size, num_classes=10, image_shape=(1, 28, 28), dtype="float32"):
    """Get network a simple multilayer perceptron.

    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    data_shape = (batch_size,) + image_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)
    data = relay.nn.batch_flatten(data)
    fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=128)
    fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
    act1 = relay.nn.relu(fc1)
    fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=64)
    fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
    act2 = relay.nn.relu(fc2)
    fc3 = relay.nn.dense(act2, relay.var("fc3_weight"), units=num_classes)
    fc3 = relay.nn.bias_add(fc3, relay.var("fc3_bias"), axis=-1)
    mlp = relay.nn.softmax(data=fc3)
    args = relay.analysis.free_vars(mlp)
    return relay.Function(args, mlp)


def get_workload(batch_size, num_classes=10, image_shape=(1, 28, 28), dtype="float32"):
    """Get benchmark workload for a simple multilayer perceptron.

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a mlp network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, num_classes, image_shape, dtype)
    return create_workload(net)
