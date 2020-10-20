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
Synthetic networks for testing purposes. Ideally, these networks are similar in
structure to real world networks, but are much smaller in order to make testing
faster.
"""
from __future__ import absolute_import
from tvm import relay
from .init import create_workload, Constant
from . import layers


def get_net(input_shape=(1, 3, 24, 12), dtype="float32", wtype=None):
    """Get synthetic testing network.

    Parameters
    ----------
    image_shape : tuple, optional
        The input shape as (batch_size, channels, height, width).

    dtype : str, optional
        The data type for the input.

    wtype : str, optional
        The data type for weights. Defaults to `dtype`.

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
    if wtype is None:
        wtype = dtype
    data = relay.var("data", shape=input_shape, dtype=dtype)
    dense_shape = [-1, input_shape[3]]
    dense = relay.nn.relu(
        relay.nn.dense(
            relay.reshape(data, dense_shape),
            relay.var("dense_weight", shape=[input_shape[3], dense_shape[1]], dtype=wtype),
        )
    )
    dense = relay.reshape_like(dense, data)
    conv_shape = [input_shape[1], input_shape[1], 3, 3]
    conv = relay.nn.softmax(
        relay.nn.conv2d(
            data,
            relay.var("conv_weight", shape=conv_shape, dtype=wtype),
            padding=1,
            kernel_size=3,
        )
    )
    added = relay.add(dense, conv)
    biased = layers.batch_norm_infer(
        relay.nn.bias_add(added, relay.var("bias", dtype=wtype)), name="batch_norm"
    )
    dense = relay.nn.relu(
        relay.nn.dense(
            relay.reshape(biased, dense_shape),
            relay.var("dense2_weight", shape=[input_shape[3], dense_shape[1]], dtype=wtype),
        )
    )
    dense = relay.reshape_like(dense, data)
    conv = relay.nn.softmax(
        relay.nn.conv2d(
            biased,
            relay.var("conv2_weight", shape=conv_shape, dtype=wtype),
            padding=1,
            kernel_size=3,
        )
    )
    added = relay.add(dense, conv)
    args = relay.analysis.free_vars(added)
    return relay.Function(args, added)


def get_workload(input_shape=(1, 3, 24, 12), dtype="float32", wtype=None):
    """Get benchmark workload for the synthetic net.

    Parameters
    ----------
    image_shape : tuple, optional
        The input shape as (batch_size, channels, height, width).

    dtype : str, optional
        The data type for the input.

    wtype : str, optional
        The data type for weights. Defaults to `dtype`.

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a synthetic network.

    params : dict of str to NDArray
        The parameters.
    """
    return create_workload(
        get_net(input_shape=input_shape, dtype=dtype, wtype=wtype),
        initializer=Constant(),
    )
