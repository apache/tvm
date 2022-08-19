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

# coding: utf-8
# pylint: disable=unused-argument

"""
Symbol of SqueezeNet

Reference:
Iandola, Forrest N., et al.
"Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." (2016).
"""

from tvm import relay
from tvm.relay.testing import create_workload

# Helpers
def _make_fire(net, squeeze_channels, expand1x1_channels, expand3x3_channels, prefix=""):
    net = _make_fire_conv(net, squeeze_channels, 1, 0, f"{prefix}/squeeze1x1")

    left = _make_fire_conv(net, expand1x1_channels, 1, 0, f"{prefix}/expand1x1")
    right = _make_fire_conv(net, expand3x3_channels, 3, 1, f"{prefix}/expand3x3")
    # NOTE : Assume NCHW layout here
    net = relay.concatenate((left, right), axis=1)
    return net


def _make_fire_conv(net, channels, kernel_size, padding=0, prefix=""):
    net = relay.nn.conv2d(
        net,
        relay.var(f"{prefix}_weight"),
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(padding, padding),
    )
    net = relay.nn.bias_add(net, relay.var(f"{prefix}_bias"))
    net = relay.nn.relu(net)
    return net


# Net
def get_net(batch_size, image_shape, num_classes, dtype):
    """Get symbol of SqueezeNet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    image_shape : tuple
        The input image shape

    num_classes: int
        The number of classification results

    dtype : str
        The data type

    """
    data_shape = (batch_size,) + image_shape
    net = relay.var("data", shape=data_shape, dtype=dtype)
    net = relay.nn.conv2d(
        net,
        relay.var("conv1_weight"),
        channels=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(0, 0),
    )
    net = relay.nn.bias_add(net, relay.var("conv1_bias"))
    net = relay.nn.relu(net)
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    net = _make_fire(net, 16, 64, 64, "fire2")
    net = _make_fire(net, 16, 64, 64, "fire3")
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    net = _make_fire(net, 32, 128, 128, "fire4")
    net = _make_fire(net, 32, 128, 128, "fire5")
    net = relay.nn.max_pool2d(net, pool_size=(3, 3), strides=(2, 2))
    net = _make_fire(net, 48, 192, 192, "fire6")
    net = _make_fire(net, 48, 192, 192, "fire7")
    net = _make_fire(net, 64, 256, 256, "fire8")
    net = _make_fire(net, 64, 256, 256, "fire9")
    net = relay.nn.dropout(net, rate=0.5)
    net = relay.nn.conv2d(net, relay.var("conv10_weight"), channels=num_classes, kernel_size=(1, 1))
    net = relay.nn.bias_add(net, relay.var("conv10_bias"))
    net = relay.nn.relu(net)
    net = relay.nn.global_avg_pool2d(net)
    net = relay.nn.softmax(net, axis=1)
    args = relay.analysis.free_vars(net)
    return relay.Function(args, net)


def get_workload(batch_size=1, image_shape=(3, 224, 224), num_classes=1000, dtype="float32"):
    """Get benchmark workload for SqueezeNet

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : relay.Function
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """

    net = get_net(batch_size, image_shape, num_classes, dtype)
    return create_workload(net)
