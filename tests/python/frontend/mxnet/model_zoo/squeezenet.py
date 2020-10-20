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
Symbol of SqueezeNet

Reference:
Iandola, Forrest N., et al.
"Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." (2016).
"""

import mxnet as mx

# Helpers
def _make_fire(net, squeeze_channels, expand1x1_channels, expand3x3_channels):
    net = _make_fire_conv(net, squeeze_channels, 1, 0)

    left = _make_fire_conv(net, expand1x1_channels, 1, 0)
    right = _make_fire_conv(net, expand3x3_channels, 3, 1)
    # NOTE : Assume NCHW layout here
    net = mx.sym.concat(left, right, dim=1)

    return net


def _make_fire_conv(net, channels, kernel_size, padding=0):
    net = mx.sym.Convolution(
        net, num_filter=channels, kernel=(kernel_size, kernel_size), pad=(padding, padding)
    )
    net = mx.sym.Activation(net, act_type="relu")
    return net


# Net
def get_symbol(num_classes=1000, version="1.0", **kwargs):
    """Get symbol of SqueezeNet

    Parameters
    ----------
    num_classes: int
        The number of classification results

    version : str, optional
        "1.0" or "1.1" of SqueezeNet
    """
    assert version in [
        "1.0",
        "1.1",
    ], "Unsupported SqueezeNet version {version}:" "1.0 or 1.1 expected".format(version=version)
    net = mx.sym.Variable("data")
    if version == "1.0":
        net = mx.sym.Convolution(net, num_filter=96, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
        net = mx.sym.Activation(net, act_type="relu")
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 16, 64, 64)
        net = _make_fire(net, 16, 64, 64)
        net = _make_fire(net, 32, 128, 128)
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 32, 128, 128)
        net = _make_fire(net, 48, 192, 192)
        net = _make_fire(net, 48, 192, 192)
        net = _make_fire(net, 64, 256, 256)
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 64, 256, 256)
    else:
        net = mx.sym.Convolution(net, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
        net = mx.sym.Activation(net, act_type="relu")
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 16, 64, 64)
        net = _make_fire(net, 16, 64, 64)
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 32, 128, 128)
        net = _make_fire(net, 32, 128, 128)
        net = mx.sym.Pooling(data=net, kernel=(3, 3), pool_type="max", stride=(2, 2))
        net = _make_fire(net, 48, 192, 192)
        net = _make_fire(net, 48, 192, 192)
        net = _make_fire(net, 64, 256, 256)
        net = _make_fire(net, 64, 256, 256)
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.Convolution(net, num_filter=num_classes, kernel=(1, 1))
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(data=net, global_pool=True, kernel=(13, 13), pool_type="avg")
    net = mx.sym.flatten(net)
    return mx.sym.softmax(net)
