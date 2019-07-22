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
Port of NNVM version of MobileNet to Relay.
"""
# pylint: disable=invalid-name

from tvm import relay
from . import layers
from .init import create_workload

def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
               padding=(1, 1), epsilon=1e-5):
    """Helper function to construct conv_bn-relu"""
    # convolution + bn + relu
    conv = layers.conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout='NCHW',
        name=name+'_conv')
    bn = layers.batch_norm_infer(data=conv, epsilon=epsilon, name=name + '_bn')
    act = relay.nn.relu(data=bn)
    return act


def separable_conv_block(data, name, depthwise_channels, pointwise_channels,
                         kernel_size=(3, 3), downsample=False, padding=(1, 1),
                         epsilon=1e-5):
    """Helper function to get a separable conv block"""
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)
    # depthwise convolution + bn + relu
    conv1 = layers.conv2d(
        data=data,
        channels=depthwise_channels,
        groups=depthwise_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        name=name+'_depthwise_conv1')
    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, name=name+'_bn1')
    act1 = relay.nn.relu(data=bn1)
    # pointwise convolution + bn + relu
    conv2 = layers.conv2d(
        data=act1,
        channels=pointwise_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout='NCHW',
        name=name + '_conv2')
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, name=name+'_bn2')
    act2 = relay.nn.relu(data=bn2)
    return act2


def mobile_net(num_classes=1000, data_shape=(1, 3, 224, 224),
               dtype='float32', alpha=1.0, is_shallow=False):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    body = conv_block(data, 'conv_block_1', int(32*alpha), strides=(2, 2))
    body = separable_conv_block(body, 'separable_conv_block_1',
                                int(32*alpha), int(64*alpha))
    body = separable_conv_block(body, 'separable_conv_block_2',
                                int(64*alpha), int(128*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_3',
                                int(128*alpha), int(128*alpha))
    body = separable_conv_block(body, 'separable_conv_block_4',
                                int(128*alpha), int(256*alpha), downsample=True)
    body = separable_conv_block(body, 'separable_conv_block_5',
                                int(256*alpha), int(256*alpha))
    body = separable_conv_block(body, 'separable_conv_block_6',
                                int(256*alpha), int(512*alpha), downsample=True)
    if is_shallow:
        body = separable_conv_block(body, 'separable_conv_block_7',
                                    int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_8',
                                    int(1024*alpha), int(1024*alpha), downsample=True)
    else:
        for i in range(7, 12):
            body = separable_conv_block(body, 'separable_conv_block_%d' % i,
                                        int(512*alpha), int(512*alpha))
        body = separable_conv_block(body, 'separable_conv_block_12',
                                    int(512*alpha), int(1024*alpha), downsample=True)
        body = separable_conv_block(body, 'separable_conv_block_13',
                                    int(1024*alpha), int(1024*alpha))
    pool = relay.nn.global_avg_pool2d(data=body)
    flatten = relay.nn.batch_flatten(data=pool)
    weight = relay.var('fc_weight')
    fc = relay.nn.dense(data=flatten, weight=weight, units=num_classes)
    softmax = relay.nn.softmax(data=fc)
    return relay.Function(relay.analysis.free_vars(softmax), softmax)


def get_workload(batch_size=1, num_classes=1000, image_shape=(3, 224, 224), dtype='float32'):
    """Get benchmark workload for mobilenet

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
    mod : tvm.relay.Module
        The relay module that contains a MobileNet network.

    params : dict of str to NDArray
        The parameters.
    """
    data_shape = tuple([batch_size] + list(image_shape))
    net = mobile_net(num_classes=num_classes, data_shape=data_shape,
                     dtype=dtype, alpha=1.0, is_shallow=False)
    return create_workload(net)
