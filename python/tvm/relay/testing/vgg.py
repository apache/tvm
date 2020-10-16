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

"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
from tvm import relay
from .init import create_workload
from . import layers as wrapper


def get_feature(internal_layer, layers, filters, batch_norm=False):
    """Get VGG feature body as stacks of convolutions."""
    for i, num in enumerate(layers):
        for j in range(num):
            internal_layer = wrapper.conv2d(
                data=internal_layer,
                kernel_size=(3, 3),
                padding=(1, 1),
                channels=filters[i],
                name="conv%s_%s" % (i + 1, j + 1),
            )
            internal_layer = relay.nn.bias_add(
                internal_layer, relay.var("conv%s_%s_bias" % (i + 1, j + 1))
            )
            if batch_norm:
                internal_layer = wrapper.batch_norm_infer(
                    data=internal_layer, name="bn%s_%s" % (i + 1, j + 1)
                )
            internal_layer = relay.nn.relu(data=internal_layer)
        internal_layer = relay.nn.max_pool2d(data=internal_layer, pool_size=(2, 2), strides=(2, 2))
    return internal_layer


def get_classifier(input_data, num_classes):
    """Get VGG classifier layers as fc layers."""
    flatten = relay.nn.batch_flatten(data=input_data)
    fc6 = wrapper.dense_add_bias(data=flatten, units=4096, name="fc6")
    relu6 = relay.nn.relu(data=fc6)
    drop6 = relay.nn.dropout(data=relu6, rate=0.5)
    fc7 = wrapper.dense_add_bias(data=drop6, units=4096, name="fc7")
    relu7 = relay.nn.relu(data=fc7)
    drop7 = relay.nn.dropout(data=relu7, rate=0.5)
    fc8 = wrapper.dense_add_bias(data=drop7, units=num_classes, name="fc8")
    return fc8


def get_net(batch_size, image_shape, num_classes, dtype, num_layers=11, batch_norm=False):
    """
    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    image_shape : tuple, optional
        The input image shape

    num_classes : int, optional
        Number of claseses

    dtype : str, optional
        The data type

    num_layers : int
        Number of layers for the variant of vgg. Options are 11, 13, 16, 19.

    batch_norm : bool, default False
        Use batch normalization.
    """
    vgg_spec = {
        11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
        13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
        16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
        19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]),
    }
    if num_layers not in vgg_spec:
        raise ValueError("Invalide num_layers {}. Choices are 11,13,16,19.".format(num_layers))
    layers, filters = vgg_spec[num_layers]
    data_shape = (batch_size,) + image_shape
    data = relay.var("data", shape=data_shape, dtype=dtype)
    feature = get_feature(data, layers, filters, batch_norm)
    classifier = get_classifier(feature, num_classes)
    symbol = relay.nn.softmax(data=classifier)
    args = relay.analysis.free_vars(symbol)
    return relay.Function(args, symbol)


def get_workload(
    batch_size,
    num_classes=1000,
    image_shape=(3, 224, 224),
    dtype="float32",
    num_layers=11,
    batch_norm=False,
):
    """Get benchmark workload for VGG nets.

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

    num_layers : int
        Number of layers for the variant of vgg. Options are 11, 13, 16, 19.

    batch_norm : bool
        Use batch normalization.

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a VGG network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, image_shape, num_classes, dtype, num_layers, batch_norm)
    return create_workload(net)
