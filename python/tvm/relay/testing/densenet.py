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

# pylint: disable=invalid-name, line-too-long
"""
Port of MxNet version of Densenet to Relay.
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/densenet.py
"""
# pylint: enable=line-too-long
from tvm import relay
from . import layers
from .init import create_workload


def _make_dense_layer(data, growth_rate, bn_size, index):
    """Single densenet layer."""
    bn1 = layers.batch_norm_infer(data, name="batch_1_%s" % index)
    relu1 = relay.nn.relu(bn1)
    conv1 = layers.conv2d(
        relu1, channels=bn_size * growth_rate, kernel_size=(1, 1), name="conv2d_1_%s" % index
    )
    bn2 = layers.batch_norm_infer(conv1, name="batch_2_" + index)
    relu2 = relay.nn.relu(bn2)
    conv2 = layers.conv2d(
        relu2, channels=growth_rate, kernel_size=(3, 3), padding=(1, 1), name="conv2d_2_%s" % index
    )
    return conv2


def _make_dense_block(data, num_layers, bn_size, growth_rate, index):
    """Makes a block of dense layers of the specified size."""
    layer_out = data
    blocks = []
    for i in range(num_layers):
        layer_out = _make_dense_layer(layer_out, growth_rate, bn_size, "%s_%s" % (index, i))
        blocks.append(layer_out)
    block_out = relay.concatenate(blocks, 1)
    return block_out


def _make_transition(data, num_output_features, index):
    """Transition between layers."""
    bn = layers.batch_norm_infer(data, name="batch_t_%s" % index)
    relu = relay.nn.relu(bn)
    conv = layers.conv2d(
        relu, channels=num_output_features, kernel_size=(1, 1), name="conv_t_%s" % index
    )
    return relay.nn.avg_pool2d(conv, pool_size=(2, 2), strides=(2, 2))


def _make_dense_net(
    num_init_features, growth_rate, block_config, data_shape, data_dtype, bn_size=4, classes=1000
):
    """Builds up a densenet."""
    data = relay.Var(
        "data", relay.TensorType(data_shape, data_dtype)
    )  # (batch_size, 3, 224, 224)))
    conv1 = layers.conv2d(
        data,
        channels=num_init_features,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=(3, 3),
        name="conv1",
    )
    bn1 = layers.batch_norm_infer(conv1, name="batch1")
    relu1 = relay.nn.relu(bn1)
    mp = relay.nn.max_pool2d(relu1, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))

    num_features = num_init_features
    layer_out = mp
    for i, num_layers in enumerate(block_config):
        layer_out = _make_dense_block(layer_out, num_layers, bn_size, growth_rate, i)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            layer_out = _make_transition(layer_out, num_features // 2, i)
            num_features = num_features // 2
    bn2 = layers.batch_norm_infer(layer_out, name="batch2")
    relu2 = relay.nn.relu(bn2)
    avg = relay.nn.avg_pool2d(relu2, pool_size=(7, 7))
    flat = relay.nn.batch_flatten(avg)

    ret = layers.dense_add_bias(flat, units=classes, name="dense")

    return relay.Function(relay.analysis.free_vars(ret), ret)


def get_workload(
    densenet_size=121, classes=1000, batch_size=4, image_shape=(3, 224, 224), dtype="float32"
):
    """Gets benchmark workload for densenet.

    Parameters
    ----------
    densenet_size : int, optional (default 121)
        Parameter for the network size. The supported sizes
        are 121, 161, 169, and 201.

    classes : int, optional (default 1000)
        The number of classes.

    batch_size : int, optional (detault 4)
        The batch size for the network.

    image_shape : shape, optional (default (3, 224, 224))
        The shape of the input data.

    dtype : data type, optional (default 'float32')
        The data type of the input data.

    Returns
    -------
    mod: tvm.IRModule
        The relay module that contains a DenseNet network.

    params : dict of str to NDArray
        The benchmark paraeters.
    """
    specs = {
        121: (64, 32, [6, 12, 24, 16]),
        161: (96, 48, [6, 12, 36, 24]),
        169: (69, 32, [6, 12, 32, 32]),
        201: (64, 32, [6, 12, 48, 32]),
    }
    bn_size = 4
    num_init_features, growth_rate, block_config = specs[densenet_size]
    data_shape = tuple([batch_size] + list(image_shape))
    net = _make_dense_net(
        num_init_features, growth_rate, block_config, data_shape, dtype, bn_size, classes
    )
    return create_workload(net)
