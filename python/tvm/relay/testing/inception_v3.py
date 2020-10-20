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
Inception V3, suitable for images with around 299 x 299

Reference:
Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision."
arXiv preprint arXiv:1512.00567 (2015).

Adopted from https://github.com/apache/incubator-mxnet/blob/
             master/example/image-classification/symbols/inception-v3.py
"""
# pylint: disable=invalid-name,missing-docstring,unused-argument
from tvm import relay
from .init import create_workload
from . import layers


def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=""):
    conv = layers.conv2d(
        data=data,
        channels=int(num_filter),
        kernel_size=kernel,
        strides=stride,
        padding=pad,
        name="%s%s_conv1" % (name, suffix),
    )

    bn = layers.batch_norm_infer(
        data=conv, epsilon=2e-5, scale=False, name="%s%s_bn" % (name, suffix)
    )
    act = relay.nn.relu(data=bn)
    return act


def Pooling(data, kernel, stride, pad, pool_type, name):
    if pool_type == "max":
        return relay.nn.max_pool2d(data=data, pool_size=kernel, strides=stride, padding=pad)
    if pool_type == "avg":
        return relay.nn.avg_pool2d(
            data=data, pool_size=kernel, strides=stride, padding=pad, count_include_pad=True
        )
    raise ValueError("Invalid pooling type: " + pool_type)


def Inception7A(
    data, num_1x1, num_3x3_red, num_3x3_1, num_3x3_2, num_5x5_red, num_5x5, pool, proj, name
):
    tower_1x1 = Conv(data, num_1x1, name=("%s_conv" % name))
    tower_5x5 = Conv(data, num_5x5_red, name=("%s_tower" % name), suffix="_conv")
    tower_5x5 = Conv(
        tower_5x5, num_5x5, kernel=(5, 5), pad=(2, 2), name=("%s_tower" % name), suffix="_conv_1"
    )
    tower_3x3 = Conv(data, num_3x3_red, name=("%s_tower_1" % name), suffix="_conv")
    tower_3x3 = Conv(
        tower_3x3,
        num_3x3_1,
        kernel=(3, 3),
        pad=(1, 1),
        name=("%s_tower_1" % name),
        suffix="_conv_1",
    )
    tower_3x3 = Conv(
        tower_3x3,
        num_3x3_2,
        kernel=(3, 3),
        pad=(1, 1),
        name=("%s_tower_1" % name),
        suffix="_conv_2",
    )
    pooling = Pooling(
        data=data,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        pool_type=pool,
        name=("%s_pool_%s_pool" % (pool, name)),
    )

    cproj = Conv(pooling, proj, name=("%s_tower_2" % name), suffix="_conv")
    concat = relay.concatenate((tower_1x1, tower_5x5, tower_3x3, cproj), axis=1)
    return concat


# First Downsample
def Inception7B(data, num_3x3, num_d3x3_red, num_d3x3_1, num_d3x3_2, pool, name):
    tower_3x3 = Conv(
        data, num_3x3, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=("%s_conv" % name)
    )
    tower_d3x3 = Conv(data, num_d3x3_red, name=("%s_tower" % name), suffix="_conv")
    tower_d3x3 = Conv(
        tower_d3x3,
        num_d3x3_1,
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        name=("%s_tower" % name),
        suffix="_conv_1",
    )
    tower_d3x3 = Conv(
        tower_d3x3,
        num_d3x3_2,
        kernel=(3, 3),
        pad=(0, 0),
        stride=(2, 2),
        name=("%s_tower" % name),
        suffix="_conv_2",
    )
    pooling = Pooling(
        data=data,
        kernel=(3, 3),
        stride=(2, 2),
        pad=(0, 0),
        pool_type="max",
        name=("max_pool_%s_pool" % name),
    )
    concat = relay.concatenate((tower_3x3, tower_d3x3, pooling), axis=1)
    return concat


def Inception7C(
    data,
    num_1x1,
    num_d7_red,
    num_d7_1,
    num_d7_2,
    num_q7_red,
    num_q7_1,
    num_q7_2,
    num_q7_3,
    num_q7_4,
    pool,
    proj,
    name,
):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=("%s_conv" % name))
    tower_d7 = Conv(data=data, num_filter=num_d7_red, name=("%s_tower" % name), suffix="_conv")
    tower_d7 = Conv(
        data=tower_d7,
        num_filter=num_d7_1,
        kernel=(1, 7),
        pad=(0, 3),
        name=("%s_tower" % name),
        suffix="_conv_1",
    )
    tower_d7 = Conv(
        data=tower_d7,
        num_filter=num_d7_2,
        kernel=(7, 1),
        pad=(3, 0),
        name=("%s_tower" % name),
        suffix="_conv_2",
    )
    tower_q7 = Conv(data=data, num_filter=num_q7_red, name=("%s_tower_1" % name), suffix="_conv")
    tower_q7 = Conv(
        data=tower_q7,
        num_filter=num_q7_1,
        kernel=(7, 1),
        pad=(3, 0),
        name=("%s_tower_1" % name),
        suffix="_conv_1",
    )
    tower_q7 = Conv(
        data=tower_q7,
        num_filter=num_q7_2,
        kernel=(1, 7),
        pad=(0, 3),
        name=("%s_tower_1" % name),
        suffix="_conv_2",
    )
    tower_q7 = Conv(
        data=tower_q7,
        num_filter=num_q7_3,
        kernel=(7, 1),
        pad=(3, 0),
        name=("%s_tower_1" % name),
        suffix="_conv_3",
    )
    tower_q7 = Conv(
        data=tower_q7,
        num_filter=num_q7_4,
        kernel=(1, 7),
        pad=(0, 3),
        name=("%s_tower_1" % name),
        suffix="_conv_4",
    )
    pooling = Pooling(
        data=data,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        pool_type=pool,
        name=("%s_pool_%s_pool" % (pool, name)),
    )
    cproj = Conv(
        data=pooling, num_filter=proj, kernel=(1, 1), name=("%s_tower_2" % name), suffix="_conv"
    )
    # concat
    concat = relay.concatenate((tower_1x1, tower_d7, tower_q7, cproj), axis=1)
    return concat


def Inception7D(
    data, num_3x3_red, num_3x3, num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3, pool, name
):
    tower_3x3 = Conv(data=data, num_filter=num_3x3_red, name=("%s_tower" % name), suffix="_conv")
    tower_3x3 = Conv(
        data=tower_3x3,
        num_filter=num_3x3,
        kernel=(3, 3),
        pad=(0, 0),
        stride=(2, 2),
        name=("%s_tower" % name),
        suffix="_conv_1",
    )
    tower_d7_3x3 = Conv(
        data=data, num_filter=num_d7_3x3_red, name=("%s_tower_1" % name), suffix="_conv"
    )
    tower_d7_3x3 = Conv(
        data=tower_d7_3x3,
        num_filter=num_d7_1,
        kernel=(1, 7),
        pad=(0, 3),
        name=("%s_tower_1" % name),
        suffix="_conv_1",
    )
    tower_d7_3x3 = Conv(
        data=tower_d7_3x3,
        num_filter=num_d7_2,
        kernel=(7, 1),
        pad=(3, 0),
        name=("%s_tower_1" % name),
        suffix="_conv_2",
    )
    tower_d7_3x3 = Conv(
        data=tower_d7_3x3,
        num_filter=num_d7_3x3,
        kernel=(3, 3),
        stride=(2, 2),
        name=("%s_tower_1" % name),
        suffix="_conv_3",
    )
    pooling = Pooling(
        data=data,
        kernel=(3, 3),
        stride=(2, 2),
        pool_type=pool,
        pad=(0, 0),
        name=("%s_pool_%s_pool" % (pool, name)),
    )
    # concat
    concat = relay.concatenate((tower_3x3, tower_d7_3x3, pooling), axis=1)
    return concat


def Inception7E(
    data,
    num_1x1,
    num_d3_red,
    num_d3_1,
    num_d3_2,
    num_3x3_d3_red,
    num_3x3,
    num_3x3_d3_1,
    num_3x3_d3_2,
    pool,
    proj,
    name,
):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=("%s_conv" % name))
    tower_d3 = Conv(data=data, num_filter=num_d3_red, name=("%s_tower" % name), suffix="_conv")
    tower_d3_a = Conv(
        data=tower_d3,
        num_filter=num_d3_1,
        kernel=(1, 3),
        pad=(0, 1),
        name=("%s_tower" % name),
        suffix="_mixed_conv",
    )
    tower_d3_b = Conv(
        data=tower_d3,
        num_filter=num_d3_2,
        kernel=(3, 1),
        pad=(1, 0),
        name=("%s_tower" % name),
        suffix="_mixed_conv_1",
    )
    tower_3x3_d3 = Conv(
        data=data, num_filter=num_3x3_d3_red, name=("%s_tower_1" % name), suffix="_conv"
    )
    tower_3x3_d3 = Conv(
        data=tower_3x3_d3,
        num_filter=num_3x3,
        kernel=(3, 3),
        pad=(1, 1),
        name=("%s_tower_1" % name),
        suffix="_conv_1",
    )
    tower_3x3_d3_a = Conv(
        data=tower_3x3_d3,
        num_filter=num_3x3_d3_1,
        kernel=(1, 3),
        pad=(0, 1),
        name=("%s_tower_1" % name),
        suffix="_mixed_conv",
    )
    tower_3x3_d3_b = Conv(
        data=tower_3x3_d3,
        num_filter=num_3x3_d3_2,
        kernel=(3, 1),
        pad=(1, 0),
        name=("%s_tower_1" % name),
        suffix="_mixed_conv_1",
    )
    pooling = Pooling(
        data=data,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        pool_type=pool,
        name=("%s_pool_%s_pool" % (pool, name)),
    )
    cproj = Conv(
        data=pooling, num_filter=proj, kernel=(1, 1), name=("%s_tower_2" % name), suffix="_conv"
    )
    # concat
    concat = relay.concatenate(
        (tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj), axis=1
    )
    return concat


def get_net(batch_size, num_classes, image_shape, dtype):
    """Get network a Inception v3 network.

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

    # stage 1
    conv = Conv(data, 32, kernel=(3, 3), stride=(2, 2), name="conv")
    conv_1 = Conv(conv, 32, kernel=(3, 3), name="conv_1")
    conv_2 = Conv(conv_1, 64, kernel=(3, 3), pad=(1, 1), name="conv_2")
    pool = Pooling(
        data=conv_2, kernel=(3, 3), stride=(2, 2), pool_type="max", pad=(0, 0), name="pool"
    )
    # stage 2
    conv_3 = Conv(pool, 80, kernel=(1, 1), name="conv_3")
    conv_4 = Conv(conv_3, 192, kernel=(3, 3), name="conv_4")
    pool1 = Pooling(
        data=conv_4, kernel=(3, 3), stride=(2, 2), pool_type="max", pad=(0, 0), name="pool1"
    )

    # stage 3
    in3a = Inception7A(pool1, 64, 64, 96, 96, 48, 64, "avg", 32, "mixed")

    in3b = Inception7A(in3a, 64, 64, 96, 96, 48, 64, "avg", 64, "mixed_1")
    in3c = Inception7A(in3b, 64, 64, 96, 96, 48, 64, "avg", 64, "mixed_2")
    in3d = Inception7B(in3c, 384, 64, 96, 96, "max", "mixed_3")
    # stage 4
    in4a = Inception7C(in3d, 192, 128, 128, 192, 128, 128, 128, 128, 192, "avg", 192, "mixed_4")
    in4b = Inception7C(in4a, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_5")
    in4c = Inception7C(in4b, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_6")
    in4d = Inception7C(in4c, 192, 192, 192, 192, 192, 192, 192, 192, 192, "avg", 192, "mixed_7")
    in4e = Inception7D(in4d, 192, 320, 192, 192, 192, 192, "max", "mixed_8")
    # stage 5
    in5a = Inception7E(in4e, 320, 384, 384, 384, 448, 384, 384, 384, "avg", 192, "mixed_9")
    in5b = Inception7E(in5a, 320, 384, 384, 384, 448, 384, 384, 384, "max", 192, "mixed_10")

    # pool
    pool = Pooling(
        data=in5b, kernel=(8, 8), stride=(1, 1), pool_type="avg", pad=(0, 0), name="global_pool"
    )

    flatten = relay.nn.batch_flatten(pool)
    fc1 = relay.nn.dense(flatten, relay.var("fc1_weight"), units=num_classes)
    fc1 = relay.nn.bias_add(fc1, relay.var("fc2_bias"), axis=-1)
    inception_v3 = relay.nn.softmax(data=fc1)
    args = relay.analysis.free_vars(inception_v3)
    return relay.Function(args, inception_v3)


def get_workload(batch_size=1, num_classes=1000, image_shape=(3, 299, 299), dtype="float32"):
    """Get benchmark workload for InceptionV3

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains an Inception V3 network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, num_classes, image_shape, dtype)
    return create_workload(net)
