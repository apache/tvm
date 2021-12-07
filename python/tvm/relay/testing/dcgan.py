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
# pylint: disable=unused-argument
"""
Net of the generator of DCGAN

Adopted from:
https://github.com/tqchen/mxnet-gan/blob/main/mxgan/generator.py

Reference:
Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional generative adversarial networks."
arXiv preprint arXiv:1511.06434 (2015).
"""
from tvm import relay

from . import layers
from .init import create_workload


def deconv2d(data, ishape, oshape, kshape, layout, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    target_shape = (oshape[-2], oshape[-1])

    pad_y = (kshape[0] - 1) // 2
    pad_x = (kshape[1] - 1) // 2
    adj_y = (target_shape[0] + 2 * pad_y - kshape[0]) % stride[0]
    adj_x = (target_shape[1] + 2 * pad_x - kshape[1]) % stride[1]

    if layout == "NCHW":
        kernel_layout = "IOHW"
    elif layout == "NHWC":
        kernel_layout = "HWOI"
    else:
        raise ValueError("Invalid layout: " + layout)

    net = layers.conv2d_transpose(
        data,
        kernel_size=kshape,
        strides=stride,
        channels=oshape[0],
        padding=(pad_y, pad_x),
        output_padding=(adj_y, adj_x),
        data_layout=layout,
        kernel_layout=kernel_layout,
        name=name,
    )
    return net


def deconv2d_bn_relu(data, prefix, **kwargs):
    """a block of deconv + batch norm + relu"""
    eps = 1e-5 + 1e-12
    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    bn_axis = kwargs.get("layout", "NCHW").index("C")
    net = layers.batch_norm_infer(
        net, epsilon=eps, scale=False, axis=bn_axis, name="%s_batch_norm" % prefix
    )
    net = relay.nn.relu(net)
    return net


def get_net(
    batch_size,
    random_len=100,
    oshape=(3, 64, 64),
    ngf=128,
    code=None,
    layout="NCHW",
    dtype="float32",
):
    """get net of dcgan generator"""
    assert oshape[-1] == 64, "Only support 64x64 image"
    assert oshape[-2] == 64, "Only support 64x64 image"

    code = relay.var("data", dtype=dtype, shape=(batch_size, random_len)) if code is None else code
    dense_weight = relay.var("dense_weight")
    dense = relay.nn.dense(code, weight=dense_weight, units=4 * 4 * ngf * 8)
    relu = relay.nn.relu(dense)
    # 4 x 4
    if layout == "NCHW":
        reshape = relay.reshape(relu, newshape=(-1, ngf * 8, 4, 4))
    elif layout == "NHWC":
        reshape = relay.reshape(relu, newshape=(-1, 4, 4, ngf * 8))
    else:
        raise ValueError("Invalid layout: " + layout)
    # 8 x 8
    dc8 = deconv2d_bn_relu(
        reshape,
        ishape=(ngf * 8, 4, 4),
        oshape=(ngf * 4, 8, 8),
        kshape=(4, 4),
        layout=layout,
        prefix="g2",
    )
    # 16x16
    dc16 = deconv2d_bn_relu(
        dc8,
        ishape=(ngf * 4, 8, 8),
        oshape=(ngf * 2, 16, 16),
        kshape=(4, 4),
        layout=layout,
        prefix="g3",
    )
    # 32x32
    dc32 = deconv2d_bn_relu(
        dc16,
        ishape=(ngf * 2, 16, 16),
        oshape=(ngf, 32, 32),
        kshape=(4, 4),
        layout=layout,
        prefix="g4",
    )
    # 64x64
    dc64 = deconv2d(
        dc32,
        ishape=(ngf, 32, 32),
        oshape=oshape[-3:],
        kshape=(4, 4),
        layout=layout,
        name="g5_deconv",
    )
    tanh = relay.tanh(dc64)

    args = relay.analysis.free_vars(tanh)
    return relay.Function(args, tanh)


def get_workload(
    batch_size, oshape=(3, 64, 64), ngf=128, random_len=100, layout="NCHW", dtype="float32"
):
    """Get benchmark workload for a DCGAN generator

    Parameters
    ----------
    batch_size : int
        The batch size used in the model
    oshape : tuple, optional
        The shape of output image, layout="CHW"
    ngf: int, optional
        The number of final feature maps in the generator
    random_len : int, optional
        The length of random input
    layout: str, optional
        The layout of conv2d transpose
    dtype : str, optional
        The data type

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a DCGAN network.
    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, random_len, oshape=oshape, ngf=ngf, layout=layout, dtype=dtype)
    return create_workload(net)
