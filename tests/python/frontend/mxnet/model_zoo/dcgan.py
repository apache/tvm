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
The MXNet symbol of DCGAN generator

Adopted from:
https://github.com/tqchen/mxnet-gan/blob/main/mxgan/generator.py

Reference:
Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional generative adversarial networks."
arXiv preprint arXiv:1511.06434 (2015).
"""

import mxnet as mx


def deconv2d(data, ishape, oshape, kshape, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    target_shape = (oshape[-2], oshape[-1])
    pad_y = (kshape[0] - 1) // 2
    pad_x = (kshape[1] - 1) // 2
    adj_y = (target_shape[0] + 2 * pad_y - kshape[0]) % stride[0]
    adj_x = (target_shape[1] + 2 * pad_x - kshape[1]) % stride[1]

    net = mx.sym.Deconvolution(
        data,
        kernel=kshape,
        stride=stride,
        pad=(pad_y, pad_x),
        adj=(adj_y, adj_x),
        num_filter=oshape[0],
        no_bias=True,
        name=name,
    )
    return net


def deconv2d_bn_relu(data, prefix, **kwargs):
    """a block of deconv + batch norm + relu"""
    eps = 1e-5 + 1e-12

    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    net = mx.sym.BatchNorm(net, eps=eps, name="%s_bn" % prefix)
    net = mx.sym.Activation(net, name="%s_act" % prefix, act_type="relu")
    return net


def get_symbol(oshape=(3, 64, 64), ngf=128, code=None):
    """get symbol of dcgan generator"""
    assert oshape[-1] == 64, "Only support 64x64 image"
    assert oshape[-2] == 64, "Only support 64x64 image"

    code = mx.sym.Variable("data") if code is None else code
    net = mx.sym.FullyConnected(
        code, name="g1", num_hidden=ngf * 8 * 4 * 4, no_bias=True, flatten=False
    )
    net = mx.sym.Activation(net, act_type="relu")
    # 4 x 4
    net = mx.sym.reshape(net, shape=(-1, ngf * 8, 4, 4))
    # 8 x 8
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 8, 4, 4), oshape=(ngf * 4, 8, 8), kshape=(4, 4), prefix="g2"
    )
    # 16x16
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 4, 8, 8), oshape=(ngf * 2, 16, 16), kshape=(4, 4), prefix="g3"
    )
    # 32x32
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 2, 16, 16), oshape=(ngf, 32, 32), kshape=(4, 4), prefix="g4"
    )
    # 64x64
    net = deconv2d(net, ishape=(ngf, 32, 32), oshape=oshape[-3:], kshape=(4, 4), name="g5_deconv")
    net = mx.sym.Activation(net, act_type="tanh")
    return net
