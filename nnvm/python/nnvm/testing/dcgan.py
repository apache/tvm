# pylint: disable=unused-argument
"""
Symbol of the generator of DCGAN

Adopted from:
https://github.com/tqchen/mxnet-gan/blob/master/mxgan/generator.py

Reference:
Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional generative adversarial networks."
arXiv preprint arXiv:1511.06434 (2015).
"""
from .. import symbol as sym
from . utils import create_workload

def deconv2d(data, ishape, oshape, kshape, name, stride=(2, 2)):
    """a deconv layer that enlarges the feature map"""
    target_shape = (oshape[-2], oshape[-1])

    pad_y = (kshape[0] - 1) // 2
    pad_x = (kshape[1] - 1) // 2
    adj_y = (target_shape[0] + 2 * pad_y - kshape[0]) % stride[0]
    adj_x = (target_shape[1] + 2 * pad_x - kshape[1]) % stride[1]

    net = sym.conv2d_transpose(data,
                               kernel_size=kshape,
                               strides=stride,
                               channels=oshape[0],
                               padding=(pad_y, pad_x),
                               output_padding=(adj_y, adj_x),
                               use_bias=False,
                               name=name)
    return net

def deconv2d_bn_relu(data, prefix, **kwargs):
    """a block of deconv + batch norm + relu"""
    eps = 1e-5 + 1e-12
    net = deconv2d(data, name="%s_deconv" % prefix, **kwargs)
    net = sym.batch_norm(net, epsilon=eps, name="%s_bn" % prefix)
    net = sym.relu(net, name="%s_act" % prefix)
    return net

def get_symbol(oshape, ngf=128, code=None):
    """get symbol of dcgan generator"""
    assert oshape[-1] == 64, "Only support 64x64 image"
    assert oshape[-2] == 64, "Only support 64x64 image"

    code = sym.Variable("data") if code is None else code
    net = sym.dense(code, name="g1", units=4*4*ngf*8, use_bias=False)
    net = sym.relu(net)
    # 4 x 4
    net = sym.reshape(net, shape=(-1, ngf * 8, 4, 4))
    # 8 x 8
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 8, 4, 4), oshape=(ngf * 4, 8, 8), kshape=(4, 4), prefix="g2")
    # 16x16
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 4, 8, 8), oshape=(ngf * 2, 16, 16), kshape=(4, 4), prefix="g3")
    # 32x32
    net = deconv2d_bn_relu(
        net, ishape=(ngf * 2, 16, 16), oshape=(ngf, 32, 32), kshape=(4, 4), prefix="g4")
    # 64x64
    net = deconv2d(
        net, ishape=(ngf, 32, 32), oshape=oshape[-3:], kshape=(4, 4), name="g5_deconv")
    net = sym.tanh(net)
    return net


def get_workload(batch_size, oshape=(3, 64, 64), ngf=128, random_len=100, dtype="float32"):
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
    dtype : str, optional
        The data type

    Returns
    -------
    net : nnvm.symbol
        The computational graph
    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol(oshape=oshape, ngf=ngf)
    return create_workload(net, batch_size, (random_len, ), dtype)
