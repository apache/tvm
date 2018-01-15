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

'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
# pylint: disable=unused-argument
from .. import symbol as sym
from . utils import create_workload

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same,
        otherwise means differ
    name : str
        Base name of the operators
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes,
        # a bit difference with origin paper
        bn1 = sym.batch_norm(data=data, epsilon=2e-5, name=name + '_bn1')
        act1 = sym.relu(data=bn1, name=name + '_relu1')
        conv1 = sym.conv2d(
            data=act1, channels=int(num_filter*0.25), kernel_size=(1, 1),
            strides=(1, 1), padding=(0, 0), use_bias=False, name=name + '_conv1')
        bn2 = sym.batch_norm(data=conv1, epsilon=2e-5, name=name + '_bn2')
        act2 = sym.relu(data=bn2, name=name + '_relu2')
        conv2 = sym.conv2d(
            data=act2, channels=int(num_filter*0.25), kernel_size=(3, 3),
            strides=stride, padding=(1, 1), use_bias=False, name=name + '_conv2')
        bn3 = sym.batch_norm(data=conv2, epsilon=2e-5, name=name + '_bn3')
        act3 = sym.relu(data=bn3, name=name + '_relu3')
        conv3 = sym.conv2d(
            data=act3, channels=num_filter, kernel_size=(1, 1),
            strides=(1, 1), padding=(0, 0), use_bias=False, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = sym.conv2d(
                data=act1, channels=num_filter, kernel_size=(1, 1),
                strides=stride, use_bias=False, name=name+'_sc')
        return sym.elemwise_add(conv3, shortcut)
    else:
        bn1 = sym.batch_norm(data=data, epsilon=2e-5, name=name + '_bn1')
        act1 = sym.relu(data=bn1, name=name + '_relu1')
        conv1 = sym.conv2d(
            data=act1, channels=num_filter, kernel_size=(3, 3),
            strides=stride, padding=(1, 1), use_bias=False, name=name + '_conv1')
        bn2 = sym.batch_norm(data=conv1, epsilon=2e-5, name=name + '_bn2')
        act2 = sym.relu(data=bn2, name=name + '_relu2')
        conv2 = sym.conv2d(
            data=act2, channels=num_filter, kernel_size=(3, 3),
            strides=(1, 1), padding=(1, 1), use_bias=False, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = sym.conv2d(
                data=act1, channels=num_filter, kernel_size=(1, 1),
                strides=stride, use_bias=False, name=name+'_sc')
        return sym.elemwise_add(conv2, shortcut)

def resnet(units, num_stages, filter_list, num_classes, image_shape,
           bottle_neck=True):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    """
    num_unit = len(units)
    assert num_unit == num_stages
    data = sym.Variable(name='data')
    data = sym.batch_norm(data=data, epsilon=2e-5, name='bn_data')
    (_, height, _) = image_shape
    if height <= 32:            # such as cifar10
        body = sym.conv2d(
            data=data, channels=filter_list[0], kernel_size=(3, 3),
            strides=(1, 1), padding=(1, 1), use_bias=False, name="conv0")
    else:                       # often expected to be 224 such as imagenet
        body = sym.conv2d(
            data=data, channels=filter_list[0], kernel_size=(7, 7),
            strides=(2, 2), padding=(3, 3), use_bias=False, name="conv0")
        body = sym.batch_norm(data=body, epsilon=2e-5, name='bn0')
        body = sym.relu(data=body, name='relu0')
        body = sym.max_pool2d(data=body, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))

    for i in range(num_stages):
        body = residual_unit(
            body, filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2),
            False, name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck)
        for j in range(units[i]-1):
            body = residual_unit(
                body, filter_list[i+1], (1, 1), True,
                name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck)
    bn1 = sym.batch_norm(data=body, epsilon=2e-5, name='bn1')
    relu1 = sym.relu(data=bn1, name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = sym.global_avg_pool2d(data=relu1, name='pool1')
    flat = sym.flatten(data=pool1)
    fc1 = sym.dense(data=flat, units=num_classes, name='fc1')
    return sym.softmax(data=fc1, name='softmax')

def get_symbol(num_classes, num_layers=50, image_shape=(3, 224, 224), **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (_, height, _) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}".format(num_layers))

    return resnet(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=num_classes,
                  image_shape=image_shape,
                  bottle_neck=bottle_neck)

def get_workload(batch_size=1, num_classes=1000, num_layers=18,
                 image_shape=(3, 224, 224), dtype="float32", **kwargs):
    """Get benchmark workload for resnet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    num_layers : int, optional
        Number of layers

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    kwargs : dict
        Extra arguments

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol(num_classes=num_classes, num_layers=num_layers,
                     image_shape=image_shape, **kwargs)
    return create_workload(net, batch_size, image_shape, dtype)
