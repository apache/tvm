..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Relay Core Tensor Operators
===========================

This page contains the list of core tensor operator primitives pre-defined in tvm.relay.
The core tensor operator primitives cover typical workloads in deep learning.
They can represent workloads in front-end frameworks and provide basic building blocks for optimization.
Since deep learning is a fast evolving field, it is possible to have operators that are not in here.


.. note::

   This document will directly list the function signature of
   these operators in the python frontend.


Overview of Operators
---------------------
**Level 1: Basic Operators**

This level enables fully connected multi-layer perceptron.

.. autosummary::
   :nosignatures:

   tvm.relay.log
   tvm.relay.sqrt
   tvm.relay.rsqrt
   tvm.relay.exp
   tvm.relay.sigmoid
   tvm.relay.add
   tvm.relay.subtract
   tvm.relay.multiply
   tvm.relay.divide
   tvm.relay.mod
   tvm.relay.tanh
   tvm.relay.concatenate
   tvm.relay.expand_dims
   tvm.relay.nn.softmax
   tvm.relay.nn.log_softmax
   tvm.relay.nn.relu
   tvm.relay.nn.dropout
   tvm.relay.nn.batch_norm
   tvm.relay.nn.bias_add


**Level 2: Convolutions**

This level enables typical convnet models.

.. autosummary::
   :nosignatures:

   tvm.relay.nn.conv2d
   tvm.relay.nn.conv2d_transpose
   tvm.relay.nn.dense
   tvm.relay.nn.max_pool2d
   tvm.relay.nn.max_pool3d
   tvm.relay.nn.avg_pool2d
   tvm.relay.nn.avg_pool3d
   tvm.relay.nn.global_max_pool2d
   tvm.relay.nn.global_avg_pool2d
   tvm.relay.nn.upsampling
   tvm.relay.nn.upsampling3d
   tvm.relay.nn.batch_flatten
   tvm.relay.nn.pad
   tvm.relay.nn.lrn
   tvm.relay.nn.l2_normalize
   tvm.relay.nn.bitpack
   tvm.relay.nn.bitserial_dense
   tvm.relay.nn.bitserial_conv2d
   tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
   tvm.relay.nn.contrib_conv2d_winograd_weight_transform
   tvm.relay.nn.contrib_conv3d_winograd_without_weight_transform
   tvm.relay.nn.contrib_conv3d_winograd_weight_transform


**Level 3: Additional Math And Transform Operators**

This level enables additional math and transform operators.

.. autosummary::
   :nosignatures:

   tvm.relay.nn.leaky_relu
   tvm.relay.nn.prelu
   tvm.relay.reshape
   tvm.relay.reshape_like
   tvm.relay.copy
   tvm.relay.transpose
   tvm.relay.squeeze
   tvm.relay.floor
   tvm.relay.ceil
   tvm.relay.sign
   tvm.relay.trunc
   tvm.relay.clip
   tvm.relay.round
   tvm.relay.abs
   tvm.relay.negative
   tvm.relay.take
   tvm.relay.zeros
   tvm.relay.zeros_like
   tvm.relay.ones
   tvm.relay.ones_like
   tvm.relay.gather_nd
   tvm.relay.full
   tvm.relay.full_like
   tvm.relay.cast
   tvm.relay.reinterpret
   tvm.relay.split
   tvm.relay.arange
   tvm.relay.stack
   tvm.relay.repeat
   tvm.relay.tile
   tvm.relay.reverse
   tvm.relay.unravel_index


**Level 4: Broadcast and Reductions**

.. autosummary::
   :nosignatures:

   tvm.relay.right_shift
   tvm.relay.left_shift
   tvm.relay.equal
   tvm.relay.not_equal
   tvm.relay.greater
   tvm.relay.greater_equal
   tvm.relay.less
   tvm.relay.less_equal
   tvm.relay.all
   tvm.relay.any
   tvm.relay.logical_and
   tvm.relay.logical_or
   tvm.relay.logical_not
   tvm.relay.logical_xor
   tvm.relay.maximum
   tvm.relay.minimum
   tvm.relay.power
   tvm.relay.where
   tvm.relay.argmax
   tvm.relay.argmin
   tvm.relay.sum
   tvm.relay.max
   tvm.relay.min
   tvm.relay.mean
   tvm.relay.variance
   tvm.relay.std
   tvm.relay.mean_variance
   tvm.relay.mean_std
   tvm.relay.prod
   tvm.relay.strided_slice
   tvm.relay.broadcast_to


**Level 5: Vision/Image Operators**

.. autosummary::
   :nosignatures:

   tvm.relay.image.resize
   tvm.relay.image.crop_and_resize
   tvm.relay.image.dilation2d
   tvm.relay.vision.multibox_prior
   tvm.relay.vision.multibox_transform_loc
   tvm.relay.vision.nms
   tvm.relay.vision.yolo_reorg


**Level 6: Algorithm Operators**

.. autosummary::
   :nosignatures:

   tvm.relay.argsort
   tvm.relay.topk


**Level 10: Temporary Operators**

This level support backpropagation of broadcast operators. It is temporary.

.. autosummary::
   :nosignatures:

   tvm.relay.broadcast_to_like
   tvm.relay.collapse_sum_like
   tvm.relay.slice_like
   tvm.relay.shape_of
   tvm.relay.ndarray_size
   tvm.relay.layout_transform
   tvm.relay.device_copy
   tvm.relay.annotation.on_device
   tvm.relay.reverse_reshape
   tvm.relay.sequence_mask
   tvm.relay.nn.batch_matmul
   tvm.relay.nn.adaptive_max_pool2d
   tvm.relay.nn.adaptive_avg_pool2d
   tvm.relay.one_hot


**Level 11: Dialect Operators**

This level supports dialect operators.

.. autosummary::
   :nosignatures:

   tvm.relay.qnn.op.requantize
   tvm.relay.qnn.op.conv2d