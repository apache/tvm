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
   tvm.relay.nn.avg_pool2d
   tvm.relay.nn.global_max_pool2d
   tvm.relay.nn.global_avg_pool2d
   tvm.relay.nn.upsampling
   tvm.relay.nn.batch_flatten
   tvm.relay.nn.pad
   tvm.relay.nn.lrn
   tvm.relay.nn.l2_normalize
   tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
   tvm.relay.nn.contrib_conv2d_winograd_weight_transform


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
   tvm.relay.split
   tvm.relay.arange
   tvm.relay.stack
   tvm.relay.repeat
   tvm.relay.tile
   tvm.relay.reverse


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
   tvm.relay.logical_and
   tvm.relay.logical_or
   tvm.relay.logical_not
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
   tvm.relay.prod
   tvm.relay.strided_slice
   tvm.relay.broadcast_to


**Level 5: Vision/Image Operators**

.. autosummary::
   :nosignatures:

   tvm.relay.image.resize
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
   tvm.relay.layout_transform
   tvm.relay.device_copy
   tvm.relay.annotation.on_device
   tvm.relay.reverse_reshape
   tvm.relay.sequence_mask
   tvm.relay.nn.batch_matmul
   tvm.relay.contrib.adaptive_max_pool2d
   tvm.relay.contrib.adaptive_avg_pool2d


Level 1 Definitions
-------------------
.. autofunction:: tvm.relay.log
.. autofunction:: tvm.relay.sqrt
.. autofunction:: tvm.relay.rsqrt
.. autofunction:: tvm.relay.exp
.. autofunction:: tvm.relay.sigmoid
.. autofunction:: tvm.relay.add
.. autofunction:: tvm.relay.subtract
.. autofunction:: tvm.relay.multiply
.. autofunction:: tvm.relay.divide
.. autofunction:: tvm.relay.mod
.. autofunction:: tvm.relay.tanh
.. autofunction:: tvm.relay.concatenate
.. autofunction:: tvm.relay.expand_dims
.. autofunction:: tvm.relay.nn.softmax
.. autofunction:: tvm.relay.nn.log_softmax
.. autofunction:: tvm.relay.nn.relu
.. autofunction:: tvm.relay.nn.dropout
.. autofunction:: tvm.relay.nn.batch_norm
.. autofunction:: tvm.relay.nn.bias_add


Level 2 Definitions
-------------------
.. autofunction:: tvm.relay.nn.conv2d
.. autofunction:: tvm.relay.nn.conv2d_transpose
.. autofunction:: tvm.relay.nn.dense
.. autofunction:: tvm.relay.nn.max_pool2d
.. autofunction:: tvm.relay.nn.avg_pool2d
.. autofunction:: tvm.relay.nn.global_max_pool2d
.. autofunction:: tvm.relay.nn.global_avg_pool2d
.. autofunction:: tvm.relay.nn.upsampling
.. autofunction:: tvm.relay.nn.batch_flatten
.. autofunction:: tvm.relay.nn.pad
.. autofunction:: tvm.relay.nn.lrn
.. autofunction:: tvm.relay.nn.l2_normalize
.. autofunction:: tvm.relay.nn.contrib_conv2d_winograd_without_weight_transform
.. autofunction:: tvm.relay.nn.contrib_conv2d_winograd_weight_transform

Level 3 Definitions
-------------------
.. autofunction:: tvm.relay.nn.leaky_relu
.. autofunction:: tvm.relay.nn.prelu
.. autofunction:: tvm.relay.reshape
.. autofunction:: tvm.relay.reshape_like
.. autofunction:: tvm.relay.copy
.. autofunction:: tvm.relay.transpose
.. autofunction:: tvm.relay.squeeze
.. autofunction:: tvm.relay.floor
.. autofunction:: tvm.relay.ceil
.. autofunction:: tvm.relay.sign
.. autofunction:: tvm.relay.trunc
.. autofunction:: tvm.relay.clip
.. autofunction:: tvm.relay.round
.. autofunction:: tvm.relay.abs
.. autofunction:: tvm.relay.negative
.. autofunction:: tvm.relay.take
.. autofunction:: tvm.relay.zeros
.. autofunction:: tvm.relay.zeros_like
.. autofunction:: tvm.relay.ones
.. autofunction:: tvm.relay.ones_like
.. autofunction:: tvm.relay.gather_nd
.. autofunction:: tvm.relay.full
.. autofunction:: tvm.relay.full_like
.. autofunction:: tvm.relay.cast
.. autofunction:: tvm.relay.split
.. autofunction:: tvm.relay.arange
.. autofunction:: tvm.relay.stack
.. autofunction:: tvm.relay.repeat
.. autofunction:: tvm.relay.tile
.. autofunction:: tvm.relay.reverse


Level 4 Definitions
-------------------
.. autofunction:: tvm.relay.right_shift
.. autofunction:: tvm.relay.left_shift
.. autofunction:: tvm.relay.equal
.. autofunction:: tvm.relay.not_equal
.. autofunction:: tvm.relay.greater
.. autofunction:: tvm.relay.greater_equal
.. autofunction:: tvm.relay.less
.. autofunction:: tvm.relay.less_equal
.. autofunction:: tvm.relay.all
.. autofunction:: tvm.relay.logical_and
.. autofunction:: tvm.relay.logical_or
.. autofunction:: tvm.relay.logical_not
.. autofunction:: tvm.relay.maximum
.. autofunction:: tvm.relay.minimum
.. autofunction:: tvm.relay.power
.. autofunction:: tvm.relay.where
.. autofunction:: tvm.relay.argmax
.. autofunction:: tvm.relay.argmin
.. autofunction:: tvm.relay.sum
.. autofunction:: tvm.relay.max
.. autofunction:: tvm.relay.min
.. autofunction:: tvm.relay.mean
.. autofunction:: tvm.relay.prod
.. autofunction:: tvm.relay.strided_slice
.. autofunction:: tvm.relay.broadcast_to


Level 5 Definitions
-------------------
.. autofunction:: tvm.relay.image.resize
.. autofunction:: tvm.relay.vision.multibox_prior
.. autofunction:: tvm.relay.vision.multibox_transform_loc
.. autofunction:: tvm.relay.vision.nms
.. autofunction:: tvm.relay.vision.yolo_reorg


Level 6 Definitions
-------------------
.. autofunction:: tvm.relay.argsort
.. autofunction:: tvm.relay.topk


Level 10 Definitions
--------------------
.. autofunction:: tvm.relay.broadcast_to_like
.. autofunction:: tvm.relay.collapse_sum_like
.. autofunction:: tvm.relay.slice_like
.. autofunction:: tvm.relay.shape_of
.. autofunction:: tvm.relay.layout_transform
.. autofunction:: tvm.relay.device_copy
.. autofunction:: tvm.relay.annotation.on_device
.. autofunction:: tvm.relay.reverse_reshape
.. autofunction:: tvm.relay.sequence_mask
.. autofunction:: tvm.relay.nn.batch_matmul
.. autofunction:: tvm.relay.contrib.adaptive_max_pool2d
.. autofunction:: tvm.relay.contrib.adaptive_avg_pool2d
