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

topi
----
.. automodule:: topi

List of operators
~~~~~~~~~~~~~~~~~

.. autosummary::

   topi.identity
   topi.negative
   topi.floor
   topi.ceil
   topi.sign
   topi.trunc
   topi.round
   topi.abs
   topi.isnan
   topi.isfinite
   topi.isinf
   topi.exp
   topi.tanh
   topi.log
   topi.sqrt
   topi.rsqrt
   topi.sigmoid
   topi.clip
   topi.cast
   topi.reinterpret
   topi.transpose
   topi.flip
   topi.strided_slice
   topi.expand_dims
   topi.reshape
   topi.unravel_index
   topi.squeeze
   topi.concatenate
   topi.split
   topi.take
   topi.gather_nd
   topi.full
   topi.full_like
   topi.nn.relu
   topi.nn.leaky_relu
   topi.nn.dilate
   topi.nn.pool
   topi.nn.global_pool
   topi.nn.adaptive_pool
   topi.nn.upsampling
   topi.nn.softmax
   topi.nn.dense
   topi.nn.batch_matmul
   topi.nn.log_softmax
   topi.nn.conv2d_nchw
   topi.nn.conv2d_hwcn
   topi.nn.depthwise_conv2d_nchw
   topi.nn.depthwise_conv2d_nhwc
   topi.nn.fifo_buffer
   topi.max
   topi.sum
   topi.min
   topi.argmax
   topi.argmin
   topi.prod
   topi.broadcast_to
   topi.add
   topi.subtract
   topi.multiply
   topi.divide
   topi.mod
   topi.maximum
   topi.minimum
   topi.power
   topi.greater
   topi.less
   topi.equal
   topi.not_equal
   topi.greater_equal
   topi.less_equal
   topi.all
   topi.any
   topi.logical_and
   topi.logical_or
   topi.logical_not
   topi.logical_xor
   topi.arange
   topi.stack
   topi.repeat
   topi.tile
   topi.shape
   topi.ndarray_size
   topi.layout_transform
   topi.image.resize
   topi.image.crop_and_resize
   topi.image.dilation2d
   topi.argsort
   topi.topk
   topi.sequence_mask
   topi.one_hot


List of schedules
~~~~~~~~~~~~~~~~~
.. autosummary::

   topi.generic.schedule_conv2d_nchw
   topi.generic.schedule_depthwise_conv2d_nchw
   topi.generic.schedule_reduce
   topi.generic.schedule_broadcast
   topi.generic.schedule_injective

topi
~~~~
.. autofunction:: topi.negative
.. autofunction:: topi.identity
.. autofunction:: topi.floor
.. autofunction:: topi.ceil
.. autofunction:: topi.sign
.. autofunction:: topi.trunc
.. autofunction:: topi.round
.. autofunction:: topi.abs
.. autofunction:: topi.isnan
.. autofunction:: topi.isfinite
.. autofunction:: topi.isinf
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.rsqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.clip
.. autofunction:: topi.cast
.. autofunction:: topi.reinterpret
.. autofunction:: topi.transpose
.. autofunction:: topi.flip
.. autofunction:: topi.strided_slice
.. autofunction:: topi.expand_dims
.. autofunction:: topi.reshape
.. autofunction:: topi.unravel_index
.. autofunction:: topi.squeeze
.. autofunction:: topi.concatenate
.. autofunction:: topi.split
.. autofunction:: topi.take
.. autofunction:: topi.gather_nd
.. autofunction:: topi.full
.. autofunction:: topi.full_like
.. autofunction:: topi.all
.. autofunction:: topi.any
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min
.. autofunction:: topi.prod
.. autofunction:: topi.broadcast_to
.. autofunction:: topi.add
.. autofunction:: topi.subtract
.. autofunction:: topi.multiply
.. autofunction:: topi.divide
.. autofunction:: topi.floor_divide
.. autofunction:: topi.mod
.. autofunction:: topi.floor_mod
.. autofunction:: topi.maximum
.. autofunction:: topi.minimum
.. autofunction:: topi.power
.. autofunction:: topi.greater
.. autofunction:: topi.less
.. autofunction:: topi.arange
.. autofunction:: topi.stack
.. autofunction:: topi.repeat
.. autofunction:: topi.tile
.. autofunction:: topi.shape
.. autofunction:: topi.ndarray_size
.. autofunction:: topi.layout_transform
.. autofunction:: topi.argsort
.. autofunction:: topi.topk
.. autofunction:: topi.sequence_mask
.. autofunction:: topi.one_hot
.. autofunction:: topi.logical_and
.. autofunction:: topi.logical_or
.. autofunction:: topi.logical_not
.. autofunction:: topi.logical_xor

topi.nn
~~~~~~~
.. autofunction:: topi.nn.relu
.. autofunction:: topi.nn.leaky_relu
.. autofunction:: topi.nn.dilate
.. autofunction:: topi.nn.pool
.. autofunction:: topi.nn.global_pool
.. autofunction:: topi.nn.upsampling
.. autofunction:: topi.nn.softmax
.. autofunction:: topi.nn.dense
.. autofunction:: topi.nn.batch_matmul
.. autofunction:: topi.nn.log_softmax
.. autofunction:: topi.nn.conv2d_nchw
.. autofunction:: topi.nn.conv2d_hwcn
.. autofunction:: topi.nn.depthwise_conv2d_nchw
.. autofunction:: topi.nn.depthwise_conv2d_nhwc
.. autofunction:: topi.nn.fifo_buffer

topi.image
~~~~~~~~~~
.. autofunction:: topi.image.resize
.. autofunction:: topi.image.crop_and_resize

topi.sparse
~~~~~~~~~~~
.. autofunction:: topi.sparse.csrmv
.. autofunction:: topi.sparse.csrmm
.. autofunction:: topi.sparse.dense

topi.generic
~~~~~~~~~~~~
.. automodule:: topi.generic

.. autofunction:: topi.generic.schedule_conv2d_nchw
.. autofunction:: topi.generic.schedule_depthwise_conv2d_nchw
.. autofunction:: topi.generic.schedule_reduce
.. autofunction:: topi.generic.schedule_broadcast
.. autofunction:: topi.generic.schedule_injective
