Core Tensor Operators
=====================

This page contains the list of core tensor operator primitives pre-defined in NNVM.
The core tensor operator primitives(``nnvm.top``) covers typical workloads in deep learning.
They can represent workloads in front-end frameworks, and provide basic building blocks for optimization.
Since deep learning is a fast evolving field and it is that possible to have operators that are not in here.
NNVM is designed for this problem and can easily new operators without changing the core library.

.. note::

   Each operator node in the graph IR contains the following two kinds of parameters.

   - inputs: positional list of input tensors
   - attrs: attributes about operator(e.g. kernel_size in conv2d)

   This document lists both inputs and attributes in the parameter field.  You can distinguish them by the marked type. The inputs are of type Tensor, while the rest parameters are attributes.
   To construct the graph with NNVM python API, a user can pass in the input Tensors as positional arguments, and attributes as keyword arguments.


Overview of Operators
---------------------
**Level 1: Basic Operators**

This level enables fully connected multi-layer perceptron.

.. autosummary::
   :nosignatures:

   nnvm.symbol.dense
   nnvm.symbol.relu
   nnvm.symbol.tanh
   nnvm.symbol.sigmoid
   nnvm.symbol.exp
   nnvm.symbol.log
   nnvm.symbol.sqrt
   nnvm.symbol.elemwise_add
   nnvm.symbol.elemwise_sub
   nnvm.symbol.elemwise_mul
   nnvm.symbol.elemwise_div
   nnvm.symbol.elemwise_sum
   nnvm.symbol.flatten
   nnvm.symbol.concatenate
   nnvm.symbol.expand_dims
   nnvm.symbol.squeeze
   nnvm.symbol.split
   nnvm.symbol.dropout
   nnvm.symbol.batch_norm
   nnvm.symbol.softmax
   nnvm.symbol.log_softmax
   nnvm.symbol.pad
   nnvm.symbol.block_grad


**Level 2: Convolutions**

This level enables typical convnet models.

.. autosummary::
   :nosignatures:

   nnvm.symbol.conv2d
   nnvm.symbol.conv2d_transpose
   nnvm.symbol.max_pool2d
   nnvm.symbol.avg_pool2d
   nnvm.symbol.global_max_pool2d
   nnvm.symbol.global_avg_pool2d


**Level 3: Additional Tensor Ops**

.. autosummary::
   :nosignatures:

   nnvm.symbol.reshape
   nnvm.symbol.copy
   nnvm.symbol.negative
   nnvm.symbol.leaky_relu
   nnvm.symbol.__add_scalar__
   nnvm.symbol.__sub_scalar__
   nnvm.symbol.__rsub_scalar__
   nnvm.symbol.__mul_scalar__
   nnvm.symbol.__div_scalar__
   nnvm.symbol.__rdiv_scalar__
   nnvm.symbol.__pow_scalar__
   nnvm.symbol.__rpow_scalar__

**Level 4: Broadcast and Reductions**

.. autosummary::
   :nosignatures:

   nnvm.symbol.transpose
   nnvm.symbol.broadcast_to
   nnvm.symbol.sum
   nnvm.symbol.min
   nnvm.symbol.max
   nnvm.symbol.broadcast_add
   nnvm.symbol.broadcast_sub
   nnvm.symbol.broadcast_mul
   nnvm.symbol.broadcast_div
   nnvm.symbol.clip
   nnvm.symbol.greater
   nnvm.symbol.less
   nnvm.symbol.expand_like
   nnvm.symbol.reshape_like
   nnvm.symbol.full
   nnvm.symbol.full_like
   nnvm.symbol.ones
   nnvm.symbol.ones_like
   nnvm.symbol.zeros
   nnvm.symbol.zeros_like

Detailed Definitions
--------------------
.. autofunction:: nnvm.symbol.dense
.. autofunction:: nnvm.symbol.relu
.. autofunction:: nnvm.symbol.tanh
.. autofunction:: nnvm.symbol.sigmoid
.. autofunction:: nnvm.symbol.exp
.. autofunction:: nnvm.symbol.log
.. autofunction:: nnvm.symbol.sqrt
.. autofunction:: nnvm.symbol.elemwise_add
.. autofunction:: nnvm.symbol.elemwise_sub
.. autofunction:: nnvm.symbol.elemwise_mul
.. autofunction:: nnvm.symbol.elemwise_div
.. autofunction:: nnvm.symbol.elemwise_sum
.. autofunction:: nnvm.symbol.flatten
.. autofunction:: nnvm.symbol.concatenate
.. autofunction:: nnvm.symbol.expand_dims
.. autofunction:: nnvm.symbol.squeeze
.. autofunction:: nnvm.symbol.split
.. autofunction:: nnvm.symbol.dropout
.. autofunction:: nnvm.symbol.batch_norm
.. autofunction:: nnvm.symbol.softmax
.. autofunction:: nnvm.symbol.log_softmax
.. autofunction:: nnvm.symbol.pad
.. autofunction:: nnvm.symbol.block_grad

.. autofunction:: nnvm.symbol.conv2d
.. autofunction:: nnvm.symbol.conv2d_transpose
.. autofunction:: nnvm.symbol.max_pool2d
.. autofunction:: nnvm.symbol.avg_pool2d
.. autofunction:: nnvm.symbol.global_max_pool2d
.. autofunction:: nnvm.symbol.global_avg_pool2d

.. autofunction:: nnvm.symbol.reshape
.. autofunction:: nnvm.symbol.copy
.. autofunction:: nnvm.symbol.negative
.. autofunction:: nnvm.symbol.leaky_relu
.. autofunction:: nnvm.symbol.__add_scalar__
.. autofunction:: nnvm.symbol.__sub_scalar__
.. autofunction:: nnvm.symbol.__rsub_scalar__
.. autofunction:: nnvm.symbol.__mul_scalar__
.. autofunction:: nnvm.symbol.__div_scalar__
.. autofunction:: nnvm.symbol.__rdiv_scalar__
.. autofunction:: nnvm.symbol.__pow_scalar__
.. autofunction:: nnvm.symbol.__rpow_scalar__

.. autofunction:: nnvm.symbol.transpose
.. autofunction:: nnvm.symbol.broadcast_to
.. autofunction:: nnvm.symbol.sum
.. autofunction:: nnvm.symbol.min
.. autofunction:: nnvm.symbol.max
.. autofunction:: nnvm.symbol.broadcast_add
.. autofunction:: nnvm.symbol.broadcast_sub
.. autofunction:: nnvm.symbol.broadcast_mul
.. autofunction:: nnvm.symbol.broadcast_div
.. autofunction:: nnvm.symbol.clip
.. autofunction:: nnvm.symbol.greater
.. autofunction:: nnvm.symbol.less
.. autofunction:: nnvm.symbol.expand_like
.. autofunction:: nnvm.symbol.reshape_like
.. autofunction:: nnvm.symbol.full
.. autofunction:: nnvm.symbol.full_like
.. autofunction:: nnvm.symbol.ones
.. autofunction:: nnvm.symbol.ones_like
.. autofunction:: nnvm.symbol.zeros
.. autofunction:: nnvm.symbol.zeros_like
