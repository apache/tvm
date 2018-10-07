Relay Core Tensor Operators
===========================

This page contains the list of core tensor operator primitives pre-defined in tvm.relay.
The core tensor operator primitives covers typical workloads in deep learning.
They can represent workloads in front-end frameworks, and provide basic building blocks for optimization.
Since deep learning is a fast evolving field and it is that possible to have operators that are not in here.


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
   tvm.relay.exp
   tvm.relay.sigmoid
   tvm.relay.add
   tvm.relay.expand_dims
   tvm.relay.concatenate
   tvm.relay.nn.softmax
   tvm.relay.nn.log_softmax
   tvm.relay.subtract
   tvm.relay.multiply
   tvm.relay.divide
   tvm.relay.mod
   tvm.relay.tanh
   tvm.relay.sigmoid


**Level 2: Convolutions**

This level enables typical convnet models.

.. autosummary::
   :nosignatures:

   tvm.relay.nn.conv2d
   tvm.relay.nn.max_pool2d
   tvm.relay.nn.avg_pool2d
   tvm.relay.nn.global_max_pool2d
   tvm.relay.nn.global_avg_pool2d
   tvm.relay.nn.upsampling
   tvm.relay.nn.batch_flatten


**Level 3: Additional Math And Transform Operators**

This level enables additional math and transform operators.

.. autosummary::
   :nosignatures:

   tvm.relay.zeros_like
   tvm.relay.ones_like
   tvm.relay.reshape
   tvm.relay.copy
   tvm.relay.transpose
   tvm.relay.floor
   tvm.relay.ceil
   tvm.relay.trunc
   tvm.relay.round
   tvm.relay.abs
   tvm.relay.negative


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
   tvm.relay.maximum
   tvm.relay.minimum
   tvm.relay.pow

**Level 5: Vision/Image Operators**

.. autosummary::
   :nosignatures:

   tvm.relay.image.resize


Level 1 Definitions
-------------------
.. autofunction:: tvm.relay.log
.. autofunction:: tvm.relay.sqrt
.. autofunction:: tvm.relay.exp
.. autofunction:: tvm.relay.sigmoid
.. autofunction:: tvm.relay.add
.. autofunction:: tvm.relay.subtract
.. autofunction:: tvm.relay.multiply
.. autofunction:: tvm.relay.divide
.. autofunction:: tvm.relay.mod
.. autofunction:: tvm.relay.tanh
.. autofunction:: tvm.relay.sigmoid
.. autofunction:: tvm.relay.concatenate
.. autofunction:: tvm.relay.nn.softmax
.. autofunction:: tvm.relay.nn.log_softmax


Level 2 Definitions
-------------------
.. autofunction:: tvm.relay.nn.conv2d
.. autofunction:: tvm.relay.nn.max_pool2d
.. autofunction:: tvm.relay.nn.avg_pool2d
.. autofunction:: tvm.relay.nn.global_max_pool2d
.. autofunction:: tvm.relay.nn.global_avg_pool2d
.. autofunction:: tvm.relay.nn.upsampling
.. autofunction:: tvm.relay.nn.batch_flatten


Level 3 Definitions
-------------------
.. autofunction:: tvm.relay.floor
.. autofunction:: tvm.relay.ceil
.. autofunction:: tvm.relay.trunc
.. autofunction:: tvm.relay.round
.. autofunction:: tvm.relay.abs
.. autofunction:: tvm.relay.negative
.. autofunction:: tvm.relay.reshape
.. autofunction:: tvm.relay.copy
.. autofunction:: tvm.relay.transpose

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
.. autofunction:: tvm.relay.maximum
.. autofunction:: tvm.relay.minimum
.. autofunction:: tvm.relay.pow

Level 5 Definitions
-------------------
.. autofunction:: tvm.relay.image.resize
