NNVM Core Primitives
====================

**Level 1: Basic Ops**

.. autosummary::
   :nosignatures:

   nnvm.symbol.dense
   nnvm.symbol.relu
   nnvm.symbol.tanh
   nnvm.symbol.sigmoid
   nnvm.symbol.exp
   nnvm.symbol.log
   nnvm.symbol.elemwise_add
   nnvm.symbol.elemwise_sub
   nnvm.symbol.elemwise_mul
   nnvm.symbol.elemwise_div
   nnvm.symbol.flatten
   nnvm.symbol.concatenate
   nnvm.symbol.split
   nnvm.symbol.dropout
   nnvm.symbol.batch_norm
   nnvm.symbol.softmax
   nnvm.symbol.log_softmax


**Level 2: Convolutions**

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


.. autofunction:: nnvm.symbol.dense
.. autofunction:: nnvm.symbol.relu
.. autofunction:: nnvm.symbol.tanh
.. autofunction:: nnvm.symbol.sigmoid
.. autofunction:: nnvm.symbol.exp
.. autofunction:: nnvm.symbol.log
.. autofunction:: nnvm.symbol.elemwise_add
.. autofunction:: nnvm.symbol.elemwise_sub
.. autofunction:: nnvm.symbol.elemwise_mul
.. autofunction:: nnvm.symbol.elemwise_div
.. autofunction:: nnvm.symbol.flatten
.. autofunction:: nnvm.symbol.concatenate
.. autofunction:: nnvm.symbol.split
.. autofunction:: nnvm.symbol.dropout
.. autofunction:: nnvm.symbol.batch_norm
.. autofunction:: nnvm.symbol.softmax
.. autofunction:: nnvm.symbol.log_softmax


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
