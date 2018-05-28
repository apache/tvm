TOPI
----
.. automodule:: topi

List of operators
~~~~~~~~~~~~~~~~~

.. autosummary::

   topi.identity
   topi.negative
   topi.exp
   topi.tanh
   topi.log
   topi.sqrt
   topi.sigmoid
   topi.clip
   topi.cast
   topi.transpose
   topi.expand_dims
   topi.reshape
   topi.squeeze
   topi.concatenate
   topi.split
   topi.full
   topi.full_like
   topi.greater
   topi.less
   topi.nn.relu
   topi.nn.leaky_relu
   topi.nn.dilate
   topi.nn.pool
   topi.nn.global_pool
   topi.nn.upsampling
   topi.nn.softmax
   topi.nn.log_softmax
   topi.nn.conv2d_nchw
   topi.nn.conv2d_hwcn
   topi.nn.depthwise_conv2d_nchw
   topi.nn.depthwise_conv2d_nhwc
   topi.max
   topi.sum
   topi.min
   topi.broadcast_to
   topi.broadcast_add
   topi.broadcast_sub
   topi.broadcast_mul
   topi.broadcast_div
   topi.broadcast_maximum
   topi.broadcast_minimum


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
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.clip
.. autofunction:: topi.cast
.. autofunction:: topi.transpose
.. autofunction:: topi.expand_dims
.. autofunction:: topi.reshape
.. autofunction:: topi.squeeze
.. autofunction:: topi.concatenate
.. autofunction:: topi.split
.. autofunction:: topi.full
.. autofunction:: topi.full_like
.. autofunction:: topi.greater
.. autofunction:: topi.less
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min
.. autofunction:: topi.broadcast_to
.. autofunction:: topi.broadcast_add
.. autofunction:: topi.broadcast_sub
.. autofunction:: topi.broadcast_mul
.. autofunction:: topi.broadcast_div
.. autofunction:: topi.broadcast_maximum
.. autofunction:: topi.broadcast_minimum


topi.nn
~~~~~~~
.. autofunction:: topi.nn.relu
.. autofunction:: topi.nn.leaky_relu
.. autofunction:: topi.nn.dilate
.. autofunction:: topi.nn.pool
.. autofunction:: topi.nn.global_pool
.. autofunction:: topi.nn.upsampling
.. autofunction:: topi.nn.softmax
.. autofunction:: topi.nn.log_softmax
.. autofunction:: topi.nn.conv2d_nchw
.. autofunction:: topi.nn.conv2d_hwcn
.. autofunction:: topi.nn.depthwise_conv2d_nchw
.. autofunction:: topi.nn.depthwise_conv2d_nhwc


topi.generic
~~~~~~~~~~~~
.. automodule:: topi.generic

.. autofunction:: topi.generic.schedule_conv2d_nchw
.. autofunction:: topi.generic.schedule_depthwise_conv2d_nchw
.. autofunction:: topi.generic.schedule_reduce
.. autofunction:: topi.generic.schedule_broadcast
.. autofunction:: topi.generic.schedule_injective
