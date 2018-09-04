TOPI
----
.. automodule:: topi

List of operators
~~~~~~~~~~~~~~~~~

.. autosummary::

   topi.identity
   topi.negative
   topi.floor
   topi.ceil
   topi.trunc
   topi.round
   topi.abs
   topi.exp
   topi.tanh
   topi.log
   topi.sqrt
   topi.sigmoid
   topi.clip
   topi.cast
   topi.transpose
   topi.flip
   topi.strided_slice
   topi.expand_dims
   topi.reshape
   topi.squeeze
   topi.concatenate
   topi.split
   topi.take
   topi.full
   topi.full_like
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
   topi.image.resize


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
.. autofunction:: topi.trunc
.. autofunction:: topi.round
.. autofunction:: topi.abs
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.clip
.. autofunction:: topi.cast
.. autofunction:: topi.transpose
.. autofunction:: topi.flip
.. autofunction:: topi.strided_slice
.. autofunction:: topi.expand_dims
.. autofunction:: topi.reshape
.. autofunction:: topi.squeeze
.. autofunction:: topi.concatenate
.. autofunction:: topi.split
.. autofunction:: topi.take
.. autofunction:: topi.full
.. autofunction:: topi.full_like
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min
.. autofunction:: topi.prod
.. autofunction:: topi.broadcast_to
.. autofunction:: topi.add
.. autofunction:: topi.subtract
.. autofunction:: topi.multiply
.. autofunction:: topi.divide
.. autofunction:: topi.mod
.. autofunction:: topi.maximum
.. autofunction:: topi.minimum
.. autofunction:: topi.power
.. autofunction:: topi.greater
.. autofunction:: topi.less

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

topi.image
~~~~~~~~~~
.. autofunction:: topi.image.resize


topi.generic
~~~~~~~~~~~~
.. automodule:: topi.generic

.. autofunction:: topi.generic.schedule_conv2d_nchw
.. autofunction:: topi.generic.schedule_depthwise_conv2d_nchw
.. autofunction:: topi.generic.schedule_reduce
.. autofunction:: topi.generic.schedule_broadcast
.. autofunction:: topi.generic.schedule_injective
