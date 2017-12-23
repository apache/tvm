TVM Operator Inventory
----------------------
.. automodule:: topi

Index
~~~~~

**List of operators**

.. autosummary::

   topi.exp
   topi.tanh
   topi.log
   topi.sqrt
   topi.sigmoid
   topi.transpose
   topi.expand_dims
   topi.nn.relu
   topi.nn.leaky_relu
   topi.nn.dilate
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


**List of schedules**

.. autosummary::

   topi.generic.schedule_conv2d_nchw
   topi.generic.schedule_depthwise_conv2d_nchw
   topi.generic.schedule_reduce
   topi.generic.schedule_broadcast
   topi.generic.schedule_injective

topi
~~~~
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.transpose
.. autofunction:: topi.expand_dims
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min
.. autofunction:: topi.broadcast_to
.. autofunction:: topi.broadcast_add
.. autofunction:: topi.broadcast_sub
.. autofunction:: topi.broadcast_mul
.. autofunction:: topi.broadcast_div


topi.nn
~~~~~~~
.. autofunction:: topi.nn.relu
.. autofunction:: topi.nn.leaky_relu
.. autofunction:: topi.nn.dilate
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
