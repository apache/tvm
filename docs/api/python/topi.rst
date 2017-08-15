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
   topi.broadcast_to
   topi.max
   topi.sum
   topi.min
   topi.nn.relu
   topi.nn.dilate
   topi.nn.scale_shift
   topi.nn.conv2d_nchw
   topi.nn.conv2d_hwcn
   topi.nn.depthwise_conv2d


**List of schedules**

.. autosummary::

   topi.cuda.schedule_conv2d_nchw
   topi.cuda.schedule_conv2d_hwcn
   topi.cuda.schedule_depthwise_conv2d
   topi.cuda.schedule_reduce
   topi.cuda.schedule_broadcast_to


topi
~~~~
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.broadcast_to
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min

topi.nn
~~~~~~~
.. autofunction:: topi.nn.relu
.. autofunction:: topi.nn.dilate
.. autofunction:: topi.nn.scale_shift
.. autofunction:: topi.nn.conv2d_nchw
.. autofunction:: topi.nn.conv2d_hwcn
.. autofunction:: topi.nn.depthwise_conv2d

topi.cuda
~~~~~~~~~
.. automodule:: topi.cuda

.. autofunction:: topi.cuda.schedule_conv2d_nchw
.. autofunction:: topi.cuda.schedule_conv2d_hwcn
.. autofunction:: topi.cuda.schedule_depthwise_conv2d
.. autofunction:: topi.cuda.schedule_reduce
.. autofunction:: topi.cuda.schedule_broadcast_to
