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
   tvm.relay.add
   tvm.relay.expand_dims

**Level 2: Convolutions**

This level enables typical convnet models.

.. autosummary::
   :nosignatures:

   tvm.relay.nn.conv2d


**Level 3: Additional Math And Transform Operators**

**Level 4: Broadcast and Reductions**


**Level 5: Vision/Image Operators**


Level 1 Definitions
-------------------
.. autofunction:: tvm.relay.log
.. autofunction:: tvm.relay.sqrt
.. autofunction:: tvm.relay.exp
.. autofunction:: tvm.relay.add


Level 2 Definitions
-------------------
.. autofunction:: tvm.relay.nn.conv2d
