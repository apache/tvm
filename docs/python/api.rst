Python API Reference
====================

tvm
---
tvm is a library root namespace contains functions for
declaring computation.

.. autofunction:: tvm.Var

.. autofunction:: tvm.convert

.. autofunction:: tvm.placeholder

.. autofunction:: tvm.compute

.. autofunction:: tvm.scan

.. autofunction:: tvm.extern

.. autofunction:: tvm.reduce_axis

.. autofunction:: tvm.sum

tvm.expr
--------
.. automodule:: tvm.expr
    :members:

tvm.tensor
----------
The `tvm.tensor` module contains declaration of Tensor
and Operation class for computation declaration.

.. autoclass:: tvm.tensor.Tensor
    :members:
    :inherited-members:

.. autoclass:: tvm.tensor.Operation
    :members:
    :inherited-members:

tvm.schedule
------------
.. autofunction:: tvm.Schedule

.. autoclass:: tvm.schedule.Schedule
    :members:

.. autoclass:: tvm.schedule.Stage
    :members:

tvm.build
---------

.. autofunction:: tvm.lower

.. autofunction:: tvm.build

Runtime Array
-------------

.. autofunction:: tvm.cpu
.. autofunction:: tvm.opencl

.. autofunction:: tvm.ndarray.array

.. autoclass:: tvm.ndarray.NDArray
    :members:
    :inherited-members:

.. autoclass:: tvm.Function
    :members:
    :inherited-members:

Compiled Module
---------------
.. autofunction:: tvm.module.load

.. autofunction:: tvm.module.load

.. autoclass:: tvm.module.Module
    :members:
    :inherited-members:
