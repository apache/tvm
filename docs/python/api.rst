Python API Reference
====================

Computation Declaration
-----------------------

.. autofunction:: tvm.Var

.. autofunction:: tvm.convert

.. autofunction:: tvm.placeholder

.. autofunction:: tvm.compute

.. autofunction:: tvm.scan

.. autofunction:: tvm.extern

Computation Schedule
--------------------
.. autofunction:: tvm.Schedule

.. autoclass:: tvm.schedule.Schedule
    :members:

.. autoclass:: tvm.schedule.Stage
    :members:

Runtime Array Manipulation
--------------------------
Every function under TVM API can be directly used under namespace tvm tvm.


.. autofunction:: tvm.cpu
.. autofunction:: tvm.opencl

.. autofunction:: tvm.ndarray.array

.. autoclass:: tvm.ndarray.NDArray
    :members:
    :inherited-members:

.. autoclass:: tvm.Function
    :members:
    :inherited-members:
