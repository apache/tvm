Python API
==========

tvm
---
tvm is a library root namespace contains functions for
declaring computation.

.. autofunction:: tvm.load_json

.. autofunction:: tvm.save_json

.. autofunction:: tvm.var

.. autofunction:: tvm.convert

.. autofunction:: tvm.placeholder

.. autofunction:: tvm.compute

.. autofunction:: tvm.scan

.. autofunction:: tvm.extern

.. autofunction:: tvm.reduce_axis

.. autofunction:: tvm.sum

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
.. autofunction:: tvm.create_schedule

.. autoclass:: tvm.schedule.Schedule
    :members:

.. autoclass:: tvm.schedule.Stage
    :members:

tvm.build
---------

.. autofunction:: tvm.lower
.. autofunction:: tvm.build

tvm.ndarray
-----------
tvm.ndarray provides a minimum runtime array API to testing out
the correctness of the program.

.. autofunction:: tvm.cpu
.. autofunction:: tvm.gpu
.. autofunction:: tvm.vpi
.. autofunction:: tvm.opencl
.. autofunction:: tvm.ndarray.array

.. autoclass:: tvm.ndarray.TVMContext
    :members:

.. autoclass:: tvm.ndarray.NDArray
    :members:
    :inherited-members:

tvm.Function
------------

.. autofunction:: tvm.register_func

.. autoclass:: tvm.Function

tvm.module
----------
.. autofunction:: tvm.module.load

.. autofunction:: tvm.module.load

.. autoclass:: tvm.module.Module
    :members:
    :inherited-members:

tvm.node
--------
tvm.node provides

.. autofunction:: tvm.register_node

.. autoclass:: tvm.node.NodeBase
    :members:

.. autoclass:: tvm.node.Node
    :members:

tvm.expr
--------
.. automodule:: tvm.expr
    :members:
