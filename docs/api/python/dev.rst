Developer API
-------------
This page contains modules that are used by developers of TVM.
Many of these APIs are PackedFunc registered in C++ backend.

tvm.node
~~~~~~~~
.. automodule:: tvm.node

.. autoclass:: tvm.node.NodeBase
    :members:

.. autoclass:: tvm.node.Node
    :members:

.. autofunction:: tvm.register_node

tvm.expr
~~~~~~~~
.. automodule:: tvm.expr
   :members:
   :undoc-members:

tvm.codegen
~~~~~~~~~~~
.. automodule:: tvm.codegen
   :members:
   :undoc-members:

tvm.stmt
~~~~~~~~
.. automodule:: tvm.stmt
   :members:
   :undoc-members:

tvm.ir_pass
~~~~~~~~~~~
.. autosummary::

   tvm.ir_pass.Inline
   tvm.ir_pass.Simplify
   tvm.ir_pass.ConvertSSA
   tvm.ir_pass.VerifySSA
   tvm.ir_pass.CanonicalSimplify
   tvm.ir_pass.StorageFlatten
   tvm.ir_pass.VectorizeLoop
   tvm.ir_pass.UnrollLoop
   tvm.ir_pass.ThreadSync
   tvm.ir_pass.StorageRewrite
   tvm.ir_pass.MakeAPI
   tvm.ir_pass.SplitHostDevice
   tvm.ir_pass.InjectVirtualThread
   tvm.ir_pass.LoopPartition
   tvm.ir_pass.RemoveNoOp
   tvm.ir_pass.SplitPipeline
   tvm.ir_pass.LowerThreadAllreduce
   tvm.ir_pass.LowerIntrin
   tvm.ir_pass.LowerTVMBuiltin
   tvm.ir_pass.NarrowChannelAccess

.. automodule:: tvm.ir_pass
   :members:

tvm.ir_builder
~~~~~~~~~~~~~~
.. automodule:: tvm.ir_builder
   :members:

tvm.make
~~~~~~~~
.. automodule:: tvm.make
   :members:
