..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Putting the VM in TVM: The Relay Virtual Machine
================================================

Relay, a new program representation, has enabled the representation and optimization of
a great breadth of machine learning programs.
Unfortunately, by supporting a more expressive set of programs, we have
introduced several new execution challenges.

Relay's interpreter can execute the full language but has notable limitations
that make it unsuited for production deployments. It is structured as an inefficient
interpreter that performs AST traversal to execute the program. This approach is conceptually
simple but inefficient, as the AST traversal heavily relies on indirection.

There are further challenges in compiling dynamic code, such as dynamic scheduling and allocation,
fully dynamic tensor shapes, and control flow. The interpreter offers simple solutions
for these, but none is sufficiently compelling or optimized.

The second execution mechanism is the existing graph executor. In order to target Relay
programs to this, we compile a small subset of them to the old graph format and execute
them on the runtime. Graph executor provides a fast execution experience but only for a very limited
subset of Relay programs.

An alternative but not-standard approach is Relay's ahead-of-time compiler,
which compiles a Relay program into a shared library containing an ahead-of-time
implementation. The ahead-of-time compiler provides compelling performance
but is difficult to extend and instrument, which can only be done by modifying the
code generation and optimization mechanisms.

The Relay virtual machine is intended to be a framework that balances these competing
approaches, providing a dynamic execution environment which can be extended, instrumented,
and integrated with other approaches like ahead-of-time compilation via a flexible extension
mechanism.

The virtual machine is designed to strike a balance between performance and flexibility
when deploying and executing Relay programs, without giving up the benefits of TVM.

Virtual machine (VM) design is a well-studied area in programming languages and systems,
and there have been various virtual machine designs for both full-fledged
and embedded programing languages.
Previous language VM designs have been heavily tailored to the execution profile of traditional programs.
Traditional programs manipulate small scalar values and consist of a large number of low-level instructions.
The sheer quantity of instructions requires instruction execution and dispatch to be extremely efficient.
In the context of machine learning we manipulate primarily tensor values, using a (relatively)
low number of high level instructions. ML programs' cost centers are expensive operator invocations,
such as GEMM or convolution, over a large input. Due to the execution profile exhibited by ML programs,
micro-optimizations present in scalar VMs are dramatically less important.

TVM has provided strong support for vision models,
but we want to grow to support a wider variety of models.
The graph executor is able to utilize the fully static nature of the input graphs to perform
aggressive optimization such as fully static allocation, and optimal memory reuse.
When we introduce models which make use of control flow, recursion, dynamic shapes, and dynamic
allocation, we must change how execution works. A virtual machine for Relay is a natural choice.

The rest of this document provides a high-level overview of the Relay
virtual machine design and its instruction set.

Design
------

The VM's design is focused on simplicity without sacrificing performance.
In order to accomplish this we have focused on designing a tensor VM rather than a scalar VM.

In the tensor VM setting, we optimize for cheap “allocation” of objects (by trying to avoid real allocation),
reuse of static fragments, and the ability to do dynamic shape (i.e jagged tensors).

Instruction Set
~~~~~~~~~~~~~~~

The choices of an instruction set and instruction representation are the most critical design decisions for a VM.
The current representation of the instructions is a tagged union containing the op-code and the data payload.  An important design decision is the level of abstraction of the instructions (RISC vs. CISC) and how they take their data (fixed-width instruction encoding vs. variable-length encoding). The current version is closer to CISC, with complex instructions like AllocTensor, and is variable-length due to the inclusion of the shape as part of the instruction. The current instruction set is very high-level and corresponds roughly to high-level operations in Relay.

Ret
^^^
**Arguments**:
::

  RegName dst
  RegName result

Returns the object in register ``result`` to caller's register ``dst``.

InvokePacked
^^^^^^^^^^^^
**Arguments**:
::

  Index packed_index
  Index arity
  Index output_size
  RegName* packed_args

Invoke the packed function denoted by ``packed_index``. The ``arity``
and ``output_size`` are used to inform the VM how many inputs and
outputs to expect. ``packed_args`` stores the list of argument registers. Note ``Index``
is an alias of ``int64_t``, and it will be used in other instructions as well.

AllocTensor
^^^^^^^^^^^
**Arguments**:
::

  RegName dst
  RegName storage
  uint32_t ndim
  int64_t* shape
  DLDataType dtype

Allocate a tensor value of using constant shape (stored in ``shape``) and ``dtype``
from the given storage block, ``storage``. The result is saved to register ``dst``.

AllocTensorReg
^^^^^^^^^^^^^^
**Arguments**:
::

  RegName dst
  RegName storage
  RegName shape_register
  DLDataType dtype

Allocate a tensor value of the appropriate shape (stored in ``shape_register``)
and ``dtype`` from the given storage block (stored in ``storage``). The result is saved to register ``dst``.

AllocStorage
^^^^^^^^^^^^
**Arguments**:
::

  RegName dst
  RegName size
  RegName alignment
  DLDataType dtype_hint

Allocate a storage block with the given ``size``, ``alignment`` and data type, ``dtype_hint``.
The allocated storage block is stored in register ``dst``.

AllocADT
^^^^^^^^
**Arguments**:
::

  RegName dst
  Index tag
  Index num_fields
  RegName* datatype_fields

Allocate a data type with the tag ``tag`` using the ``num_fields`` entries
from registers ``datatype_fields``. The result is saved to register ``dst``.

AllocClosure
^^^^^^^^^^^^
**Arguments**:
::

  RegName dst
  Index clo_index
  Index num_freevar
  RegName* free_vars;

Allocate a closure with the VMFunction at ``clo_index`` as
its code, and the ``num_freevar`` entries from registers in
``free_vars``. The result is saved to register ``dst``.

GetField
^^^^^^^^
**Arguments**:
::

  RegName dst
  RegName object
  Index field_index

Get the field value with index ``field_index`` from ``object``. And saves the result to register ``dst``.

If
^^
**Arguments**:
::

  RegName test
  RegName target
  Index true_offset
  Index false_offset

Check if the object at register ``test`` is equal to ``target``.
If equal, relative jump by ``true_offset``, else relative
jump by ``false_offset``.

GetTag
^^^^^^
**Arguments**:
::

  RegName object
  RegName dst

Get the object tag for ADT object in register ``object``. And saves the reult to register ``dst``.

Fatal
^^^^^
Fail the virtual machine execution.

Goto
^^^^
**Arguments**:
::

  Index pc_offset

Relative unconditional jump by ``pc_offset``.

Invoke
^^^^^^
**Arguments**:
::

  Index func_index

Invoke function at ``func_index``, consumes the number of arguments contained in the VMFunction's
arity field.

InvokeClosure
^^^^^^^^^^^^^
**Arguments**:
::

    RegName closure
    Index num_closure_args
    RegName* closure_args

Invokes ``closure``, consuming the number of arguments declared in the closure's VMFunction.

LoadConst
^^^^^^^^^
**Arguments**:
::

  RegName dst
  Index const_index

Load the constant at ``const_index`` from the constant pool. The result is saved to register ``dst``.

LoadConsti
^^^^^^^^^^
**Arguments**:
::

  Index val
  RegName dst

Load the constant integer ``val`` to register ``dst``. The result is a 0-rank tensor.

Object Representation
~~~~~~~~~~~~~~~~~~~~~
We leverage the object protocol to represent the objects that are used by the
VM.

Currently, three types of objects, ``NDArray``, ``ADT``, and ``Closure`` objects, are used
to represent tensor, tuple/list, and closure data, respectively. More details
for each of them can be found at `include/tvm/runtime/ndarray.h`_,
`include/tvm/runtime/vm/vm.h`_, and `include/tvm/runtime/container.h`_, respectively.

.. _include/tvm/runtime/ndarray.h: https://github.com/apache/tvm/blob/main/include/tvm/runtime/ndarray.h

.. _include/tvm/runtime/vm/vm.h: https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/vm.h

.. _include/tvm/runtime/container.h: https://github.com/apache/tvm/blob/main/include/tvm/runtime/container.h

Stack and State
~~~~~~~~~~~~~~~

The Relay VM maintains a stack frame, which contains information about how to resume the
previous call. Registers are allocated in a continuous space (virtual register file) for each function.

We keep track of a set of Relay functions we have called, a pointer into its bytecode, an offset into the byte code (known as the program counter).

.. code-block:: c

    struct VirtualMachine {
      ...
      std::vector<VMFrame> frames;
      ...
      // Current function.
      size_t func_index;
      // Pointer into the current function's instructions.
      const Instruction* code;
      // Current program counter relative to the code pointer.
      size_t pc;
      ...
    };


Dispatch Loop
~~~~~~~~~~~~~
A critical piece of a VM is the dispatch loop. The dispatch loop usually dominates the execution time of a
virtual machine, but we have experimentally found this not to be the case for Relay. We have just implemented
a simple ``switch``/``goto`` dispatch loop which dispatches based on instruction op code.

This loop is implemented by ``VirtualMachine::Run()``.

VM Compiler
~~~~~~~~~~~

An important part of this infrastructure is a compiler from Relay's full IR into a sequence of bytecode.
The VM compiler transforms a ``tvm::relay::Module`` into a ``tvm::relay::vm::Executable``. The executable
contains a set of compiled functions, the compiled functions are contained in ``tvm::relay::vm::Function``.
The functions contain metadata about the function as well as its compiled bytecode. The emitted executable
object then can be loaded and run by a ``tvm::relay::vm::VirtualMachine`` object. For full definitions of the
data structures, please see `include/tvm/runtime/vm/executable.h`_ and `include/tvm/runtime/vm/vm.h`_.

.. _include/tvm/runtime/vm/executable.h: https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/executable.h

Optimizations
~~~~~~~~~~~~~

There are quite a few optimizations required by the VM compiler. Each of them
is implemented as a pass which is managed by the Relay pass manager.

Optimizations marked with `TODO` are not implemented yet.

- A-Normal Form
- Lambda Lift (see `src/relay/vm/lambda_lift.cc`_)
- Inline Primitives (see `src/relay/vm/inline_primitives.cc`_)
- Constant Pool Layout (see `src/relay/backend/vm/compiler.cc`_)
- Tail Call Optimization (TODO)
- Liveness Analysis (TODO)

.. _src/relay/vm/lambda_lift.cc: https://github.com/apache/tvm/blob/main/src/relay/backend/vm/lambda_lift.cc

.. _src/relay/vm/inline_primitives.cc: https://github.com/apache/tvm/blob/main/src/relay/backend/vm/inline_primitives.cc

.. _src/relay/backend/vm/compiler.cc: https://github.com/apache/tvm/blob/main/src/relay/backend/vm/compiler.cc

Serialization
~~~~~~~~~~~~~

Serializing and deserializing the executable generated by the Relay VM compiler is a must as
we may want to save the model to the disk and perform inference later. Previously, Relay has produced
a serialized form in a json file for the graph executor. However, the same format is not directly
applicable to the VM as it emits bytecode instead of graph-style programs.
Serialization of an executable essentially needs to handle both model specific
(i.e. weights and kernels) and VM related (i.e. bytecode and global function names) data.

For kernels, we can conveniently leverage existing TVM infra to save and load
the compiled library module. Here we only focus on serializing other several
components in a binary format that is organized with the following sections in order.

- Global section. This section contains the globals (function names) used by the virtual machine.

- Constant section. This section is used to store the constant pool (i.e. weights of the model)
  for a virtual machine.

- Primitive name section. This section is introduced to accommodate the list of primitive
  operator names that will be invoked by the virtual machine, i.e. the names
  starting with ``fused_``. The primitive names are used as symbols to look up
  function pointers in the compiled kernel library.

- Code section. The VM functions, including bytecode, are sitting in this section. The dispatching
  loop iterates through this section to fetch instructions for execution.

Hence, unlike the graph executor artifact that contains weight (.params), graph json (.json),
and compiled kernel library (.so), the serialized executable artifact is composed of the Relay
object file (.ro) and the compiled kernel library (.so).

A ``save`` function is implemented to store the executable to the disk and
serialize it into the above format. Meanwhile, a ``load_exec`` function is used to
load the serialized kernel binary and executable related binary code, which will be again used to
instantiate a VM object. Please refer to the `test_vm_serialization.py`_ file for more
examples.

.. _test_vm_serialization.py: https://github.com/apache/tvm/blob/main/tests/python/relay/test_vm_serialization.py

Unresolved Questions
~~~~~~~~~~~~~~~~~~~~

How do we handle dynamic shapes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic shape support is ongoing work in TVM as we upgrade Relay, TVM's compiler.  For the most recent updates on
dynamic shape support, we recommend following updates in TVM's Discuss forum (https://discuss.tvm.apache.org/).

How can we modify the VM to support JIT compilation of certain code paths?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the code generation space there are still many tradeoffs to be analyzed and the VM is designed
to be very flexible so we can modify it for future experiments.

How do we support heterogenous execution?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Heterogenous execution should work out of the box assuming we have annotated the appropriate device copies.
In order to do this properly we need to run the device annotation and copying passes.
