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

The second execution mechanism is the existing graph runtime. In order to target Relay
programs to this, we compile a small subset of them to the old graph format and execute
them on the runtime. Graph runtime provides a fast execution experience but only for a very limited
subset of Relay programs.

An alternative but not-standard approach is Relay's ahead-of-time compiler,
which compiles a Relay program into a shared library containing an ahead-
of-time implementation. The ahead-of-time compiler provides compelling performance
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
The graph runtime is able to utilize the fully static nature of the input graphs to perform
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

Returns the object in register `result` to caller's register `dst`.

InvokePacked
^^^^^^^^^^^^
**Arguments**:
::
  size_t packed_index
  size_t arity
  size_t output_size
  RegName* packed_args

Invoke the packed function denoted by `packed_index`. The `arity`
and `output_size` are used to inform the VM how many inputs and
outputs to expect. `packed_args` stores the list of argument registers.

AllocTensor
^^^^^^^^^^^
**Arguments**:
::
  RegName dst
  RegName shape_register
  size_t ndim
  DLDataType dtype

Allocate a tensor value of the appropriate shape (stored in `shape_register`) and `dtype`. The result
is saved to register `dst`.

AllocDatatype
^^^^^^^^^^^^^
**Arguments**:
::
  RegName dst
  size_t tag
  size_t num_fields
  RegName* datatype_fields

Allocate a data type with the tag `tag` using the `num_fields` entries
from registers `datatype_fields`. The result is saved to register `dst`.

AllocClosure
^^^^^^^^^^^^
**Arguments**:
::
  RegName dst
  size_t clo_index
  size_t num_freevar
  RegName* free_vars;

Allocate a closure with the VMFunction at `clo_index` as
its code, and the `num_freevar` entries from registers in
`free_vars`. The result is saved to register `dst`.

GetField
^^^^^^^^
**Arguments**:
::
  RegName dst
  RegName object
  size_t field_index

Get the field value with index `field_index` from `object`. And saves the result to register `dst`.

If
^^
**Arguments**:
::
  RegName test
  RegName target
  size_t true_offset
  size_t false_offset

Check if the object at register `test` is equal to `target`.
If equal, relative jump by `true_offset`, else relative
jump by `false_offset`.

GetTagi
^^^^^^^
**Arguments**:
::
  RegName object
  RegName dst

Get the object tag for Datatype object in register `object`. And saves the reult to register `dst`.

Fatal
^^^^^
Fail the virtual machine execution.

Goto
^^^^
**Arguments**:
::
  size_t pc_offset

Relative unconditional jump by `pc_offset`.

Invoke
^^^^^^
**Arguments**:
::
  size_t func_index

Invoke function at `func_index`, consumes the number of arguments contained in the VMFunction's
arity field.

InvokeClosure
^^^^^^^^^^^^^
**Arguments**:
::
    RegName closure
    size_t closure_args_num
    RegName* closure_args

Invokes `closure`, consuming the number of arguments declared in the closure's VMFunction.

LoadConst
^^^^^^^^^
**Arguments**:
::
  RegName dst
  size_t const_index

Load the constant at `const_index` from the constant pool. The result is saved to register `dst`.

LoadConsti
^^^^^^^^^^
**Arguments**:
::
  size_t val
  RegName dst

Load the constant integer `val` to register `dst`. The result is a 0-rank tensor.

Object Representation
~~~~~~~~~~~~~~~~~~~~~
We use a simple object representation that uses shared pointers and tagging.
There is a huge space of possible object representations trade-offs, but we
believe micro-optimizing this code has little to no effect on the end-to-end performance.

::

    struct ObjectCell {
      ObjectTag tag;
      ...
    };

    struct Object {
      std::shared_ptr<ObjectCell> ptr;
      ...
    }

See `include/tvm/runtime/vm.h` for more details.

Currently, we support 3 types of objects: tensors, data types, and closures.

::

    VMObject VMTensor(const tvm::runtime::NDArray& data);
    VMObject VMDatatype(size_t tag, const std::vector<VMObject>& fields);
    VMObject VMClosure(size_t func_index, std::vector<VMObject> free_vars);


Stack and State
~~~~~~~~~~~~~~~

The Relay VM maintains a stack frame, which contains information about how to resume the
previous call. Registers are allocated in a continuous space (virtual register file) for each function.

We keep track of a set of Relay functions we have called, a pointer into its bytecode, an offset into the byte code (known as the program counter).

::

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
a simple `switch`/`goto` dispatch loop which dispatches based on instruction op code.

This loop is implemented by `VirtualMachine::Run()`.

VM Compiler
~~~~~~~~~~~

An important part of this infrastructure is a compiler from Relay's full IR into a sequence of bytecode.
The VM compiler transforms a `tvm::relay::Module` into a `tvm::relay::vm::VirtualMachine`. The virtual
machine contains a set of compiled functions, the compiled functions are contained in `tvm::relay::vm::Function`. The functions contain metadata about the the function as well as its compiled bytecode. For full definitions of the data structures see `vm.h`.

Optimizations
~~~~~~~~~~~~~

There are quite a few optimizations required by the VM compiler.

We have implemented them in the old pass style, but plan to port them to
the new pass manager (#2546) before merging.

Optimizations marked with `TODO` are not implemented yet.

- A-Normal Form
- Lambda Lift (see `src/relay/vm/lambda_lift.cc`)
- Inline Primitives (see `src/relay/vm/inline_primitives.cc`)
- Inliner (see `src/relay/pass/inliner.cc`)
- Constant Pool Layout (see `src/relay/backend/vm/compiler.cc`)
- ADT Tag Allocation (see `src/relay/backend/vm/compiler.cc`)
- Tail Call Optimization (TODO)
- Liveness Analysis (TODO)

Serialization
~~~~~~~~~~~~~

A final and yet-to-be-implemented part of the VM design is serialization. The accompanying PR will introduce both the bytecode and its serialization, as well as VM-level serialization. The design premise is that a VM can be efficiently stored to disk and resumed at a later time. This would also allow us to efficiently schedule many models on to a single machine in order to obtain good utilization.

Unresolved Questions
~~~~~~~~~~~~~~~~~~~~

How do we handle dynamic shapes?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TODO

How can we modify the VM to support JIT compilation of certain code paths?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the code generation space there are still many tradeoffs to be analyzed and the VM is designed
to be very flexible so we can modify it for future experiments.

How do we support heterogenous execution?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Heterogenous execution should work out of the box assuming we have annotated the appropriate device copies.
In order to do this properly we need to run the device annotation and copying passes. 
