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

.. _microtvm-design:

**************************
microTVM Design Document
**************************

.. contents:: Table of Contents
    :depth: 3

Background
===========

TVM is a model deployment framework that has demonstrated good performance across a wide range of
models on traditional operating systems. Given TVM's layered approach to compilation, it is a
natural extension to target bare metal devices. While most of the compilation flow does not need to
change for a proof-of-concept implementation on such devices, the runtime cannot depend on:

* **Virtual Memory**, and by extension any system-provided ``malloc``. Additionally, bare metal
  devices typically have very limited memory (measured in KB). Because of this, libraries designed
  for such platforms typically need to be more judicious in using memory, and need to release
  memory when it is not in use.
* Traditional OS abstractions, such as **files**, **libraries**, and **kernel functions**. Some
  projects implement support for these, but they are by no means standard.
* Support for programming languages other than **C**.

Such changes require a different approach from the TVM C++ runtime typically used on traditional
Operating Systems.

Typical Use
===========

This section discusses our vision of the "typical" microTVM use case. Each component used to achieve
this typical use case is intended to be designed for flexibility, but this unifying vision serves to
motivate the inclusion of each part of the design.

.. figure:: https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_workflow.svg
   :align: center
   :width: 85%

The parts of this process are described below:

#. **Model Import**. The user imports an existing model or describes a new model to TVM, producing a
   *Relay module*.

#. **Model Transformations**. The user can apply transformations, such as quantization, to the
   model. After each transformation, the user should still have a Relay module.

#. **Compilation** (Scheduling and Code Generation). TVM implements each operator into Tensor IR by
   assigning a schedule and schedule configuration to each Relay operator. Then, code (C source or
   compiled object) is generated for each operator.

#. **Integration**. The generated code is integrated along with the TVM C Runtime library into a
   user-supplied binary project. In some cases (such as when the project is standardized across
   multiple SoC/development boards), this process is handled automatically.

#. **Deployment**. The project is built and the residual firmware binary is flashed onto the device.
   Model inference is driven either by TVM using an on-device RPC server, or on the device using the
   on-device Graph Executor.

Design Goals
============

microTVM aims to achieve these design goals:

1. **Portable Code**. microTVM can translate any Relay model into C code that can compile with only
   a C standard library.
2. **Minimal Overhead**. microTVM generates target-specific, highly optimized code. As much overhead
   from the runtime should be removed.
3. **Accessible Code**. microTVM considers C source code as a first-class output mechanism so that
   it is easier for a firmware engineer to understand and tweak.

Overview
========

microTVM requires changes at all levels of the TVM compiler stack. The following sub-sections enumerate
these changes at a high level, and follow-on sections discuss the specifics in more detail.

Modeling Target Platforms
-------------------------

TVM's search-based optimization approach allows it to largely avoid system-level modeling of targets
in favor of experimental results. However, some modeling is necessary in order to ensure TVM is
comparing apples-to-apples search results, and to avoid wasting time during the search by attempting
to compile invalid code for a target.

microTVM models these parts of the target:

* The CPU used, through the ``-mcpu`` and ``-march`` target flags.
* The presence or absence of accelerators, through the device components of the target (Currently
  only the absence of accelerators can be expressed, but this mechanism should extend well).

microTVM aims to model these parts of the target in the future:

* Memory, modeled as a set of disjoint memory spaces, each with a label and size and prefetch/flush
  behavior. Some memory may be shared with accelerators.
* Target runtime configuration (i.e. clock tree configuration, clock speed, etc). This is intended
  only to contribute to the AutoTVM schedule key and not for any other use.

At this time, TVM does not intend to model:

* Size, type, or relationship of caches, with the exception of prefetching or cache flushing.


TVM Targets for microTVM
-------------------------

A central data structure in the compilation process is the ``tvm::target::Target`` class. TVM uses
Target to decide which TIR schedules to enable and how to configure the code generator. The Target
class should also uniquely identify the generated code for a particular operator, as autotuning
logs use it to rank measured performance (but see Future Work).

Targets are currently represented as strings structured similarly to command-line arguments. An
example target is shown below:

    ``c -keys=arm_cpu -mcpu=cortex-m7 -model=stm32f746xx``

The relevant parts to microTVM are:

 * Code generator (``llvm`` or ``c``)
 * ``-mcpu=cortex-m7``: used by TOPI to enable Cortex-M schedules, and, when the C source code
   generator is selected, included in the output as a comment to help identify the code and
   configure the downstream C compiler.

Runtime and Executor configuration for microTVM
-----------------------------------------------

When using microTVM, it's important to use the C Runtime (``Runtime('crt')``), which is the runtime that works best on micro devices rather than the more dynamic C++ Runtime. Alongside this, there are two executors which you could use in combination with the C runtime:

* ``Executor("aot")`` - The Ahead of Time (AOT) executor precompiles the network into a runnable function which you can add directly into your micro application
* ``Executor("graph", {"link-params": True})`` - The Graph executor provides a JSON representation of your network and requires the C Runtime's system library to be generated to find functions in the function registry (``Runtime("crt", {"system-lib": True})``). ``{"link-params":True}`` enables parameters to be linked into the generated files rather than provided externally.

These are specified when building a runtime module: ``relay.build(..., runtime=..., executor=...)``.

Writing Schedules for microTVM
------------------------------

For operations scheduled on the CPU, microTVM initially plans to make use of specialized
instructions and extern (i.e. hand-optimized) functions to achieve good performance. In TVM, this
approach is generally accomplished through tensorization, in which TVM breaks a computation into
small pieces, and a TIR extern function accelerates each small piece.

TVM currently accommodates both approaches using ``tir.call_extern``. First, a pragma is attached to
the schedule defining the extern function in portable C.

    ``sched[output].pragma(n, "import_c", "void call_asm(int32_t* a, int32_t* b) { /* ... */ }")``

Next, ``tensorize`` is used to split the computation.

    ``sched[output].tensorize(owi, gemm)``

There are a couple of caveats to this approach, all which could be resolved by linking generated
code against external libraries:

* Inline assembly is compiler-specific. While Clang and GCC have standardized on one syntax, this
  may not be portable to other compilers. SDKs solve this by conditionally including a header file
  depending on the compiler being used. However, taking this approach means that the generated code
  needs additional compiler flags (i.e. ``-Isystempath/to/header``).
* It may be helpful to reference helper functions from the generated code (e.g. to inline common
  sequences of hand-optimized assembly).
* Finally, the extern function invoked may be wholly written in an external library. If those
  functions can be wholly inlined, this caveat is the same as the previous. If not, then additional
  C code needs to be compiled and linked against the operator.

At present, microTVM presumes that all eligible schedules can be compiled. This means that the user-
supplied project (see next section) must include all libraries that are used by the generated code.
When not using autotuning, TVM randomly chooses a fallback schedule, so all libraries would need to
be supported. When using autotuning, TVM selects the best-performing schedule, so only that library
is needed. There isn't currently a way to force TVM to pick a particular schedule outside of
autotuning logs, but that would be a good addition.

Finally, when using the ``llvm`` backend, the process is similar except that LLVM bitcode is included
in the generated code (with an ``import_llvm`` pragma). LLVM bitcode provides a portable way to call
inline assembly. However, it may be more complex to call external C functions, and helper functions
are of course not easy to use from LLVM bitcode.

Executing Models
----------------

The TVM compiler traditionally outputs three pieces:

1. Model operator implementations, as discussed above;
2. A model execution graph, encoded as JSON; and
3. Simplified parameters.

To correctly execute the model, a Graph Executor needs to reconstruct the graph in memory, load the
parameters, and then invoke the operator implementations in the correct order.

microTVM supports two ways to do this:

1. **Host-Driven**. The Graph Executor can run on the host and carry out execution by issuing
   commands to the device using an RPC link with a UART-like transport.
2. **Standalone**. A C Graph Executor is available to be compiled on-device, but it is not
   particularly memory efficient. This way enables standalone execution without any attached host.

Host-Driven is designed for experimenting with models on-device and, like AutoTVM, uses the RPC server to
drive computation on-device. Standalone is intended for deployment.

Host-Driven Execution
^^^^^^^^^^^^^^^^^^^^^

In Host-Driven execution, the firmware binary is the following:

1. Generated operator implementations from TVM.
2. The TVM C runtime.
3. SoC-specific initialization.
4. The TVM RPC server.
5. (optional) Simplified Parameters.

This firmware image is flashed onto the device and a GraphExecutor instance is created on the host.
The GraphExecutor drives execution by sending RPC commands over a UART:

.. figure:: https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_host_driven.svg
   :align: center
   :width: 85%

Standalone Execution
^^^^^^^^^^^^^^^^^^^^

In Standalone execution, the GraphExecutor is instantiated on device:

.. figure:: https://raw.githubusercontent.com/tvmai/web-data/main/images/dev/microtvm_standalone.svg
   :align: center
   :width: 85%

microTVM Firmware
------------------

We can now discuss how microTVM firmware should behave. An important task common to both model
execution strategies is configuring the SoC to match the way it performs in production. microTVM
considers this task project- and SoC-dependent. Whether for AutoTVM, host-driven model inference, or
in standalone deployment, the user is expected to supply a project whose main() does the following:

1. Configure the SoC to match deployment performance.
2. Initialize the TVM C Runtime.

When configuring for host-driven inference or AutoTVM, the remaining tasks are well-defined:

3. Initialize a transport (i.e. a UART) for use with the TVM RPC server.
4. Launch the TVM RPC Server.

When configuring for standalone deployment, the firmware needs to:

1. Instantiate the system library by calling the ``runtime.SystemLib`` PackedFunc.
2. Instantiate a GraphExecutor passing the system library module.
3. Configure parameters and inputs as needed.
4. Run the model.

Parts of a microTVM Binary
--------------------------

To summarize, a microTVM firwmare binary image must contain these parts:

1. Operator implementations, produced by TVM.
2. The TVM C runtime library, supplied by TVM as a static library.
3. SoC Initialization, supplied by the user.

For Host-driven model execution, firmware also needs:

4. The TVM RPC Server library.

For Standalone model execution, firmware also needs:

4. The TVM C GraphExecutor library, supplied by TVM as a static library.
5. The remaining compiler outputs (Simplified Parameters and Graph JSON).

The Automated Build Flow
------------------------

Once code generation is complete, ``tvm.relay.build`` returns a ``tvm.runtime.Module`` and the
user can save the generated C source or binary object to a ``.c`` or ``.o`` file. From this point, TVM
can theoretically step back and the user can compile and run the code separately.

However, for AutoTVM, TVM needs some automated flow to handle the following tasks:

1. Integrate operator implementations, the TVM C Runtime library, and the TVM RPC Server library into the
   firmware project containing user-supplied SoC Initialization.
2. Build the resulting project.
3. Program the built firmware onto a (specific) attached device.
4. Identify the serial port or other transport to be used by TVM to drive remote execution.

At present, TVM expects the user to supply an implementation of the ``tvm.micro.Compiler``,
``tvm.micro.Flasher``, and ``tvm.micro.Transport`` interfaces. TVM then:

1. Builds each piece separately as a library.
2. Builds the libraries into a binary firmware image.
3. Programs the firmware image onto an attached device.
4. Opens a serial port to serve as the RPC server transport.

This design was chosen to reduce build times for microTVM (the common libraries need to be built
only once per candidate operator implemmentation). In practice, these projects are extremely small
and compile relatively quickly. Compared with the added complexity of this tighter build integration
with TVM, the performance gains are likely not worth it. A future design will consolidate the build
tasks into a single step and narrow the interface to provide a better integration.

Measuring operator performance
------------------------------

The TVM C runtime depends on user-supplied functions to measure time on-device. Users should implement
``TVMPlatformTimerStart`` and ``TVMPlatformTimerStop``. These functions should measure wall clock time, so there
are some pitfalls in implementing these functions:

1. If the CPU could halt or sleep during a computation (i.e. if it is being done on an accelerator),
   a cycle counter should likely not be used as these tend to stop counting while the CPU is asleep.
2. The granularity of these functions can be relaxed as needed to extend the range of the timer
   device. However, if granularity is too coarse, a sub-optimal schedule may be used.
3. An error should be raised if the timer overflows.
4. The timer should not interrupt computation unless absolutely necessary. Doing so may affect the
   accuracy of the results.
5. Calibrating the output against a wall clock is ideal, but it will likely be too cumbersome. A
   future PR could enable some characterization of the platform timer by, e.g., measuring the internal
   oscillator against a reference such as an external crystal.

Future Work
===========

Ahead-of-Time Runtime
----------------------

A limitation of the Graph Executor is the amount of memory overhead required in parsing the JSON.
The current implementation contributes significantly to the dynamic memory usage of microTVM,
limiting its utility. An ahead-of-time runtime can avoid the need for any Graph JSON parsing and
improve inference speed by generating C code to call the generated operator implementations directly
rather than relying on a data-driven approach with the Graph Executor.

Memory Planning
----------------

The current memory planner attempts to limit the number of ``TVMBackendDeviceAlloc()`` calls
issued for intermediate tensors only. Because scratchpads can vary widely, and because the planner
coalesces memory allocations within 16x of each other, this strategy typically results in high
peak memory usage.

Heterogeneous Execution
-----------------------

Newer Cortex-M SoCs can contain multiple CPUs and onboard ML accelerators.


Autotuning Target
-----------------

As discussed previously,
