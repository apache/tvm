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

Design and Architecture
=======================

This document is intended for developers who want to understand the
architecture of TVM and/or actively develop on the project.
This page is organized as follows:

- The `Example Compilation Flow`_ gives an overview of the steps that TVM takes to turn a high level description of a model into a deployable module.
  To get started, please read this section first.

- The `Logical Architecture Components`_ section describes the logical components.
  The sections after are specific guides focused on each logical component, organized
  by the component's name.

- The :ref:`Device/Target Interactions <tvm-target-specific-overview>`
  page describes how TVM interacts with each supported physical device
  and code-generation target.

- Feel free to also check out the :ref:`dev-how-to` for useful development tips.

This guide provides a few complementary views of the architecture.
First, we review a single end-to-end compilation flow and discuss the key data structures and the transformations.
This runtime-based view focuses on the interactions of each components when running the compiler.
Then we will review the logical modules of the codebase and their relationship. This part provides a static overarching view of the design.


Example Compilation Flow
------------------------

In this guide, we will study an example compilation flow in the compiler. The figure below shows the flow. At a high-level, it contains several steps:

- Import: The frontend component ingests a model into an IRModule, which contains a collection of functions that internally represent the model.
- Transformation: The compiler transforms an IRModule to another functionally equivalent or approximately
  equivalent(e.g. in the case of quantization) IRModule. Many of the transformations are target (backend) independent.
  We also allow target to affect the configuration of the transformation pipeline.
- Target Translation: The compiler translates(codegen) the IRModule to an executable format specified by the target.
  The target translation result is encapsulated as a `runtime.Module` that can be exported, loaded, and executed on the target runtime environment.
- Runtime Execution: the user loads back a `runtime.Module` and runs the compiled functions in the supported runtime environment.


.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_dyn_workflow.svg
   :align: center
   :width: 85%


Key data structures
~~~~~~~~~~~~~~~~~~~

One of the best ways to design and understand a complex system is to identify the key data structures and APIs that
manipulate (transform) these data structures. Once we identified the key data structures, we can then breakdown a system into logical
components that either define a collection of key data structures or transformations among the data structures.

**IRModule** is the primary data structure used across the entire stack. An IRModule (intermediate representation module)
contains a collection of functions. Currently, we support two primary variants of functions.

- **relay::Function** is a high-level functional program representation. A relay.Function usually corresponds to an end-to-end model.
  You can view a relay.Function as a computational graph with additional support for control-flow, recursion, and complex data structures.
- **tir::PrimFunc** is a low-level program representation that contains elements including loop-nest choices, multi-dimensional load/store,
  threading, and vector/tensor instructions. It is usually used to represent an operator program that executes a (possibly-fused) layer in a model.

During the compilation, a relay function may be lowered to multiple tir::PrimFunc functions and a top-level function that calls into
those tir::PrimFunc functions.

Transformations
~~~~~~~~~~~~~~~

Now that we have covered the key data structures, let us talk about the transformations. Each transformation could serve one of the following purposes:

- optimization: transform a program to an equivalent, possibly more optimized version.
- lowering: transform a program to a lower-level representation that is closer to the target.

**relay/transform** contains a collection of passes that optimize the model. The optimizations include common program
optimizations such as constant folding and dead-code elimination, and tensor-computation specific passes such as layout
transformation and scaling factor folding.

Near the end of the relay optimization pipeline, we will run a pass(FuseOps) to break the end-to-end function(e.g. MobileNet)
into sub-function(e.g. conv2d-relu) segments. We call these segments of functions.
This process helps us to divide the original problem into two sub-problems:

- Compilation and optimization for each sub-function.
- Overall execution structure: we need to do a sequence of calls into the generated sub-functions to execute the whole model.

We use the low-level tir phase to compile and optimize each sub-functions. For specific targets, we may also directly go to the target translation
phase and use external code generators.

There are a few different ways(in relay/backend) to handle the calls into the overall execution problem. For simple models with known shapes and no control flow, we can lower to a graph executor that stores the execution structure in a graph. We also support a virtual machine backend for dynamic executions. Finally, we plan to support ahead of time compilation that compiles the high-level execution structure into the executable and generated primitive functions. All of these execution modes are encapsulated by a unified **runtime.Module** interface, which we will discuss in the latter part of the guide.

**tir/transform** contains transformation passes for TIR level functions. Many tir passes serve the purpose of lowering. For example, there are passes to flatten multi-dimensional access to one-dimensional pointer access, to expand the intrinsics into target-specific ones, and to decorate the function entry to meet the runtime calling convention. Of course, there are also optimizations passes, such as access index simplification and dead code elimination.

Many low-level optimizations can be handled in the target phase by the LLVM, CUDA C, and other target compilers. As a result, we leave low-level optimizations such as register allocation to the downstream compilers and only focus on optimizations that are not covered by them.

Search-space and Learning-based Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformation passes we described so far are deterministic and rule-based. One design goal of the TVM stack is to support high-performance code optimizations for different hardware platforms. To do so, we will need to investigate as many optimization choices as possible, including but not limited to, multi-dimensional tensor access, loop tiling behavior, special accelerator memory hierarchy, and threading.

It is hard to define a heuristic to make all of the choices. Instead, we will take a search and learning-based approach.
We first define a collection of actions we can take to transform a program. Example actions include loop transformations, inlining,
vectorization. We call these actions **scheduling primitives**. The collection of scheduling primitives defines a search space of possible
optimizations we can make to a program. The system then searches over different possible scheduling
sequence to pick the best scheduling combination.
The search procedure is usually guided by a machine learning algorithm.

We can record the best schedule sequence for an (possibly-fused) operator once the search is completed. The compiler can then just lookup the best
schedule sequence and apply it to the program. Notably, this schedule application phase is **exactly like** the rule-based transformations,
enabling us to share the same interface convention with tradition passes.

We use search based optimizations to handle the initial tir function generation problem. This part of the module is called AutoTVM(auto_scheduler).
We expect to expand the learning-based transformations to more areas as we continue to develop the TVM stack.

Target Translation
~~~~~~~~~~~~~~~~~~

The target translation phase transforms an IRModule to the corresponding target executable format.
For backends such as x86 and ARM, we use the LLVM IRBuilder to build in-memory LLVM IR.
We can also generate source-level languages such as CUDA C and OpenCL.
Finally, we support direct translations of a Relay function (sub-graph) to specific targets via external code generators.
It is important that the final code generation phase is as lightweight as possible. Vast majority of transformations
and lowering should be performed before the target translation phase.

We also provide a Target structure to specify the compilation target.
The transformations before the target translation phase can also be affected by the target — for example,
a target's vector length would change the vectorization behavior.


Runtime Execution
~~~~~~~~~~~~~~~~~

The main goal of TVM's runtime is to provide a minimal API for loading and executing the compiled artifact in a language of their choice, including Python, C++, Rust, Go, Java, and JavaScript. The code snippet below shows such an example in Python:

.. code-block:: python

    import tvm
    # Example runtime execution program in python, with type annotated
    mod: tvm.runtime.Module = tvm.runtime.load_module("compiled_artifact.so")
    arr: tvm.runtime.NDArray = tvm.nd.array([1, 2, 3], device=tvm.cuda(0))
    fun: tvm.runtime.PackedFunc = mod["addone"]
    fun(arr)
    print(arr.numpy())


:py:class:`tvm.runtime.Module` encapsulates the result of compilation. A runtime.Module contains a GetFunction method to obtain PackedFuncs by name.

:py:class:`tvm.runtime.PackedFunc` is a type-erased function interface for both the generated functions. A runtime.PackedFunc can take arguments and return values with the
following types: POD types(int, float), string, runtime.PackedFunc, runtime.Module, runtime.NDArray, and other sub-classes of runtime.Object.

:py:class:`tvm.runtime.Module` and :py:class:`tvm.runtime.PackedFunc` are powerful mechanisms to modularize the runtime. For example, to get the above `addone` function on CUDA, we can use LLVM to generate the host-side code to compute the launching parameters(e.g. size of the thread groups) and then call into another PackedFunc from a CUDAModule that is backed by the CUDA driver API. The same mechanism can be used for OpenCL kernels.

The above example only deals with a simple `addone` function. The code snippet below gives an example of an end-to-end model execution using the same interface:

.. code-block:: python

   import tvm
   # Example runtime execution program in python, with types annotated
   factory: tvm.runtime.Module = tvm.runtime.load_module("resnet18.so")
   # Create a stateful graph execution module for resnet18 on cuda(0)
   gmod: tvm.runtime.Module = factory["resnet18"](tvm.cuda(0))
   data: tvm.runtime.NDArray = get_input_data()
   # set input
   gmod["set_input"](0, data)
   # execute the model
   gmod["run"]()
   # get the output
   result = gmod["get_output"](0).numpy()

The main take away is that runtime.Module and runtime.PackedFunc are sufficient to encapsulate both operator level programs (such as addone), as well as the end-to-end models.

Summary and Discussions
~~~~~~~~~~~~~~~~~~~~~~~

In summary, the key data structures in the compilation flows are:

- IRModule: contains relay.Function and tir.PrimFunc
- runtime.Module: contains runtime.PackedFunc

Most parts of the compilation are transformations among the key data structures.

- relay/transform and tir/transform are determinstic rule-based transformations
- auto_scheduler and autotvm contains the search-based transformations

Finally, the compilation flow example is only a typical use-case of the TVM stack.
We expose these key data structures and transformations to python and C++ APIs. As a result, you can use TVM just like the way you use numpy,
except that the data structure of interest changes from the numpy.ndarray to tvm.IRModule. Here are some example use-cases:

- Directly construct IRModule using the python API.
- Compose a custom set of transformations(e.g. customize quantization).
- Manipulate the IR directly using TVM's python API.


Logical Architecture Components
-------------------------------

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_static_overview.svg
   :align: center
   :width: 85%

   TVM Architecture Diagram

The above figure shows the major logical components in the project. Please read the following sections
for information about the components and their relations.


tvm/support
-----------
The support module contains the most common utilities for the infrastructure, such as generic arena allocator, socket, and logging.


tvm/runtime
-----------

The runtime serves as the foundation of the TVM stack. It provides the mechanism to load and execute compiled artifacts.
The runtime defines a stable standard set of C APIs to interface with frontend languages such as Python and Rust.

`runtime::Object` is one of the primary data structures in TVM runtime besides the `runtime::PackedFunc`.
It is a reference-counted base class with a type index to support runtime type checking and downcasting.
The object system allows the developer to introduce new data structures to the runtime, such as Array, Map, and new IR data structures.

Besides deployment use-cases, the compiler itself also makes heavy use of TVM's runtime mechanism.
All of the IR data structures are subclasses of `runtime::Object`, as a result, they can be directly accessed and manipulated from the Python frontend.
We use the PackedFunc mechanism to expose various APIs to the frontend.

Runtime support for different hardware backends are defined in subdirectories of runtime(e.g. runtime/opencl).
These hardware-specific runtime modules define APIs for device memory allocation and device function serialization.

`runtime/rpc` implements an RPC support for PackedFunc. We can use the RPC mechanism to send a cross-compiled library to a remote
device and benchmark the execution performance. The rpc infrastructure enables data collection from a wide range of hardware backends
for learning-based optimizations.


.. toctree::
   :maxdepth: 1

   runtime


.. toctree::
   :maxdepth: 1

   debugger
   virtual_machine
   introduction_to_module_serialization
   device_target_interactions



tvm/node
--------
The node module adds additional features on top of the `runtime::Object` for IR data structures.
The main features include reflection, serialization, structural equivalence, and hashing.

Thanks to the node module, we can directly access any field of the TVM's IRNode by their name in Python.

.. code-block:: python

    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Add(x, x)
    # a and b are fields of a tir.Add node
    # we can directly use the field name to access the IR structures
    assert y.a == x


We can also serialize arbitrary IR node into a JSON format, and load them back.
The ability to save/store, and inspect an IR node provides a foundation for making the compiler more accessible.


tvm/ir
------
The `tvm/ir` folder contains the unified data structure and interfaces across for all IR function variants.
The components in `tvm/ir` are shared by `tvm/relay` and `tvm/tir`, notable ones include

- IRModule
- Type
- PassContext and Pass
- Op

Different variants of functions(e.g. relay.Function and tir.PrimFunc) can co-exist in an IRModule.
While these variants may not have the same content representation, they use the same data structure to represent types.
As a consequence, we use the same data structure to represent function (type) signatures of these variants.
The unified type system allows one function variant to call another function
once we clearly define the calling convention. This opens doors for future cross-function-variant optimizations.

We also provide a unified PassContext for configuring the pass behavior, and common composite passes to execute a pass pipeline.
The following code snippet gives an example of PassContext configuration.

.. code-block:: python

    # configure the behavior of the tir.UnrollLoop pass
    with tvm.transform.PassContext(config={"tir.UnrollLoop": { "auto_max_step": 10 }}):
        # code affected by the pass context


Op is the common class to represent all system-defined primitive operator/intrinsics.
Developers can register new Ops as well as their additional attributes(e.g. whether the Op is elementwise) to the system.

.. toctree::
   :maxdepth: 1

   pass_infra


tvm/target
----------
The target module contains all the code generators that translate an IRModule to a target runtime.Module.
It also provides a common `Target` class that describes the target.

.. TODO(tvm-team) add a target json description example once the new target API stablizes.


The compilation pipeline can be customized according to the target by querying the attribute information
in the target and builtin information registered to each target id(cuda, opencl).

.. toctree::
   :maxdepth: 1

   device_target_interactions

tvm/tir
-------

TIR contains the definition of the low-level program representations. We use `tir::PrimFunc` to represent functions that can be transformed by TIR passes.
Besides the IR data structures, the tir module also defines a set of builtin intrinsics and their attributes via the common Op registry, as well as transformation passes in `tir/transform`.

tvm/arith
---------

This module is closely tied to the TIR. One of the key problems in the low-level code generation is the analysis of the indices'
arithmetic properties — the positiveness, variable bound, and the integer set that describes the iterator space. arith module provides
a collection of tools that do (primarily integer) analysis. A TIR pass can use these analyses to simplify and optimize the code.

tvm/te
------

The name te stands for "tensor expression". This is a domain-specific language module that allows us to construct `tir::PrimFunc` variants quickly by writing tensor expressions.
Importantly, a tensor expression itself is not a self-contained function that can be stored into IRModule. Instead, it is a fragment of IR that we can stitch together to build an IRModule.

`te/schedule` provides a collection of scheduling primitives to control the function being generated. In the future, we might bring some of
these scheduling components to the a `tir::PrimFunc` itself.

.. toctree::
   :maxdepth: 1

   inferbound
   hybrid_script

tvm/topi
--------
While possible to construct operators directly via TIR or tensor expressions (TE) for each use case it is tedious to do so.
`topi` (Tensor operator inventory) provides a set of pre-defined operators (in TE or TIR) defined by
numpy and found in common deep learning workloads. We also provide a collection of common schedule templates to obtain performant implementations across different target platforms.


tvm/relay
---------
Relay is the high-level functional IR used to represent full models. Various optimizations are defined in `relay.transform`. The Relay compiler defines multiple dialects,
and each dialect is designed to support specific styles of optimization. Notable ones include QNN(for importing pre-quantized models), VM(for lowering to dynamic virtual machine),
memory(for memory optimization).

.. toctree::
   :maxdepth: 1

   relay_intro
   relay_op_strategy
   convert_layout


tvm/autotvm
-----------

AutoTVM and AutoScheduler are both components which automate search based program optimization. This is rapidly evolving and primarily consists of:

- Cost models and feature extraction.
- A record format for storing program benchmark results for cost model construction.
- A set of search policies over program transformations.

Automated program optimization is still an active research field. As a result, we have attempted to modularize the design so that researchers may quickly modify a
component or apply their own algorithms via the Python bindings, and
customize the search and plugin their algorithms from the Python binding.

.. toctree::
   :maxdepth: 1

   benchmark

Frontends
---------
Frontends ingest models from different frameworks into the TVM stack.
:py:mod:`tvm.relay.frontend` is the namespace for model ingestion APIs.

.. toctree::
   :maxdepth: 1

   frontend/tensorflow


Security
---------
.. toctree::
   :maxdepth: 1

   security


microTVM
--------
.. toctree::
   :maxdepth: 1

   microtvm_design
   microtvm_project_api
   model_library_format
