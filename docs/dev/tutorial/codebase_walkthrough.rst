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

=======================================
TVM Codebase Walkthrough by Example
=======================================

Getting to know a new codebase can be a challenge. This is especially true for a codebase like that of TVM, where different components interact in non-obvious ways. In this guide, we try to illustrate the key elements that comprise a compilation pipeline with a simple example. For each important step, we show where in the codebase it is implemented. The purpose is to let new developers and interested users dive into the codebase more quickly.

*******************************************
Codebase Structure Overview
*******************************************

At the root of the TVM repository, we have following subdirectories that together comprise a bulk of the codebase.

- ``src`` - C++ code for operator compilation and deployment runtimes.
- ``src/relay`` - Implementation of Relay, a new functional IR for deep learning framework.
- ``python`` - Python frontend that wraps C++ functions and objects implemented in ``src``.
- ``src/topi`` - Compute definitions and backend schedules for standard neural network operators.

Using standard Deep Learning terminology, ``src/relay`` is the component that manages a computational graph, and nodes in a graph are compiled and executed using infrastructure implemented in the rest of ``src``. ``python`` provides python bindings for the C++ API and driver code that users can use to execute compilation. Operators corresponding to each node are registered in ``src/relay/op``. Implementations of operators are in ``topi``, and they are coded in either C++ or Python.

When a user invokes graph compilation by ``relay.build(...)``, the following sequence of actions happens for each node in the graph:

- Look up an operator implementation by querying the operator registry
- Generate a compute expression and a schedule for the operator
- Compile the operator into object code

One of the interesting aspects of the TVM codebase is that interoperability between C++ and Python is not unidirectional. Typically, all code that performs heavy lifting is implemented in C++, and Python bindings are provided for the user interface. This is also true in TVM, but in the TVM codebase, C++ code can also call into functions defined in a Python module. For example, the convolution operator is implemented in Python, and its implementation is invoked from C++ code in Relay.

*******************************************
Vector Add Example
*******************************************

We use a simple example that uses the low level TVM API directly. The example is vector addition, which is covered in detail in :ref:`tutorial-tensor-expr-get-started`

::

   n = 1024
   A = tvm.te.placeholder((n,), name='A')
   B = tvm.te.placeholder((n,), name='B')
   C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")

Here, types of ``A``, ``B``, ``C`` are ``tvm.tensor.Tensor``, defined in ``python/tvm/te/tensor.py``. The Python ``Tensor`` is backed by C++ ``Tensor``, implemented in ``include/tvm/te/tensor.h`` and ``src/te/tensor.cc``. All Python types in TVM can be thought of as a handle to the underlying C++ type with the same name. If you look at the definition of Python ``Tensor`` type below, you can see it is a subclass of ``Object``.

::

   @register_object
   class Tensor(Object, _expr.ExprOp):
       """Tensor object, to construct, see function.Tensor"""

       def __call__(self, *indices):
          ...

The object protocol is the basis of exposing C++ types to frontend languages, including Python. The way TVM implements Python wrapping is not straightforward. It is briefly covered in :ref:`tvm-runtime-system`, and details are in ``python/tvm/_ffi/`` if you are interested.

We use the ``TVM_REGISTER_*`` macro to expose C++ functions to frontend languages, in the form of a :ref:`tvm-runtime-system-packed-func`. A ``PackedFunc`` is another mechanism by which TVM implements interoperability between C++ and Python. In particular, this is what makes calling Python functions from the C++ codebase very easy.
You can also checkout `FFI Navigator <https://github.com/tqchen/ffi-navigator>`_ which allows you to navigate between python and c++ FFI calls.

A ``Tensor`` object has an ``Operation`` object associated with it, defined in ``python/tvm/te/tensor.py``, ``include/tvm/te/operation.h``, and ``src/tvm/te/operation`` subdirectory. A ``Tensor`` is an output of its ``Operation`` object. Each ``Operation`` object has in turn ``input_tensors()`` method, which returns a list of input ``Tensor`` to it. This way we can keep track of dependencies between ``Operation``.

We pass the operation corresponding to the output tensor ``C`` to ``tvm.te.create_schedule()`` function in ``python/tvm/te/schedule.py``.

::

   s = tvm.te.create_schedule(C.op)

This function is mapped to the C++ function in ``include/tvm/schedule.h``.

::

   inline Schedule create_schedule(Array<Operation> ops) {
     return Schedule(ops);
   }

``Schedule`` consists of collections of ``Stage`` and output ``Operation``.

``Stage`` corresponds to one ``Operation``. In the vector add example above, there are two placeholder ops and one compute op, so the schedule ``s`` contains three stages. Each ``Stage`` holds information about a loop nest structure, types of each loop (``Parallel``, ``Vectorized``, ``Unrolled``), and where to execute its computation in the loop nest of the next ``Stage``, if any.

``Schedule`` and ``Stage`` are defined in ``tvm/python/te/schedule.py``, ``include/tvm/te/schedule.h``, and ``src/te/schedule/schedule_ops.cc``.

To keep it simple, we call ``tvm.build(...)`` on the default schedule created by ``create_schedule()`` function above, and we must add necessary thread bindings to make it runnable on GPU.

::

   target = "cuda"
   bx, tx = s[C].split(C.op.axis[0], factor=64)
   s[C].bind(bx, tvm.te.thread_axis("blockIdx.x"))
   s[C].bind(tx, tvm.te.thread_axis("threadIdx.x"))
   fadd = tvm.build(s, [A, B, C], target)

``tvm.build()``, defined in ``python/tvm/driver/build_module.py``, takes a schedule, input and output ``Tensor``, and a target, and returns a :py:class:`tvm.runtime.Module` object. A :py:class:`tvm.runtime.Module` object contains a compiled function which can be invoked with function call syntax.

The process of ``tvm.build()`` can be divided into two steps:

- Lowering, where a high level, initial loop nest structures are transformed into a final, low level IR
- Code generation, where target machine code is generated from the low level IR

Lowering is done by ``tvm.lower()`` function, defined in ``python/tvm/build_module.py``. First, bound inference is performed, and an initial loop nest structure is created.

::

   def lower(sch,
             args,
             name="default_function",
             binds=None,
             simple_mode=False):
      ...
      bounds = schedule.InferBound(sch)
      stmt = schedule.ScheduleOps(sch, bounds)
      ...

Bound inference is the process where all loop bounds and sizes of intermediate buffers are inferred. If you target the CUDA backend and you use shared memory, its required minimum size is automatically determined here. Bound inference is implemented in ``src/te/schedule/bound.cc``, ``src/te/schedule/graph.cc`` and ``src/te/schedule/message_passing.cc``. For more information on how bound inference works, see :ref:`dev-InferBound-Pass`.


``stmt``, which is the output of ``ScheduleOps()``, represents an initial loop nest structure. If you have applied ``reorder`` or ``split`` primitives to your schedule, then the initial loop nest already reflects those changes. ``ScheduleOps()`` is defined in ``src/te/schedule/schedule_ops.cc``.

Next, we apply a number of lowering passes to ``stmt``. These passes are implemented in ``src/tir/pass`` subdirectory. For example, if you have applied ``vectorize`` or ``unroll`` primitives to your schedule, they are applied in loop vectorization and unrolling passes below.

::

     ...
     stmt = ir_pass.VectorizeLoop(stmt)
     ...
     stmt = ir_pass.UnrollLoop(
         stmt,
         cfg.auto_unroll_max_step,
         cfg.auto_unroll_max_depth,
         cfg.auto_unroll_max_extent,
         cfg.unroll_explicit)
     ...

After lowering is done, ``build()`` function generates target machine code from the lowered function. This code can contain SSE or AVX instructions if you target x86, or PTX instructions for CUDA target. In addition to target specific machine code, TVM also generates host side code that is responsible for memory management, kernel launch etc.

Code generation is done by ``build_module()`` function, defined in ``python/tvm/target/codegen.py``. On the C++ side, code generation is implemented in ``src/target/codegen`` subdirectory. ``build_module()`` Python function will reach ``Build()`` function below in ``src/target/codegen/codegen.cc``:



The ``Build()`` function looks up the code generator for the given target in the ``PackedFunc`` registry, and invokes the function found. For example, ``codegen.build_cuda`` function is registered in ``src/codegen/build_cuda_on.cc``, like this:

::

   TVM_REGISTER_GLOBAL("codegen.build_cuda")
   .set_body([](TVMArgs args, TVMRetValue* rv) {
       *rv = BuildCUDA(args[0]);
     });

The ``BuildCUDA()`` above generates CUDA kernel source from the lowered IR using ``CodeGenCUDA`` class defined in ``src/codegen/codegen_cuda.cc``, and compile the kernel using NVRTC. If you target a backend that uses LLVM, which includes x86, ARM, NVPTX and AMDGPU, code generation is done primarily by ``CodeGenLLVM`` class defined in ``src/codegen/llvm/codegen_llvm.cc``. ``CodeGenLLVM`` translates TVM IR into LLVM IR, runs a number of LLVM optimization passes, and generates target machine code.

The ``Build()`` function in ``src/codegen/codegen.cc`` returns a ``runtime::Module`` object, defined in ``include/tvm/runtime/module.h`` and ``src/runtime/module.cc``. A ``Module`` object is a container for the underlying target specific ``ModuleNode`` object. Each backend implements a subclass of ``ModuleNode`` to add target specific runtime API calls. For example, the CUDA backend implements ``CUDAModuleNode`` class in ``src/runtime/cuda/cuda_module.cc``, which manages the CUDA driver API. The ``BuildCUDA()`` function above wraps ``CUDAModuleNode`` with ``runtime::Module`` and return it to the Python side. The LLVM backend implements ``LLVMModuleNode`` in ``src/codegen/llvm/llvm_module.cc``, which handles JIT execution of compiled code. Other subclasses of ``ModuleNode`` can be found under subdirectories of ``src/runtime`` corresponding to each backend.

The returned module, which can be thought of as a combination of a compiled function and a device API, can be invoked on TVM's NDArray objects.

::

   dev = tvm.device(target, 0)
   a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
   b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
   c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
   fadd(a, b, c)
   output = c.numpy()

Under the hood, TVM allocates device memory and manages memory transfers automatically. To do that, each backend needs to subclass ``DeviceAPI`` class, defined in ``include/tvm/runtime/device_api.h``, and override memory management methods to use device specific API. For example, the CUDA backend implements ``CUDADeviceAPI`` in ``src/runtime/cuda/cuda_device_api.cc`` to use ``cudaMalloc``, ``cudaMemcpy`` etc.

The first time you invoke the compiled module with ``fadd(a, b, c)``, ``GetFunction()`` method of ``ModuleNode`` is called to get a ``PackedFunc`` that can be used for a kernel call. For example, in ``src/runtime/cuda/cuda_module.cc`` the CUDA backend implements ``CUDAModuleNode::GetFunction()`` like this:

::

   PackedFunc CUDAModuleNode::GetFunction(
         const std::string& name,
         const std::shared_ptr<ModuleNode>& sptr_to_self) {
     auto it = fmap_.find(name);
     const FunctionInfo& info = it->second;
     CUDAWrappedFunc f;
     f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
     return PackFuncVoidAddr(f, info.arg_types);
   }

The ``PackedFunc``'s overloaded ``operator()`` will be called, which in turn calls ``operator()`` of ``CUDAWrappedFunc`` in ``src/runtime/cuda/cuda_module.cc``, where finally we see the ``cuLaunchKernel`` driver call:

::

   class CUDAWrappedFunc {
    public:
     void Init(...)
     ...
     void operator()(TVMArgs args,
                     TVMRetValue* rv,
                     void** void_args) const {
       int device_id;
       CUDA_CALL(cudaGetDevice(&device_id));
       if (fcache_[device_id] == nullptr) {
         fcache_[device_id] = m_->GetFunc(device_id, func_name_);
       }
       CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
       ThreadWorkLoad wl = launch_param_config_.Extract(args);
       CUresult result = cuLaunchKernel(
           fcache_[device_id],
           wl.grid_dim(0),
           wl.grid_dim(1),
           wl.grid_dim(2),
           wl.block_dim(0),
           wl.block_dim(1),
           wl.block_dim(2),
           0, strm, void_args, 0);
     }
   };

This concludes an overview of how TVM compiles and executes a function. Although we did not detail TOPI or Relay, in the end, all neural network operators go through the same compilation process as above. You are encouraged to dive into the details of the rest of the codebase.
