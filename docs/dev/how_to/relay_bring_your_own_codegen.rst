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

.. _relay-bring-your-own-codegen:

=============================
Bring Your Own Codegen To TVM
=============================

As the number of hardware devices targeted by deep learning workloads keeps increasing, the required knowledge for users to achieve high performance on various devices keeps increasing as well. To free data scientists from worrying about the performance when developing a new model, hardware backend providers either provide libraries such as DNNL(Intel OneDNN) or cuDNN with many commonly used deep learning operators, or provide frameworks such as TensorRT to let users describe their models in a certain way to achieve high performance. However, users have to learn a new programming interface when they attempt to work on a new library or device. As a result, the demand for a unified programming interface becomes more and more important to 1) let all users and hardware backend providers stand on the same page, and 2) provide a feasible solution to allow specialized hardware or library to only support widely used operators with extremely high performance, but fallback unsupported operators to general devices like CPU/GPU.

In this developer guide, we demonstrate how you, as a hardware backend provider, can easily implement your own codegen and register it as a Relay backend compiler to support your hardware device/library. This guide covers two types of codegen based on different graph representations you need:

**1. You want to generate C code.**

If your hardware already has a well-optimized C/C++ library, such as Intel CBLAS/MKL to CPU and NVIDIA CUBLAS to GPU, then this is what you are looking for. Fortunately, C source code module is fully compatible with TVM runtime module, which means the generated code could be compiled by any C/C++ compiler with proper compilation flags, so the only task you have is to implement a codegen that generates C code for subgraphs and a C source module to integrate into TVM runtime module. We will demonstrate how to implement a C code generator for your hardware in the following section.

**2. You want to generate any other graph representations.**

Your hardware may require other forms of graph representation, such as JSON. In this case, you need to implement not only a codegen but also a customized TVM runtime module to let TVM runtime know how this graph representation should be executed. If you already have a complete graph execution engine for your hardware, such as TensorRT for GPU, then this is a solution you can consider.

After you finish the codegen and runtime, you can then let your customers annotate their models with your customized tag to make use of them. The tutorial for end-users to annotate and launch a specific codegen is **here (TBA)**.

*********************
Implement a C Codegen
*********************

In this part, we demonstrate how to implement a codegen that generates C code with pre-implemented operator functions. To simplify, our example codegen does not depend on third-party libraries. Instead, we manually implement two macros in C:

.. code-block:: c++

    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)         \
        extern "C" void p_ID_(float* a, float* b, float* out) { \
            for (int64_t i = 0; i < p_DIM1_; ++i) {             \
                out[i] = a[i] p_OP_ b[i];                       \
            }                                                   \
        }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
        extern "C" void p_ID_(float* a, float* b, float* out) {   \
            for (int64_t i = 0; i < p_DIM1_; ++i) {               \
                for (int64_t j = 0; j < p_DIM2_; ++j) {           \
                    int64_t k = i * p_DIM2_ + j;                  \
                    out[k] = a[k] p_OP_ b[k];                     \
                }                                                 \
            }                                                     \
        }

With the two macros, we can generate binary operators for 1-D and 2-D tensors. For example, given a subgraph as follows. Assuming all inputs are 2-D tensors with shape (10, 10).

::

    c_compiler_input0
           |
          add <-- c_compiler_input1
           |
        subtract <-- c_compiler_input2
           |
        multiply <-- c_compiler_input3
           |
          out

Our goal is to generate the following compilable code to execute the subgraph:

.. code-block:: c++

    #include <tvm/runtime/c_runtime_api.h>
    #include <tvm/runtime/packed_func.h>
    #include <dlpack/dlpack.h>
    #include <cstdint>
    #include <cstring>
    #include <iostream>

    #define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          out[i] = a[i] p_OP_ b[i];                           \
        }                                                     \
      }

    #define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          for (int64_t j = 0; j < p_DIM2_; ++j) {             \
            int64_t k = i * p_DIM2_ + j;                      \
            out[k] = a[k] p_OP_ b[k];                         \
          }                                                   \
        }                                                     \
      }

    // Note 1
    GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
    GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
    GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);

    // Note 2
    extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1,
                           float* gcc_input2, float* gcc_input3, float* out) {
      float* buf_0 = (float*)malloc(4 * 100);
      float* buf_1 = (float*)malloc(4 * 100);
      gcc_0_2(gcc_input0, gcc_input1, buf_0);
      gcc_0_1(buf_0, gcc_input2, buf_1);
      gcc_0_0(buf_1, gcc_input3, out);
      free(buf_0);
      free(buf_1);
    }

    // Note 3
    extern "C" int gcc_0_wrapper(DLTensor* arg0, DLTensor* arg1, DLTensor* arg2,
                                 DLTensor* arg3, DLTensor* out) {
      gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
             static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
             static_cast<float*>(out->data));
      return 0;
    }
    TVM_DLL_EXPORT_TYPED_FUNC(gcc_0, gcc_0_wrapper);

Here we highlight the notes marked in the above code:

* **Note 1** is the function implementation for the three nodes in the subgraph.

* **Note 2** is a function to execute the subgraph by allocating intermediate buffers and invoking corresponding functions.

* **Note 3** is a TVM runtime compatible wrapper function. It accepts a list of input tensors and one output tensor (the last argument), casts them to the right data type, and invokes the subgraph function described in Note 2. In addition, ``TVM_DLL_EXPORT_TYPED_FUNC`` is a TVM macro that generates another function ``gcc_0`` with unified the function arguments by packing all tensors to ``TVMArgs``. As a result, the TVM runtime can directly invoke ``gcc_0`` to execute the subgraph without additional efforts. With the above code generated, TVM is able to compile it along with the rest parts of the graph and export a single library for deployment.

In the rest of this section, we will implement a codegen step-by-step to generate the above code. Your own codegen has to be located at ``src/relay/backend/contrib/<your-codegen-name>/``. In our example, we name our codegen "codegen_c" and put it under `/src/relay/backend/contrib/codegen_c/ <https://github.com/apache/tvm/blob/main/src/relay/backend/contrib/codegen_c/codegen.cc>`_. Feel free to check this file for a complete implementation.

Specifically, we are going to implement two classes in this file and here is their relationship:

::

                       subgraph                                subgraph
  TVM backend -----------------------------> CSourceCodegen -------------> CodegenC
         ^                                       |    ^                       |
         |                                       |    |                       |
         ----------------------------------------      ------------------------
            generated C source runtime module              generated C code

When TVM backend finds a function (subgraph) in a Relay graph is annotated with the registered compiler tag (``ccompiler`` in this example), TVM backend invokes ``CSourceCodegen`` and passes the subgraph. ``CSourceCodegen``'s member function ``CreateCSourceModule`` will 1) generate C code for the subgraph, and 2) wrap the generated C code to a C source runtime module for TVM backend to compile and deploy. In particular, the C code generation is transparent to the ``CodegenC`` class because it provides many useful utilities to ease the code generation implementation. The following sections will implement these two classes in the bottom-up order.

Implement CodegenC
==================

In ``src/relay/backend/contrib/codegen_c/codegen.cc``, we first create a codegen class skeleton under the namespace of ``tvm.relay.contrib``:

.. code-block:: c++

    #include <tvm/relay/expr_functor.h>
    #include <tvm/relay/transform.h>
    #include <tvm/relay/type.h>
    #include <tvm/runtime/module.h>
    #include <tvm/runtime/object.h>

    #include <fstream>
    #include <sstream>

    #include "codegen_c.h"

    namespace tvm {
    namespace relay {
    namespace contrib {

    class CodegenC : public ExprVisitor, public CodegenCBase {
      public:
        explicit CodegenC(const std::string& id) { this->ext_func_id_ = id; }

        void VisitExpr_(const VarNode* node) { ; }
        void VisitExpr_(const CallNode* call) final { ; }
        std::string JIT() { ; }

      private:
        /*! \brief The function id that represents a C source function. */
        std::string ext_func_id_ = "";
        /*! \brief The index of a wrapped C function. */
        int func_idx = 0;
        /*! \brief The index of allocated buffers. */
        int buf_idx_ = 0;
        /*! \brief The arguments of a C compiler compatible function. */
        std::vector<std::string> ext_func_args_;
        /*! \brief The statements of a C compiler compatible function. */
        std::vector<std::string> ext_func_body;
        /*! \brief The declaration statements of a C compiler compatible function. */
        std::vector<std::string> func_decl_;
        /*! \brief The declaration statements of buffers. */
        std::vector<std::string> buf_decl_;
        /*! \brief The name and index pairs for output. */
        std::vector<std::pair<std::string, int>> out_;
    }

The ``CodegenC`` class inherits two classes: ``ExprVisitor`` provides abilities to traverse subgraphs and collects the required information and generate subgraph functions such as ``gcc_0_``; ``CodegenCBase`` provides abilities and utilities to generate wrapper functions such as ``gcc_0`` in the above example. As can be seen, we only need to implement three functions in this codegen class to make it work.

Code Generation for Operators
-----------------------------

We first implement ``VisitExpr_(const CallNode* call)``. This function visits all call nodes when traversing the subgraph. Each call node contains an operator that we want to offload to your hardware. As a result, we need to generate the corresponding C code with correct operators in topological order. We implement this function step-by-step as follows.

**1. Generate the function declaration**

Example Result: ``GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);``

To generate the function declaration, as shown above, we need 1) a function name (e.g., ``gcc_0_0``), 2) the type of operator (e.g., ``*``), and 3) the input tensor shape (e.g., ``(10, 10)``). Fortunately, this information can be obtained easily from ``CallNode``:

.. code-block:: c++

    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    // Generate a unique function name you like.
    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration string.
    macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

    // Check the operator type.
    if (IsOp(call, "add")) {
      macro_stream << "+";
    } else if (IsOp(call, "subtract")) {
      macro_stream << "-";
    } else if (IsOp(call, "multiply")) {
      macro_stream << "*";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    // Extract the input tensor shape.
    auto in_shape = GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      macro_stream << ", " << in_shape[i];
    }
    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

As can be seen, we push the generated code to class member variables ``func_decl_``. It means after we finish traversing the entire subgraph, we have collected all required function declarations and the only thing we need to do is having them compiled by GCC. The rest implementation of ``VisitExpr_(const CallNode* call)`` also follow this concept.

**2. Generate the function call**

Example Result: ``gcc_0_0(buf_1, gcc_input3, out);``

After generating the function declaration, we need to generate a function call with proper inputs and outputs. To know which inputs or buffers we should put when calling this function, we have to visit its arguments:

.. code-block:: c++

    bool first = true;
    decl_stream << func_name << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      VisitExpr(call->args[i]); // Note 1
      for (auto out : out_) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.first;
      }
    }
    // Note 2

Again, we want to highlight the notes in the above code:

**Note 1**: ``VisitExpr(call->args[i])`` is a recursive call to visit arguments of the current function. An argument could be an output of another node or an input tensor. In our example implementation, we make sure every node updates a class variable ``out_`` before leaving the visitor. Here is an illustration:

::

        arg_node                 arg_node <- Visit arg (Note 1)       arg_node
           |                        |                                    |
       curr_node <- Process      curr_node                            curr_node <- Put "buf_0" as an input buffer

      (a) out_ = {}            (b) out_ = {}                   (c) out_ = {("buf_0", 20)}


We can see in the above figure, class variable ``out_`` is empty before visiting the argument node, and it was filled with the output buffer name and size of ``arg_node``. As a result, when we finished visiting the argument node, we know the proper input buffer we should put by looking at ``out_``. You will find out how we update ``out_`` at the end of this section as well as the next section.

**Note 2**: You may notice that we did not close the function call string in this step. The current function call string looks like: ``gcc_0_0(buf_1, gcc_input3``. This is because we have not put the last argument (i.e., the output) to this call. The output of a function call could be either an allocated temporary buffer or the subgraph output tensor. For simplify, in this example, we allocate an output buffer for every call node (next step) and copy the result in the very last buffer to the output tensor.

**3. Generate the output buffer**

Example Result: ``float* buf_0 = (float*)malloc(4 * 100);``

As mentioned in the previous step, in addition to the subgraph input and output tensors, we may also need buffers to keep the intermediate results. To generate the buffer, we extract the shape information to determine the buffer type and size:

.. code-block:: c++

    // This example only supports single output.
    auto type_node = call->checked_type().as<TensorTypeNode>();
    ICHECK(type_node != nullptr && runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
          << "Only support single output tensor with float type";

    // Generate a unique buffer name.
    std::string out = "buf_" + std::to_string(buf_idx_++);

    // Extract the shape to be the buffer size.
    auto out_shape = GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }

    // Make the buffer allocation and push to the buffer declarations.
    buf_stream << "float* " << out << " = (float*)std::malloc(4 * " << out_size << ");";
    buf_decl_.push_back(buf_stream.str());

After we have allocated the output buffer, we can now close the function call string and push the generated function call to a class variable ``ext_func_body``.

.. code-block:: c++

    decl_stream << ", " << out << ");";
    ext_func_body.push_back(decl_stream.str());

**4. Update output buffer**

To let the next node, which accepts the output of the current call node as its input, know which buffer it should take, we need to update the class variable ``out_`` before leaving this visit function:

.. code-block:: c++

    out_.clear();
    out_.push_back({out, out_size});

Congratulations! we have finished the most difficult function in this class. In the next two sections, we just need to make up some minor missing parts in this function.

Code Generation for Input Variables
-----------------------------------

Recall that we collected the input buffer information by visiting the arguments of a call node (2nd step in the previous section), and handled the case when its argument is another call node (4th step). In this section, we demonstrate how to handle other nodes by taking ``VarNode`` as an example.

``VarNode`` represents input tensors in a model. The only but important information it has is a name hint (e.g., ``data``, ``weight``, etc). When visiting a ``VarNode``, we simply update class variable ``out_`` to pass the name hint so that the descendant call nodes can generate the correct function call.

.. code-block:: c++

  void VisitExpr_(const VarNode* node) {
    ext_func_args_.push_back(node->name_hint());
    out_.clear();
    out_.push_back({node->name_hint(), 0});
  }

Note that in this example we assume the subgraph we are offloading has only call nodes and variable nodes. If your subgraphs contain other types of nodes, such as ``TupleNode``, then you also need to visit them and bypass the output buffer information.

Code Emitting
-------------

The final part in this codegen class is a ``JIT`` function that emits a C function for the subgraph and uses the C code we just generated as the function body. Remember, in addition to the subgraph function we generated in the previous sections, we also need a wrapper function with a unified argument for TVM runtime to invoke and pass data. Fortunately, the base class we inherited already provides an implementation, ``JitImpl``, to generate the function. For example, we can invoke ``JitImpl`` as follows:

.. code-block:: c++

  JitImpl("gcc_0" /* Subgraph symbol (ID) */,
          {"gcc_input0", "gcc_input1", "gcc_input2", "gcc_input3"} /* Input arguments */,
          {"float *buf_0 = (float*)malloc(4 * 20)", ...} /* Buffer allocations */,
          {"gcc_0_2(gcc_input0, gcc_input1, buf_0);"} /* Function body */,
          {"out"} /* Output */);

The above call will generate three functions (one from the TVM wrapper macro):

1. The subgraph function ``gcc_0_`` (with one more underline at the end of the function name) with all C code we generated to execute a subgraph.

2. The wrapper function ``gcc_0__wrapper_`` with a list of ``DLTensor`` arguments that casts data to the right type and invokes ``gcc_0_``.

3. The TVM runtime compatible function ``gcc_0`` with TVM unified function arguments that unpacks TVM packed tensors and invokes ``gcc_0__wrapper_``.

Accordingly, the only thing we need in ``JIT`` implementation is passing all subgraph function code we generated to ``JitImpl``:

.. code-block:: c++

  std::string JIT() {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out_);
  }

All variables (``ext_func_id``, etc) we passed are class variables and were filled when we traversed the subgraph.

Implement CSourceCodegen
========================

Again, let's create a class skeleton and implement the required functions. Note that it inherits ``CSourceModuleCodegenBase``

.. code-block:: c++

  class CSourceCodegen : public CSourceModuleCodegenBase {
   public:
    // Pass a subgraph function, and generate the C code.
    void GenCFunc(const Function& func) { ; }

    // Use GenCFunc to generate the C code and wrap it as a C source module.
    runtime::Module CreateCSourceModule(const NodeRef& ref) override { ; }

   private:
    std::ostringstream code_stream_;
  };

Implement GenCFunc
------------------

``GenCFunc`` simply uses the ``CodegenC`` we just implemented to traverse a Relay function (subgraph) and obtains the generated C code. The builtin function ``GetExtSymbol`` retrieves a unique symbol name (e.g., ``gcc_0``) in the Relay function and we **must** use it as the C function name, because this symbol is going to be used for DSO runtime lookup.

.. code-block:: c++

  void GenCFunc(const Function& func) {
    ICHECK(func.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    CodeGenC builder(sid);
    builder.VisitExpr(func->body);
    code_stream_ << builder.JIT();
  }

Implement CreateCSourceModule
-----------------------------

This function creates a runtime module for the external library. In this example, we create a CSourceModule that can be directly compiled and linked together with a TVM generated DSOModule. After you have implemented ``CodegenC``, implementing this function is relatively straightforward:

.. code-block:: c++

  runtime::Module CreateCSourceModule(const NodeRef& ref) override {
    // Create headers
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <iostream>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <stdio.h>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";

    // Append some common macro for operator definition.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)       \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          out[i] = a[i] p_OP_ b[i];                           \
        }                                                     \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
      extern "C" void p_ID_(float* a, float* b, float* out) {     \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                   \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                 \
            int64_t k = i * p_DIM2_ + j;                          \
            out[k] = a[k] p_OP_ b[k];                             \
          }                                                       \
        }                                                         \
      }
    )op_macro";

    code_stream_ << operator_macro << "\n\n";

    // Generate C code for the subgraph.
    if (ref->IsInstance<FunctionNode>()) {
      GenCFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<relay::ModuleNode>()) {
      relay::Module mod = Downcast<relay::Module>(ref);
      for (const auto& it : mod->functions) {
        GenCFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("module.csource_module_create");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc");
  }

Register Your Codegen
=====================

The last step is registering your codegen to TVM backend. We first implement a simple function to invoke our codegen and generate a runtime module.

.. code-block:: c++

  runtime::Module CCompiler(const NodeRef& ref) {
    CSourceCodegen csource;
    return csource.CreateCSourceModule(ref);
  }

Finally, we register this function to TVM backend:

.. code-block:: c++

  TVM_REGISTER_GLOBAL("relay.ext.ccompiler").set_body_typed(CCompiler);

where ``ccompiler`` is a customized tag to let TVM know this is the codegen it should use to generate and offload subgraphs when the subgraph is annotated with ``ccompiler``.

Finally, a good practice is to set up a CMake configuration flag to include your compiler only for your customers. We first create a cmake file: ``cmake/modules/contrib/CODEGENC.cmake``:

.. code-block:: cmake

  if(USE_CODEGENC)
    file(GLOB CSOURCE_RELAY_CONTRIB_SRC src/relay/backend/contrib/codegen_c/codegen.cc)
    list(APPEND COMPILER_SRCS ${CSOURCE_RELAY_CONTRIB_SRC})
  endif(USE_CODEGENC)

So that users can configure whether to include your compiler when configuring TVM using ``config.cmake``:

.. code-block:: cmake

  set(USE_CODEGENC ON)

*******************************************
Implement a Codegen for Your Representation
*******************************************

Although we have demonstrated how to implement a C codegen, your hardware may require other forms of graph representation, such as JSON. In this case, you could modify ``CodegenC`` class we have implemented to generate your own graph representation and implement a customized runtime module to let TVM runtime know how this graph representation should be executed.

To simplify, we define a graph representation named "ExampleJSON" in this guide. ExampleJSON does not mean the real JSON but just a simple representation for graphs without a control flow. For example, assuming we have the following subgraph named ``subgraph_0``:

::

         input0
           |
          add <-- input1
           |
        subtract <-- input2
           |
        multiply <-- input3
           |
          out

Then the ExampleJON of this subgraph looks like:

.. code-block:: none

  subgraph_0
    input 0 10 10
    input 1 10 10
    input 2 10 10
    input 3 10 10
    add 4 inputs: 0 1 shape: 10 10
    sub 5 inputs: 4 2 shape: 10 10
    mul 6 inputs: 5 3 shape: 10 10

The ``input`` keyword declares an input tensor with its ID and shape; while the other statements describes computations in ``<op> <output ID> inputs: [input ID] shape: [shape]`` syntax.

In this section, our goal is to implement the following customized TVM runtime module to execute ExampleJSON graphs.

.. code-block:: c++

  runtime::Module ExampleJsonCompiler(const NodeRef& ref) {
      ExampleJsonCodeGen codegen(ref);
      std::string code = codegen.gen(); // Note 1
      const auto* pf = runtime::Registry::Get("module.examplejson_module_create"); // Note 2
      ICHECK(pf != nullptr) << "Cannot find ExampleJson module to create the external runtime module";
      return (*pf)(code);
  }
  TVM_REGISTER_GLOBAL("relay.ext.examplejsoncompiler").set_body_typed(ExampleJsonCompiler);

**Note 1**: We will implement a customized codegen later to generate a ExampleJSON code string by taking a subgraph.

**Note 2**: This line obtains a pointer to a function for creating the customized runtime module. You can see that it takes subgraph code in ExampleJSON format we just generated and initializes a runtime module.

In the following sections, we are going to introduce 1) how to implement ``ExampleJsonCodeGen`` and 2) how to implement and register ``examplejson_module_create``.

Implement ExampleJsonCodeGen
============================

Similar to the C codegen, we also derive ``ExampleJsonCodeGen`` from ``ExprVisitor`` to make use of visitor patterns for subgraph traversing. On the other hand, we do not have to inherit ``CodegenCBase`` because we do not need TVM C++ wrappers. The codegen class is implemented as follows:

.. code-block:: c++

    #include <tvm/relay/expr_functor.h>
    #include <tvm/relay/transform.h>
    #include <tvm/relay/type.h>
    #include <tvm/runtime/module.h>
    #include <tvm/runtime/object.h>

    #include <fstream>
    #include <sstream>

    namespace tvm {
    namespace relay {
    namespace contrib {

    class ExampleJsonCodeGen : public ExprVisitor {
      public:
        explicit ExampleJsonCodeGen();

        // Note 1
        void VisitExpr_(const VarNode* node) { /* Skip in this example. */ }
        void VisitExpr_(const CallNode* call) final { /* Skip in this example. */ }

        // Note 2
        std::string gen(NodeRef& ref) {
            this->code = "";
            if (ref->IsInstance<FunctionNode>()) {
                this->visit(Downcast<Function>(ref));
            } else if (ref->IsInstance<relay::ModuleNode>()) {
                relay::Module mod = Downcast<relay::Module>(ref);
                for (const auto& it : mod->functions) {
                    this->visit(Downcast<Function>(it.second));
                }
            } else {
                LOG(FATAL) << "The input ref is expected to be a Relay function or module";
            }
            return this->code;
        }

      private:
          /*! \brief The function id that represents a C source function. */
         std::string code;
    }

**Note 1**: We again implement corresponding visitor functions to generate ExampleJSON code and store it to a class variable ``code`` (we skip the visitor function implementation in this example as their concepts are basically the same as C codegen). After finished the graph visiting, we should have an ExampleJSON graph in ``code``.

**Note 2**: We define an internal API ``gen`` to take a subgraph and generate a ExampleJSON code. This API can be in an arbitrary name you prefer.

The next step is to implement a customized runtime to make use of the output of ``ExampleJsonCodeGen``.

Implement a Customized Runtime
==============================

In this section, we will implement a customized TVM runtime step-by-step and register it to TVM runtime modules. The customized runtime should be located at ``src/runtime/contrib/<your-runtime-name>/``. In our example, we name our runtime "example_ext_runtime".

Again, we first define a customized runtime class as follows. The class has to be derived from TVM ``ModuleNode`` in order to be compatible with other TVM runtime modules.

.. code-block:: c++

	#include <dmlc/logging.h>
	#include <tvm/runtime/c_runtime_api.h>
	#include <tvm/runtime/memory.h>
	#include <tvm/runtime/module.h>
	#include <tvm/runtime/ndarray.h>
	#include <tvm/runtime/object.h>
	#include <tvm/runtime/packed_func.h>
	#include <tvm/runtime/registry.h>

	#include <fstream>
	#include <cmath>
	#include <map>
	#include <sstream>
	#include <string>
	#include <vector>

	namespace tvm {
	namespace runtime {
	class ExampleJsonModule : public ModuleNode {
	 public:
	  explicit ExampleJsonModule(std::string graph_json);

	  PackedFunc GetFunction(const std::string& name,
	                         const ObjectPtr<Object>& sptr_to_self) final;

	  const char* type_key() const { return "examplejson"; }

	  void SaveToBinary(dmlc::Stream* stream) final;

	  static Module LoadFromBinary(void* strm);

	  static Module Create(const std::string& path);

          std::string GetSource(const std::string& format = "");

          void Run(int id, const std::vector<int>& inputs, int output);

          void ParseJson(const std::string& json);

	 private:
	  /* \brief The json string that represents a computational graph. */
	  std::string graph_json_;
	  /* \brief The subgraph that being processed. */
	  std::string curr_subgraph_;
	  /*! \brief A simple graph from subgraph id to node entries. */
	  std::map<std::string, std::vector<NodeEntry>> graph_;
	  /* \brief A simple pool to contain the tensor for each node in the graph. */
	  std::vector<NDArray> data_entry_;
	  /* \brief A mapping from node id to op name. */
	  std::vector<std::string> op_id_;
	};

In particular, there are some functions derived from ``ModuleNode`` that we must implement in ``ExampleJsonModule``:

* Constructor: The constructor of this class should accept a subgraph (in your representation), process and store it in any format you like. The saved subgraph could be used by the following two functions.

* ``GetFunction``: This is the most important function in this class. When TVM runtime wants to execute a subgraph with your compiler tag, TVM runtime invokes this function from your customized runtime module. It provides the function name as well as runtime arguments, and ``GetFunction`` should return a packed function implementation for TVM runtime to execute.

* ``SaveToBinary`` and ``LoadFromBinary``: ``SaveToBinary`` serialize the runtime module to a binary format for later deployment. This function will be called by TVM when users use ``export_library`` API. On the other hand, since we are now using our own graph representation, we have to make sure that ``LoadFromBinary`` is able to construct the same runtime module by taking the serialized binary generated by ``SaveToBinary``.

* ``GetSource`` (optional): If you would like to see the generated ExampleJSON code, you can implement this function to dump it; otherwise you can skip the implementation.

Other functions and class variables will be introduced along with the implementation of above must-have functions.

Implement Constructor
---------------------

.. code-block:: c++

    explicit ExampleJsonModule(std::string graph_json) {
      this->graph_json_ = graph_json;
      ParseJson(this->graph_json_);
    }

Then, we implement ``ParseJson`` to parse a subgraph in ExampleJSON format and construct a graph in memory for later usage. Since we do not support subgraph with branches in this example, we simply use an array to store every nodes in a subgraph in order.

.. code-block:: c++

  void ParseJson(const std::string& json) {
    std::string line;
    std::string curr_subgraph;
    std::stringstream ss(json);

    while (std::getline(ss, line, '\n')) {
      std::stringstream ss2(line);
      std::string token;
      int id = 0;

      ss2 >> token;
      if (token.find("subgraph_") != std::string::npos) {
        curr_subgraph = token;
        continue;
      }

      ss2 >> id;
      if (op_id_.size() <= static_cast<size_t>(id)) {
        op_id_.resize(id + 1);
        data_entry_.resize(id + 1);
      }

      int64_t total_elements = 1;
      std::vector<int64_t> shape;
      if (token == "input") {
        int64_t size = 0;
        while (ss2 >> size) {
          total_elements *= size;
          shape.push_back(size);
        }
      } else {
        op_id_[id] = token; // Note 1
        bool shape_data = false;
        NodeEntry entry;
        while (ss2 >> token) {
          if (token == "shape:") {
            shape_data = true;
          } else if (shape_data) {
            total_elements *= std::stoll(token);
            shape.push_back(std::stoll(token));
          } else if (token != "inputs:") {
            entry.inputs.push_back(std::stoi(token));
          }
        }
        entry.id = id;
        entry.output = id;
        graph_[curr_subgraph].push_back(entry); // Note 2
      }
      DLDevice dev;
      dev.device_type = static_cast<DLDeviceType>(1);
      dev.device_id = 0;
      data_entry_[id] = NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, dev); // Note 3
    }
  }

**Note 1**: We use a class variable ``op_id_`` to map from subgraph node ID to the operator name (e.g., ``add``) so that we can invoke the corresponding operator function in runtime.

**Note 2**: We use a class variable ``graph_`` to map from subgraph name to an array of nodes. ``GetFunction`` will query graph nodes by a subgraph ID in runtime.

**Note 3**: We use a class variable `data_entry_` to map from a subgraph node ID to a tensor data placeholder. We will put inputs and outputs to the corresponding data entry in runtime.

Implement GetFunction
---------------------

After the construction, we should have the above class variables ready. We then implement ``GetFunction`` to provide executable subgraph functions to TVM runtime:

.. code-block:: c++

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    if (this->graph_.find(name) != this->graph_.end()) {
      this->curr_subgraph_ = name;
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

        // Copy input tensors to corresponding data entries.
        for (auto i = 0; i < args.size(); ++i) {
          ICHECK(args[i].type_code() == kNDArrayContainer || args[i].type_code() == kArrayHandle)
              << "Expect NDArray or DLTensor as inputs\n";
          if (args[i].type_code() == kArrayHandle) {
            DLTensor* arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          } else {
            NDArray arg = args[i];
            this->data_entry_[i].CopyFrom(arg);
          }
        }

        // Execute the subgraph.
        for (const auto& it : this->graph_[this->curr_subgraph_]) {
          this->Run(it.id, it.inputs, it.output);
        }
        ICHECK_GT(graph_.count(this->curr_subgraph_), 0U);

        // Copy the output from a data entry back to TVM runtime argument.
        auto out_idx = graph_[this->curr_subgraph_].back().output;
        if (args[args.size() - 1].type_code() == kArrayHandle) {
          DLTensor* arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        } else {
          NDArray arg = args[args.size() - 1];
          this->data_entry_[out_idx].CopyTo(arg);
        }
        *rv = data_entry_.back();
      });
    } else {
      LOG(FATAL) << "Unknown subgraph: " << name << "\n";
      return PackedFunc();
    }
  }

As can be seen, ``GetFunction`` is composed of three major parts. The first part copies data from TVM runtime arguments to the corresponding data entries we assigned in the constructor. The second part executes the subgraph with ``Run`` function (will implement later) and saves the results to another data entry. The third part copies the results from the output data entry back to the corresponding TVM runtime argument for output.

Implement Run
-------------

Now let's implement ``Run`` function. This function accepts 1) a subgraph ID, 2) a list of input data entry indexs, and 3) an output data entry index.

.. code-block:: c++

  void Run(int id, const std::vector<int>& inputs, int output) {
    // Make a list data entry indexs.
    std::vector<int> args(inputs.begin(), inputs.end());
    args.push_back(output);

    // Initialize data holders.
    std::vector<TVMValue> values(args.size());
    std::vector<int> type_codes(args.size());

    // Initialize a TVM arg setter with TVMValue and its type code.
    TVMArgsSetter setter(values.data(), type_codes.data());

    // Set each argument to its corresponding data entry.
    if (op_id_[id] == "add" || op_id_[id] == "sub" || op_id_[id] == "mul") {
      for (size_t i = 0; i < args.size(); i++) {
        setter(i, data_entry_[args[i]]);
      }
    }

    // Invoke the corresponding operator function.
    if (op_id_[id] == "add") {
      Add(values.data(), type_codes.data(), args.size());
    } else if (op_id_[id] == "sub") {
      Sub(values.data(), type_codes.data(), args.size());
    } else if (op_id_[id] == "mul") {
      Mul(values.data(), type_codes.data(), args.size());
    } else {
      LOG(FATAL) << "Unknown op: " << op_id_[id] << "\n";
    }
  }

``Run`` function mainly has two parts. The first part allocates a list of ``TVMValue``, and maps corresponding data entry blocks. This will become the arguments of our operator functions. The second part than invokes our operator functions. Although we use the same C functions as the previous example, you can replace ``Add``, ``Sub``, and ``Mul`` with your own engine. You only need to make sure your engine stores the results to the last argument so that they can be transferred back to TVM runtime.

With above functions implemented, our customized codegen and runtime can now execute subgraphs. The last step is registering an API (``examplejson_module_create``) to create this module:

.. code-block:: c++

  TVM_REGISTER_GLOBAL("module.examplejson_module_create")
  .set_body_typed([](std::string code){
      auto n = make_object<ExampleJsonModule>(code);
      return runtime::Module(n);
  });

Implement SaveToBinary and LoadFromBinary
-----------------------------------------

So far we have implemented the main features of a customized runtime so that it can be used as other TVM runtimes. However, when users want to save the built runtime to a disk for deployment, TVM has no idea about how to save it. This is the reason we want to implement ``SaveToBinary`` and ``LoadFromBinary``, which tell TVM how should this customized runtime be persist and restored.

We first implement ``SaveToBinary`` function to allow users to save this module in disk.

.. code-block:: c++

  void SaveToBinary(dmlc::Stream* stream) final {
      stream->Write(this->graph_json_);
  }

We can find that this function is pretty simple. Recall that the only argument we took in constructor is a subgraph representation, meaning that we only need a subgraph representation to construct/recover this customized runtime module. As a result, ``SaveToBinary`` simply writes the subgraph to an output DMLC stream. That is, when users use ``export_library`` API to export the module, the customized module will be an ExampleJSON stream of a subgraph.

Similarity, ``LoadFromBinary`` reads the subgraph stream and re-constructs the customized runtime module:

.. code-block:: c++

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string graph_json;
    stream->Read(&graph_json);
    auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
    return Module(n);
  }

We also need to register this function to enable the corresponding Python API:

.. code-block:: c++

  TVM_REGISTER_GLOBAL("module.loadbinary_examplejson")
  .set_body_typed(ExampleJsonModule::LoadFromBinary);

The above registration means when users call ``tvm.runtime.load_module(lib_path)`` API and the exported library has an ExampleJSON stream, our ``LoadFromBinary`` will be invoked to create the same customized runtime module.

In addition, if you want to support module creation directly from an ExampleJSON file, you can also implement a simple function and register a Python API as follows:

.. code-block:: c++

  static Module Create(const std::string& path) {
      std::ifstream filep;
      filep.open(path, std::ios::in);
      std::string graph_json;
      std::string line;
      while (std::getline(filep, line)) {
          graph_json += line;
          graph_json += "\n";
      }
      filep.close();
      auto n = tvm::runtime::make_object<ExampleJsonModule>(graph_json);
      return Module(n);
  }

  TVM_REGISTER_GLOBAL("module.loadfile_examplejson")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = ExampleJsonModule::Create(args[0]);
  });

It means users can manually write/modify an ExampleJSON file, and use Python API ``tvm.runtime.load_module("mysubgraph.examplejson", "examplejson")`` to construct a customized module.

*******
Summary
*******

In summary, here is a checklist for you to refer:

* A codegen class derived from ``ExprVisitor`` and ``CodegenCBase`` (only for C codegen) with following functions.

  * ``VisitExpr_(const CallNode* call)`` to collect call node information.
  * Other visitor functions you needed to collect subgraph information.
  * ``JIT`` to generate subgraph code.
  * Register codegen.

* A function to create ``CSourceModule`` (for C codegen).

* A runtime module class derived from ``ModuleNode`` with following functions (for your graph representation).

  * Constructor.
  * ``GetFunction`` to generate a TVM runtime compatible ``PackedFunc``.
  * ``Run`` to execute a subgraph.
  * Register a runtime creation API.
  * ``SaveToBinary`` and ``LoadFromBinary`` to serialize/deserialize customized runtime module.
  * Register ``LoadFromBinary`` API to support ``tvm.runtime.load_module(your_module_lib_path)``.
  * (optional) ``Create`` to support customized runtime module construction from subgraph file in your representation.

* An annotator to annotate a user Relay program to make use of your compiler and runtime (TBA).
