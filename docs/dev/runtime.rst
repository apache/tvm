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

.. _tvm-runtime-system:

TVM Runtime System
==================

TVM supports multiple programming languages for the compiler stack development and deployment.
In this note, we explain the key elements of the TVM runtime.

.. image:: https://tvm.apache.org/images/release/tvm_flexible.png

We need to satisfy quite a few interesting requirements:

- Deployment: invoke the compiled function from python/javascript/c++ language.
- Debug: define a function in python and call that from a compiled function.
- Link: write driver code to call device specific code (CUDA) and call it from compiled host function.
- Prototype: define an IR pass from python and call that from C++ backend.
- Expose: compiler stack developed in c++ to front-end (i.e, python)
- Experiment: ship a compiled function to an embedded device to directly run there.

We want to be able to define a function from any language and call from another.
We also want the runtime core to be minimal to deploy to embedded devices.

.. _tvm-runtime-system-packed-func:

PackedFunc
----------

`PackedFunc`_ is a simple but elegant solution
we find to solve the challenges listed. The following code block provides an example in C++

.. _PackedFunc: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/packed_func.h

.. code:: c

    #include <tvm/runtime/packed_func.h>

    void MyAdd(TVMArgs args, TVMRetValue* rv) {
      // automatically convert arguments to desired type.
      int a = args[0];
      int b = args[1];
      // automatically assign value return to rv
      *rv = a + b;
    }

    void CallPacked() {
      PackedFunc myadd = PackedFunc(MyAdd);
      // get back 3
      int c = myadd(1, 2);
    }

In the above codeblock, we defined a PackedFunc MyAdd. It takes two arguments
: ``args`` represents input arguments and ``rv`` represents return value.
The function is type-erased, which means that the function signature does not restrict which input type to pass in or type to return.
Under the hood, when we call a PackedFunc, it packs the input arguments to TVMArgs on stack,
and gets the result back via TVMRetValue.

Thanks to template tricks in C++, we can call a PackedFunc just like a normal function. Because of its type-erased nature, we can call a PackedFunc from dynamic languages like python, without additional glue code for each new type function created.
The following example registers PackedFunc in C++ and calls from python.

.. code:: c

    // register a global packed function in c++
    TVM_REGISTER_GLOBAL("myadd")
    .set_body(MyAdd);

.. code:: python

    import tvm

    myadd = tvm.get_global_func("myadd")
    # prints 3
    print(myadd(1, 2))

Most of the magic of PackedFunc lies in ``TVMArgs`` and ``TVMRetValue`` structure.
We restrict a list of possible types which can be passed.
Here are the common ones:

- int, float and string
- PackedFunc itself
- Module for compiled modules
- DLTensor* for tensor object exchange
- TVM Object to represent any object in IR

The restriction makes the implementation simple without the need of serialization.
Despite being minimum, the PackedFunc is sufficient for the use-case of deep learning deployment as
most functions only take DLTensor or numbers.

Since one PackedFunc can take another PackedFunc as an argument,
we can pass functions from python (as PackedFunc) to C++.

.. code:: c

    TVM_REGISTER_GLOBAL("callhello")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      PackedFunc f = args[0];
      f("hello world");
    });

.. code:: python

    import tvm

    def callback(msg):
      print(msg)

    # convert to PackedFunc
    f = tvm.convert(callback)
    callhello = tvm.get_global_func("callhello")
    # prints hello world
    callhello(f)

TVM provides a `minimum C API`_,
which allows us to embed the PackedFunc into any languages. Besides python, so far we supported
`java`_ and `javascript`_.
This philosophy of embedded API is very like Lua, except that we don't have a new language but use C++.

.. _minimum C API: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/c_runtime_api.h
.. _java: https://github.com/apache/incubator-tvm/tree/master/jvm
.. _javascript: https://github.com/apache/incubator-tvm/tree/master/web


One fun fact about PackedFunc is that we use it for both compiler and deployment stack.

- All TVM's compiler pass functions are exposed to frontend as PackedFunc, see `here`_
- The compiled module also returns the compiled function as PackedFunc

.. _here: https://github.com/apache/incubator-tvm/tree/master/src/api

To keep the runtime minimum, we isolated the IR Object support from the deployment runtime. The resulting runtime takes around 200K - 600K depending on how many runtime driver modules (e.g., CUDA) get included.

The overhead of calling into PackedFunc vs. a normal function is small, as it is only saving a few values on the stack.
So it is OK as long as we don't wrap small functions.
In summary, the PackedFunc is the universal glue in TVM where we use it extensively to support our compiler and deployment.

Module
------

Since TVM supports multiple types of devices, we need to support different type of drivers.
We have to use the driver API to load the kernel, set up the argument in packed format and perform kernel launch.
We also need to patch up the driver API so that the exposed functions are threadsafe.
So we often need to implement these driver glues in C++ and expose them to the user.
We can certainly not do it for each type of functions, so again PackedFunc is our answer.

TVM defines the compiled object as `Module`_.
The user can get the compiled function from Module as PackedFunc.
The generated compiled code can dynamically get function from Module in runtime. It caches the function handle in the first call and reuses in subsequent calls. We use this to link device code and callback into any PackedFunc(e.g., python) from generated code.

.. _Module: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/module.h

The ModuleNode is an abstract class that can be implemented by each type of device.
So far we support modules for CUDA, Metal, OpenCL and loading dynamic shared libraries. This abstraction makes introduction
of new device easy, and we do not need to redo the host code generation for each type of device.

Remote Deployment
-----------------

The PackedFunc and Module system also makes it easy to ship the function into remote devices directly.
Under the hood, we have an RPCModule that serializes the arguments to do the data movement and launches the computation on the remote.

.. image:: https://tvm.apache.org/images/release/tvm_rpc.png

The RPC server itself is minimum and can be bundled into the runtime. We can start a minimum TVM
RPC server on iPhone/android/raspberry pi or even the browser. The cross compilation on server and shipping of the module for testing can be done in the same script. Checkout
:ref:`tutorial-cross-compilation-and-rpc` for more details.


This instant feedback gives us a lot of advantages. For example, to test the correctness of generated code on iPhone, we no longer have to write test-cases in swift/objective-c from scratch -- We can use RPC to execute on iPhone, copy the result back and do verification on the host via numpy. We can also do the profiling using the same script.

TVM Object and Compiler Stack
-----------------------------

As we mentioned earlier, we build compiler stack API on top of the PackedFunc runtime system.
We faced a constant changing of the compiler API for the need of research. We need a new language object or IR node whenever we want to test out new primitives.
However, we don't want to change our API from time to time. Besides that, we also want to

- be able to serialize any language object and IRs
- be able to explore, print, and manipulate the IR objects in front-end language to do quick prototyping.

We introduced a base class, called `Object`_ to solve this problem.
All the language object in the compiler stack is a subclass of ``Object``. Each object contains a string type_key that uniquely identifies
the type of object. We choose string instead of int as type key so new ``Object`` class can be added in the decentralized fashion without
adding the code back to the central repo. To ease the speed of dispatching, we allocate an integer type_index at runtime for each type_key.

.. _Object: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/object.h

Since usually one ``Object`` could be referenced in multiple places in the language, we use a shared_ptr to keep
track of reference. We use ``ObjectRef`` class to represent a reference to the ``Object``.
We can roughly view ``ObjectRef`` class as shared_ptr to the ``Object`` container.
We can also define subclass ``ObjectRef`` to hold each subtypes of ``Object``. Each subclass of ``Object`` needs to define the VisitAttr function.

.. code:: c

    class AttrVisitor {
    public:
      virtual void Visit(const char* key, double* value) = 0;
      virtual void Visit(const char* key, int64_t* value) = 0;
      virtual void Visit(const char* key, uint64_t* value) = 0;
      virtual void Visit(const char* key, int* value) = 0;
      virtual void Visit(const char* key, bool* value) = 0;
      virtual void Visit(const char* key, std::string* value) = 0;
      virtual void Visit(const char* key, void** value) = 0;
      virtual void Visit(const char* key, Type* value) = 0;
      virtual void Visit(const char* key, ObjectRef* value) = 0;
      // ...
    };

    class BaseAttrsNode : public Object {
    public:
      virtual void VisitAttrs(AttrVisitor* v) {}
      // ...
    };

Each ``Object`` subclass will override this to visit its members. Here is an example implementation of TensorNode.

.. code:: c

    class TensorNode : public Object {
    public:
      /*! \brief The shape of the tensor */
      Array<Expr> shape;
      /*! \brief data type in the content of the tensor */
      Type dtype;
      /*! \brief the source operation, can be None */
      Operation op;
      /*! \brief the output index from source operation */
      int value_index{0};
      /*! \brief constructor */
      TensorNode() {}

      void VisitAttrs(AttrVisitor* v) final {
        v->Visit("shape", &shape);
        v->Visit("dtype", &dtype);
        v->Visit("op", &op);
        v->Visit("value_index", &value_index);
      }
    };

In the above examples, both ``Operation`` and ``Array<Expr>`` are ObjectRef.
The VisitAttrs gives us a reflection API to visit each member of the object.
We can use this function to visit the node and serialize any language object recursively.
It also allows us to get members of an object easily in front-end language.
For example, in the following code, we accessed the op field of the TensorNode.

.. code:: python

    import tvm
    from tvm import te

    x = te.placeholder((3,4), name="x")
    # access the op field of TensorNode
    print(x.op.name)

New ``Object`` can be added to C++ without changing the front-end runtime, making it easy to make extensions to the compiler stack.
Note that this is not the fastest way to expose members to front-end language, but might be one of the simplest
approaches possible. We also find that it fits our purposes as we mainly use python for testing and prototyping and still use c++
to do the heavy lifting job.

Implementation Details
----------------------

Each argument in PackedFunc contains a union value `TVMValue`_
and a type code. This design allows the dynamically typed language to convert to the corresponding type directly, and statically typed language to
do runtime type checking during conversion.

.. _TVMValue: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/c_runtime_api.h#L122

The relevant files are

- `packed_func.h`_ for C++ API
- `c_runtime_api.cc`_ for C API and how to provide callback.

.. _packed_func.h: https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/packed_func.h
.. _c_runtime_api.cc: https://github.com/apache/incubator-tvm/blob/master/src/runtime/c_runtime_api.cc#L262

To support extension types, we used a registry system to register type related information, like support of any
in C++, see `Extension types`_ for more details.

.. _Extension types: https://github.com/apache/incubator-tvm/tree/master/apps/extension
