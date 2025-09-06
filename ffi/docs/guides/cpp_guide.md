<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# C++ Guide

This guide introduces the tvm-ffi C++ API.
We provide C++ API on top of the stable C ABI to provide a type-safe and efficient way to work with the tvm-ffi.
The C++ API is designed to abstract away the complexity of the C ABI while maintaining full compatibility.
The C++ API builds around the following key concepts:

- **Any and AnyView**: Type-erased containers that can hold values of any supported type in tvm-ffi.
- **Function**: A type-erased "packed" function that can be invoked like normal functions.
- **Objects and ObjectRefs**: Reference-counted objects to manage on-heap data types.

Code examples in this guide use `EXPECT_EQ` for demonstration purposes, which is a testing framework macro. In actual applications, you would use standard C++ assertions or error handling.
You can find runnable code of the examples under tests/cpp/test_example.cc.

## Any and AnyView

`Any` and `AnyView` are the foundation of tvm-ffi, providing
ways to store values that are compatible with the ffi system.
The following example shows how we can interact with Any and AnyView.

```cpp

#include <tvm/ffi/any.h>

void ExampleAny() {
  namespace ffi = tvm::ffi;
  // Create an Any from various types
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  ffi::Any int_value = 42;
  ffi::Any float_value = 3.14;
  ffi::Any string_value = "hello world";

  // AnyView provides a lightweight view without ownership
  ffi::AnyView view = int_value;
  // we can cast Any/AnyView to a specific type
  int extracted = view.cast<int>();
  EXPECT_EQ(extracted, 42);

  // If we are not sure about the type
  // we can use as to get an optional value
  std::optional<int> maybe_int = view.as<int>();
  if (maybe_int.has_value()) {
    EXPECT_EQ(maybe_int.value(), 42);
  }
  // Try cast is another version that will try to run the type
  // conversion even if the type does not exactly match
  std::optional<int> maybe_int_try = view.try_cast<int>();
  if (maybe_int_try.has_value()) {
    EXPECT_EQ(maybe_int_try.value(), 42);
  }
}
```

At a high level, we can perform the following operations:

- We can store a value into Any, under the hood, Any will record the type of the value by its type_index.
- We can fetch a value from Any or AnyView using the `cast` function.
- If we are unsure about the type in Any, we can use `as` or `try_cast` function to get an optional value.

Under the hood, Any and AnyView store the value via the ABI convention and also manage the reference
counting correctly when the stored value is an on-heap object.

## Object and ObjectRef

The tvm-ffi object system provides the foundation for all managed, reference-counted objects
in the system. It enables type safety, cross-language compatibility, and efficient memory management.

The object system is built around three key classes: Object, ObjectPtr, and ObjectRef.
The `Object` class is the base class of all heap-allocated objects. It contains a common header
that includes the `type_index`, reference counter and deleter for the object.
Users do not need to explicitly manage these fields as part of the C++ API. Instead,
they are automatically managed through a smart pointer `ObjectPtr` which points
to a heap-allocated object instance.
The following code shows an example object and the creation of an `ObjectPtr`:

```cpp
#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>

class MyIntPairObj : public tvm::ffi::Object {
 public:
  int64_t a;
  int64_t b;

  MyIntPairObj() = default;
  MyIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  // Required: declare type information
  // to register a dynamic type index through the system
  static constexpr const char* _type_key = "example.MyIntPair";
  // This macro registers the class with the FFI system to set up the right type index
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(MyIntPairObj, tvm::ffi::Object);
};

void ExampleObjectPtr() {
  namespace ffi = tvm::ffi;
  // make_object automatically sets up the deleter correctly
  // This function creates a new ObjectPtr with proper memory management
  // It handles allocation, initialization, and sets up the reference counting system
  ffi::ObjectPtr<MyIntPairObj> obj = ffi::make_object<MyIntPairObj>(100, 200);
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(obj->a, 100);
  EXPECT_EQ(obj->b, 200);
}
```

We typically provide a reference class that wraps the ObjectPtr.
The `ObjectRef` base class provides the interface and reference counting
functionality for these wrapper classes.
```cpp
#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>

class MyIntPair : public tvm::ffi::ObjectRef {
 public:
  // Constructor
  explicit MyIntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<MyIntPairObj>(a, b);
  }

  // Required: define object reference methods
  // This macro provides the necessary methods for ObjectRef functionality
  TVM_FFI_DEFINE_OBJECT_REF_METHODS(MyIntPair, tvm::ffi::ObjectRef, MyIntPairObj);
};

void ExampleObjectRef() {
  namespace ffi = tvm::ffi;
  MyIntPair pair(100, 200);
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(pair->a, 100);
  EXPECT_EQ(pair->b, 200);
}
```

**Note:** The ObjectRef provides a user-friendly interface while ObjectPtr handles the low-level memory management.
The ObjectRef acts as a smart pointer wrapper that automatically manages the ObjectPtr lifecycle.

The overall implementation pattern is as follows:
- **Object Class**: Inherits from `ffi::Object`, stores data and implements the core functionality.
- **ObjectPtr**: Smart pointer that manages the Object lifecycle and reference counting.
- **Ref Class**: Inherits from `ffi::ObjectRef`, provides a user-friendly interface and automatic memory management.

This design ensures efficient memory management while providing a clean API for users. Once we define an ObjectRef class,
we can integrate it with the Any, AnyView and Functions.

```cpp
#include <tvm/ffi/object.h>
#include <tvm/ffi/any.h>

void ExampleObjectRefAny() {
  namespace ffi = tvm::ffi;
  MyIntPair pair(100, 200);
  ffi::Any any = pair;
  MyIntPair pair2 = any.cast<MyIntPair>();
  // Note: EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(pair2->a, 100);
  EXPECT_EQ(pair2->b, 200);
}

```

Under the hood, ObjectPtr manages the lifecycle of the object through the same mechanism as shared pointers. We designed
the object to be intrusive, which means the reference counter and type index metadata are embedded at the header of each object.
This design allows us to allocate the control block and object memory together. As we will see in future sections,
all of our heap-allocated classes such as Function, on-heap String, Array and Map are managed using subclasses of Object,
and the user-facing classes such as Function are ObjectRefs.


We provide a collection of built-in object and reference types, which are sufficient for common cases.
Developers can also bring new object types as shown in the example of this section. We provide mechanisms
to expose these objects to other language bindings such as Python.


## Function

The `Function` class provides a type-safe way to create and invoke callable objects
through tvm-ffi ABI convention. We can create a `ffi::Function` from an existing typed lambda function.

```cpp
#include <tvm/ffi/function.h>

void ExampleFunctionFromTyped() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fadd1 = ffi::Function::FromTyped(
    [](const int a) -> int { return a + 1; }
  );
  int b = fadd1(1).cast<int>();
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(b, 2);
}
```

Under the hood, tvm-ffi leverages Any and AnyView to create a unified ABI for
all functions. The following example demonstrates the low-level way of defining
a "packed" function for the same `fadd1`.

```cpp
void ExampleFunctionFromPacked() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fadd1 = ffi::Function::FromPacked(
    [](const ffi::AnyView* args, int32_t num_args, ffi::Any* rv) {
      // Check that we have exactly one argument
      TVM_FFI_ICHECK_EQ(num_args, 1);
      int a = args[0].cast<int>();
      *rv = a + 1;
    }
  );
  int b = fadd1(1).cast<int>();
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(b, 2);
}
```

At a high level, `ffi::Function` implements function calling by the following convention:
- The arguments are passed through an on-stack array of `ffi::AnyView`
- Return values are passed through `ffi::Any`

Because the return value is `ffi::Any`, we need to explicitly call `cast` to convert the return
value to the desirable type. Importantly, `ffi::Function` itself is a value type that is compatible
with tvm-ffi, which means we can pass it as an argument and return values. The following code shows
an example of passing a function as an argument and applying it inside.

```cpp
void ExampleFunctionPassFunction() {
  namespace ffi = tvm::ffi;
  // Create a function from a typed lambda
  ffi::Function fapply = ffi::Function::FromTyped(
      [](const ffi::Function f, ffi::Any param) { return f(param.cast<int>()); });
  ffi::Function fadd1 = ffi::Function::FromTyped(  //
      [](const int a) -> int { return a + 1; });
  int b = fapply(fadd1, 2).cast<int>();
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(b, 3);
}
```

This pattern is very powerful because we can construct `ffi::Function` not only from C++,
but from any languages that expose to the tvm-ffi ABI. For example, this means we can easily call functions
passed in or registered from Python for quick debugging or other purposes.


### Global Function Registry

Besides creating functions locally, tvm-ffi provides a global function registry that allows
functions to be registered and called across different modules and languages.
The following code shows an example

```cpp
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

void ExampleGlobalFunctionRegistry() {
  namespace ffi = tvm::ffi;
  ffi::reflection::GlobalDef().def("xyz.add1", [](const int a) -> int { return a + 1; });
  ffi::Function fadd1 = ffi::Function::GetGlobalRequired("xyz.add1");
  int b = fadd1(1).cast<int>();
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(b, 2);
}
```

You can also access and register global functions from the Python API.

### Exporting as Library Symbol

Besides the API that allows registration of functions into the global table,
we also provide a macro to export static functions as `TVMFFISafeCallType` symbols in a dynamic library.

```c++
void AddOne(DLTensor* x, DLTensor* y) {
  // ... implementation omitted ...
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, my_ffi_extension::AddOne);
```

The new `add_one` takes the signature of `TVMFFISafeCallType` and can be wrapped as `ffi::Function`
through the C++ `ffi::Module` API.

```cpp
ffi::Module mod = ffi::Module::LoadFromFile("path/to/export_lib.so");
ffi::Function func = mod->GetFunction("add_one").value();
```

## Error Handling

We provide a specific `ffi::Error` type that is also made compatible with the ffi ABI.
We also provide a macro `TVM_FFI_THROW` to simplify the error throwing step.

```cpp
// file: cpp/test_example.cc
#include <tvm/ffi/error.h>

void FuncThrowError() {
  namespace ffi = tvm::ffi;
  TVM_FFI_THROW(TypeError) << "test0";
}

void ExampleErrorHandling() {
  namespace ffi = tvm::ffi;
  try {
    FuncThrowError();
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
    EXPECT_EQ(e.message(), "test0");
    std::cout << e.traceback() << std::endl;
  }
}
```
The structured error class records kind, message and traceback that can be mapped to
Pythonic style error types and tracebacks. The traceback follows the Python style,
tvm-ffi will try to preserve the traceback when possible. In the above example,
you can see the traceback output as
```
... more lines omitted
File "cpp/test_example.cc", line 106, in ExampleErrorHandling
File "cpp/test_example.cc", line 100, in void FuncThrowError()
```

The ffi ABI provides minimal but sufficient mechanisms to propagate these errors across
language boundaries.
So when we call the function from Python, the Error will be translated into a corresponding
Error type. Similarly, when we call a Python callback from C++, the error will be translated
into the right error kind and message.


## Tensor

For many use cases, we do not need to manage the nd-array/Tensor memory.
In such cases, `DLTensor*` can be used as the function arguments.
There can be cases for a managed container for multi-dimensional arrays.
`ffi::Tensor` is a minimal container to provide such support.
Notably, specific logic of device allocations and array operations are non-goals
of the FFI. Instead, we provide minimal generic API `ffi::Tensor::FromNDAlloc`
to enable flexible customization of Tensor allocation.

```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) {
    tensor->data = malloc(tvm::ffi::GetDataSize(*tensor));
  }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

void ExampleTensor() {
  namespace ffi = tvm::ffi;
  ffi::Shape shape = {1, 2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  ffi::Tensor tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
  // now tensor is a managed tensor
}
```

The above example shows how we define `CPUNDAlloc` that customizes `AllocData`
and `FreeData` behavior. The CPUNDAlloc struct will be kept alive with the Tensor object.
This pattern allows us to implement various Tensor allocations using the same API:

- For CUDA allocation, we can change malloc to cudaMalloc
- For memory-pool based allocation, we can update `CPUNDAlloc` to keep a strong reference to the pool,
  so we can keep memory-pool alive when the array is alive.

**Working with Shapes** As you may have noticed in the example, we have a `ffi::Shape` container that is used
to represent the shapes in nd-array. This container allows us to have compact and efficient representation
of managed shapes and we provide quick conversions from standard vector types.

### DLPack Conversion

We provide first-class DLPack support to the `ffi::Tensor` that enables efficient exchange
through the DLPack Protocol.

```cpp
#include <tvm/ffi/container/tensor.h>

void ExampleTensorDLPack() {
  namespace ffi = tvm::ffi;
  ffi::Shape shape = {1, 2, 3};
  DLDataType dtype = {kDLFloat, 32, 1};
  DLDevice device = {kDLCPU, 0};
  ffi::Tensor tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
  // convert to DLManagedTensorVersioned
  DLManagedTensorVersioned* dlpack = nd.ToDLPackVersioned();
  // load back from DLManagedTensorVersioned
  ffi::Tensor tensor2 = ffi::Tensor::FromDLPackVersioned(dlpack);
}
```

These APIs are also available through the C APIs
`TVMFFITensorFromDLPackVersioned` and `TVMFFITensorToDLPackVersioned`.

## String and Bytes

The tvm-ffi provides first-class support for `String` and `Bytes` types that are efficient,
FFI-compatible, and interoperable with standard C++ string types.

```cpp
#include <tvm/ffi/string.h>

void ExampleString() {
  namespace ffi = tvm::ffi;
  ffi::String str = "hello world";
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(str.size(), 11);
  std::string std_str = str;
  EXPECT_EQ(std_str, "hello world");
}
```

Alternatively, users can always directly use `std::string` in function arguments, conversion
will happen automatically.

**Rationale:** We need to have separate Bytes and String so they map well to corresponding Python types.
`ffi::String` is backed by a possibly managed object that makes it more compatible with the Object system.

## Container Types

To enable effective passing and storing of collections of values that are compatible with tvm-ffi,
we provide several built-in container types.

### Array

`Array<T>` provides an array data type that can be used as function arguments.
When we use `Array<T>` as an argument of a Function, it will
perform runtime checks of the elements to ensure the values match the expected type.

```cpp
#include <tvm/ffi/container/array.h>


void ExampleArray() {
  namespace ffi = tvm::ffi;
  ffi::Array<int> numbers = {1, 2, 3};
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(numbers.size(), 3);
  EXPECT_EQ(numbers[0], 1);

  ffi::Function head = ffi::Function::FromTyped([](const ffi::Array<int> a) {
    return a[0];
  });
  EXPECT_EQ(head(numbers).cast<int>(), 1);

  try {
    // throw an error because 2.2 is not int
    head(ffi::Array<ffi::Any>({1, 2.2}));
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
  }
}
```

Under the hood, Array is backed by a reference-counted Object `ArrayObj` that stores
a collection of Any values. Note that conversion from Any to `Array<T>` will result in
runtime checks of elements because the type index only indicates `ArrayObj` as the backing storage.
If you want to defer such checks at the FFI function boundary, consider using `Array<Any>` instead.
When passing lists and tuples from Python, the values will be converted to `Array<Any>` before
being passed into the Function.

**Performance note:** Repeatedly converting Any to `Array<T>` can incur repeated
checking overhead at each element. Consider using `Array<Any>` to defer checking or only run conversion once.

### Tuple

`Tuple<Types...>` provides type-safe fixed-size collections.

```cpp
#include <tvm/ffi/container/tuple.h>

void ExampleTuple() {
  namespace ffi = tvm::ffi;
  ffi::Tuple<int, ffi::String, bool> tup(42, "hello", true);

  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(tup.get<0>(), 42);
  EXPECT_EQ(tup.get<1>(), "hello");
  EXPECT_EQ(tup.get<2>(), true);
}
```

Under the hood, Tuple is backed by the same `ArrayObj` as the Array container.
This enables zero-cost exchange with input arguments.

**Rationale:** This design unifies the conversion rules from Python list/tuple to
Array/Tuple. We always need a container representation for tuples
to be stored in Any.

### Map

`Map<K, V>` provides a key-value based hashmap container that can accept dict-style parameters.

```cpp
#include <tvm/ffi/container/map.h>

void ExampleMap() {
  namespace ffi = tvm::ffi;

  ffi::Map<ffi::String, int> map0 = {{"Alice", 100}, {"Bob", 95}};

  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(map0.size(), 2);
  EXPECT_EQ(map0.at("Alice"), 100);
  EXPECT_EQ(map0.count("Alice"), 1);
}
```


Under the hood, Map is backed by a reference-counted Object `MapObj` that stores
a collection of Any values. The implementation provides a SmallMap variant that stores
values as an array and another variant that is based on a hashmap. The Map preserves insertion
order like Python dictionaries. Conversion from Any to `Map<K, V>` will result in
runtime checks of its elements because the type index only indicates `MapObj` as the backing storage.
If you want to defer such checks at the FFI function boundary, consider using `Map<Any, Any>` instead.
When passing dictionaries from Python, the values will be converted to `Map<Any, Any>` before
being passed into the Function.

**Performance note:** Repeatedly converting Any to `Map<K, V>` can incur repeated
checking overhead at each element. Consider using `Map<Any, Any>` to defer checking or only run conversion once.

### Optional

`Optional<T>` provides a safe way to handle values that may or may not exist.
We specialize Optional for `ffi::String` and Object types to be more compact,
using nullptr to indicate non-existence.

```cpp
#include <tvm/ffi/container/optional.h>

void ExampleOptional() {
  namespace ffi = tvm::ffi;
  ffi::Optional<int> opt0 = 100;
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(opt0.has_value(), true);
  EXPECT_EQ(opt0.value(), 100);

  ffi::Optional<ffi::String> opt1;
  EXPECT_EQ(opt1.has_value(), false);
  EXPECT_EQ(opt1.value_or("default"), "default");
}
```


### Variant

`Variant<Types...>` provides a type-safe union of different types.

```cpp
#include <tvm/ffi/container/variant.h>

void ExampleVariant() {
  namespace ffi = tvm::ffi;
  ffi::Variant<int, ffi::String> var0 = 100;
  // EXPECT_EQ is used here for demonstration purposes (testing framework)
  EXPECT_EQ(var0.get<int>(), 100);

  var0 = ffi::String("hello");
  std::optional<ffi::String> maybe_str = var0.as<ffi::String>();
  EXPECT_EQ(maybe_str.value(), "hello");

  std::optional<int> maybe_int2 = var0.as<int>();
  EXPECT_EQ(maybe_int2.has_value(), false);
}
```

Under the hood, Variant is a wrapper around Any that restricts the type to the specific types in the list.
