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
# ABI Overview

This section provides an overview of the ABI convention of TVM FFI. The ABI
is designed around the following key principles:

- **Stable C ABI:** Core ABI is defined on top of a stable C ABI.
- **Minimal and efficient:** Keep things simple when possible and bring close-to-metal efficiency.
- **Focus on machine learning systems:** while also ensuring reasonable extensibility.

To explain the concepts in the following sections, we will write in **low-level C/C++ code** when possible,
so the code itself illustrates the low-level semantics of how to work with the ABI convention.
These can serve as references for how to build language bindings and compiler codegen for the ABI.

```{note}
The authoritative ABI specifications are defined in [tvm/ffi/c_api.h](https://github.com/apache/tvm/blob/main/ffi/include/tvm/ffi/c_api.h) for core ABI,
and [tvm/ffi/extra/c_env_api.h](https://github.com/apache/tvm/blob/main/ffi/include/tvm/ffi/extra/c_env_api.h) for extra support features
such as stream handling. This document provides explanations about design concepts and rationales.
```

## Simplified Example

Before diving into details, it is helpful to review at a high level
what happens when a function is called in TVM FFI ABI.
One main design goal here is to represent all kinds of functions in a single
unified C signature. Please review the following
simplified code example that illustrates the key idea:

```c++
// simplified struct for TVMFFIAny
typedef struct TVMFFIAny {
  int32_t type_index;
  uint32_t zero_padding;
  // union values
  union {
    int64_t v_int64;       // integers
    double v_float64;      // floating-point numbers
    const char* v_c_str;   // raw C-string
  };
};

// This is the signature of TVM FFI function ABI
typedef int (*TVMFFISafeCallType)(
   void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
);

// An example function signature
int MyFunc(const char* param0, int param1);

// This is what MyFunc looks like when exposed through TVM FFI ABI
int MyFuncTVMFFISafeCall(
  void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
) {
  assert(args[0].type_index == kTVMFFIRawStr);
  assert(args[1].type_index == kTVMFFInt);
  result->type_index = kTVMFFInt;
  result->v_int64 = MyFunc(args[0].v_c_str, args[1].v_int64);
  // return value indicates no error occurred
  return 0;
}

// This is how we call the MyFuncTVMFFISafeCall
// this can happen on the caller side in another language (e.g. python)
int CallTVMFFISafeCall(const char* param0, int param1) {
  // arguments on stack
  TVMFFIAny args[2], result;
  args[0].type_index = kTVMFFIRawStr;
  args[0].v_c_str = param0;
  args[1].type_index = kTVMFFInt;
  args[1].v_int64 = param1;
  result.type_index = kTVMFFINone;
  // In this case we do not need handle
  // handle is used to hold closure pointers
  void* handle = nullptr;
  int num_args = 2;
  MyFuncTVMFFISafeCall(handle, args, num_args, &result);
  return result.v_int64;
}
```

At a high level, the `TVMFFISafeCallType` signature does the following things:
- Arguments and return values are stored in structured `TVMFFIAny`
  - Each value comes with a `type_index` to indicate its type
  - Values are stored in union fields, depending on the specific type.
- Caller can explicitly store the type index and value into
  a stack of `TVMFFIAny`.
- Callee can load the parameters from args and check their type indices.

In this way, the same `TVMFFISafeCallType` can be used to represent any function
that contains an arbitrary number of arguments and types that can be identified by `type_index`.
Of course, this is a simplified example and we did not touch on specific details
like Any value format and error handling. The following sections will provide a more systematic
treatment of each of these specific topics.
You can keep this example in mind as the overall picture and refine it as you read through
the following sections.


## TVMFFIAny Storage Format

To start with, we need a mechanism to store the values that are passed across machine learning frameworks.
It achieves this using a core data structure called TVMFFIAny.

```c++
typedef struct TVMFFIAny {
  int32_t type_index;
  union {  // 4 bytes
    uint32_t zero_padding;
    uint32_t small_str_len;
  };
  // union values
  union {
    int64_t v_int64;       // integers
    double v_float64;      // floating-point numbers
    void* v_ptr;           // typeless pointers
    const char* v_c_str;   // raw C-string
    TVMFFIObject* v_obj;   // ref counted objects
    DLDataType v_dtype;    // data type
    DLDevice v_device;     // device
    char v_bytes[8];       // small string
    ...
  };
} TVMFFIAny;
```

TVMFFIAny is a 16-byte C structure that follows the design principle of tagged-union:

- `type_index` helps us identify the type being stored.
- The value union part is designed to store the value:
  - Small POD values (like integers and floats) are stored directly as "on-stack" values.
  - `v_obj` can also point to a managed heap-allocated object, which we will discuss next.
- The second field stores metadata for small strings.


### Storing a POD Value

There are many values that are plain-old-data types. In such cases, we store them directly
on-stack in the value part of the TVMFFIAny. The following example shows how to store
an int.

```c++
void SetIntValue(TVMFFIAny* any, int value) {
  // must zero the entire space first
  any->type_index = kTVMFFIInt;
  any->zero_padding = 0;
  any->v_int64 = value;
}
```

:::{note}

We **must zero the content that is not being used** by
the current value type. The following example shows a common place
where mistakes can be made when we forget to zero the value field
on 32-bit platforms (where pointers only fill the 32-bit part of the value).

```c++
void SetOpaquePtrValue(TVMFFIAny* any, void* opaque_ptr) {
  any->type_index = kTVMFFIOpaquePtr;
  // must zero the padding
  any->zero_padding = 0;
  // the zeroing is needed for 32-bit platforms!
  any->v_uint64 = 0;
  any->v_ptr = opaque_ptr;
}
```

**Rationale:** Such invariants allow us to directly compare
and hash TVMFFIAny in bytes for quick equality checks without going through
type index switching.
:::


## Object Storage Format

When TVMFFIAny points to a heap-allocated object (such as n-dimensional arrays),
we adopt a unified object storage format, defined as follows:

```c++
typedef struct TVMFFIObject {
  int32_t type_index;
  uint32_t weak_ref_count;
  uint64_t strong_ref_count;
  union {
    void (*deleter)(struct TVMFFIObject* self, int flags);
    int64_t __ensure_align;
  };
} TVMFFIObject;
```

`TVMFFIObject` defines a common 24-byte intrusive header that all in-memory objects share:

- `type_index` helps us identify the type being stored, which is consistent with `TVMFFIAny.type_index`.
- `weak_ref_count` stores the weak atomic reference counter of the object.
- `strong_ref_count` stores the strong atomic reference counter of the object.
- `deleter` should be called when either the strong or weak ref counter goes to zero.
  - The flags are set to indicate the event of either weak or strong going to zero, or both.
  - When `strong_ref_count` gets to zero, the deleter needs to call the destructor of the object.
  - When `weak_ref_count` gets to zero, the deleter needs to free the memory allocated by self.

**Rationales:** There are several considerations when designing the data structure:
- `type_index` enables runtime dynamic type checking and casting.
- We introduce weak/strong ref counters so we can be compatible with systems that need weak pointers.
- The weak ref counter is kept as 32-bit so we can pack the object header as 24 bytes.
- `deleter` ensures that objects allocated from one language/runtime can be safely deleted in another.

The object format provides a unified way to manage object life-cycle and dynamic type casting
for heap-allocated objects, including Shape, Tensor,
Function, Array, Map and other custom objects.


### DLPack Compatible Tensor

We provide first-class support for DLPack raw unmanaged pointer support as well as a managed Tensor object that
directly adopts the DLPack DLTensor layout. The overall layout of the Tensor object is as follows:

```c++
struct TensorObj: public ffi::Object, public DLTensor {
};
```

That means we can read out the array buffer information from an `TVMFFIAny`
in the following way:

```c++
DLTensor* ReadDLTensorPtr(const TVMFFIAny *value) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    return static_cast<DLTensor*>(value->v_ptr);
  }
  assert(value->type_index == kTVMFFITensor);
  return reinterpret_cast<DLTensor*>(
    reinterpret_cast<char*>(value->v_obj) + sizeof(TVMFFIObject));
}
```
The above code can be used as a reference to implement compiler codegen for data.
Note that the C++ API automatically handles such conversion.

### Advanced: Dynamic Type Index

The `TVMFFITypeIndex` defines a set of type indices. Each built-in type has a corresponding statically
assigned type index that is defined in the enum. Static type indices should be sufficient for most
library use cases.
For advanced use cases we also support user-defined objects whose `type_index` are assigned at startup time
by calling `TVMFFITypeGetOrAllocIndex` with a unique
`type_key` string. This design allows us to enable decentralized extension of the objects as long as the `type_key`
values are unique by appending namespace prefix to the key.

## AnyView and Managed Any

An `TVMFFIAny` can either be treated as a strongly managed value (corresponding to `ffi::Any` in C++),
or an unmanaged value (corresponding to `ffi::AnyView` in C++).
- For POD types, there is no difference between the two
- For object types, copying of AnyView should not change reference counters, while copying and deletion
  of managed Any should result in increase and decrease of strong reference counters.
- When we convert AnyView to Any, we will convert raw C string `const char*` and `const TVMFFIByteArray*`
  into their managed counterparts (String and Bytes).
- C API function `TVMFFIAnyViewToOwnedAny` is provided to perform such conversion.

Unless the user is writing a compiler backend that needs low-level C style access, we encourage use of the
C++ API to automatically manage conversion and casting between normal types and Any. The following code
shows some example usage of the C++ API.

```c++
#include <tvm/ffi/any.h>

void AnyExample() {
  namespace ffi = tvm::ffi;
  // Here is a managed any
  ffi::Any value = "hello world";
  // explicit cast to a specific type
  ffi::String str_value = value.cast<ffi::String>();
  // copy int to value
  value = 1;
  // copy into a view
  ffi::AnyView view = value;
  // cast view back to int
  std::cout << "Value is " << view.cast<int>() << std::endl;
}
```

`ffi::Any` can serve as a container type to hold managed values that can be recognized by the TVM FFI system.
They can be composed with container structures such as `Map<String, Any>`, `Array<Any>` to represent various
broad patterns in APIs that may appear in ML systems.

## Function Calling Convention

As discussed in the overview, we need to consider foreign function calls as first-class citizens. We adopt a single standard C function as follows:

```c++
typedef int (*TVMFFISafeCallType)(
   void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
);
```

The handle contains the pointer to the function object itself, allowing us to support closures. args and num_args describe the input arguments and results store the return value. When args and results contain heap-managed objects, we expect the caller to own args and result.

```{note}
Before calling the function, caller must set `result->type_index` to be kTVMFFINone, or any type index that do not corresponds
to an on-heap object.

**Rationale:** Simplifies callee implementation as initial state of result can be viewed as managed Any.
```

We call this approach a packed function, as it provides a single signature to represent all functions in a "type-erased" way. It saves the need to declare and jit shim for each FFI function call while maintaining reasonable efficiency. This mechanism enables the following scenarios:
- Calling from Dynamic Languages (e.g., Python): we provide a tvm_ffi binding that prepares the args based on dynamically examining Python arguments passed in.
- Calling from Static Languages (e.g., C++): For static languages, we can leverage C++ templates to directly instantiate the arguments on the stack, saving the need for dynamic examination.
- Dynamic language Callbacks: the signature enables us to easily bring dynamic language (Python) callbacks as ffi::Function, as we can take each argument and convert to the dynamic values.
- Efficiency: In practice, we find this approach is sufficient for machine learning focused workloads. For example, we can get to microsecond level overhead for Python/C++ calls, which is generally similar to overhead for eager mode. When both sides of calls are static languages, the overhead will go down to tens of nanoseconds. As a side note, although we did not find it necessary, the signature still leaves room for link time optimization (LTO), when both sides are static languages with a known symbol and linked into a single binary when we inline the callee into caller side and the stack argument memory passing into register passing.

We support first-class Function objects that allow us to also pass function/closures from different places around, enabling cool usages such as quick python callback for prototyping, and dynamic Functor creation for driver-based kernel launching.


## Error Handling

Most TVM FFI C API calls, including `TVMFFISafeCallType` uses the return value to
indicate whether an error happens. When an error happens during a function call,
a non-zero value will be returned. The callee needs also to set the error through `TVMFFIErrorSetRaisedFromCStr` or `TVMFFIErrorSetRaised` API, which stores
the error on a thread-local storage.

```c++
// Example function that raises an error
int ErrorFunc(void* handle, const TVMFFIAny* args, int num_args, TVMFFIAny *result) {
  const char* error_kind = "RuntimeError";
  const char* error_msg = "error message";
  // set the thread-local error state
  TVMFFIErrorSetRaisedFromCStr(error_kind, error_msg);
  return -1;
}
```

The caller can retrieve the error from thread-local error storage
using `TVMFFIErrorMoveFromRaised` function.
The ABI stores Error also as a specific Object,
the overall error object is stored as follows
```c++
typedef struct {
  /*! \brief The kind of the error. */
  TVMFFIByteArray kind;
  /*! \brief The message of the error. */
  TVMFFIByteArray message;
  /*! \brief The traceback of the error. */
  TVMFFIByteArray traceback;
  /*!
   * \brief Function handle to update the traceback of the error.
   * \param self The self object handle.
   * \param traceback The traceback to update.
   */
  void (*update_traceback)(TVMFFIObjectHandle self, const TVMFFIByteArray* traceback);
} TVMFFIErrorCell;

// error object
class ErrorObj : public ffi::Object, public TVMFFIErrorCell {
};
```

The error object stores kind, message and traceback as string. When possible,
we store the traceback in the same format of python traceback (see an example as follows):
```
File "src/extension.cc", line 45, in void my_ffi_extension::RaiseError(tvm::ffi::String)
```

We provide C++ object `ffi::Error` that can be throwed as exception in c++ environment. When we encounter
the C ABI boundary, we will catch the error and call `TVMFFIErrorSetRaised` to propagate the error
to the caller safely.
`TVMFFIErrorSetRaisedFromCStr` is a convenient method to set error directly from C string and can be useful in compiler backend construction to implement features such as assert.

**Rationales:** The error object contains minimal but sufficient information to reconstruct structured
error in python side. We opt-for thread-local error state as it simplifies overall support.

## String and Bytes

The ABI supports strings and bytes as first-class citizens. A string can take multiple forms that are identified by
its `type_index`.

- `kTVMFFIRawStr`: raw C string terminated by `\0`.
- `kTVMFFISmallStr`: small string, the length is stored in `small_str_len` and data is stored in `v_bytes`.
- `kTVMFFIStr`: on-heap string object for strings that are longer than 7 characters.

The following code shows the layout of the on-heap string object.
```c++
// span-like data structure to store header and length
typedef struct {
  const char* data;
  size_t size;
} TVMFFIByteArray;

// showcase the layout of the on-heap string.
class StringObj : public ffi::Object, public TVMFFIByteArray {
};
```

The following code shows how to read a string from `TVMFFIAny`
```c++
TVMFFIByteArray ReadString(const TVMFFIAny *value) {
  TVMFFIByteArray ret;
  if (value->type_index == kTVMFFIRawStr) {
    ret.data = value->v_c_str;
    ret.size = strlen(ret.data);
  } else if (value->type_index == kTVMFFISmallStr) {
    ret.data = value->v_bytes;
    ret.size = value->small_str_len;
  } else {
    assert(value->type_index == kTVMFFIStr);
    ret = *reinterpret_cast<TVMFFIByteArray*>(
      reinterpret_cast<char*>(value->v_obj) + sizeof(TVMFFIObject));
  }
  return ret;
}
```

Similarly, we have type indices to represent bytes. The C++ API provides classes
`ffi::String` and `ffi::Bytes` to enable the automatic conversion of these values with Any storage format.

**Rationales:** Separate string and bytes enable clear mappings from the Python side. Small string allows us to
store short names on-stack. To favor 8-byte alignment (v_bytes) and keep things simple, we did not further
pack characters into the `small_len` field.
