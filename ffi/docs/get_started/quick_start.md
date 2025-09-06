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
# Quick Start

This is a quick start guide explaining the basic features and usage of tvm-ffi.
The source code can be found at `examples/quick_start` in the project source.

## Build and Run the Example

Let us first get started by build and run the example. The example will show us:

- How to expose c++ functions as tvm ffi ABI function
- How to load and run tvm-ffi based library from python
- How to load and run tvm-ffi based library from c++


Before starting, ensure you have:

- TVM FFI installed following [installation](./install.md)
- C++ compiler with C++17 support
- CMake 3.18 or later
- (Optional) CUDA toolkit for GPU examples
- (Optional) PyTorch for checking torch integrations

Then obtain a copy of the tvm-ffi source code.

```bash
git clone https://github.com/apache/tvm --recursive
cd tvm/ffi
```

The examples are now in the example folder, you can quickly build
the example using the following command.
```bash
cd examples/quick_start
cmake -B build -S .
cmake --build build
```

After the build finishes, you can run the python examples by
```
python run_example.py
```

You can also run the c++ example

```
./build/example
```

## Walk through the Example

Now we have quickly try things out. Let us now walk through the details of the example.
Specifically, in this example, we create a simple "add one" operation that adds 1 to each element of an input
tensor and expose that function as TVM FFI compatible function. The key file structures are as follows:

```
examples/quick_start/
├── src/
│   ├── add_one_cpu.cc      # CPU implementation
│   ├── add_one_cuda.cu     # CUDA implementation
│   └── run_example.cc      # C++ usage example
├── run_example.py          # Python usage example
├── run_example.sh          # Build and run script
└── CMakeLists.txt          # Build configuration
```

### CPU Implementation

```cpp
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

namespace tvm_ffi_example {

void AddOne(DLTensor* x, DLTensor* y) {
  // Validate inputs
  TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
  DLDataType f32_dtype{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
  TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
  TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
  TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";

  // Perform the computation
  for (int i = 0; i < x->shape[0]; ++i) {
    static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
  }
}

// Expose the function through TVM FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, tvm_ffi_example::AddOne);
}
```

**Key Points:**
- Functions take `DLTensor*` parameters for cross-language compatibility
- The `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro exposes the function with a given name

### CUDA Implementation

```cpp
void AddOneCUDA(DLTensor* x, DLTensor* y) {
  // Validation (same as CPU version)
  // ...

  int64_t n = x->shape[0];
  int64_t nthread_per_block = 256;
  int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;

  // Get current CUDA stream from environment
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetCurrentStream(x->device.device_type, x->device.device_id));

  // Launch kernel
  AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(
      static_cast<float*>(x->data), static_cast<float*>(y->data), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvm_ffi_example::AddOneCUDA);
```

**Key Points:**
- We use `TVMFFIEnvGetCurrentStream` to obtain the current stream from the environement
- When invoking ffi Function from python end with PyTorch tensor as argument,
  the stream will be populated with torch's current stream.


### Working with PyTorch

Atfer build, we will create library such as `build/add_one_cuda.so`, that can be loaded by
with api {py:func}`tvm_ffi.load_module` that returns a {py:class}`tvm_ffi.Module`
Then the function will become available as property of the loaded module.
The tensor arguments in the ffi functions automatically consumes `torch.Tensor`. The following code shows how
to use the function in torch.

```python
import torch
import tvm_ffi

if torch.cuda.is_available():
    mod = tvm_ffi.load_module("build/add_one_cuda.so")

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    # TVM FFI automatically handles CUDA streams
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        mod.add_one_cuda(x, y)
    stream.synchronize()
```

### Working with Python Data Arrays

TVM FFI functions works automaticaly with python data arrays that are compatible with dlpack.
The following examples how to use the function with numpy.

```python
import tvm_ffi
import numpy as np

# Load the compiled module
mod = tvm_ffi.load_module("build/add_one_cpu.so")

# Create input and output arrays
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.empty_like(x)

# Call the function
mod.add_one_cpu(x, y)
print("Result:", y)  # [2, 3, 4, 5, 6]
```

### Working with C++

One important design goal of tvm-ffi is to be universally portable.
As a result, the result libraries do not have explicit dependencies in python
and can be loaded in other language environments, such as c++. The following code
shows how to run the example exported function in C++.

```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

void CallAddOne(DLTensor* x, DLTensor *y) {
  namespace ffi = tvm::ffi;
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cpu.so");
  ffi::Function add_one_cpu = mod->GetFunction("add_one_cpu").value();
  add_one_cpu(x, y);
}
```

## Summary Key Concepts

- **TVM_FFI_DLL_EXPORT_TYPED_FUNC** exposes a c++ function into tvm-ffi C ABI
- **DLTensor** is a universal tensor structure that enables zero-copy exchange of array data
- **Module loading** is provided by tvm ffi APIs in multiple languages.
