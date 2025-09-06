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
# Packaging

This guide explains how to package a tvm-ffi-based library into a Python ABI-agnostic wheel.
It demonstrates both source-level builds (for cross-compilation) and builds based on pre-shipped shared libraries.
At a high level, packaging with tvm-ffi offers several benefits:

- **ABI-agnostic wheels**: Works across different Python versions with minimal dependency.
- **Universally deployable**: Build once with tvm-ffi and ship to different environments, including Python and non-Python environments.

While this guide shows how to build a wheel package, the resulting `my_ffi_extension.so` is agnostic
to Python, comes with minimal dependencies, and can be used in other deployment scenarios.

## Build and Run the Example

Let's start by building and running the example.
First, obtain a copy of the tvm-ffi source code.

```bash
git clone https://github.com/apache/tvm --recursive
cd tvm/ffi
```

The examples are now in the examples folder. You can quickly build
and install the example using the following command.
```bash
cd examples/packaging
pip install -v .
```

Then you can run examples that leverage the built wheel package.

```bash
python run_example.py add_one
```

## Setup pyproject.toml

A typical tvm-ffi-based project has the following structure:

```
├── CMakeLists.txt          # CMake build configuration
├── pyproject.toml          # Python packaging configuration
├── src/
│   └── extension.cc        # C++ source code
├── python/
│   └── my_ffi_extension/
│       ├── __init__.py     # Python package initialization
│       ├── base.py         # Library loading logic
│       └── _ffi_api.py     # FFI API registration
└── README.md               # Project documentation
```

The `pyproject.toml` file configures the build system and project metadata.

```toml
[project]
name = "my-ffi-extension"
version = "0.1.0"
# ... more project metadata omitted ...

[build-system]
requires = ["scikit-build-core>=0.10.0", "apache-tvm-ffi"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
# ABI-agnostic wheel
wheel.py-api = "py3"
# ... more build configuration omitted ...
```

We use scikit-build-core for building the wheel. Make sure you add tvm-ffi as a build-system requirement.
Importantly, we should set `wheel.py-api` to `py3` to indicate it is ABI-generic.

## Setup CMakeLists.txt

The CMakeLists.txt handles the build and linking of the project.
There are two ways you can build with tvm-ffi:

- Link the pre-built `libtvm_ffi` shipped from the pip package
- Build tvm-ffi from source

For common cases, using the pre-built library and linking tvm_ffi_shared is sufficient.
To build with the pre-built library, you can do:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_ffi_extension)

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
# find the prebuilt package
find_package(tvm_ffi CONFIG REQUIRED)

# ... more cmake configuration omitted ...

# linking the library
target_link_libraries(my_ffi_extension tvm_ffi_shared)
```

There are cases where one may want to cross-compile or bundle part of tvm_ffi objects directly
into the project. In such cases, you should build from source.

```cmake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --sourcedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
# add the shipped source code as a cmake subdirectory
add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)

# ... more cmake configuration omitted ...

# linking the library
target_link_libraries(my_ffi_extension tvm_ffi_shared)
```
Note that it is always safe to build from source, and the extra cost of building tvm-ffi is small
because tvm-ffi is a lightweight library. If you are in doubt,
you can always choose to build tvm-ffi from source.
In Python or other cases when we dynamically load libtvm_ffi shipped with the dedicated pip package,
you do not need to ship libtvm_ffi.so in your package even if you build tvm-ffi from source.
The built objects are only used to supply the linking information.

## Exposing C++ Functions

The C++ implementation is defined in `src/extension.cc`.
There are two ways one can expose a function in C++ to the FFI library.
First, `TVM_FFI_DLL_EXPORT_TYPED_FUNC` can be used to expose the function directly as a C symbol that follows the tvm-ffi ABI,
which can later be accessed via `tvm_ffi.load_module`.

Here's a basic example of the function implementation:

```c++
void AddOne(DLTensor* x, DLTensor* y) {
  // ... implementation omitted ...
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, my_ffi_extension::AddOne);
```

We can also register a function into the global function table with a given name:

```c++
void RaiseError(ffi::String msg) {
  TVM_FFI_THROW(RuntimeError) << msg;
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("my_ffi_extension.raise_error", RaiseError);
});
```

Make sure to have a unique name across all registered functions when registering a global function.
Always prefix with a package namespace name to avoid name collisions.
The function can then be found via `tvm_ffi.get_global_func(name)`
and is expected to stay throughout the lifetime of the program.

We recommend using `TVM_FFI_DLL_EXPORT_TYPED_FUNC` for functions that are supposed to be dynamically
loaded (such as JIT scenarios) so they won't be exposed to the global function table.

## Library Loading in Python

The base module handles loading the compiled extension:

```python
import tvm_ffi
import os
import sys

def _load_lib():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    # Platform-specific library names
    if sys.platform.startswith("win32"):
        lib_name = "my_ffi_extension.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "my_ffi_extension.dylib"
    else:
        lib_name = "my_ffi_extension.so"

    lib_path = os.path.join(file_dir, lib_name)
    return tvm_ffi.load_module(lib_path)

_LIB = _load_lib()
```

Effectively, it leverages the `tvm_ffi.load_module` call to load the library
extension DLL shipped along with the package. The `_ffi_api.py` contains a function
call to `tvm_ffi.init_ffi_api` that registers all global functions prefixed
with `my_ffi_extension` into the module.

```python
# _ffi_api.py
import tvm_ffi
from .base import _LIB

# Register all global functions prefixed with 'my_ffi_extension.'
# This makes functions registered via TVM_FFI_STATIC_INIT_BLOCK available
tvm_ffi.init_ffi_api("my_ffi_extension", __name__)
```

Then we can redirect the calls to the related functions.

```python
from .base import _LIB
from . import _ffi_api

def add_one(x, y):
    # ... docstring omitted ...
    return _LIB.add_one(x, y)

def raise_error(msg):
    # ... docstring omitted ...
    return _ffi_api.raise_error(msg)
```

## Build and Use the Package

First, build the wheel:
```bash
pip wheel -v -w dist .
```

Then install the built wheel:
```bash
pip install dist/*.whl
```

Then you can try it out:

```python
import torch
import my_ffi_extension

# Create input and output tensors
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.empty_like(x)

# Call the function
my_ffi_extension.add_one(x, y)
print(y)  # Output: tensor([2., 3., 4., 5., 6.])
```

You can also run the following command to see how errors are raised and propagated
across language boundaries:

```python
python run_example.py raise_error
```

When possible, tvm-ffi will try to preserve tracebacks across language boundaries. You will see tracebacks like:
```
File "src/extension.cc", line 45, in void my_ffi_extension::RaiseError(tvm::ffi::String)
```

## Wheel Auditing

When using `auditwheel`, exclude `libtvm_ffi` as it will be shipped with the `tvm_ffi` package.

```bash
auditwheel repair --exclude libtvm_ffi.so dist/*.whl
```

As long as you import `tvm_ffi` first before loading the library, the symbols will be available.
