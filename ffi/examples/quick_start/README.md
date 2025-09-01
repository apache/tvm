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

# Getting Started with TVM FFI

This example demonstrates how to use tvm-ffi to expose a universal function
that can be loaded in different environments.

The example implements a simple "add one" operation that adds 1 to each element
of an input tensor, showing how to create C++ functions callable from Python.

You can run this quick start example by:

```bash
# ensure you installed tvm-ffi first
pip install -e ../..

# Build and run the complete example
./run_example.sh
```

At a high level, the `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro helps to expose
a C++ function into the TVM FFI C ABI convention for functions.
Then the function can be accessed by different environments and languages
that interface with the TVM FFI. The current example shows how to do so
in Python and C++.

## Key Files

- `src/add_one_cpu.cc` - CPU implementation of the add_one function
- `src/add_one_cuda.cu` - CUDA implementation for GPU operations
- `run_example.py` - Python example showing how to call the functions
- `run_example.cc` - C++ example demonstrating the same functionality

## Compile without CMake

You can also compile the modules directly using
flags provided by the `tvm-ffi-config` tool.

```bash
g++ -shared -fPIC `tvm-ffi-config --cxxflags`  \
    src/add_one_cpu.cc -o build/add_one_cpu.so \
    `tvm-ffi-config --ldflags` `tvm-ffi-config --libs`
```
