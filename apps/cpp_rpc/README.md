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

# TVM RPC Server
This folder contains a simple recipe to make RPC server in c++.

## Usage (Non-Windows)
- Configure the tvm cmake build with `config.cmake` ensuring that `USE_CPP_RPC` is set to `ON` in the config.
- If cross compiling for Android, add the following options to the cmake config or specify them when invoking cmake:
```
  # Whether to build the C++ RPC server binary
  set(USE_CPP_RPC ON)
  # Path to the Android NDK cmake toolchain
  set(CMAKE_TOOLCHAIN_FILE $ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake)
  # The Android ABI and platform to target
  set(ANDROID_ABI "arm64-v8a")
  set(ANDROID_PLATFORM android-28)
  ```
- Similarly, if cross compiling for embedded Linux add the following options to cmake config:
```
  # Needed to ensure pthread is linked
  set(OS Linux)
  # Path to the desired C++ cross compiler
  set(CMAKE_CXX_COMPILER /path/to/cross/compiler/executable)
```
- If you need to build cpp_rpc with OpenCL support, specify variable `USE_OPENCL` in the config:
  ```
  set(USE_OPENCL ON)
  ```
  In this case [OpenCL-wrapper](../../src/runtime/opencl/opencl_wrapper) or OpenCL installed to your system will be used.
  When OpenCL-wrapper is used, it will dynamically load OpenCL library on the device.
  If the device doesn't have OpenCL library on it, then you'll see in the runtime that OpenCL library cannot be opened.

  If linking against a custom device OpenCL library is needed, in the config specify the path to the OpenCL SDK containing the include/CL headers and lib/ or lib64/libOpenCL.so:
```
  set(USE_OPENCL /path/to/opencl-sdk)
```

- From within the configured tvm build directory, compile `tvm_runtime` and the `tvm_rpc` server:
```
  cd $TVM_ROOT/build
  make -jN tvm_runtime tvm_rpc
```
- Use `./tvm_rpc server` to start the RPC server

## Usage (Windows)
- Configure the tvm cmake build with `config.cmake` ensuring that `USE_CPP_RPC` is set to `ON` in the config.
- Install [LLVM pre-build binaries](https://releases.llvm.org/download.html), making sure to select the option to add it to the PATH.
- Verify Python 3.6 or newer is installed and in the PATH.
- Use `<tvm_output_dir>\tvm_rpc.exe` to start the RPC server

## How it works
- The tvm runtime dll is linked along with this executable and when the RPC server starts it will load the tvm runtime library.

```
Command line usage
 server       - Start the server
--host        - The hostname of the server, Default=0.0.0.0
--port        - The port of the RPC, Default=9090
--port-end    - The end search port of the RPC, Default=9199
--tracker     - The RPC tracker address in host:port format e.g. 10.1.1.2:9190 Default=""
--key         - The key used to identify the device type in tracker. Default=""
--custom-addr - Custom IP Address to Report to RPC Tracker. Default=""
--silent      - Whether to run in silent mode. Default=False
  Example
  ./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 --tracker=127.0.0.1:9190 --key=rasp
```

## Note
Currently support is only there for Linux / Android / Windows environment and proxy mode isn't supported currently.
