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

# Hexagon backend runtime

The Hexagon runtime implements the functionality necessary for executing ML
models on Hexagon hardware (or emulation).

The prerequisite is to have Hexagon SDK installed, version 4.0.0 or later.

It is also recommended to use as recent version of LLVM as possible, version
7.0.0 being the minimum (based on community feedback).

### Compiling TVM with support for Hexagon for host (x86)

TVM running on host can serve as a cross-compiler that produces machine code
for Hexagon. To enable that, certain elements of both, the compiler and the
runtime need to include Hexagon-specific functionality. For the compiler, it
is code generation, and for the runtime, it is the ability to represent
modules with Hexagon code. Since Hexagon codegen is based on LLVM, LLVM
codegen needs to be enabled as well. The set of cmake options to enable
Hexagon support is
```
USE_LLVM=llvm-config
USE_HEXAGON=ON
USE_HEXAGON_SDK=/path/to/sdk
```

### Compiling TVM runtime for non-x86

Aside from x86, there are two other platforms where support for Hexagon may
be relevant. One of them is obviously Hexagon itself, the other one is
Android. Neither of these platforms supports the compiler side of TVM, only
runtime, and so the only compiler-related cmake option from the x86 build
above can be omitted: USE_LLVM.

Additionally, for Android, set the toolchain and target flags:
```
ANDROID_ABI=aarch64-v8a
ANDROID_PLATFORM=android-28
CMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake
USE_HEXAGON=ON
USE_HEXAGON_ARCH=v65|v66|v68|v69|v73
USE_HEXAGON_SDK=/path/to/sdk
```

Building for Hexagon requires setting the C/C++ compiler to `hexagon-clang/++`:
```
CMAKE_C_COMPILER=hexagon-clang
CMAKE_CXX_COMPILER=hexagon-clang++
USE_HEXAGON=ON
USE_HEXAGON_ARCH=v65|v66|v68|v69|v73
USE_HEXAGON_SDK=/path/to/sdk
USE_RPC=OFF
USE_LIBBACKTRACE=OFF
```

As mentioned before, only build the `runtime` component (e.g. `make runtime`).

Please note that the Hexagon SDK version needs to support the architecture
specified in `USE_HEXAGON_ARCH`.
