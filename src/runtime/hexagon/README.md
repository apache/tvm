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

The Hexagon runtime is a part of the TVM runtime that facilitates communication between a host and a Hexagon device. There are two types of host/device arrangements that are supported:
- X86/Linux host running Hexagon simulator,
- Android/AArch64 host running on a physical device containing a Hexagon module (i.e. CSDP or ADSP).

The TVM runtime that contains Hexagon runtime is the one executing on host.  In either case, there will need to be a separate TVM runtime (i.e.  the `libtvm_runtime.so` library) compiled for execution on Hexagon.

The prerequisite is to have Hexagon SDK installed, preferably version 3.5.0 or later. The Hexagon SDK can be downloaded from https://developer.qualcomm.com/software/hexagon-dsp-sdk.

It is also recommended to use as recent version of LLVM as possible, version 7.0.0 being the minimum (based on community feedback).

### Compiling TVM runtime for x86

This will use Hexagon simulator, which is provided in the Hexagon SDK.

When configuring TVM (cmake), set the following variables:
```
USE_LLVM=llvm-config
USE_HEXAGON_DEVICE=sim
USE_HEXAGON_SDK=/path/to/sdk
```

You can then build the entire TVM with the usual command (e.g. `make`).

### Compiling TVM runtime for Android

This will use FastRPC mechanism to communicate between the AArch64 host and Hexagon.

When configuring TVM (cmake), set the following variables:
```
USE_LLVM=llvm-config
USE_HEXAGON_DEVICE=device
USE_HEXAGON_SDK=/path/to/sdk
```

You will need Android clang toolchain to compile the runtime.  It is provided in Android NDK r19 or newer.

Set the C/C++ compiler to the Android clang for aarch64, and pass `-DCMAKE_CXX_FLAGS='-stdlib=libc++'` to the cmake command.

Only build the `runtime` component of TVM (e.g. `make runtime`), building the entire TVM will not work.

### Compiling TVM runtime for Hexagon

The TVM runtime executing on Hexagon does not need to have support for Hexagon device in it (as it is only for communication between host and Hexagon device). In fact, it's only needed for basic services (like thread control), and so it should not contain support for any devices.

When configuring TVM (cmake), set the following variables:
```
USE_RPC=OFF
USE_LLVM=OFF
USE_HEXAGON_DEVICE=OFF
USE_HEXAGON_SDK=/path/to/sdk
```

Please note that while suport for a Hexagon device is disabled, the Hexagon SDK is still needed and the path to it needs to be passed to cmake.

Set the C/C++ compiler to `hexagon-clang` (included in the Hexagon SDK), and set `CMAKE_CXX_FLAGS='-stdlib=libc++'`.

As in the case of Android, only build the `runtime` component (e.g.  `make runtime`).

