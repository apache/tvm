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

# Hexagon IDL libraries

This directory hosts IDL files and their implementations to offload TVM kernels to Hexagon via FastRPC. The implementations can be used to generate stub and skel libraries.

### Prerequisites

1. Android NDK version r19c or later.
2. Hexagon SDK version 3.5.0 or later.

Android NDK can be downloaded from https://developer.android.com/ndk.
Hexagon SDK is available at //developer.qualcomm.com/software/hexagon-dsp-sdk.

### Configuring

Skel and stub libraries need to be configured and built separately. Please use different subdirectories for each. Otherwise the cmake cache from one configuration can interfere with the next.

For skel libraries, set
```
FASTRPC_LIBS=SKEL
HEXAGON_SDK_ROOT=/path/to/sdk
CMAKE_C_COMPILER=hexagon-clang
CMAKE_CXX_COMPILER=hexagon-clang++
HEXAGON_ARCH= one of v60, v62, v65, v66
```

Please note that support for older versions of the Hexagon processor may be removed from the future versions of the Hexagon toolchain.


For stub libraries, set
```
FASTRPC_LIBS=STUB
HEXAGON_SDK_ROOT=/path/to/sdk
CMAKE_C_COMPILER=aarch64-linux-android28-clang      # or later
CMAKE_CXX_COMPILER=aarch64-linux-android28-clang++  # or later
```

### Building

In each instance, simple `make` command will create header files `fastrpc/include/tvm_remote.h` and `fastrpc/include/tvm_remote_nd.h`. These headers are needed to compile the TVM runtime for Android (and the stub/skel libraries themselves).
