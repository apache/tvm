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
# Hexagon Proxy RPC server

The proxy RPC server for Hexagon is a wrapper which takes standard TVM RPC calls from a python host
to a remote Android device and forwards them across FastRPC to Hexagon. This RPC flow will be replaced 
by running a minimal RPC server directly on Hexagon. For now we provide a prototype forwarding RPC server 
for host driven execution on Hexagon.

## Compilation

Project inventory: 
* Android
  * libtvm_runtime.so (containing HexagonHostDeviceAPI src/runtime/Hexagon/proxy_rpc/device_api.cc)
  * tvm_rpc (C++ RPC server)
  * librpc_env (Hexagon specific RPC proxy environment)
    
* Hexagon
  * libhexagon_proxy_rpc_skel.so (Hexagon device code containing FastRPC endpoints for the Hexagon Proxy RPC server)

All Android and Hexagon device artifacts will be placed in `apps_hexagon_proxy_rpc` from which they can be pushed
to an attached `adb` device.

### Prerequisites

1. Android NDK version r19c or later.
2. Hexagon SDK version 4.0.0 or later.

Android NDK can be downloaded from https://developer.android.com/ndk.
Hexagon SDK is available at //developer.qualcomm.com/software/Hexagon-dsp-sdk.

### Compilation with TVM

Building the Hexagon Proxy RPC as a component of the main TVM build
used for Hexagon codegen can be achieved by setting `USE_HEXAGON_PROXY_RPC=ON`.
A minimal  example invocation for compiling TVM along with the Hexagon Proxy RPC server 
is included below:

```
cmake -DCMAKE_C_COMPILER=/path/to/clang \
      -DCMAKE_CXX_COMPILER=/path/to/clang++ \
      -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
      -DCMAKE_CXX_STANDARD=14 \
      -DUSE_RPC=ON \
      -DUSE_LLVM=/path/to/llvm/bin/llvm-config \
      -DUSE_HEXAGON_PROXY_RPC=ON \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DUSE_ANDROID_TOOLCHAIN=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
      -DUSE_HEXAGON_ARCH=v65|v66|v68 \
      -DUSE_HEXAGON_SDK=/path/to/Hexagon/SDK \
      -DUSE_HEXAGON_TOOLCHAIN=/path/to/Hexagon/toolchain/ ..
```

where `v65|v66|v68` means "one of" these architecture versions.
The Hexagon proxy RPC application (tvm_rpc) is an android binary and thus requires the use 
of an android toolchain for compilation. Similarly, the Hexagon tvm runtime 
requires the use of the Hexagon toolchain and depends on the Hexagon SDK. The 
resulting Hexagon launcher binaries can be found in the `apps_Hexagon_launcher`
subdirectory of the cmake build directory. The above command
will build support for Hexagon codegen in the TVM library that requires 
`USE_LLVM` to be set to an llvm-config that has the Hexagon target built in.


# Disclaimer

The Hexagon proxy RPC is intended for use with prototyping and does not utilize any
performance acceleration, as such the measured performance may be very poor.
