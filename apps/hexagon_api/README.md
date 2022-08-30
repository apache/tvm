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

# Hexagon API app

This is a meta-app that build the necessary binaries for use with
the `HexagonLauncher` utility from `tvm.contrib.hexagon`.

It will build the TVM runtime for Android, the RPC server application
for Android, and the RPC library for Hexagon with the TVM runtime for
Hexagon built into it.

## Configuration

There is a set of configuration variables that are required for cmake:
- `ANDROID_ABI`: Set this to `arm64-v8a`.
- `ANDROID_PLATFORM`: This can be `android-28`.
- `USE_ANDROID_TOOLCHAIN`: The path to the Android toolchain file, i.e.
`android.toolchain.cmake`. This file is a part of the Android NDK.
- `USE_HEXAGON_ARCH`: The version string of the Hexagon architecture
to use, i.e. vNN. The typical setting would be `v68` or later.
- `USE_HEXAGON_SDK`: The path to the Hexagon SDK. Set this path in such
a way that `${USE_HEXAGON_SDK}/setup_sdk_env.source` exists.
- `USE_HEXAGON_TOOLCHAIN`: Path to Hexagon toolchain. It can be the
Hexagon toolchain included in the SDK, for example
`${USE_HEXAGON_TOOLCHAIN}/tools/HEXAGON_Tools/x.y.z/Tools`.  The `x.y.z`
in the path is the toolchain version number, which is specific to the
version of the SDK.

Additionally, the variable `USE_OUTPUT_BINARY_DIR` can be set to indicate
the location where the generated binaries will be placed. If not set, it
defaults to `hexagon_rpc` subdirectory in the current build directory.


## Build

The build will generate the following binaries:
- `tvm_runtime.so`: TVM runtime for Android (shared library).
- `tvm_rpc_android`: RPC server for Android.
- `libhexagon_rpc_skel.so`: RPC library for Hexagon.
- `libtvm_runtime.a`: TVM runtime for Hexagon (static library).

The RPC library for Hexagon contains the TVM runtime, so the static
TVM runtime for Hexagon is not strictly necessary.
