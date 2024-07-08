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

# Test TVM on Hexagon
This document explains various pieces that are involved in testing TVM on an Android device which includes Hexagon DSP or Hexagon simulator.

## What is HexagonLauncherRPC?
HexagonLauncherRPC is a class to handle interactions with an Android phone which includes Hexagon DSP or Hexagon simulator to run a TVMModule(function/operation/graph) on Hexagon. HexagonLauncherRPC reuses [minRPC](https://github.com/apache/tvm/tree/main/src/runtime/minrpc) implementation to set up an RPC connection from host (your local machine) to Hexagon target, and it is passed through Android RPC server.

## Build Required Tools/Libraries
To build TVM for Hexagon and run tests you need to run multiple steps which includes preparing required tools, setting up environment variables and building various versions of TVM. Alternatively, you can skip these instructions and use docker image which has pre-installed required tools. We highly recommend to use docker, especially if this is your first time working with Hexagon. For instructions on using docker image follow ["use hexagon docker image"](#use-hexagon-docker-image).

- Build TVMRuntime library and C++ RPC server for Android.
- Build minRPC server along with FastRPC for Hexagon.
- Build TVM library with Hexagon support for host machine.
- Build TVMRuntime library and RPC server for host machine.

First, ensure to export Clang libraries to `LD_LIBRARY_PATH` and Hexagon toolchain to `HEXAGON_TOOLCHAIN`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"Path to `llvm-clang/lib` sub-directory. Currently we use LLVM-13 in TVM CI."

export HEXAGON_TOOLCHAIN="Path to Hexagon toolchain. It can be the Hexagon toolchain included in the SDK, for example `HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/x.y.z/Tools`.  The `x.y.z` in the path is the toolchain version number, which is specific to the version of the SDK."
```

You can find more information about downloading [Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk).

First build Hexagon API application under `apps/hexagon_api`. This step will generate `tvm_rpc_android` and `libtvm_runtime.so` to run on Android. Also, it generates `libtvm_runtime.a` `libtvm_runtime.so`, `libhexagon_rpc_skel.so` and `libhexagon_rpc_sim.so` to run on Hexagon device or Hexagon simulator.

**Note:** To get the most updated instructions, please take a look at [task_build_hexagon_api.sh](https://github.com/apache/tvm/blob/main/tests/scripts/task_build_hexagon_api.sh).

```bash
cd apps/hexagon_api
mkdir build
cd build
cmake -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-28 \
        -DUSE_ANDROID_TOOLCHAIN="path to `android-ndk/build/cmake/android.toolchain.cmake` file" \
        -DUSE_HEXAGON_ARCH=v65|v66|v68|v69|v73|v75 \
        -DUSE_HEXAGON_SDK="path to Hexagon SDK" \
        -DUSE_HEXAGON_TOOLCHAIN="path to Hexagon toolchain `Tools` sub-directory which explained above" \
        -DUSE_OUTPUT_BINARY_DIR="path to `build/hexagon_api_output` which is a sub-directory of `tvm`" ..

make -j2
```

Next, we need to build TVM on host with RPC and Hexagon dependencies. To do that follow these commands.

**Note:** To get the most recent configs for this step, please take a look at [task_config_build_hexagon.sh](https://github.com/apache/tvm/blob/main/tests/scripts/task_config_build_hexagon.sh).

```bash
cd tvm
mkdir build
cd build
cmake -DUSE_LLVM="path to `llvm/bin/llvm-config`" \
        -DUSE_RPC=ON \
        -DCMAKE_CXX_COMPILER="path to `clang++` executable" \
        -DUSE_HEXAGON_SDK="path to Hexagon SDK" \
        -DUSE_HEXAGON=ON ..

make -j2
```

## Use Hexagon Docker Image
To use hexagon docker image, install TVM and Hexagon API follow these steps from your TVM home directory:

```bash
# Log in to docker image
./docker/bash.sh ci_hexagon

# Build TVM
rm -rf build
./tests/scripts/task_config_build_hexagon.sh build
cd build
cmake ..
make -j2

# Build Hexagon API
cd ..
./tests/scripts/task_build_hexagon_api.sh
```

Now that you have built required tools, you can jump to [run test examples](#run-tests).

## Run Tests
You have the options of running Hexagon test on real hardware or on Hexagon simulator. Also, depending on whether you decided to use Hexagon docker image or not we will explain both cases here.

**Note: You can always find updated instructions based on this [script](https://github.com/apache/tvm/blob/main/tests/scripts/task_python_hexagon.sh).**

### Only follow these steps if running tests outside of docker
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"path to `llvm-clang/lib` sub-directory"

export HEXAGON_TOOLCHAIN="Path to Hexagon toolchain. It can be the Hexagon toolchain included in the HexagonSDK, for example `HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/x.y.z/Tools`.  The `x.y.z` in the path is the toolchain version number, which is specific to the version of the SDK."

export PYTHONPATH=$PYTHONPATH:"path to `tvm/python`"
```

### Now, follow these steps
**Note:** If you are using Hexagon docker image, first step is to log into the Hexagon docker image. Following these commands you will log in to the most recent version of Hexagon docker image on your TVM local branch. Since we have already built TVM for hexagon, we can just log in and use it. From your TVM home directory:

```bash
./docker/bash.sh ci_hexagon
```

Now, you need to export few environment variables and execute following commands:

```bash
# Run RPC Tracker in the background
export TVM_TRACKER_HOST="Your host IP address or 0.0.0.0"
export TVM_TRACKER_PORT="Port number of your choice."
python -m tvm.exec.rpc_tracker --host $TVM_TRACKER_HOST --port $TVM_TRACKER_PORT&

# Only For real hardware testing
export ANDROID_SERIAL_NUMBER="You can get this number by running 'adb devices' command"

# Only For simulator testing
export HEXAGON_SHARED_LINK_FLAGS="-Lbuild/hexagon_api_output -lhexagon_rpc_sim"
export ANDROID_SERIAL_NUMBER="simulator"
```

Finally, to run a Hexagon Launcher tests you can run:
```bash
pytest tests/python/contrib/test_hexagon/test_launcher.py
```
