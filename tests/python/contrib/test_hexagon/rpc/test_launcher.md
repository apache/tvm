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

# HexagonLauncher
HexagonLauncher is a class to handle interactions with an Android phone which includes Hexagon DSP to run a TVMModule(function/operation/graph) on Hexagon. HexagonLauncher reuses minRPC implementation to setup an RPC connection from host (your local machine) to Hexagon target which is passed through Android RPC server.

## Build Required Tools/Libraries
Here are the steps that are taken to prepare a runtime on a Hexagon device to test any model.

- Build TVMRuntime library and C++ RPC server for Android.
- Build minRPC server along with FastRPC for Hexagon.
- Build TVM library with Hexagon support for host machine.
- Build TVMRuntime library and C++ RPC server for host machine.

To build these pieces, you can use a cmake command as follow.

```bash
cmake -DUSE_HEXAGON_RPC=ON \
        -DUSE_ANDROID_TOOLCHAIN=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
        -DANDROID_PLATFORM=android-28 \
        -DANDROID_ABI=arm64-v8a \
        -DUSE_HEXAGON_ARCH=v65|v66|v68 \
        -DUSE_HEXAGON_SDK=/path/to/Hexagon/SDK \
        -DUSE_HEXAGON_TOOLCHAIN=/path/to/Hexagon/toolchain/ \
        -DUSE_LLVM=/path/to/llvm/bin/llvm-config \
        -DUSE_CPP_RPC=ON \
        -DCMAKE_CXX_COMPILER=/path/to/clang++ \    
        -DCMAKE_CXX_FLAGS='-stdlib=libc++' ..
```

## Testing Using HexagonLauncher
Before starting a test you need to run an RPC tracker on your local machine and export HOST and PORT as environment variables. Also, you need to export Clang libraries to `LD_LIBRARY_PATH` and Hexagon toolchain to `HEXAGON_TOOLCHAIN`.

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/clang++/lib"
export HEXAGON_TOOLCHAIN="/path/to/Hexagon/toolchain/"

export TVM_TRACKER_HOST="0.0.0.0"
export TVM_TRACKER_PORT=9192
python -m tvm.exec.rpc_tracker --host $TVM_TRACKER_HOST --port $TVM_TRACKER_PORT
```

Now, follow these steps to create an RPC session from host to Hexagon.

```python
# create an HexagonLauncher instance
launcher = HexagonLauncher(serial_number="Serial number taken from `adb devices` command")

# Create a workspace directory for this test on Android.
# Upload required Android artifacts including TVMRuntime library and RPC server to Android workspace.
# Uses port `forward` and `reverse` to open connection on certain ports that TVM uses to connect to RPC tracker.
# Execute `android_bash.sh` on Android which creates two RPC servers and connects them to RPC tracker running on host machine. 
launcher.android_run_rpc(rpc_tracker_host="TVM_TRACKER_HOST", rpc_tracker_port="TVM_TRACKER_PORT")

# Upload Hexagon RPC libraries to Android workspace.
launcher.hexagon_setup()

# Create an RPC session from host to Hexagon.
remote_kw = {
    "host": "TVM_TRACKER_HOST",
    "port": "TVM_TRACKER_PORT",
    "priority": 0,
    "timeout": 60,
}
launcher.hexagon_session_setup(remote_kw)

# Upload TVMModule binary file to Android remote.
launcher.upload("Path to DSO binary file on host", "DSO filename on Android remote")
```

- To execute a single function/operator on Hexagon, follow these steps.
    ```python
    # Enter session.
    with launcher.session as sess:
        # dlopen DSO binary file on Hexagon.
        mod = launcher.get_module(dso_binary)
        # Use mod to run function/operator on Hexagon...
    ```
- Or, follow these steps to create a GraphExecutor and run a JSON graph.
    ```python
    graph_mod = launcher.get_local_graph_executor(lowered, dso_binary)
    graph_mod.set_input(...)
    graph_mod.run(...)
    ```
