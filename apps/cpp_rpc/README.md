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
- Build tvm runtime
- Make the rpc executable [Makefile](Makefile).
  `make CXX=/path/to/cross compiler g++/ TVM_RUNTIME_DIR=/path/to/tvm runtime library directory/ OS=Linux`
  if you want to compile it for embedded Linux, you should add `OS=Linux`.
  if the target os is Android, you doesn't need to pass OS argument.
  You could cross compile the TVM runtime like this:
```
  cd tvm
  mkdir arm_runtime
  cp cmake/config.cmake arm_runtime
  cd arm_runtime
  cmake .. -DCMAKE_CXX_COMPILER="/path/to/cross compiler g++/"
  make runtime
```
- Use `./tvm_rpc server` to start the RPC server

## Usage (Windows)
- Build tvm with the argument -DUSE_CPP_RPC
- Install [LLVM pre-build binaries](https://releases.llvm.org/download.html), making sure to select the option to add it to the PATH.
- Verify Python 3.6 or newer is installed and in the PATH.
- Use `<tmv_output_dir>\tvm_rpc.exe` to start the RPC server

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
Currently support is only there for Linux / Android / Windows environment and proxy mode doesn't be supported currently.