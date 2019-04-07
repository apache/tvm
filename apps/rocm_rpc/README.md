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

# TVM ROCm RPC

This folder contains a simple recipe to make RPC work together with ROCm.TVM's RPC server relies on process
fork to create a new process for each incoming session.
Like CUDA, opencl driver, the runtime ROCm runtime is not fork-safe.
A typical CUDA or opencl driver will initialize lazily
and we can use normal TVM RPC server because we won't touch the driver API before we fork a new session.
However, the current ROCm runtime eagerly initialize during startup and will directly cause error during fork.
This folder provides a workaround to this problem.

## Usage
- Build tvm **without** rocm (it is important to exclude rocm from runtime)
- Modify the ROCM_PATH to be the correct path the current [Makefile](Makefile)
- Type make to build lib/libtvm_runtime_rocm.so, which is a standalone dll module
- Use [start_rpc_server.sh](start_rpc_server.sh) to start the RPC server

## How it works
- The RPC server starts without ROCm dependency.
- lib/libtvm_runtim_rocm.so is dynamically loaded only after the fork.

## Note
With ROCm RPC, we can build AMDGPU program from a machine without AMD GPU
and remotely upload and execute on a AMDGPU machine.
Please note that you will need to set the gfx version correctly(via ```-model``` or ```-mcpu```)
because we can no longer query the GPU version dynamically during runtime.


```python
import tvm
from tvm.contrib import rpc

# set mcpu explicitly to be the gpu version.
target = "rocm -mcpu=gfx900"
remote = rpc.connect(server_host, server_port)
mod = tvm.build(s, args, target)
mod.export_library("mylib.so")

remote.upload("mylib.so")
foo = remote.load_module("mylib.so")
# same as normal RPC
```
