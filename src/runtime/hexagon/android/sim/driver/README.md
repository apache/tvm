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

# Hexagon simulator driver

The driver (`sim_dev` executable) is the process running on the Hexagon simulator that handles the Hexagon-side communication with the TVM runtime running on x86. The location of `sim_dev` should be added to `PATH` before running any python code that uses Hexagon. The `sim_dev` executable is not intended to be run by users, it is automatically loaded by the simulator control code (in `hexagon_device_sim.cc`).

### Prerequisites

1. Hexagon C/C++ toolchain (such as the one in Hexagon SDK version 3.5.0 or later).

Hexagon SDK is available at //developer.qualcomm.com/software/hexagon-dsp-sdk.

### Configuring

Set
```
CMAKE_C_COMPILER=hexagon-clang
CMAKE_CXX_COMPILER=hexagon-clang++
```

### Building

There are no special options required for `make` (or the tool selected with `cmake`). The location of the resulting binary `sim_dev` should be added to `PATH`.
