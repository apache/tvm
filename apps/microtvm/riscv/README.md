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


Running TVM on bare metal Risc-V CPU
====================================

This folder contains an example of how to use TVM to run a model
on bare metal Risc-V CPU.

Prerequisites
-------------
If the demo is run in the ci_riscv Docker container provided with TVM, then the following
software will already be installed.

If the demo is not run in the ci_riscv Docker container, then you will need the following:
- Software required to build and run the demo
  - [RISC-V gcc toolchain (baremetal)](https://github.com/riscv-collab/riscv-gnu-toolchain)
  - [RISC-V QEMU](https://github.com/riscv-collab/riscv-gnu-toolchain)
  - [RISC-V ISA Simulator Spike](https://github.com/riscv-software-src/riscv-isa-sim)

You will also need TVM which can either be:
  - Built from source (see [Install from Source](https://tvm.apache.org/docs/install/from_source.html))
    - When building from source, the following need to be set in config.cmake:
      - set(USE_MICRO ON)
      - set(USE_LLVM ON)

Running the demo application
----------------------------
Type the following command to run the bare metal demo application ([src/main.c](./src/main.c)):

```bash
./run_demo.sh <Model name> [Bitness] [Simulator]
```

There are four models available in demo:
  - DS_CNN_S (float)
  - DS_CNN_M (float)
  - KWS_MICRONET_M (int8)
  - PERSON_DETECT (int8)

Bittnes are `32` or `64`, simulator can be `spike` or `qemu`. By default `64`/`qemu` is used.

This will:
- Download a model
- Use tvmc to compile the model for selected Risc-V arch
- Download an image / audio to run the model on
- Create a C header file inputs.c containing the image data as a C array
- Create a C header file outputs.c containing a C array where the output of inference will be stored
- Build the demo application
- Run the demo application on selected simulator
- The application will report whether expected result is reached
