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

VTA TSIM Installation
======================

*TSIM* is a cycle-accurate hardware simulation environment that can be invoked and managed directly from TVM. It aims to enable cycle accurate simulation of deep learning accelerators including VTA.
This simulation environment can be used in both OSX and Linux.
There are two dependencies required to make *TSIM* works: [Verilator](https://www.veripool.org/wiki/verilator) and [sbt](https://www.scala-sbt.org/) for accelerators designed in [Chisel3](https://github.com/freechipsproject/chisel3).

## OSX Dependencies

Install `sbt` and `verilator` using [Homebrew](https://brew.sh/).

```bash
brew install verilator sbt
```

## Linux Dependencies

Add `sbt` to package manager (Ubuntu).

```bash
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
```

Install `sbt` and `verilator`.

```bash
sudo apt install verilator sbt
```

Verilator version check

```bash
verilator --version
```

the supported version of Verilator should be at least 4.012, 
if homebrew (OSX) or package-manager (Linux) does not support that version,
please install Verilator 4.012 or later from binary or source base on following
instruction of Verilator wiki.  

https://www.veripool.org/projects/verilator/wiki/Installing

## Setup in TVM

1. Install `verilator` and `sbt` as described above
2. Get tvm `git clone https://github.com/dmlc/tvm.git`
3. Change VTA target in `tvm/vta/config/vta_config.json` from `sim` to `tsim`
4. Build [tvm](https://docs.tvm.ai/install/from_source.html#build-the-shared-library)

## How to run VTA TSIM examples

There are two sample VTA accelerators, add-a-constant, designed in Chisel3 and Verilog to show how *TSIM* works.
The default `TARGET` language for these two implementations is Verilog. The following instructions show
how to run both of them:

* Test Verilog backend
    * Go to `<tvm-root>/vta/apps/tsim_example`
    * Run `make`

* Test Chisel3 backend
    * Go to `<tvm-root>/vta/apps/tsim_example`
    * Run `make run_chisel`

* Some pointers
    * Verilog and Chisel3 tests in `<tvm-root>/vta/apps/tsim_example/tests/python`
    * Verilog accelerator backend `<tvm-root>/vta/apps/tsim_example/hardware/verilog`
    * Chisel3 accelerator backend `<tvm-root>/vta/apps/tsim_example/hardware/chisel`
    * Software C++ driver (backend) that handles the accelerator `<tvm-root>/vta/apps/tsim_example/src/driver.cc`
    * Software Python driver (frontend) that handles the accelerator `<tvm-root>/vta/apps/tsim_example/python/accel`
