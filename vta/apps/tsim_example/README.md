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

## Setup in TVM

1. Install `verilator` and `sbt` as described above
2. Enable VTA TSIM by turning on the switch `USE_VTA_TSIM` in config.cmake
3. Build tvm

## How to run VTA TSIM examples

There are two sample VTA accelerators (add-by-one) designed in Chisel3 and Verilog to show how *TSIM* works.
These examples are located at `<tvm-root>/vta/apps/tsim_example`.

* Instructions
    * Open `<tvm-root>/vta/apps/tsim_example/python/tsim/config.json`
    * Change `TARGET` from `verilog` to `chisel`, depending on what language backend you would like to test
    * Go to `tvm/vta/apps/tsim`
    * Run `make`

* Some pointers
    * Build cmake script for software library`<tvm-root>/vta/apps/tsim_example/cmake/modules/sw.cmake`
    * Build cmake script for hardware library`<tvm-root>/vta/apps/tsim_example/cmake/modules/hw.cmake`
    * Software driver that handles the accelerator `<tvm-root>/vta/apps/tsim_example/src/driver.cc`
    * Add-by-one accelerator in Verilog `<tvm-root>/vta/apps/tsim_example/hardware/verilog`
    * Add-by-one accelerator in Chisel3 `<tvm-root>/vta/apps/tsim_example/hardware/chisel`
