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

VTA TSIM Application 
======================
Prior to this application, please take a look at `<tvm-root>/vta/apps/tsim_example` for installation
This is an application that performs Bit Serial Multiplication for GEMM utilizing TSIM.

**Bit Serial Multiplication for GEMM:**

General Matrix Multiplications (GEMM), are mostly calculated by repeatly calculating the dot product for each pair of vectors.
The dot product is calculated by summing every product of the vector pair.
We approach this operation with slicing and shifting, like how basic multiplication works, each vector elements before we accumulate them.
We can sufficiently reduce the cycles required to perform a gemm given that the data bit width is small. This GEMM application uses TSIM for future accerlerator prototypes.

* Test Chisel3 backend with bit serial GEMM
    * Go to `<tvm-root>/vta/apps/gemm`
    * Run `make`

* If you have already compiled chisel backend (i.e. ran `make`) 
    * Bit Serial test with another input set, run `make serial`
    * Bit parallel test with another input set, run `make parallel`

* Some steps for creating your own custom TSIM application
    * Go to `<tvm-root>/vta/apps/gemm`
    * Create custom circuit within `./hardware/chisel/src/scala.main/accel/Compute.scala`
    * Map the according Registers in `./hardware/chisel/src/scala.main/accel/RegFile.scala`
    * Create your test script
    * Map the registers in `./src/driver.cc` and link it with both `RegFile.scala` and the test script
    * Understanding of `<tvm-root>/vta/apps/tsim_example`, which performs add by one to a vector, is highly encouraged to create a more complex application

* Some pointers
    * Chisel3 tests in `<tvm-root>/vta/apps/gemm/tests/python`
    * Chisel3 accelerator backend `<tvm-root>/vta/apps/gemm/hardware/chisel`
    * Software C++ driver (backend) that handles the accelerator `<tvm-root>/vta/apps/gemm/src/driver.cc`
    * Software Python driver (frontend) that handles the accelerator `<tvm-root>/vta/apps/gemm/python/accel`
