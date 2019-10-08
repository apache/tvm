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

* Test Chisel3 backend
    * Go to `<tvm-root>/vta/apps/serialLoadMM`
    * Run `make`

* Testing on compiled backend
    * If you have already compile chisel backend and want test with another input set, run `make test`

* Some steps for creating your own custom TSIM application
    * Go to `<tvm-root>/vta/apps/serialLoadMM`
    * Create custom circuit within `./hardware/chisel/src/scala.main/accel/Compute.scala`
    * Map the according Registers in `./hardware/chisel/src/scala.main/accel/RegFile.scala`
    * Map the registers in `./src/driver.cc` and link it with the python test script
    * Create your python test script
    * Understanding of `<tvm-root>/vta/apps/tsim_example`, which performs add by one to a vector, is essential to create a more complex application

* Some pointers
    * Chisel3 tests in `<tvm-root>/vta/apps/serialLoadMM/tests/python`
    * Chisel3 accelerator backend `<tvm-root>/vta/apps/serialLoadMM/hardware/chisel`
    * Software C++ driver (backend) that handles the accelerator `<tvm-root>/vta/apps/serialLoadMM/src/driver.cc`
    * Software Python driver (frontend) that handles the accelerator `<tvm-root>/vta/apps/serialLoadMM/python/accel`
