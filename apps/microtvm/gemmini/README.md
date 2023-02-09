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

This directory contains code to create code for the Gemmini accelerator using microTVM. These tests are then executed on the Spike RISC-V ISA simulator.

In order to use this correctly, the Spike simulator has to be installed. This can be done by following the steps found on the [Chipyard](https://chipyard.readthedocs.io/en/stable/) repository. The instructions to also install the patch of the Spike simulator that adds the Gemmini functional simulator can be found in the [Gemmini](https://github.com/ucb-bar/gemmini) repository.
