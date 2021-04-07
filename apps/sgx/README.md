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

## Setup

1. [Install the Fortanix Enclave Development Platform](https://edp.fortanix.com/docs/installation/guide/)
2. `rustup component add llvm-tools-preview` to get `llvm-ar` and `llvm-objcopy`
3. `pip install numpy decorator psutil`
4. `cargo run` to start the enclave TCP server
5. Send a 28x28 "image" to the enclave model server using `head -c $((28*28*4)) /dev/urandom | nc 127.0.0.1 4242 | python read_results.py`
