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

# Code Organization

Header files in include are public APIs that share across modules.
There can be internal header files within each module that sit in src.

## Modules
- arith: Arithmetic expression and set simplification.
- auto\_scheduler: The template-free auto-tuning module.
- autotvm: The template-based auto-tuning module.
- contrib: Contrib extension libraries.
- driver: Compilation driver APIs.
- ir: Common IR infrastructure.
- node: The base infra for IR/AST nodes that is dialect independent.
- relay: Relay IR, high-level optimizations.
- runtime: Minimum runtime related codes.
- support: Internal support utilities.
- target: Hardware targets.
- tir: Tensor IR, low-level optimizations.
- te: Tensor expression DSL.
- topi: Tensor Operator Inventory.
