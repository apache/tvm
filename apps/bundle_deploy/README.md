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


How to Bundle TVM Modules
=========================

This folder contains an example on how to bundle a TVM module (with the required
interpreter runtime modules such as `runtime::GraphExecutor`, the graph JSON, and
the params) into a single, self-contained shared object (`bundle.so`) which
exposes a C API wrapping the appropriate `runtime::GraphExecutor` instance.

This is useful for cases where we'd like to avoid deploying the TVM runtime
components to the target host in advance - instead, we simply deploy the bundled
shared-object to the host, which embeds both the model and the runtime
components. The bundle should only depend on libc/libc++.

It also contains an example code (`demo.cc`) to load this shared object and
invoke the packaged TVM model instance. This is a dependency-free binary that
uses the functionality packaged in `bundle.so` (which means that `bundle.so` can
be deployed lazily at runtime, instead of at compile time) to invoke TVM
functionality.

Type the following command to run the sample code under the current folder,
after building TVM first.

```bash
make demo_dynamic
```

This will:

- Download the mobilenet0.25 model from the MXNet Gluon Model Zoo
- Compile the model with Relay
- Build a `bundle.so` shared object containing the model specification and
  parameters
- Build a `demo_dynamic` executable that `dlopen`'s `bundle.so` (or `bundle_c.so` in
  terms of the MISRA-C runtime), instantiates the contained graph executor,
  and invokes the `GraphExecutor::Run` function on a cat image, then prints
  the output results.

Type the following command to run the sample code with static linking.

```bash
make demo_static
```

This will:
- Download the mobilenet0.25 model from the MXNet Gluon Model Zoo
- Compile the model with Relay and outputs `model.o`
- Build a `bundle_static.o` object containing the runtime functions
- Build a `demo_static` executable which has static link to `bundle_static.o` and
  `model.o`, functions on a cat image, then prints the output results.
