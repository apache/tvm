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

# TVM

This crate provides an idiomatic Rust API for [Apache TVM](https://github.com/apache/tvm).
The code works on **Stable Rust** and is tested against `rustc 1.47`.

You can find the API Documentation [here](https://tvm.apache.org/docs/api/rust/tvm/index.html).

## What Does This Crate Offer?

The goal of this crate is to provide bindings to both the TVM compiler and runtime
APIs. First train your **Deep Learning** model using any major framework such as
[PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).
Then use **TVM** to build and deploy optimized model artifacts on a supported devices such as CPU, GPU, OpenCL and specialized accelerators.

The Rust bindings are composed of a few crates:
- The [tvm](https://tvm.apache.org/docs/api/rust/tvm/index.html) crate which exposes Rust bindings to
  both the compiler and runtime.
- The [tvm_macros](https://tvm.apache.org/docs/api/rust/tvm/index.html) crate which provides macros
  which generate unsafe boilerplate for TVM's data structures.
- The [tvm_rt](https://tvm.apache.org/docs/api/rust/tvm_rt/index.html) crate which exposes Rust
  bindings to the TVM runtime APIs.
- The [tvm_sys] crate which provides raw bindings and linkage to the TVM C++ library.
- The [tvm_graph_rt] crate which implements a version of the TVM graph executor in Rust vs. C++.

These crates have been recently refactored and reflect a much different philosophy than
previous bindings, as well as much increased support for more of the TVM API including
exposing all of the compiler internals.

These are still very much in development and should not be considered stable, but contributions
and usage is welcome and encouraged. If you want to discuss design issues check our Discourse
[forum](https://discuss.tvm.ai) and for bug reports check our GitHub [repository](https://github.com/apache/tvm).

## Install

Please follow the TVM [install](https://tvm.apache.org/docs/install/index.html) instructions, `export TVM_HOME=/path/to/tvm` and add `libtvm_runtime` to your `LD_LIBRARY_PATH`.

*Note:* To run the end-to-end examples and tests, `tvm` and `topi` need to be added to your `PYTHONPATH` or it's automatic via an Anaconda environment when it is installed individually.

### Disclaimers

*Apache TVM is a top level project from the Apache software foundation. Please refer to the official Apache TVM website for Apache source releases. Apache TVM, Apache, the Apache feather, and the Apache TVM project logo are either trademarks or registered trademarks of the Apache Software Foundation.*
