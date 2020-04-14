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

<img src=https://raw.githubusercontent.com/apache/incubator-tvm-site/master/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tvm.ai/buildStatus/icon?job=tvm/master)](https://ci.tvm.ai/job/tvm/job/master/)
[![WinMacBuild](https://github.com/apache/incubator-tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/incubator-tvm/actions?query=workflow%3AWinMacBuild)

Apache(incubating) TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity focused deep learning frameworks, and the performance or efficiency oriented hardware backends. TVM provides the following main features:

* Compilation of deep learning models in Keras, MXNet, PyTorch, TensorFlow, TFLite, CoreML, ONNX, Caffe2, DarkNet into minimum deployable modules on diverse hardware backends.
* Infrastructure to automatically generate and optimize tensor operators on more backend with better performance.


License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Checkout the [Contributor Guide](https://tvm.apache.org/docs/contribute/)

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
  originates from Halide. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
