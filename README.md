<img src=https://raw.githubusercontent.com/tqchen/tvmlang.org/master/images/logo/tvm-logo-small.png width=128/> Tensor IR Stack for Deep Learning Systems
==============================================

[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=dmlc/tvm/master)](http://mode-gpu.cs.washington.edu:8080/job/dmlc/job/tvm/job/master/)

[Installation](docs/how_to/install.md) |
[Documentation](http://docs.tvmlang.org) |
[Tutorials](http://tutorials.tvmlang.org) |
[Operator Inventory](topi) |
[FAQ](docs/faq.md) |
[Contributors](CONTRIBUTORS.md) |
[Community](http://tvmlang.org/community.html) |
[Release Notes](NEWS.md)

TVM is a Tensor intermediate representation(IR) stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.
Checkout the [tvm stack homepage](http://tvmlang.org/)  for more information.

License
-------
© Contributors Licensed under an [Apache-2.0](https://github.com/dmlc/tvm/blob/master/LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.

- [Contributor Guide](docs/how_to/contribute.md)
- Please add your name to [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Please also update [NEWS.md](NEWS.md) on changes and improvements in API and codes.

Acknowledgement
---------------
We learnt a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): TVM uses [HalideIR](https://github.com/dmlc/HalideIR) as data structure for
  arithematic simplification and low level lowering. We also learnt and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
