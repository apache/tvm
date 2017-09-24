# NNVM: Graph IR Stack for Deep Learning Systems

[![Build Status](https://travis-ci.org/dmlc/nnvm.svg?branch=master)](https://travis-ci.org/dmlc/nnvm)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

NNVM is a reusable computational graph optimization and compilation stack for deep learning systems.
NNVM provides modules to:

- Represent deep learning workloads from front-end frameworks via a graph IR.
- Optimize computation graphs to improve performance.
- Compile into executable modules and deploy to different hardware backends with minimum dependency.

NNVM is designed to add new frontend, operators and graph optimizations in a decentralized fashion without changing the core interface. NNVM is part of [TVM stack](https://github.com/dmlc/tvm), which provides an end to end IR compilation stack for deploying deep learning workloads into different hardware backends

## Links
- [TinyFlow](https://github.com/tqchen/tinyflow) on how you can use  NNVM to build a TensorFlow like API.
- [Apache MXNet](http://mxnet.io/) uses NNVM as a backend.

