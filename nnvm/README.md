# NNVM: Build deep learning system by parts

[![Build Status](https://travis-ci.org/dmlc/nnvm.svg?branch=master)](https://travis-ci.org/dmlc/nnvm)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

NNVM is not a deep learning library. It is a modular,
decentralized and lightweight part to help build deep learning libraries.

## What is it

While most deep learning systems offer end to end solutions,
it is interesting to  assemble a deep learning system by parts.
The goal is to enable user to customize optimizations, target platforms and set of operators they care about.
We believe that the decentralized modular system is an interesting direction.

The hope is that effective parts can be assembled together just like you assemble your own desktops.
So the customized deep learning solution can be minimax, minimum in terms of dependencies,
while maximizing the users' need.

NNVM offers one such part, it provides a generic way to do
computation graph optimization such as memory reduction, device allocation and more
while being agnostic to the operator interface definition and how operators are executed.
NNVM is inspired by LLVM, aiming to be a high level intermediate representation library
for neural nets and computation graphs generation and optimizations.

See [Overview](docs/overview.md) for an introduction on what it provides.

## Example
See [TinyFlow](https://github.com/tqchen/tinyflow) on how you can build a TensorFlow API with NNVM and Torch.
 
## Why build learning system by parts

This is essentially ***Unix philosophy*** applied to machine learning system.

- Essential parts can be assembled in minimum way for embedding systems.
- Developers can hack the parts they need and compose with other well defined parts.
- Decentralized modules enable new extensions creators to own their project
  without creating a monolithic version.

Deep learning system itself is not necessary one part, for example
here are some relative independent parts that can be isolated

- Computation graph definition, manipulation.
- Computation graph intermediate optimization.
- Computation graph execution.
- Operator kernel libraries.
- Imperative task scheduling and parallel task coordination.

We hope that there will be more modular parts in the future,
so system building can be fun and rewarding.

## Links

[MXNet](https://github.com/dmlc/mxnet) is moving to NNVM as its intermediate
representation layer for symbolic graphs.
