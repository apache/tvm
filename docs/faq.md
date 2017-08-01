Frequently Asked Questions
==========================
This document contains frequently asked questions.

How to Install
--------------
See [Installation](https://github.com/dmlc/tvm/blob/master/docs/how_to/install.md)

TVM's relation to XLA
---------------------
They has different abstraction level.
XLA is a higher level tensor algebra DSL, the system defines codegen and loop transformation
rules for each kernels. TVM is an low level array index based DSL that give the loop transformation
primitives to the user. In terms of design philosophy, TVM aims to be directly used by developers
and provide general support for different framework via DLPack.
See also [This Issue](https://github.com/dmlc/tvm/issues/151)

TVM's relation to libDNN cuDNN
------------------------------
TVM can incorporate these library as external calls. One goal of TVM is to be able to
generate high performing kernels. We will evolve TVM an incremental manner as
we learn from the technics of manual kernel crafting and add these as primitives in DSL.
