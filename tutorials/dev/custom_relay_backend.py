# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""

.. _tutorial-custom-relay-backend

Design a New Relay Backend for Third-Parties
============================================
**Author**: `Zhi Chen <https://github.com/zhiics>`_, `Cody Hao Yu <https:://github.com/comaniac>`_

As the hardware devices targeted by deep learning workloads keep increasing, the required knowledge
for users to achieve high performance on vary devices keeps increasing as well. To free data scientists
from worrying about the performance when developing a new model, hardware vendors either provide
libraries such as MKLDNN or cuDNN with many commonly used deep learning operators, or provide frameworks
such as TensorRT to let users describle their models in a certain way to achieve high performance.
However, users have to learn a new programming interface when they attempt to work on a new libaray
or device. As a result, the demeand of a unified programming interface becomes more and more important
to 1) let all users and hardware vendors stand on the same page, and 2) provide a feasbile solution to
allow a specialized hardware or library to only support widely used operators with extremely high
perofrmance, but fallback unsupported operators to general devices like CPU/GPU.

In this tutorial, we introduce how a hardware vendor can easily implement a Relay backend to support
a specialized hardware device/library. It mainly takes three steps: 1) define whether an operator is
supported, 2) specify how to compile and serialize the supported operators, and 3) specify how to
execute the compiled operators on a certain device. We will demonstrate how to add a new backend that
uses GCC compiler to execute a subgraph of a model. Note that you will need to add the specialized Relay
backend to the TVM codebase and rebuild TVM for enabling.

"""

######################################################################
# Define The Supported Operators
# ------------------------------
# The first step is to define which operators are supported by our backend.
# We first create a new Python file at python/relay/backend/op/contrib/gcc/extern_op.py,
# and implement a set of boolean functions with corresponding operator names. A boolean
# function should return `True` if we allow it to be executed by our backend; `False`
# otherwise.

from __future__ import absolute_import

def conv2d(attrs, args):
    """Check if the external codegen should be used.
    """
    return False

def subtract(attrs, args):
    """Check if the external codegen should be used.
    """
    return True

def add(attrs, args):
    """Check if the external codegen should be used.
    """
    return True

def multiply(attrs, args):
    """Check if the external codegen should be used.
    """
    return True

######################################################################
# Note that since we include `attrs` and `args` into the function signature, we can
# define more complicate rules. For example, we can only support conv2d with float32
# data type or with kernel size 1x1.

######################################################################
# In the last step of the first part, we
# create python/relay/backend/op/contrib/gcc/__init__.py to allow rule functions to
# be used by the TVM.

from __future__ import absolute_import as _abs
from .extern_op import *

######################################################################
# Implement The Codegen
# ---------------------
# The second and the thrid step are implemented in C++ instead of Python.
# Specifically, we create src/relay/backend/contrib/gcc/codegen.cc and
# implement the codegen and runtime dispatcher here. For the codegen,
# we need to implement two functions: `CompileExternalLib()` and `Build()`.
# `Build()` accepts a Relay subgraph and generate the library or device code
# accordingly. In the GCC example, we implement a Relay IR visitor to generate
# C++ code for subgraphs.

######################################################################
# In addition `CompileExternalLib()` is used for specifying how to generate and
# serialize an external library for the generated device code (C++ in this
# example). The generated library/executable binary can either be materialized
# to disk and load back during runtime, or stored in memory directly for
# later usage.

######################################################################
# Implement The Runtime Dispather
# -------------------------------
# The last step is invoking the generated external library in runtime.
# Specifically, we need to implement `GetFunction()` in codegen.cc.
# The function takes a subgraph name and returns a `PackedFunc` that
# executes the subgraph with runtime input data. If the subgraph is
# compiled by `Build` in advance and the shared library or executable
# binary is available, then we can invoke it here.
# `GetFunction()` will be invoked by Relay runtime, including interpreter,
# graph runtime, and VM, meaning that this one implemtation works for all
# kinds of Relay runtimes.


