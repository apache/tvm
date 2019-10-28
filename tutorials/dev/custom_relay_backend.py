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

Bring Your Own Codegen To TVM
============================================
**Author**: `Zhi Chen <https://github.com/zhiics>`_, `Cody Hao Yu <https:://github.com/comaniac>`_

As the hardware devices targeted by deep learning workloads keep increasing, the required knowledge
for users to achieve high performance on various devices keeps increasing as well. To free data
scientists from worrying about the performance when developing a new model, hardware vendors either
provide libraries such as MKLDNN or cuDNN with many commonly used deep learning operators,
or provide frameworks such as TensorRT to let users describle their models in a certain way to
achieve high performance. However, users have to learn a new programming interface when they
attempt to work on a new libaray or device. As a result, the demand of a unified programming
interface becomes more and more important to 1) let all users and hardware vendors stand on the
same page, and 2) provide a feasible solution to allow a specialized hardware or library to only
support widely used operators with extremely high perofrmance, but fallback unsupported operators
to general devices like CPU/GPU.

In this tutorial, we demonstrate how a hardware vendor can easily implement
a Relay backend to support a specialized hardware device/library. It mainly
takes three steps: 1) define whether an operator is supported under a given
template, 2) specify how to compile and serialize the supported operators so
that it can ingest TVM specific data format, e.g. NDArray, and 3) specify how
to execute the compiled operators on a certain device. We will demonstrate how
to add a new backend that uses open source compilers (e.g. GCC, LLVM, etc) or any
proprietary compilers to execute a subgraph of a model without the exposure of
the IP of customer's codegen tool chain. Note that you will need to add the
specialized Relay backend to the TVM codebase and rebuild TVM for enabling.

"""

######################################################################
# Define The Supported Operators
# ------------------------------
# The first step is to define which operators are supported by your backend.
# A templated is provided to ease vendor's effort to add the supported
# operators.
#
# For example, We create a new Python file at python/relay/backend/op/contrib/gcc/extern_op.py,
# and implement a set of boolean functions with corresponding operator names. A boolean
# function should return `True` if we allow it to be executed by the given backend; `False`
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
# Note that since we include `attrs` and `args` into the function signature, we
# can define more complicated rules. For example, we can only support conv2d
# with float32 data type or with kernel size 1x1. In addition, the vendors can
# also check the attributes associated with a given operator to decide if it is
# supported by checking the fields in `attrs`. In a even more complicated but
# interesting scenario, we also allow developers to check the sequence of
# operators through iterating on the `agrs`. However, this is only
# unidirectional as only the inputs are visible.
#
# After annotating whether an operator can be executed on the given backend.
# Users can directly invoke the partitioning pass to separate the graph into
# multiple segments. The C++ backend implements a partitioning pass to fullfil
# the task and creates subgraphs/sub-functions with *External* attribute,
# indicating that this function will be handled by external codegen tool.
# Therefore, Relay passes should skip optimizations on them.

######################################################################
# Customize Subgraph Annotations
# ------------------------------
# In addition to specifying a set of rules for supported operators, we can also implement
# a Relay IR mutator to find the supported subgraphs, which may include multiple operators,
# for the target backend. Here we implement an annotator that includes an entire Relay graph
# to be offloaded. Specifically, we are going to do two tasks:
# - insert `aubgraph_begin` after all input variables
# - insert `subgraph_end` before the primary output. For example, given a Relay graph as follows:
#       input_a
#          |
#         add    --- input_b
#          |
#       subtract --- input_c
#          |
#       multiply --- input_d
#          |
#         out
#
# Our goal is to mutate the graph to the following:
#
#       input_a
#          |
#     subgraph_begin
#          |
#         add    --- subgraph_begin --- input_b
#          |
#       subtract --- subgraph_begin --- input_c
#          |
#       multiply --- subgraph_begin --- input_d
#          |
#      subgraph_end
#          |
#         out
#
# The implementation is shown as follows. As can be seen, the annotator is derived from
# `ExprMutator` that traverses a Relay graph and allows we to mutate it. We know that all ops
# are `call` nodes in Relay graph, so we override the call node mutator `visit_call` in
# `ExprMutator` and insert annotations.

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.annotation import subgraph_begin, subgraph_end

class WholeGraphAnnotator(ExprMutator):
    """
    An annotator that creates a subgraph for an entire graph.
    """
    def __init__(self, compiler):
        super(WholeGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = subgraph_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = subgraph_end(new_call, self.compiler)
        return new_call

######################################################################
# Finally, we apply the annotator to our workload. Let's first build a Relay
# function:

input_a = relay.var('a', shape=(10, 10))
input_b = relay.var('b', shape=(10, 10))
input_c = relay.var('c', shape=(10, 10))
input_d = relay.var('d', shape=(10, 10))

temp_1 = relay.add(input_a, input_b)
temp_2 = relay.subtract(temp_1, input_c)
out = relay.multiply(temp_2, input_d)
func = relay.Function([input_a, input_b, input_c, input_d], out)

######################################################################
# The above Relay function results in the following IR:

print(func)

######################################################################
# Then we apply the annotator to the IR and partition the graph:

mod = relay.Module()
mod['main'] = WholeGraphAnnotator('gcc').visit(func)
mod = relay.transform.PartitionGraph()(mod)

######################################################################
# Accordingly, the IR is transformed to the following. We can see that the
# entire Relay graph is enclosed in a function with `External="gcc"` attribute.
# It indicates that this function will be offloaded to an external backend
# during the runtime.

print(mod['main'])

######################################################################
# Implement The Codegen
# ---------------------
# The second and the third step are implemented in C++ instead of Python.
# Specifically, we create src/relay/backend/contrib/gcc/codegen.cc and
# implement the codegen and runtime dispatcher here. For the codegen,
# we need to implement two functions: `CompileExternalLib()` and `Build()`.
# `Build()` accepts a Relay module or subgraph and generate the library or device
# code accordingly. In the GCC example, we implement a Relay IR visitor to generate
# C++ code for subgraphs.

######################################################################
# In addition `CompileExternalLib()` is used for specifying how to generate and
# serialize an external library for the generated device code (C++ in this
# example). The generated library/executable binary can either be materialized
# to disk and load back during runtime, or stored in memory directly for
# later usage using whatever user defined mechanism. In the GCC case, the
# stand system calls e.g. dlopen/dlsym or LoadLibrary/GetProcAddress are used
# for Linux and Windows, respectively.

######################################################################
# Implement The Runtime Dispather
# -------------------------------
# The last step is invoking the generated external library in runtime.
# Specifically, we need to implement tvm runtime `Module` compatible
# `GetFunction()` in codegen.cc. The function takes a subgraph name and returns
# a `PackedFunc` that executes the subgraph with runtime input data. Note that
# the runtime data in TVM is provided in the tvm `NDArray` format. It's
# vendors' repsonsiblity to deserialize it into the format that they library
# can ingest. For example, we unpack it and extract the raw pointers for
# MKL-DNN. If the subgraph is compiled by `Build` in advance and the shared
# library or executable binary is available, then we can invoke it here.
#
# `GetFunction()` will be invoked by Relay runtime, including interpreter,
# graph runtime, and VM, meaning that this one implemtation works for all
# kinds of Relay runtimes.

######################################################################
# Add Codegen to TVM Building Process
# -----------------------------------
# Finally, we include the implemented codegen to the cmake config so that
# it will be built along with the TVM. To do so, we add two lines to
# cmake/modules/contrib/Extern.cmake:
# file(GLOB GCC_RELAY_CONTRIB_SRC src/relay/backend/contrib/gcc/codegen.cc)
# list(APPEND COMPILER_SRCS ${GCC_RELAY_CONTRIB_SRC})


######################################################################
# We can now test the correctness of the external GCC backend:
#
# .. note::
#     The complete GCC backend implementation is in the TVM codebase
#     so we can directly use it in this tutorial for demonstration.

import numpy as np

a_data = np.random.rand(10, 10).astype('float32')
b_data = np.random.rand(10, 10).astype('float32')
c_data = np.random.rand(10, 10).astype('float32')
d_data = np.random.rand(10, 10).astype('float32')

ex = relay.create_executor('debug', mod=mod, ctx=tvm.cpu(0))
result = ex.evaluate()(a_data, b_data, c_data, d_data)
tvm.testing.assert_allclose(result.asnumpy(), (a_data + b_data - c_data) * d_data)

print('Results are correct!')
