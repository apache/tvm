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
Writing a Customized Pass
=========================
**Author**: `Jian Weng <https://were.github.io>`_

TVM is a framework that abstracts away the heterogenity of machine learning accelerators.
Sometimes users may want customize some analysis and IR transformations
to adapt TVM to their own specialized hardware. This tutorial helps users write
a customized pass in TVM.

Prerequisites
-------------

Before reading this tutorial, we assume readers have already known these topics well:

- Writing an algorithm in TVM and schedule it. Otherwise, see example tutorials like
  :ref:`opt-gemm`.
- The basic structure of HalideIR. Otherwise, see ``HalideIR/src/ir/IR.h`` to learn what
  attributes of IR nodes are defined.
- Visitor design pattern. Otherwise, check the
  `Python AST module <https://docs.python.org/3/library/ast.html>`_ to see how an AST
  visitor is implemented.
- How a Schedule is lowered to either an IRModule class or a LLVM module. Otherwise,
  take a look at ``python/tvm/build_module.py`` to get some basics.

"""
import tvm
from tvm import te
import numpy as np

######################################################################
# We first write a very simple vector add and build it with the default schedule. Then, we use
# our customized lowering pass to manipulate the IR directly instead of using schedule primitives.
#

n = tvm.tir.const(128, "int32")
a = te.placeholder((n, ), name="a")
b = te.placeholder((n, ), name="b")
c = te.compute((n, ), lambda i: a[i] + b[i], name='c')

sch = te.create_schedule(c.op)
ir  = tvm.lower(sch, [a, b, c])
print(ir)

######################################################################
# Writing a Pass
# --------------
# Essentially, an "IR transformation pass" is a function which maps a statement to a new statement.
# Thus, we define this vectorize function and implement it step by step.
#

######################################################################
# TVM already provides two class for users to both analyze and transform IR.
#
# IR Visitor
# ~~~~~~~~~~
# We can use ``tvm.tir.ir_pass.PostOrderVisit(stmt, func)`` to gather information from the Halide IR.
# ``func`` is a function callback. This function will be called before exiting the current IR node,
# i.e. post-order visit. Then we leverage side effects to store the result of IR visit, because the
# return value of ``func`` will be ignored.
#
# .. note::
#
#     You MUST use some array to store the result of IR visit. Even the value is a single variable.
#     This is mainly due to the constraints in the Python-C runtime. The variable values will be
#     refreshed every recursion but the array values will be preserved.
#

loops = []
def find_width8(op):
    """ Find all the 'For' nodes whose extent can be divided by 8. """
    if isinstance(op, tvm.tir.For):
        if isinstance(op.extent, tvm.tir.IntImm):
            if op.extent.value % 8 == 0:
                loops.append(op)

#####################################################################
# IR Transformation
# ~~~~~~~~~~~~~~~~~
# The transformation interface is slightly different from the visitor interface. There is only a
# post-order callback in the visitor, but transformation visitor supports both a pre-order and a
# post-order callback. If you want to keep the origin IR node, just return None. If you want to
# change the current node to some node, use TVM IR maker interface to build it and return
# this value.
#
# .. note::
#
#     If the pre-order function is called and returns a value which is not None, the post-order
#     function will be skipped.
#

def vectorize8(op):
    """ Split can vectorize the loops found in `find_width8`. """
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = te.var(name + '.outer'), te.var(name + '.inner')
        body = tvm.tir.ir_pass.Substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.tir.For(li, 0, 8, tvm.tir.For.Vectorized, 0, body)
        body = tvm.tir.For(lo, 0, extent // 8, tvm.tir.For.Serial, 0, body)
        return body
    return None

@tvm.tir.transform.prim_func_pass(opt_level=0)
def vectorize(f, mod, ctx):
    global loops

    tvm.tir.ir_pass.PostOrderVisit(f.body, find_width8)

    if not loops:
        return sf

    # The last list arugment indicates what kinds of nodes will be transformed.
    # Thus, in this case only `For` nodes will call `vectorize8`
    return f.with_body(
        tvm.tir.ir_pass.IRTransform(f.body, None, vectorize8, ['For']))


#####################################################################
# Glue to Lowering
# ----------------
# So far, we are done with writing this IR transformation pass. What we need to do next is to glue
# this pass to TVM's lower pass.
#
# In TVM, there is a property called ``BuildConfig``. You can use this property to customize your
# own lowering options. In this case, we inject the pass written above into the TVM standard lowering
# pass by feeding **a list of tuple** as argument to ``add_lower_pass``. "Tuple" indicates different
# phases of lowering. In TVM, there are four phases of lowering and user-customized ones will be
# called after each phase is done.
#
# .. note::
#     Here are the essential transformations done by each phase:
#       - Phase 0 generates the raw IR and loop levels.
#       - Phase 1 flattens the array storage.
#       - Phase 2 transforms loops, like unroll, vectorization and thread-binding.
#       - Phase 3 does some cleanup work.
#
# Thus, a good place to put this transformation pass is just after Phase 1.
#

with tvm.target.build_config(add_lower_pass=[(1, vectorize)]) as cfg:
    print(tvm.lower(sch, [a, b, c]))

#####################################################################
# Quick View
# ----------
# This tutorial gives a quick view of writing a customized IR transformation pass:
# - Use ``tvm.tir.ir_pass.PostOrderVisit`` to gather information on each IR nodes.
# - Use ``tvm.tir.ir_pass.IRTransform`` to transform IR nodes.
# - Wrap up two above to write an IR-transformation function.
# - Use ``tvm.target.build_config`` to put this function to TVM lowering pass
#
