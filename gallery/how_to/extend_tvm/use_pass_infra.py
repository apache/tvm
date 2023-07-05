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
# pylint: disable=line-too-long
"""
.. _tutorial-use-pass-infra:

How to Use TVM Pass Infra
=========================
**Author**: `Zhi Chen <https://github.com/zhiics>`_

As the number of optimization passes increases in Relay/tir, it becomes intractable to
execute them and maintain their dependencies manually. Therefore, we have
introduced an infrastructure to manage the optimization passes and make it
applicable to different layers of the IR in the TVM stack.

The optimizations of a Relay/tir program could be applied at various granularity,
namely function-level and module-level using :py:class:`tvm.relay.transform.FunctionPass`/
:py:class:`tvm.tir.transform.PrimFuncPass` and :py:class:`tvm.transform.ModulePass`
respectively. Or users can rely on :py:class:`tvm.transform.Sequential` to apply a sequence of passes
on a Relay/tir program where the dependencies between passes can be resolved by the
pass infra. For more details about each type of these passes, please refer to
the :ref:`pass-infra`

This tutorial mainly demonstrates how developers can use the pass infra to perform
a certain optimization and create an optimization pipeline for a Relay program.
The same approach can be used for tir as well.
"""


import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

###############################################################################
# Create An Example Relay Program
# -------------------------------
# First of all, we create a simple Relay program for the tutorial. This program
# will be used by various optimizations of the examples in this tutorial.
# Similarly, users can write a tir primitive function and apply the tir passes.


def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)


###############################################################################
# Optimize the Program
# --------------------
# Now we would like to optimize the program. Relay features a host of
# optimizations. We will select some of them to apply on this example program.
#
# There are multiple ways to optimize a Relay program. Below we will provide
# examples for each of them.
#
# Manually Apply Optimization Passes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Let's first create a relay Module which contains one or multiple Relay
# functions for optimization.
f = example()
mod = tvm.IRModule.from_expr(f)

# Now we can apply constant folding on the module.
# fold_const here is a callback that doesn't take any parameters.
fold_const = relay.transform.FoldConstant()
# Then, we can invoke the pass on the given module. Note that the constant
# folding pass works at the function-level. That being said, each function in
# the module will be applied with the optimization. Users don't need to iterate
# through individual functions manually to apply this pass.
mod = fold_const(mod)
# We can see from the updated program that the constants are folded.
print(mod)

###############################################################################
# More optimizations can be applied in the similar manner. For instance, we can
# eliminate the common expressions that used by `z` and `z1`.
mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)

###############################################################################
# Some optimizations, such as fusion, are parametric as well. For example,
# opt level 0 will not allow operators to be fused together. Users can pass the
# `fuse_opt_level` to enable this.
mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

# We can observe that the optimized module contains functions that only have
# a signle primitive op.
print(mod)

###############################################################################
# Use Sequential to Apply a Sequence of Passes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Applying passes as above is actually tedious and it may require users to have
# better understanding about the dependencies between them. For example, fusion
# currently doesn't work well on let bindings. Therefore, we would not be able
# to fuse operators that were fusable if :py:func:`relay.transform.ToANormalForm` is applied before
# fusion, as this pass generates let bindings for each expression to
# canonicalize a Relay program.
#
# Relay, hence, provides :py:class:`tvm.transform.Sequential` to alleviate developers from handling
# these issues explicitly by specifying the required passes of each pass and
# packing them as a whole to execute. For example, the same passes can now be
# applied using the sequential style as the following. :py:class:`tvm.transform.Sequential` is
# similar to `torch.nn.sequential <https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential>`_
# and `mxnet.gluon.block <https://mxnet.apache.org/api/python/docs/_modules/mxnet/gluon/block.html>`_.
# For example, `torch.nn.sequential` is used to contain a sequence of PyTorch
# `Modules` that will be added to build a network. It focuses on the network
# layers. Instead, the :py:class:`tvm.transform.Sequential` in our pass infra works on the optimizing
# pass.

# Now let's execute some passes through :py:class:`tvm.transform.Sequential`
f = example()
mod = tvm.IRModule.from_expr(f)
# Glob the interested passes.
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)
mod1 = seq(mod)
print(mod1)

###############################################################################
# From the transformed Relay program, we can see that there are still two
# identical addition operations. This is because ``EliminateCommonSubexpr``
# was not actually performed. The reason is because only the passes that have
# optimization level less or equal to 2 will be executed by default under
# :py:class:`tvm.transform.Sequential`. The pass infra,
# however, provides a configuration interface
# for users to customize the optimization level that they want to execute.

with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)
print(mod2)

###############################################################################
# Now we can see that only one of the two identical additions is kept.
#
# In addition, users can selectively disable some passes using the
# `disabled_pass` config, which is similar to the `-fno-xxx` option used the
# general purpose compilers, such as Clang and GCC. For example, we can disable
# EliminateCommonSubexpr as following. The printed module will again show two
# identical addition operations.

with tvm.transform.PassContext(opt_level=3, disabled_pass=["EliminateCommonSubexpr"]):
    mod3 = seq(mod)
print(mod3)

##############################################################################
# Implement a Pass Using Python Decorator
# ------------------------------------------
# The next example illustrates how we can orchestrate a customized optimization
# pipeline through the pass infra using Python decorators. This functionality
# greatly eases the implementation of passes. For example, users can simply
# define a decorated class to do function-level optimizations as the following
# example shows. `transform_function` wraps a class to replace all constants
# with a multiple of `c`. Later on, each function in a given module will be
# visited and each constant in the function will be replaced when we invoke the
# customized pass.


@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)


f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print(mod3)

##############################################################################
# Debug a Pass
# ------------
# TVM provides users a plug-and-play style debugging pass that print the IR
# after a certain pass is done through a special pass (``PrintIR``) to dump the IR of the
# whole module. A slightly modified version of the sequential pass example
# could be like the following to enable IR dumping for ``FoldConstant`` optimization.

f = example()
mod = tvm.IRModule.from_expr(f)
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(),
    ]
)

###############################################################################
# By inserting the ``PrintIR`` pass after ``FoldConstant``, the pass infra will
# dump out the module IR when ``FoldConstant`` is done. Users can plug in this
# pass after any pass they want to debug for viewing the optimization effect.
#
# There is a more flexible debugging mechanism. One can implement a ``PassInstrument``
# class to execute arbitrary code not only before and/or after each pass but also
# at entering/exiting ``PassContext``. See :ref:`pass_instrument_cpp_backend`
# for more details.
#
# Here we use :py::func`tvm.instrument.pass_instrument` decorator to implement
# a PassInsturment class printing IR before execution of each passes:


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)


with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        # Perform the optimizations.
        mod = seq(mod)
print(mod)

print("done")

##############################################################################
# Summary
# -------
# This tutorial has covered how we can write and invoke passes in TVM more
# conveniently using the pass infra. Different ways of invoking a pass are also
# discussed. Using :py:class:`tvm.transform.Sequential` can largely help
# users to ease the work of handling multiple optimization passes and their
# dependencies. In addition, an example is provided to illustrate
# how we can debug a pass using the ``PrintIR`` and tracing.
