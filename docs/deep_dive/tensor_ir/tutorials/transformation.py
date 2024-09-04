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
.. _tir-transform:

Transformation
--------------
In this section, we will get to the main ingredients of the compilation flows -
transformations of primitive tensor functions.
"""

######################################################################
# In the :ref:`previous section <tir-learning>`, we have given an example of how to write
# ``mm_relu`` using TensorIR. In practice, there can be multiple ways to implement
# the same functionality, and each implementation can result in different performance.
#
# .. note::
#   This tutorial primarily illustrates the application of TensorIR Transformation,
#   rather than delving into optimization techniques.
#
# First, let's take a look at the implementation of ``mm_relu`` in the previous section:

import tvm
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        Y = T.alloc_buffer((128, 128))
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


######################################################################
# Before we transform the function, let's first evaluate the performance of the
# original implementation.

import numpy as np

a_np = np.random.uniform(size=(128, 128)).astype("float32")
b_np = np.random.uniform(size=(128, 128)).astype("float32")
c_np = a_np @ b_np

a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.array(np.zeros((128, 128), dtype="float32"))


def evaluate(mod: tvm.IRModule):
    lib = tvm.build(mod, target="llvm")
    # check correctness
    lib(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), c_np, rtol=1e-5)
    # evaluate performance
    f_timer = lib.time_evaluator("main", tvm.cpu())
    print(f_timer(a_nd, b_nd, c_nd))


evaluate(MyModule)

######################################################################
# Initialization Schedule
# ***********************
# We initiate the process of code transformation by establishing a Schedule helper class,
# utilizing the provided **MyModule** as input.

sch = tvm.tir.Schedule(MyModule)

######################################################################
# Loop Tiling
# ***********
# Subsequently, we execute the requisite operations to acquire a reference to
# block **Y** and its associated loops.

block_Y = sch.get_block("Y")
i, j, k = sch.get_loops(block_Y)

######################################################################
# We now proceed to execute the transformations. The initial modification involves
# splitting loop ``j`` into two separate loops, with the inner loop possessing a
# length of 4. It is crucial to understand that the transformation process is procedural;
# thus, inadvertent execution of the block twice will yield an error stating the
# non-existence of variable ``j``.

j0, j1 = sch.split(j, factors=[None, 8])

######################################################################
# The outcome of the transformation can be examined, as it is retained within ``sch.mod``.

sch.mod.show()

######################################################################
# Following the initial transformation phase, two supplementary loops, ``j_0`` and ``j_1``,
# have been generated with respective ranges of 32 and 4. The subsequent
# action involves reordering these two loops.

sch.reorder(j0, k, j1)
sch.mod.show()
evaluate(sch.mod)

######################################################################
# Leverage Localities
# *******************
# Subsequently, we will execute two additional transformation steps to achieve a different
# variant. First, we employ a primitive known as **reverse_compute_at** to relocate block
# **C** to an inner loop of **Y**.

block_C = sch.get_block("C")
sch.reverse_compute_at(block_C, j0)
sch.mod.show()

######################################################################
# Rewrite Reduction
# *****************
# Until now, the reduction initialization and update step have been maintained together
# within a single block body. This amalgamated form facilitates loop transformations,
# as the outer loops ``i``, ``j`` of initialization and updates generally need to remain
# synchronized.
#
# Following the loop transformations, we can segregate the initialization of Y's elements
# from the reduction update via the **decompose_reduction** primitive.

sch.decompose_reduction(block_Y, k)
sch.mod.show()
evaluate(sch.mod)

######################################################################
# Trace the Transformation
# ************************
# TensorIR schedule is a procedural language, and the transformation is executed in a
# step-by-step manner. We can trace the transformation by printing the schedule or the
# history of the schedule.
#
# We've already see the schedule by printing ``sch.mod``. We can also print the history
# of the schedule by ``sch.trace``.

sch.trace.show()

######################################################################
# Alternatively, we can output the IRModule in conjunction with the historical trace.

sch.show()
