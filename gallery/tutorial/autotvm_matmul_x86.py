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
.. _tutorial-autotvm-matmul-x86:

Optimizing Operators with Schedule Templates and AutoTVM
========================================================
**Authors**:
`Lianmin Zheng <https://github.com/merrymercy>`_,
`Chris Hoge <https://github.com/hogepodge>`_

In this tutorial, we show how the TVM Tensor Expression (TE) language
can be used to write schedule templates that can be searched by AutoTVM to
find the optimal schedule. This process is called Auto-Tuning, which helps
automate the process of optimizing tensor computation.

This tutorial builds on the previous :doc:`tutorial on how to write a matrix
multiplication using TE <tensor_expr_get_started>`.

There are two steps in auto-tuning.

- The first step is defining a search space.
- The second step is running a search algorithm to explore through this space.

In this tutorial, you can learn how to perform these two steps in TVM. The whole
workflow is illustrated by a matrix multiplication example.

.. note::
  Note that this tutorial will not run on Windows or recent versions of macOS.
  To get it to run, you will need to wrap the body of this tutorial in a
  :code:`if __name__ == "__main__":` block.
"""

################################################################################
# Install dependencies
# --------------------
# To use autotvm package in TVM, we need to install some extra dependencies.
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost cloudpickle
#
# To make TVM run faster in tuning, it is recommended to use cython as FFI of
# TVM. In the root directory of TVM, execute:
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Begin by importing the required packages.


import logging
import sys

import numpy as np
import tvm
from tvm import te
import tvm.testing

# the module is called `autotvm`
from tvm import autotvm

################################################################################
# Basic Matrix Multiplication with TE
# -----------------------------------
# Recall the basic implementation of matrix multiplication using TE. We write
# it down here with a few changes. We will wrap the multiplication in a python
# function definition. For simplicity, we will focus our attention on a split
# optimization, using a fixed value that defines the block size of the
# reordering.


def matmul_basic(N, L, M, dtype):

    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    yo, yi = s[C].split(y, 8)
    xo, xi = s[C].split(x, 8)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


################################################################################
# Matrix Multiplication with AutoTVM
# ----------------------------------
# In the previous schedule code, we use a constant "8" as the tiling factor.
# However, it might not be the best one because the best tiling factor depends
# on real hardware environment and input shape.
#
# If you want the schedule code to be portable across a wider range of input
# shapes and target hardware, it is better to define a set of candidate values
# and pick the best one according to the measurement results on target
# hardware.
#
# In autotvm, we can define a tunable parameter, or a "knob" for such kind of
# value.

################################################################################
# A Basic Matrix Multiplication Template
# --------------------------------------
# We begin with an example of how to create a tunable parameter set for the
# block size of the `split` scheduling operation.

# Matmul V1: List candidate values
@autotvm.template("tutorial/matmul_v1")  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


################################################################################
# Here we make four modifications to the previous schedule code and get a
# tunable "template". We can explain the modifications one by one.
#
# 1. Use a decorator to mark this function as a simple template.
# 2. Get a config object: You can regard this :code:`cfg` as an argument of
#    this function but we obtain it in a different way. With this argument, this
#    function is no longer a deterministic schedule. Instead, we can pass
#    different configurations to this function and get different schedules. A
#    function that uses a configuration object like this is called a "template".
#
#    To make the template function more compact, we can do two things to define
#    the parameter search space within a single function.
#
#    1. Define a search space across a set values. This is done by making
#       :code:`cfg` a :any:`ConfigSpace` object. It will collect all of the
#       tunable knobs in this function and build a search space from it.
#    2. Schedule according to an entity in this space. This is done by making
#       :code:`cfg` a :any:`ConfigEntity` object. When it is a
#       :any:`ConfigEntity`, it will ignore all space definition API (namely,
#       :code:`cfg.define_XXXXX(...)`). Instead, it will store deterministic
#       values for all tunable knobs, and we schedule according to these values.
#
#    During auto-tuning, we will first call this template with a
#    :any:`ConfigSpace` object to build the search space. Then we call this
#    template with different :any:`ConfigEntity` in the built space to get
#    different schedules. Finally we will measure the code generated by
#    different schedules and pick the best one.
#
# 3. Define two tunable knobs. The first one is :code:`tile_y` with 5 possible
#    values. The second one is :code:`tile_x` with a same list of possible values.
#    These two knobs are independent, so they span a search space with size 25 =
#    5x5.
# 4. The configuration knobs are passed to the :code:`split` schedule
#    operation, allowing us to schedule according to the 5x5 deterministic values
#    we previously defined in :code:`cfg`.

################################################################################
# A Matrix Multiplication Template with the Advanced Parameter API
# ----------------------------------------------------------------
# In the previous template, we manually listed all of the possible values for a
# knob. This is the lowest level API to define the space, and gives an explicit
# enumeration of the parameter space to search. However, we also provide
# another set of APIs that can make the definition of the search space easier
# and smarter. Where possible, we recommend you use this higher-level API
#
# In the following example, we use :any:`ConfigSpace.define_split` to define a
# split knob. It will enumerate all the possible ways to split an axis and
# construct the space.
#
# We also have :any:`ConfigSpace.define_reorder` for reorder knob and
# :any:`ConfigSpace.define_annotate` for annotation like unroll, vectorization,
# thread binding. When the high level API cannot meet your requirements, you
# can always fall back to using the low level API.


@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


################################################################################
# .. admonition:: More Explanation on :code:`cfg.define_split`
#
#  In this template, :code:`cfg.define_split("tile_y", y, num_outputs=2)` will
#  enumerate all possible combinations that can split axis y into two axes with
#  factors of the length of y. For example, if the length of y is 32 and we
#  want to split it into two axes using factors of 32, then there are 6
#  possible values for (length of outer axis, length of inner axis) pair,
#  namely (32, 1), (16, 2), (8, 4), (4, 8), (2, 16) or (1, 32). These are all 6
#  possible values of `tile_y`.
#
#  During scheduling, :code:`cfg["tile_y"]` is a :code:`SplitEntity` object.
#  We stores the lengths of outer axes and inner axes in
#  :code:`cfg['tile_y'].size` (a tuple with two elements).  In this template,
#  we apply it by using :code:`yo, yi = cfg['tile_y'].apply(s, C, y)`.
#  Actually, this is equivalent to :code:`yo, yi = s[C].split(y,
#  cfg["tile_y"].size[1])` or  :code:`yo, yi = s[C].split(y,
#  nparts=cfg['tile_y"].size[0])`
#
#  The advantage of using cfg.apply API is that it makes multi-level splits
#  (that is, when num_outputs >= 3) easier.

################################################################################
# Step 2: Use AutoTVM to Optimize the Matrix Multiplication
# ---------------------------------------------------------
# In Step 1, we wrote a matrix multiplication template that allowed us to
# parameterize the block size used in the `split` schedule. We can now conduct
# a search over this parameter space. The next step is to pick a tuner to guide
# the exploration of this space.
#
# Auto-tuners in TVM
# ~~~~~~~~~~~~~~~~~~
# The job for a tuner can be described by following pseudo code
#
#   .. code-block:: c
#
#    ct = 0
#    while ct < max_number_of_trials:
#        propose a batch of configs
#        measure this batch of configs on real hardware and get results
#        ct += batch_size
#
# When proposing the next batch of configs, the tuner can take different
# strategies. Some of the tuner strategies provided by TVM include:
#
# * :any:`tvm.autotvm.tuner.RandomTuner`: Enumerate the space in a random order
# * :any:`tvm.autotvm.tuner.GridSearchTuner`: Enumerate the space in a grid search order
# * :any:`tvm.autotvm.tuner.GATuner`: Using genetic algorithm to search through the space
# * :any:`tvm.autotvm.tuner.XGBTuner`: Uses a model based method. Train a XGBoost model to
#   predict the speed of lowered IR and pick the next batch according to the
#   prediction.
#
# You can choose the tuner according to the size of your space, your time
# budget and other factors.  For example, if your space is very small (less
# than 1000), a grid-search tuner or a random tuner is good enough. If your
# space is at the level of 10^9 (this is the space size of a conv2d operator on
# CUDA GPU), XGBoostTuner can explore more efficiently and find better configs.

################################################################################
# Begin tuning
# ~~~~~~~~~~~~
# Here we continue our matrix multiplication example. First we create a tuning
# task. We can also inspect the initialized search space. In this case, for a
# 512x512 square matrix multiplication, the space size is 10x10=100 Note that
# the task and search space are independent of the tuner picked.

N, L, M = 512, 512, 512
task = autotvm.task.create("tutorial/matmul", args=(N, L, M, "float32"), target="llvm")
print(task.config_space)

################################################################################
# Then we need to define how to measure the generated code and pick a tuner.
# Since our space is small, a random tuner is just okay.
#
# We only make 10 trials in this tutorial for demonstration. In practice, you
# can do more trials according to your time budget. We will log the tuning
# results into a log file. This file can be used to choose the best
# configuration discovered by the tuner later.

# logging config (for printing tuning log to the screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

################################################################################
# There are two steps for measuring a config: build and run. By default, we use
# all CPU cores to compile program. We then measure them sequentially. To help
# reduce variance, we take 5 measurements and average them.
measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("matmul.log")],
)

################################################################################
# With tuning completed, we can choose the configuration from the log file that
# has the best measured performance and compile the schedule with the
# corresponding parameters. We also do a quick verification that the schedule is
# producing correct answers.  We can call the function :code:`matmul` directly
# under the :any:`autotvm.apply_history_best` context. When we call this
# function, it will query the dispatch context with its argument and get the
# best config with the same argument.

# apply history best from log file
with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = matmul(N, L, M, "float32")
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

################################################################################
# Final Notes and Summary
# -----------------------
# In this tutorial, we have shown how to build operator templates that allow
# TVM to search a parameter space and choose optimized schedule configurations.
# To gain a deeper understanding of how this works, we recommend expanding on
# this example by adding new search parameters to the schedule based on
# schedule operations demonstrated in the :ref: `Getting Started With Tensor
# Expressions <tensor_expr_get_started>_` tutorial. In the upcoming sections, we
# will demonstrate the AutoScheduler, a method for TVM to optimize common
# operators without the need for the user to provide a user-defined template.
