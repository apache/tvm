..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _relay-op-strategy:

Relay Operator Strategy
=======================

In order to lower Relay operators to the implementations defined in TOPI
library, a compute and schedule function need to be registered to each Relay
operator.  However, compute and schedule functions are usually specialized for
each target, and further, even for the same target, we may have multiple
algorithms and implementations available. To deal with the complexity, we
introduce operator strategy to allow developers to define a flexible lowering
strategy for each operator and target.


Operator Strategy Design
------------------------

The basic element in operator strategy is an ``OpImplementation``. It includes
the a pair of compute and schedule function, the name of the implementation,
and a priority level (the use of priority level is explained in
`Select Implementation from Op Strategy`_).

The ``OpStrategy`` includes a list of ``OpSpecialization``. Each ``OpSpecialization``
contains a list of ``OpImplementation`` associated with a ``SpecializedCondition``
(see definition in ``include/tvm/te/schedule.h``).  The ``SpecializedCondition``
can be null, indicating the implementations are generally applicable;
otherwise, the implementations are only considered when the specialized
condition is satisfied. ``SpecializedCondition`` consists of a list
of clauses defined in Tensor Expression in conjunctive normal form (CNF) and
only supports conditions on tensor shapes.

Last, a ``FTVMStrategy`` function is registered to each Relay operator.
``FTVMStrategy`` is a generic function (see ``include/tvm/target/generic_func.h``),
that can be overwritten for each target. The function signature is

.. code:: c

    OpStrategy(const Attrs& attrs, const Array<Tensor>& inputs, const Type& out_type, const Target& target)

that the function returns an ``OpStrategy`` given the op attributes, input
tensors, output types, and target to compile to.



Register Strategy for An Operator
---------------------------------

There are three methods to register a strategy function for an operator,
defined in ``python/tvm/relay/op/op.py``.

First, for operators that have injective, broadcast, or reduction pattern, we
can call ``register_injective_schedule``, ``register_broadcast_schedule``, and
``register_reduce_schedule`` repsectively. The schedule function for these
patterns are already registered by each target and can be applied to these
operators. We assume the compute function should be the same across all targets,
and ``FTVMCompute`` needs to be registered to the op before invoking register
schedule.

.. code:: python

    register_broadcast_schedule("add")

Second, for operators that doesn't have these common patterns mentioned before,
but also have the same compute function for all targets, we can use
``register_schedule`` API. Before that, we need to first define the
``FTVMSchedule`` function as follows:

.. code:: python

    # add to python/tvm/relay/op/strategy/generic.py
    @generic_func
    def schedule_pool(attrs, outs, target):
        ...

    # add to each target file in python/tvm/relay/op/strategy, e.g., x86.py, cuda.py, etc.
    @schedule_pool.register("cpu")
    def schedule_pool_cpu(attrs, outs, target):
        ...

Now that we've created the ``FTVMSchedule`` for this new operator, we can
register the strategy using ``register_schedule``:

.. code:: python

    register_schedule("nn.max_pool2d", strategy.schedule_pool)

Third, for most comprehensive usage of op strategy, we can allow operator to use
different implementations for both compute and schedule for different targets.
In python, ``OpStrategy`` provides only one API, adding an implementation to the
strategy:

.. code:: python

    def add_implementation(self, compute, schedule, name="default", plevel=10)

Now let's define the ``FTVMStrategy`` function as follows:

.. code:: python

    # add to python/tvm/relay/op/strategy/generic.py
    @override_native_generic_func("topk_strategy")
    def topk_strategy(attrs, inputs, out_type, target):
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_topk(topi.topk),
            wrap_topi_schedule(topi.generic.schedule_topk),
            name="topk.generic")
        return strategy

    # add to each target file in python/tvm/relay/op/strategy, e.g., x86.py, cuda.py, etc.
    @dense_strategy.register(["cuda", "gpu"])
    def topk_strategy_cuda(attrs, inputs, out_type, target):
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_my_new_op(topi.cuda.topk),
            wrap_topi_schedule(topi.cuda.schedule_topk),
            name="topk.cuda")
        return strategy

In this example, we use two wrapper functions that wrap the topi compute and
schedule function to conform with the required function signature. Usually we
need to write a customized compute wrap function to retrieve different fields
from op attributes. After that, we can register this strategy to the new
operator with

.. code:: python

    register_strategy("topk", strategy.topk_strategy)


Advanced Strategy Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

The example above only shows the very basic strategy function.
In this part, we will show a few advanced ways to define op strategy.

First, we can add multiple implementations that use different algorithms to the
same operator:

.. code:: python

    strategy.add_implementation(
        wrap_compute_conv2d(topi.cuda.conv2d_nchw),
        wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
        name="conv2d_nchw.cuda",
        plevel=10)

    if winograd_condition:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
            wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
            name="conv2d_nchw_winograd.cuda",
            plevel=15)

In this example, we add two implementations to the conv2d strategy where
winograd algorithm is only added when ``winograd_condition`` is true.
The implementation ``"conv2d_nchw_winograd.cuda"`` will be used to compile
conv2d when ``winograd_condition`` is true as it has higher
priority level (this could be changed if certain implementation is an AutoTVM
template. See `Select Implementation from Op Strategy`_ for more
details). Otherwise, ``"conv2d_nchw.cuda"`` is used.

We can extend the example above to third party library implementation. For
example, we can add the implementation that invokes kernel in the cblas
library when cblas is included in the target.

.. code:: python

    if "cblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_cblas),
            wrap_topi_schedule(topi.x86.schedule_dense_cblas),
            name="dense_cblas.x86",
            plevel=15)


Further, we can add implementation specialized for a certain range of shapes.
The code below shows an example of dense strategy that adds an implementation
that is specialized for ``m`` greater than 16. The main difference between
hardcode python condition like examples above and specialized condition is that
it allows TVM to generate multiple kernels when the input tensors have symbolic
shapes. The compile engine will generate a dispatch function that invokes the
specialized kernel when the corresponding condition is met; otherwise,
invoke the kernel that has no associated specialized condition (``dense_common``
in this example). This part is still work in progress. More details will be
provided after it is done.

.. code:: python

    def dense_strategy(attrs, inputs, out_type, target):
        m = inputs[0].shape[0]
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_dense(dense_compute1),
            wrap_topi_schedule(dense_schedule1),
            name="dense_common")

        with tvm.te.SpecializedCondition(m > 16):
            strategy.add_implementation(
                wrap_compute_dense(dense_compute2),
                wrap_topi_schedule(dense_schedule2),
                name="dense_for_large_m",
                plevel=15)

        return strategy


Register Strategy for A New Target
----------------------------------

There are two ways to register strategies for a new target. The more
straightforward one is adding a new target file in the directory
``python/tvm/relay/op/strategy``. You only need to customize the strategy for
ops that have been implemented for this new target and reuse the generic
strategies for the rest.

Alternatively, you can also register the strategy for the new target outside the
TVM python library. The following code snippet shows an example how to do
so. You can find more examples in ``vta/python/vta/top/op.py``.

.. code:: python

    @relay.op.strategy.someop_strategy.register("mytarget")
    def someop_strategy_mytarget(attrs, inputs, out_type, target):
        ...


Select Implementation from Op Strategy
--------------------------------------

During the compilation, Relay compile engine needs to determine which
implementation to use for an operator when there are multiple. The selection
policy works as follows.

When the input tensors to an operator or a fused op all have constant shapes,
the compile engine first finds the best implementation based on AutoTVM tuning
logs. If there is no implementation that is an AutoTVM template or all AutoTVM
templates have fallback configs, the implementation with highest priority level
will then be chosen. Implementations with same priority level in this case leads
to an undefined behavior, and any of them might be selected.

The selection policy for ops with symbolic input shapes is still work in
progess. Currently, if any input tensor has a symbolic shape, only the
implementation with highest priority level will be used for this operator. This
will be updated after the implemention finishes.

For debug purpose, you can add the following lines before you compile the Relay
model to learn which implementation is used for each operator.

.. code:: python

    logging.getLogger("compile_engine").setLevel(logging.INFO)
    logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))
