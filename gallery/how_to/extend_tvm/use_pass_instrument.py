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
.. _tutorial-use-pass-instrument:

How to Use TVM Pass Instrument
==============================
**Author**: `Chi-Wei Wang <https://github.com/chiwwang>`_

As more and more passes are implemented, it becomes useful to instrument
pass execution, analyze per-pass effects, and observe various events.

We can instrument passes by providing a list of :py:class:`tvm.ir.instrument.PassInstrument`
instances to :py:class:`tvm.transform.PassContext`. We provide a pass instrument
for collecting timing information (:py:class:`tvm.ir.instrument.PassTimingInstrument`),
but an extension mechanism is available via the :py:func:`tvm.instrument.pass_instrument` decorator.

This tutorial demonstrates how developers can use ``PassContext`` to instrument
passes. Please also refer to the :ref:`pass-infra`.
"""

import tvm
import tvm.relay as relay
from tvm.relay.testing import resnet
from tvm.contrib.download import download_testdata
from tvm.relay.build_module import bind_params_by_name
from tvm.ir.instrument import (
    PassTimingInstrument,
    pass_instrument,
)


###############################################################################
# Create An Example Relay Program
# -------------------------------
# We use pre-defined resnet-18 network in Relay.
batch_size = 1
num_of_image_class = 1000
image_shape = (3, 224, 224)
output_shape = (batch_size, num_of_image_class)
relay_mod, relay_params = resnet.get_workload(num_layers=18, batch_size=1, image_shape=image_shape)
print("Printing the IR module...")
print(relay_mod.astext(show_meta_data=False))


###############################################################################
# Create PassContext With Instruments
# -----------------------------------
# To run all passes with an instrument, pass it via the ``instruments`` argument to
# the ``PassContext`` constructor. A built-in ``PassTimingInstrument`` is used to
# profile the execution time of each passes.
timing_inst = PassTimingInstrument()
with tvm.transform.PassContext(instruments=[timing_inst]):
    relay_mod = relay.transform.InferType()(relay_mod)
    relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
    # before exiting the context, get profile results.
    profiles = timing_inst.render()
print("Printing results of timing profile...")
print(profiles)


###############################################################################
# Use Current PassContext With Instruments
# ----------------------------------------
# One can also use the current ``PassContext`` and register
# ``PassInstrument`` instances by ``override_instruments`` method.
# Note that ``override_instruments`` executes ``exit_pass_ctx`` method
# if any instrument already exists. Then it switches to new instruments
# and calls ``enter_pass_ctx`` method of new instruments.
# Refer to following sections and :py:func:`tvm.instrument.pass_instrument` for these methods.
cur_pass_ctx = tvm.transform.PassContext.current()
cur_pass_ctx.override_instruments([timing_inst])
relay_mod = relay.transform.InferType()(relay_mod)
relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
profiles = timing_inst.render()
print("Printing results of timing profile...")
print(profiles)


###############################################################################
# Register empty list to clear existing instruments.
#
# Note that ``exit_pass_ctx`` of ``PassTimingInstrument`` is called.
# Profiles are cleared so nothing is printed.
cur_pass_ctx.override_instruments([])
# Uncomment the call to .render() to see a warning like:
# Warning: no passes have been profiled, did you enable pass profiling?
# profiles = timing_inst.render()


###############################################################################
# Create Customized Instrument Class
# ----------------------------------
# A customized instrument class can be created using the
# :py:func:`tvm.instrument.pass_instrument` decorator.
#
# Let's create an instrument class which calculates the change in number of
# occurrences of each operator caused by each pass. We can look at ``op.name`` to
# find the name of each operator. And we do this before and after passes to calculate the difference.


@pass_instrument
class RelayCallNodeDiffer:
    def __init__(self):
        self._op_diff = []
        # Passes can be nested.
        # Use stack to make sure we get correct before/after pairs.
        self._op_cnt_before_stack = []

    def enter_pass_ctx(self):
        self._op_diff = []
        self._op_cnt_before_stack = []

    def exit_pass_ctx(self):
        assert len(self._op_cnt_before_stack) == 0, "The stack is not empty. Something wrong."

    def run_before_pass(self, mod, info):
        self._op_cnt_before_stack.append((info.name, self._count_nodes(mod)))

    def run_after_pass(self, mod, info):
        # Pop out the latest recorded pass.
        name_before, op_to_cnt_before = self._op_cnt_before_stack.pop()
        assert name_before == info.name, "name_before: {}, info.name: {} doesn't match".format(
            name_before, info.name
        )
        cur_depth = len(self._op_cnt_before_stack)
        op_to_cnt_after = self._count_nodes(mod)
        op_diff = self._diff(op_to_cnt_after, op_to_cnt_before)
        # only record passes causing differences.
        if op_diff:
            self._op_diff.append((cur_depth, info.name, op_diff))

    def get_pass_to_op_diff(self):
        """
        return [
          (depth, pass_name, {op_name: diff_num, ...}), ...
        ]
        """
        return self._op_diff

    @staticmethod
    def _count_nodes(mod):
        """Count the number of occurrences of each operator in the module"""
        ret = {}

        def visit(node):
            if isinstance(node, relay.expr.Call):
                if hasattr(node.op, "name"):
                    op_name = node.op.name
                else:
                    # Some CallNode may not have 'name' such as relay.Function
                    return
                ret[op_name] = ret.get(op_name, 0) + 1

        relay.analysis.post_order_visit(mod["main"], visit)
        return ret

    @staticmethod
    def _diff(d_after, d_before):
        """Calculate the difference of two dictionary along their keys.
        The result is values in d_after minus values in d_before.
        """
        ret = {}
        key_after, key_before = set(d_after), set(d_before)
        for k in key_before & key_after:
            tmp = d_after[k] - d_before[k]
            if tmp:
                ret[k] = d_after[k] - d_before[k]
        for k in key_after - key_before:
            ret[k] = d_after[k]
        for k in key_before - key_after:
            ret[k] = -d_before[k]
        return ret


###############################################################################
# Apply Passes and Multiple Instrument Classes
# --------------------------------------------
# We can use multiple instrument classes in a ``PassContext``.
# However, it should be noted that instrument methods are executed sequentially,
# obeying the order of ``instruments`` argument.
# So for instrument classes like ``PassTimingInstrument``, it is inevitable to
# count-up the execution time of other instrument classes to the final
# profile result.
call_node_inst = RelayCallNodeDiffer()
desired_layouts = {
    "nn.conv2d": ["NHWC", "HWIO"],
}
pass_seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.ConvertLayout(desired_layouts),
        relay.transform.FoldConstant(),
    ]
)
relay_mod["main"] = bind_params_by_name(relay_mod["main"], relay_params)
# timing_inst is put after call_node_inst.
# So the execution time of ``call_node.inst.run_after_pass()`` is also counted.
with tvm.transform.PassContext(opt_level=3, instruments=[call_node_inst, timing_inst]):
    relay_mod = pass_seq(relay_mod)
    profiles = timing_inst.render()
# Uncomment the next line to see timing-profile results.
# print(profiles)


###############################################################################
# We can see how many CallNode increase/decrease per op type.
from pprint import pprint

print("Printing the change in number of occurrences of each operator caused by each pass...")
pprint(call_node_inst.get_pass_to_op_diff())


###############################################################################
# Exception Handling
# ------------------
# Let's see what happens if an exception occurs in a method of a ``PassInstrument``.
#
# Define ``PassInstrument`` classes which raise exceptions in enter/exit ``PassContext``:
class PassExampleBase:
    def __init__(self, name):
        self._name = name

    def enter_pass_ctx(self):
        print(self._name, "enter_pass_ctx")

    def exit_pass_ctx(self):
        print(self._name, "exit_pass_ctx")

    def should_run(self, mod, info):
        print(self._name, "should_run")
        return True

    def run_before_pass(self, mod, pass_info):
        print(self._name, "run_before_pass")

    def run_after_pass(self, mod, pass_info):
        print(self._name, "run_after_pass")


@pass_instrument
class PassFine(PassExampleBase):
    pass


@pass_instrument
class PassBadEnterCtx(PassExampleBase):
    def enter_pass_ctx(self):
        print(self._name, "bad enter_pass_ctx!!!")
        raise ValueError("{} bad enter_pass_ctx".format(self._name))


@pass_instrument
class PassBadExitCtx(PassExampleBase):
    def exit_pass_ctx(self):
        print(self._name, "bad exit_pass_ctx!!!")
        raise ValueError("{} bad exit_pass_ctx".format(self._name))


###############################################################################
# If an exception occurs in ``enter_pass_ctx``, ``PassContext`` will disable the pass
# instrumentation. And it will run the ``exit_pass_ctx`` of each ``PassInstrument``
# which successfully finished ``enter_pass_ctx``.
#
# In following example, we can see ``exit_pass_ctx`` of `PassFine_0` is executed after exception.
demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadEnterCtx("PassBadEnterCtx"),
        PassFine("PassFine_1"),
    ]
)
try:
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])

###############################################################################
# Exceptions in ``PassInstrument`` instances cause all instruments of the current ``PassContext``
# to be cleared, so nothing is printed when ``override_instruments`` is called.
demo_ctx.override_instruments([])  # no PassFine_0 exit_pass_ctx printed....etc

###############################################################################
# If an exception occurs in ``exit_pass_ctx``, then the pass instrument is disabled.
# Then exception is propagated. That means ``PassInstrument`` instances registered
# after the one throwing the exception do not execute ``exit_pass_ctx``.
demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadExitCtx("PassBadExitCtx"),
        PassFine("PassFine_1"),
    ]
)
try:
    # PassFine_1 execute enter_pass_ctx, but not exit_pass_ctx.
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])

###############################################################################
# Exceptions occurred in ``should_run``, ``run_before_pass``, ``run_after_pass``
# are not handled explicitly -- we rely on the context manager (the ``with`` syntax)
# to exit ``PassContext`` safely.
#
# We use ``run_before_pass`` as an example:
@pass_instrument
class PassBadRunBefore(PassExampleBase):
    def run_before_pass(self, mod, pass_info):
        print(self._name, "bad run_before_pass!!!")
        raise ValueError("{} bad run_before_pass".format(self._name))


demo_ctx = tvm.transform.PassContext(
    instruments=[
        PassFine("PassFine_0"),
        PassBadRunBefore("PassBadRunBefore"),
        PassFine("PassFine_1"),
    ]
)
try:
    # All exit_pass_ctx are called.
    with demo_ctx:
        relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])

###############################################################################
# Also note that pass instrumentation is not disable. So if we call
# ``override_instruments``, the ``exit_pass_ctx`` of old registered ``PassInstrument``
# is called.
demo_ctx.override_instruments([])

###############################################################################
# If we don't wrap pass execution with ``with`` syntax, ``exit_pass_ctx`` is not
# called. Let try this with current ``PassContext``:
cur_pass_ctx = tvm.transform.PassContext.current()
cur_pass_ctx.override_instruments(
    [
        PassFine("PassFine_0"),
        PassBadRunBefore("PassBadRunBefore"),
        PassFine("PassFine_1"),
    ]
)

###############################################################################
# Then call passes. ``exit_pass_ctx`` is not executed after the exception,
# as expectation.
try:
    # No ``exit_pass_ctx`` got executed.
    relay_mod = relay.transform.InferType()(relay_mod)
except ValueError as ex:
    print("Catching", str(ex).split("\n")[-1])

###############################################################################
# Clear instruments.
cur_pass_ctx.override_instruments([])
