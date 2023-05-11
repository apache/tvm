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

import tvm
import tvm.testing
from tvm.script import tir as T
import pytest

# ---------------------------------------------------------------------------------------------------
# ABOUT THIS FILE:
# ---------------------------------------------------------------------------------------------------
# We (cconvey / OctoML) are working on a sequence of PRs to allow a single TIR primfunc's
# AST to be sliced into multiple partitiones, where each partition will be converted into
# a new TIR primfunc. (See https://en.wikipedia.org/wiki/Program_slicing).
#
# The unit tests below provide a roadmap for that sequence of PRs; each PR should allow
# one more of these tests to pass.
#
# NOTE: These unit tests may change as work progresses.  They aren't meant to
# indicate hard requirements.

# NOTE! The `tvm.testing.CompareBeforeAfter` class provides TWO useful mechanisms for
# these tests:
#
# (a) It lets us specify code snippets which are valid Python, but which aren't YET
#     recognized as valid TVMScript.  This allows unit tests for new constructs,
#     e.g. 'call_tir(...)' to simply be disabled rather than fully commented out.
#
# (b) It lets us structurally compare the TIR bodies of two primfuncs.
#
#     Note that some of the tests below will require the structural comparison of
#     two entire IRModules, not just primfuncs.  This will require adding functionality
#     to the `CompareBeforeAfter` class, or implementing that level of comparison within
#     the individual unit tests.
#
# Some of the unit tests below which require whole-IRModule comparison.  For expedience
# we simply comment out the (early draft) bodies of those unit tests, rather than
# hacking their structure to get the benefits of (a).


# ---------------------------------------------------------------------------------------------------
# 'CALL_TIR' (AND RELATED) CAVEATS:
# ---------------------------------------------------------------------------------------------------
# (c) "call_tir" is a placeholder name.
#     The TVM "Relax" effort also defines a node named "call_tir", which is likely
#     become something different from what we're calling "call_tir" here.  So
#     we may rename *this* "call_tir" during implementation.
#
# (d) For "call_tir" calls, the syntax/semantics for passing buffer regions is still
#     an active area of development.  So that detail of these unit tests is likely
#     to change.
#
# (e) The specific string "extract_as_subroutine" used to annotate some IR Blocks,
#     i.e., `T.annotate("extract_as_subroutine", ...)`, may change as work progresses.


# ---------------------------------------------------------------------------------------------------
# step 1: Simply passes Python / TVMScript parsing.
# ---------------------------------------------------------------------------------------------------
#
#   The only requirement for this test is that the TVMScript parser
#   doesn't raise an error when encountering `T.call_tir(foo)`,
#   where "foo" is a syntactically valid TVMScript function name.
#
#   NOTE! The role of this unit test should evolve as follows:
#   1) Initially the test should fail, because we haven't yet changed the TVMScript
#      parser to support 'call_tir'.
#
#   2) Initial TVMScript support for 'call_tir' will be minimal, essentially ignoring
#      it.  This test should pass once that change is made.
#
#   3) As support for 'call_tir' becomes more complete, this test should once again
#      fail, because the specified callee doesn't exist.  This test should be updated
#      to once again expect failure.
@pytest.mark.xfail(reason="Awaiting TVMScript support for 'call_tir' token.", strict=True)
class TestParseCallTIR(tvm.testing.CompareBeforeAfter):
    """
    Simply confirm that the TIR node `call_tir` doesn't interfere with
    the successful parsing of the TVMScript.
    """

    def before():
        T.call_tir(add_one)
        T.evalute(0)

    def expected():
        T.evaluate(0)

    # Provide a trivial 'transform' pass to satisfy the requirements of
    # tvm.testing.CompareBeforeAfter.
    transform = tvm.tir.transform.prim_func_pass(lambda func, _mod, _ctx: func, 0)


# ---------------------------------------------------------------------------------------------------
# step 2: transform annotated block ==> separate primfuncs + call_tir
#
# NOTE: This early-draft version of the unit test contains pseudocode to compare entire IRModule
# objects, analogously to how tvm.testing.CompareBeforeAfter compares two primfuncs.
# TVM's testing infrastructure currently has no such functionality, and it will need to be added
# (or approximated) to make this unit test useable.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="Awaiting TVMScript support for 'call_tir' and T.annotation(\"extract_as_subroutine\").",
    strict=True,
)
class TestAnnotateAndSliceTIR(tvm.testing.CompareBeforeAfter):
    # def test_annotate_and_slice():
    #    @tvm.script.ir_module
    #    class irmod_before:
    #        @T.prim_func
    #        def main(A: T.Buffer((1,), "int8"):
    #            #A = T.match_buffer(a, (1,), "int8")
    #            A[0] = 0
    #            with T.block("block_foo"): # optional: give this block a name, perhaps for testing?
    #                # NOTE: nice to have: human control over name used for the generated callee
    #                T.annotate("extract_as_subroutine", "add_one")
    #                A[0] += 1
    #                return 42
    #
    #    @tvm.script.ir_module
    #    class irmod_after:
    #        @T.prim_func
    #        def main():
    #            A = T.buffer[[1], "int8"]
    #            A[0] = 0
    #            with T.block("block_foo"):
    #                call_tir(add_one, A)
    #
    #        @T.prim_func
    #        def add_one(X: T.buffer[[1], "int8"]):
    #            X[0] += 1
    pass


# ---------------------------------------------------------------------------------------------------
# step 3: transform call_tir ==> packed call
# ---------------------------------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="Awaiting TVMScript support for lowering of 'T.call_tir' to 'T.call_packed'.",
    strict=True,
)
class TestLowerCallTir(tvm.testing.CompareBeforeAfter):
    # @tvm.script.ir_module
    # class test_lower_before:
    #    @T.prim_func
    #    def main():
    #        A = T.buffer[[1], "int8"]
    #        A[0] = 0
    #        with T.block():
    #            call_tir(add_one, A)
    #
    #    @T.prim_func
    #    def add_one(X: T.buffer[[1], "int8"]):
    #        X[0] += 1
    #
    # @tvm.script.ir_module
    # class test_lower_after:
    #    @T.prim_func
    #    def main():
    #        A = T.buffer[[1], "int8"]
    #        A[0] = 0
    #        with T.block():
    #            # TODO: figure out the right TVMScript thing to do here
    #            call_packed(add_one, A)  # not sure about this function / interface
    #
    #    @T.prim_func
    #    def add_one(X: T.buffer[[1], "int8"]):
    #        X[0] += 1
    #
    # TODO(cconvey): additional test logic needed.
    # NOTE(lunderberg): Will also need a `transform` defined here.
    #      I think we'll want it to occur in `tvm.tir.transform.MakePackedAPI`.
    pass


# ---------------------------------------------------------------------------------------------------
# step 4: end-to-end functionality
# ---------------------------------------------------------------------------------------------------


@pytest.mark.xfail(reason="Awaiting end-to-end support for Primfunc slicing.", strict=True)
class TestPrimfuncSlicingEndToEnd(tvm.testing.CompareBeforeAfter):
    # @tvm.script.ir_module
    # class test_annotate_before:
    #    @T.prim_func
    #    def main():
    #        A = T.buffer[[1], "int8"]
    #        A[0] = 0
    #        with T.block(): # optional: give this block a name, perhaps for testing?
    #            # NOTE: nice to have: human control over name used for the generated callee
    #            T.annotate("extract_as_subroutine", "add_one")
    #            A[0] += 1
    #        assert(A[0] == 1)
    #
    # TODO(cconvey): additional test logic needed:
    #     Starting with the IRModule shown above, end up with a running test that
    #     module actually increments A[0] on Hexagon and x86-64 Linux.
    #
    # NOTE(lunderberg): We can use the function calls currently generated by `SplitHostDevice` as a template
    #     (see https://github.com/apache/tvm/blob/9a673faa74ed7cd715a4e011716bcce3fd2158b6/src/tir/transforms/split_host_device.cc#L336).
    #     Overall, we'll want to output a Call node with the operation builtin::tvm_call_packed().
    pass
