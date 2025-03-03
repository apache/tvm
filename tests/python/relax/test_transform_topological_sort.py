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
from tvm.script import ir as I, relax as R


class BaseCompare(tvm.testing.CompareBeforeAfter):
    def transform(self):
        return tvm.relax.transform.TopologicalSort(
            order=self.order,
            direction=self.direction,
        )


class TestDepthFirstFromInputs(BaseCompare):
    """Sort DataflowBlock bindings with DFS, starting from inputs

    Starting with the inputs to the DataflowBlock, sort the variable
    bindings according to their occurrence in a depth-first search.
    """

    order = "depth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D


class TestDepthFirstFromInputWithConstant(BaseCompare):
    """Topological sort must produce legal ordering.

    Here, both `C1` and `C2` use the input tensor `A`.  However, they
    also use the tensors `B1` and `B2`.  The bindings for `C1` and
    `C2` may not be emitted until after all their inputs have been
    emitted.

    In addition, the bindings `B1` and `B2` do not require any of the
    function inputs to compute.  If the DFS only used the function
    parameters as the initial search nodes, it would fail to output
    these variable bindings.
    """

    order = "depth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.const(1)
                B2 = R.const(2)
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D2 = R.add(A, C2)
                D1 = R.add(A, C1)
                E = R.add(D1, D2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.const(1)
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                B2 = R.const(2)
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(D1, D2)
                R.output(E)
            return E


class TestDepthFirstFromInputWithMultipleInputs(BaseCompare):
    """Use parameter order for deterministic sort

    Here, both `C1` and `C2` use the input tensor `A`, as well as
    input tensors `B1` and `B2`, respectively.  Since `B1` appears
    before `B2`, `C1` should be sorted before `C2`.
    """

    order = "depth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B1: R.Tensor, B2: R.Tensor):
            with R.dataflow():
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D2 = R.add(A, C2)
                D1 = R.add(A, C1)
                E = R.add(D1, D2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B1: R.Tensor, B2: R.Tensor):
            with R.dataflow():
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(D1, D2)
                R.output(E)
            return E


class TestDepthFirstBreakTiesByExistingOrder(BaseCompare):
    """If DFS is ambiguous, provide deterministic output

    Here, both `B1` and `B2` use the input tensor `A`.  Since there
    are no other inputs for `B1` or `B2`, they remain in the same
    relative order as the input function, and `B1` is emitted before
    `B2`.  The DFS then continues, placing `C1` immediately after
    `B1`.
    """

    order = "depth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D


class TestDepthFirstFromOutput(BaseCompare):
    """Sort DataflowBlock bindings with DFS, starting from outputs

    Starting with the outputs to the DataflowBlock, sort the variable
    bindings according to their occurrence in a depth-first search.

    Like `TestDepthFirstFromInputs`, but perform the search starting
    at the output.
    """

    order = "depth-first"
    direction = "from-outputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D


class TestDepthFirstFromOutputTupleWithBinding(BaseCompare):
    """A dataflow block may produce multiple outputs

    If a dataflow block produces multiple outputs, the result should
    be sorted according to the order in which the outputs are used.
    Here, `C1` is used before `C2`, so the expressions required to
    compute `C1` are moved before the expressions required to compute
    `C2`.
    """

    order = "depth-first"
    direction = "from-outputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                R.output(C1, C2)
            gv = (C1, C2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                R.output(C1, C2)
            gv = (C1, C2)
            return gv


class TestDepthFirstFromOutputTupleWithoutBinding(BaseCompare):
    """A dataflow block may produce multiple outputs

    Like `TestDepthFirstFromOutputTupleWithBinding`, but the
    DataflowBlock's outputs are not used as part of a variable
    binding.  Because in-line tuples are not normalized to variable
    bindings, this case must be handled explicitly.
    """

    order = "depth-first"
    direction = "from-outputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                R.output(C1, C2)
            return (C1, C2)

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                R.output(C1, C2)
            return (C1, C2)


class TestDepthFirstFromOutputWithUnusedVariables(BaseCompare):
    """Sort DataflowBlock bindings with DFS, starting from outputs

    The variables `D1` and `D2` are unused, but must still appear
    within the output DataflowBlock.

    This is analogous to `TestDepthFirstFromInputWithConstant`.
    Similar to how a DFS starting from the function inputs can
    accidentally skip expressions with no inputs, a DFS starting from
    the function outputs can accidentally skip expressions that do not
    contribute to the output.
    """

    order = "depth-first"
    direction = "from-outputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                D2 = R.add(A, C2)
                E = R.add(C1, C2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(C1, C2)
                R.output(E)
            return E


class TestDepthFirstFromInputWithUnusedParameters(BaseCompare):
    """Sort DataflowBlock bindings with DFS, starting from inputs

    Functions may accept parameters that are not used.
    """

    order = "depth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, Unused: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, Unused: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D


class TestBreadthFirst(BaseCompare):
    order = "breadth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D


class TestBreadthFirstBreakTiesByExistingOrder(BaseCompare):
    order = "breadth-first"
    direction = "from-inputs"

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                D = R.add(C2, C1)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D


if __name__ == "__main__":
    tvm.testing.main()
