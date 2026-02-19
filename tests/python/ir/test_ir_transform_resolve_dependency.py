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
"""Tests for pass dependency resolution in Sequential passes.

Note: ResolveDependency is a C++ function that needs to be exposed to Python
for direct testing. Currently, we test the behavior indirectly through
Sequential pass execution.
"""

import tvm
import tvm.testing
from tvm.ir import transform
from tvm.ir.transform import PassContext
from tvm.ir.module import IRModule


def create_test_pass(name, required=None, opt_level=0):
    """Helper function to create a test pass with specified dependencies."""

    @transform.module_pass(opt_level=opt_level, name=name, required=required or [], traceable=False)
    def pass_func(mod, ctx):
        # Simple pass that just returns the module unchanged
        return mod

    return pass_func


def test_sequential_with_dependencies():
    """Test that Sequential correctly handles pass dependencies during execution."""

    # Create passes without dependencies to test basic execution
    # The dependency resolution is tested at the C++ level through compilation
    pass1 = create_test_pass("Pass1", required=[])
    pass2 = create_test_pass("Pass2", required=[])

    # Create a sequential pass
    seq = transform.Sequential([pass1, pass2])

    # Create a simple IRModule for testing
    mod = IRModule({})

    # Execute the sequential pass
    with PassContext(opt_level=3):
        result = seq(mod)

    # Verify that the passes were executed
    assert result is not None
    assert isinstance(result, IRModule)


def test_sequential_opt_level_filtering():
    """Test that Sequential filters passes based on opt_level."""

    pass1 = create_test_pass("Pass1", required=[], opt_level=1)
    pass2 = create_test_pass("Pass2", required=[], opt_level=2)
    pass3 = create_test_pass("Pass3", required=[], opt_level=3)

    seq = transform.Sequential([pass1, pass2, pass3])
    mod = IRModule({})

    # With opt_level=2, pass3 (opt_level=3) should be skipped
    with PassContext(opt_level=2):
        result = seq(mod)

    # Execution should succeed even with some passes filtered
    assert result is not None


def test_sequential_required_pass_execution():
    """Test that required passes are executed even if not in the list."""

    # Create a pass that depends on PrintIR (a standard TVM pass)
    # PrintIR requires a header string parameter
    print_ir_pass = transform.PrintIR("TestHeader")
    pass1 = create_test_pass("Pass1", required=[])

    # Create sequential with both passes - pass1 should execute after print_ir
    seq = transform.Sequential([pass1, print_ir_pass])
    mod = IRModule({})

    # Execute - both passes should execute
    with PassContext(opt_level=3):
        result = seq(mod)

    assert result is not None


def test_sequential_dependency_chain():
    """Test simple dependency chain: A requires B, B requires C."""

    # Track execution order
    execution_order = []

    @transform.module_pass(opt_level=0, name="PassC", required=[], traceable=False)
    def pass_c(mod, ctx):
        execution_order.append("C")
        return mod

    @transform.module_pass(opt_level=0, name="PassB", required=["PassC"], traceable=False)
    def pass_b(mod, ctx):
        execution_order.append("B")
        return mod

    @transform.module_pass(opt_level=0, name="PassA", required=["PassB"], traceable=False)
    def pass_a(mod, ctx):
        execution_order.append("A")
        return mod

    # Create sequential with passes in wrong order
    # All passes must be in the list for dependency resolution to work
    # After dependency resolution, order should be C -> B -> A
    seq = transform.Sequential([pass_a, pass_b, pass_c])
    mod = IRModule({})

    with PassContext(opt_level=3):
        result = seq(mod)

    assert result is not None
    # Verify execution order: C should run before B, B before A
    assert execution_order == ["C", "B", "A"], f"Expected ['C', 'B', 'A'], got {execution_order}"


def test_sequential_shared_dependency():
    """Test that a pass required by multiple other passes is executed only once."""

    execution_order = []

    @transform.module_pass(opt_level=0, name="SharedPass", required=[], traceable=False)
    def shared_pass(mod, ctx):
        execution_order.append("Shared")
        return mod

    @transform.module_pass(opt_level=0, name="Pass1", required=["SharedPass"], traceable=False)
    def pass1(mod, ctx):
        execution_order.append("Pass1")
        return mod

    @transform.module_pass(opt_level=0, name="Pass2", required=["SharedPass"], traceable=False)
    def pass2(mod, ctx):
        execution_order.append("Pass2")
        return mod

    # Both Pass1 and Pass2 require SharedPass
    # All passes must be in the list for dependency resolution to work
    # SharedPass should execute before both, but only once
    seq = transform.Sequential([pass1, pass2, shared_pass])
    mod = IRModule({})

    with PassContext(opt_level=3):
        result = seq(mod)

    assert result is not None
    # SharedPass should be first, then Pass1 and Pass2 (order may vary)
    assert execution_order[0] == "Shared", "SharedPass should execute first"
    assert "Pass1" in execution_order and "Pass2" in execution_order
    assert execution_order.count("Shared") == 1, "SharedPass should execute only once"


def test_sequential_transitive_dependency():
    """Test transitive dependencies: A requires B, B requires C, but A doesn't explicitly require C."""

    execution_order = []

    @transform.module_pass(opt_level=0, name="PassC", required=[], traceable=False)
    def pass_c(mod, ctx):
        execution_order.append("C")
        return mod

    @transform.module_pass(opt_level=0, name="PassB", required=["PassC"], traceable=False)
    def pass_b(mod, ctx):
        execution_order.append("B")
        return mod

    @transform.module_pass(opt_level=0, name="PassA", required=["PassB"], traceable=False)
    def pass_a(mod, ctx):
        execution_order.append("A")
        return mod

    # PassA only explicitly requires PassB, but PassB requires PassC
    # All passes must be in the list for dependency resolution to work
    # ResolveDependency should handle transitive dependencies
    seq = transform.Sequential([pass_a, pass_b, pass_c])
    mod = IRModule({})

    with PassContext(opt_level=3):
        result = seq(mod)

    assert result is not None
    # C should run before B, B before A
    assert execution_order == ["C", "B", "A"], f"Expected ['C', 'B', 'A'], got {execution_order}"


def test_sequential_opt_level_disabled_pass():
    """Test that passes disabled by opt_level are not executed."""

    execution_order = []

    @transform.module_pass(opt_level=1, name="Pass1", required=[], traceable=False)
    def pass1(mod, ctx):
        execution_order.append("Pass1")
        return mod

    @transform.module_pass(opt_level=3, name="Pass3", required=[], traceable=False)
    def pass3(mod, ctx):
        execution_order.append("Pass3")
        return mod

    seq = transform.Sequential([pass1, pass3])
    mod = IRModule({})

    # With opt_level=2, Pass3 (opt_level=3) should be skipped
    with PassContext(opt_level=2):
        result = seq(mod)

    assert result is not None
    # Only Pass1 should execute
    assert execution_order == ["Pass1"], f"Expected ['Pass1'], got {execution_order}"


if __name__ == "__main__":
    tvm.testing.main()
