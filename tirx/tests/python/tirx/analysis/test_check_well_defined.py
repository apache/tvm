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

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_pass_simple():
    @T.function
    def element_wise(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        B = T.alloc_tensor(ty_args=[T.Tensor((128, 128), "float32")])
        for i, j in T.grid(128, 128):
            B[i, j] = A[i, j] * 2.0
        for i, j in T.grid(128, 128):
            C[i, j] = B[i, j] * 2.0

    assert tvm_tirx.tirx.analysis.verify_well_formed(element_wise)
    assert tvm_tirx.tirx.analysis.verify_well_formed(tvm_tirx.IRModule.from_expr(element_wise))


def test_error_for_out_of_scope_usage():
    """A variable may not be used after its scope ends.

    With flat Bind semantics, Bind vars are visible to all subsequent
    siblings in the same SeqStmt. True out-of-scope usage occurs when
    the Bind is inside a child scope (e.g., ForNode body) and the
    variable is used outside that scope.
    """
    i = tvm_tirx.tirx.Var("i", "int32")
    # Bind i inside a For loop body
    for_stmt = tvm_tirx.ir.For(
        tvm_tirx.tirx.Var("j", "int32"),
        0,
        1,
        tvm_tirx.ir.ForKind.DEFAULT,
        tvm_tirx.ir.SeqStmt([tvm_tirx.ir.Bind(i, 42), tvm_tirx.ir.Evaluate(i)]),
    )
    # Use i outside the For loop — this is out of scope
    body = tvm_tirx.ir.SeqStmt([for_stmt, tvm_tirx.ir.Evaluate(i)])
    func = tvm_tirx.tirx.Function([], body)

    with pytest.raises(
        (ValueError, RuntimeError),
        match="Invalid use of undefined variable i at .* no longer in-scope.",
    ):
        tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_error_for_nested_rebind_usage():
    """A variable may not be re-defined within the initial scope"""

    i = T.dynamic("i", "int32")

    @T.function(check_well_formed=False)
    def func():
        T.bind(42, var=i)
        T.bind(42, var=i)
        T.evaluate(i)

    with pytest.raises(
        (ValueError, RuntimeError),
        match="ill-formed, due to multiple nested definitions of variable i",
    ):
        tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_error_for_repeated_binding():
    """A variable may not be re-defined in the same flat scope.

    With flat Bind semantics, sequential Bind of the same variable in the
    same SeqStmt is treated as a nested redefinition (since the first Bind's
    scope extends to all subsequent siblings).
    """

    i = T.dynamic("i", "int32")

    @T.function(check_well_formed=False)
    def func():
        T.bind(42, var=i)
        T.evaluate(i)
        T.bind(17, var=i)
        T.evaluate(i)

    with pytest.raises(
        (ValueError, RuntimeError),
        match="multiple nested definitions of variable i",
    ):
        tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_error_for_cross_function_reuse():
    """A variable may not be re-defined in another function"""

    i = T.dynamic("i", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.function
        def func1():
            T.bind(42, var=i)
            T.evaluate(i)

        @T.function
        def func2():
            T.bind(42, var=i)
            T.evaluate(i)

    with pytest.raises((ValueError, RuntimeError), match="multiple definitions of variable i"):
        tvm_tirx.tirx.analysis.verify_well_formed(mod)


def test_sibling_launch_bindings_are_well_formed():
    """Sibling launch regions each introduce a fresh lexical binding."""

    @T.function
    def func(A: T.Tensor([256], "float32")):
        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 1.0

        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 2.0

    tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_reuse_of_region_parameter_across_functions_is_ill_formed():
    """An explicit region parameter may not be defined across multiple functions."""

    threadIdx_x = T.dynamic("threadIdx_x", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm_tirx.ir.StringImm("threadIdx.x"), 256],
                body_params=[threadIdx_x],
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm_tirx.ir.StringImm("threadIdx.x"), 256],
                body_params=[threadIdx_x],
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    with pytest.raises(
        (ValueError, RuntimeError),
        match="multiple definitions of variable threadIdx_x",
    ):
        tvm_tirx.tirx.analysis.verify_well_formed(mod)


def test_multiple_tensor_arguments_may_share_allocation():
    """Tensor signatures may re-use a data argument

    Like the shape/strides/elem_offset fields in a tensor, the first
    occurrence of a `tensor->data` field defines it, and the
    occurrences are usages of that definition.
    """

    @I.ir_module
    class mod:
        @T.function
        def func(A: T.Tensor([256], "float32"), B: T.Tensor([256], "float32", data=A.data)):  # noqa: F821
            pass

    tvm_tirx.tirx.analysis.verify_well_formed(mod)


def test_error_message_without_previous_definition_location():
    """Test case 1: Error message without 'It was first defined at'

    This tests the scenario where it == end(), so the error message should contain
    'TIR is ill-formed, due to multiple definitions of variable' but should NOT
    contain 'It was first defined at' since the iterator is invalid.

    With flat Bind semantics, sequential redefinitions in the same SeqStmt
    are treated as nested definitions, and the first definition location
    IS known, so the message includes location info.
    """

    x = T.dynamic("x", "int32")

    @T.function(check_well_formed=False)
    def func():
        T.bind(42, var=x)
        T.evaluate(x)

        T.bind(99, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        tvm_tirx.tirx.analysis.verify_well_formed(func, assert_mode=True)

    error_msg = str(exc_info.value)

    assert "TIR is ill-formed" in error_msg
    assert "multiple nested definitions of variable" in error_msg


def test_error_message_with_previous_definition_location():
    """Test case 2: Error message with 'It was first defined at'

    This tests the scenario where it != end(), so the error message should contain
    both 'TIR is ill-formed, due to multiple definitions of variable' and should also
    contain 'It was first defined at' with the location information.
    """

    x = T.dynamic("x", "int32")

    @T.function(check_well_formed=False)
    def func():
        T.bind(42, var=x)
        T.bind(99, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        tvm_tirx.tirx.analysis.verify_well_formed(func, assert_mode=True)

    error_msg = str(exc_info.value)

    assert "TIR is ill-formed" in error_msg
    assert "multiple nested definitions of variable" in error_msg

    # should contains location information since it != end()
    assert "It was first defined at" in error_msg
    assert "was re-defined at" in error_msg


def test_sequential_redefinition_with_location():
    """Test case 2b: Sequential redefinition that includes location info

    This tests the previously_defined_ path where it != end().
    With flat Bind semantics, sequential redefinitions in the same SeqStmt
    are treated as nested definitions with location info.
    """

    x = T.dynamic("x", "int32")

    @T.function(check_well_formed=False)
    def func():
        T.bind(1, var=x)
        T.evaluate(x)

        T.bind(2, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        tvm_tirx.tirx.analysis.verify_well_formed(func, assert_mode=True)

    error_msg = str(exc_info.value)

    assert "TIR is ill-formed" in error_msg
    assert "multiple nested definitions of variable" in error_msg
    assert "It was first defined at" in error_msg
    assert "was re-defined at" in error_msg


def test_tensor_param_is_well_formed():
    """TensorType-annotated parameters are in scope for the body."""

    @T.function
    def func(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        for i in T.grid(128):
            B[i] = A[i] * 2.0

    tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_decl_tensor_is_well_formed():
    """A DeclTensor statement introduces a tensor into scope for its body."""

    @T.function
    def func(A: T.Tensor((128,), "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        for i in T.grid(128):
            B[i] = A[i] * 2.0

    tvm_tirx.tirx.analysis.verify_well_formed(func)


def test_alloc_tensor_is_well_formed():
    """Allocation introduces a tensor into the function scope."""

    @I.ir_module
    class mod:
        @T.function
        def func(A: T.Tensor((128,), "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor([128], "float32")])
            for i in T.grid(128):
                B[i] = A[i] * 2.0

    tvm_tirx.tirx.analysis.verify_well_formed(mod)


def test_tensor_load_asserted_type_matches_source_and_indices():
    @T.function
    def func():
        tensor = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
        T.evaluate(tensor[0])

    serialized = tvm_tirx.ir.save_json(func)
    round_tripped = tvm_tirx.ir.load_json(serialized)
    assert tvm_tirx.tirx.analysis.verify_well_formed(round_tripped, assert_mode=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(tvm_tirx.tirx.analysis.verify_well_formed, func, False).result()

    graph = json.loads(serialized)
    load = next(node for node in graph["nodes"] if node["type"] == "ir.TensorLoad")
    int32_type_index = next(
        index
        for index, node in enumerate(graph["nodes"])
        if node["type"] == "ir.PrimType" and node["data"]["dtype"] == "int32"
    )
    load["data"]["ty"] = int32_type_index
    malformed = tvm_tirx.ir.load_json(json.dumps(graph))

    assert not tvm_tirx.tirx.analysis.verify_well_formed(malformed, assert_mode=False)
    with pytest.raises(RuntimeError, match="asserts result type"):
        tvm_tirx.tirx.analysis.verify_well_formed(malformed)


def test_tensor_load_malformed_indices_return_false_without_asserting():
    tensor = tvm_tirx.tirx.decl_tensor((4, 4), "float32")
    vector_index = tvm_tirx.ir.prim.vector_expr.Ramp(0, 1, 4)
    load = tvm_tirx.ir.TensorLoad(tensor, [0, vector_index])
    func = tvm_tirx.tirx.Function([tensor], tvm_tirx.ir.Evaluate(load))

    graph = json.loads(tvm_tirx.ir.save_json(func))
    load_node = next(node for node in graph["nodes"] if node["type"] == "ir.TensorLoad")
    indices = graph["nodes"][load_node["data"]["indices"]]["data"]

    rank_mismatch_graph = json.loads(json.dumps(graph))
    rank_mismatch_indices = rank_mismatch_graph["nodes"][load_node["data"]["indices"]]["data"]
    rank_mismatch_indices.pop()
    rank_mismatch = tvm_tirx.ir.load_json(json.dumps(rank_mismatch_graph))
    assert not tvm_tirx.tirx.analysis.verify_well_formed(rank_mismatch, assert_mode=False)
    with pytest.raises(RuntimeError, match="indexes 2-dimensional tensor"):
        tvm_tirx.tirx.analysis.verify_well_formed(rank_mismatch)

    indices[0], indices[1] = indices[1], indices[0]
    non_final_vector = tvm_tirx.ir.load_json(json.dumps(graph))
    assert not tvm_tirx.tirx.analysis.verify_well_formed(non_final_vector, assert_mode=False)
    with pytest.raises(RuntimeError, match="only the final index"):
        tvm_tirx.tirx.analysis.verify_well_formed(non_final_vector)


def test_verify_unique_definitions():
    x = tvm_tirx.tirx.Var("x", "int32")
    y = tvm_tirx.tirx.Var("tindex", "int32")
    z = tvm_tirx.ir.Evaluate(x + y)
    assert tvm_tirx.tirx.analysis.verify_well_formed(tvm_tirx.tirx.Function([x, y], z))

    assert not tvm_tirx.tirx.analysis.verify_well_formed(
        tvm_tirx.tirx.Function([x, y], tvm_tirx.ir.SeqStmt([tvm_tirx.ir.Bind(x, 1), z])),
        assert_mode=False,
    )


def test_verify_repeated_let_definitions():
    x = tvm_tirx.tirx.Var("x", "int32")
    z1 = tvm_tirx.tirx.Let(x, 1, x + 1)
    z2 = tvm_tirx.tirx.Let(x, 2, x + 2)

    # Each occurrence must define a distinct variable, even for a shared Let.
    assert not tvm_tirx.tirx.analysis.verify_well_formed(
        tvm_tirx.tirx.Function([], tvm_tirx.ir.Evaluate(z1 + z1)), assert_mode=False
    )
    assert not tvm_tirx.tirx.analysis.verify_well_formed(
        tvm_tirx.tirx.Function([], tvm_tirx.ir.Evaluate(z1 * z2)), assert_mode=False
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
