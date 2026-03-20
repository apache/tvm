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
# ruff: noqa: F841

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def test_pass_simple():
    @T.prim_func
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        B = T.sblock_alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.sblock("C"):
                # It's a opaque block , so it can use outside variables
                C[i, j] = B[i, j] * 2.0

    assert tvm.tirx.analysis.verify_well_formed(element_wise)
    assert tvm.tirx.analysis.verify_well_formed(tvm.IRModule.from_expr(element_wise))


def test_fail_use_out_loop_var():
    @T.prim_func(check_well_formed=False)
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                # we cannot use `i` since it's defined outside the block
                B[vi, vj] = A[i, vj] * 2.0

    assert not tvm.tirx.analysis.verify_well_formed(element_wise, assert_mode=False)


def test_error_for_out_of_scope_usage():
    """A variable may not be used after its scope ends.

    With flat Bind semantics, Bind vars are visible to all subsequent
    siblings in the same SeqStmt. True out-of-scope usage occurs when
    the Bind is inside a child scope (e.g., ForNode body) and the
    variable is used outside that scope.
    """
    i = tvm.tirx.Var("i", "int32")
    # Bind i inside a For loop body
    for_stmt = tvm.tirx.For(
        tvm.tirx.Var("j", "int32"),
        0,
        1,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.SeqStmt([tvm.tirx.Bind(i, 42), tvm.tirx.Evaluate(i)]),
    )
    # Use i outside the For loop — this is out of scope
    body = tvm.tirx.SeqStmt([for_stmt, tvm.tirx.Evaluate(i)])
    func = tvm.tirx.PrimFunc([], body)

    with pytest.raises(
        ValueError, match="Invalid use of undefined variable i at .* no longer in-scope."
    ):
        tvm.tirx.analysis.verify_well_formed(func)


def test_error_for_nested_rebind_usage():
    """A variable may not be re-defined within the initial scope"""

    @T.prim_func(check_well_formed=False)
    def func():
        i = T.int32()
        T.bind(42, var=i)
        T.bind(42, var=i)
        T.evaluate(i)

    with pytest.raises(
        ValueError, match="ill-formed, due to multiple nested definitions of variable i"
    ):
        tvm.tirx.analysis.verify_well_formed(func)


def test_error_for_repeated_binding():
    """A variable may not be re-defined in the same flat scope.

    With flat Bind semantics, sequential Bind of the same variable in the
    same SeqStmt is treated as a nested redefinition (since the first Bind's
    scope extends to all subsequent siblings).
    """

    @T.prim_func(check_well_formed=False)
    def func():
        i = T.int32()
        T.bind(42, var=i)
        T.evaluate(i)
        T.bind(17, var=i)
        T.evaluate(i)

    with pytest.raises(ValueError, match="multiple nested definitions of variable i"):
        tvm.tirx.analysis.verify_well_formed(func)


def test_error_for_cross_function_reuse():
    """A variable may not be re-defined in another function"""

    i = tvm.tirx.Var("i", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.prim_func
        def func1():
            T.bind(42, var=i)
            T.evaluate(i)

        @T.prim_func
        def func2():
            T.bind(42, var=i)
            T.evaluate(i)

    with pytest.raises(ValueError, match="multiple definitions of variable i"):
        tvm.tirx.analysis.verify_well_formed(mod)


def test_reuse_of_env_thread_in_function_is_well_formed():
    """An env thread may be reused within a PrimFunc

    The `T.env_thread` has unique semantics, and may be defined at
    multiple locations without the TIR being considered ill-formed.
    """

    @T.prim_func
    def func(A: T.Buffer([256], "float32")):
        threadIdx_x = T.env_thread("threadIdx.x")
        with T.launch_thread(threadIdx_x, 256):
            A[threadIdx_x] = A[threadIdx_x] + 1.0

        with T.launch_thread(threadIdx_x, 256):
            A[threadIdx_x] = A[threadIdx_x] + 2.0

    tvm.tirx.analysis.verify_well_formed(func)


def test_reuse_of_env_thread_in_function_is_mandatory():
    """An env thread may be reused within a PrimFunc

    Not only are environment threads allowed to have multiple
    definition sites, it is mandatory for them to have multiple
    definition sites.  If a PrimFunc contains more than one
    `"thread_extent"` with the same name, but with different `tirx.Var`
    instances, it is ill-formed.
    """

    @T.prim_func
    def func(A: T.Buffer([256], "float32")):
        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 1.0

        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 2.0

    tvm.tirx.analysis.verify_well_formed(func)


def test_reuse_of_env_thread_across_functions_is_ill_formed():
    """An env thread may not be reused across PrimFunc

    However, each function must have its own `tirx.Var` representing
    the environment thread, and may not share these variables across
    PrimFuncs.
    """

    threadIdx_x = tvm.tirx.Var("threadIdx_x", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    with pytest.raises(ValueError, match="multiple definitions of variable threadIdx_x"):
        tvm.tirx.analysis.verify_well_formed(mod)


def test_multiple_buffer_arguments_may_share_allocation():
    """T.match_buffer may re-use a data argument

    Like the shape/strides/elem_offset fields in a buffer, the first
    occurrence of a `buffer->data` field defines it, and the
    occurrences are usages of that definition.
    """

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A_handle: T.handle, B_handle: T.handle):
            A = T.match_buffer(A_handle, [256], "float32")
            B = T.match_buffer(B_handle, [256], "float32", data=A.data)

            pass

    tvm.tirx.analysis.verify_well_formed(mod)


def test_block_match_buffer_defines_buffer_obj():
    """In a block, T.match_buffer defines a buffer view"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "float32")):
            for iters in T.grid(16, 16, 16, 16):
                with T.sblock("compute"):
                    tile_i, tile_j, i, j = T.axis.remap("SSSS", iters)
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                    )
                    B[i, j] = 0.0

    tvm.tirx.analysis.verify_well_formed(mod)


def test_block_match_buffer_defines_symbolic_variables():
    """In a block, T.match_buffer may define symbolic variables"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "int32")):
            for iters in T.grid(16, 16, 16, 16):
                with T.sblock("compute"):
                    tile_i, tile_j, i, j = T.axis.remap("SSSS", iters)

                    elem_offset = T.int32()
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                        elem_offset=elem_offset,
                    )

                    B[i, j] = elem_offset

    tvm.tirx.analysis.verify_well_formed(mod)


def test_error_message_without_previous_definition_location():
    """Test case 1: Error message without 'It was first defined at'

    This tests the scenario where it == end(), so the error message should contain
    'TIR is ill-formed, due to multiple definitions of variable' but should NOT
    contain 'It was first defined at' since the iterator is invalid.

    With flat Bind semantics, sequential redefinitions in the same SeqStmt
    are treated as nested definitions, and the first definition location
    IS known, so the message includes location info.
    """

    @T.prim_func(check_well_formed=False)
    def func():
        x = T.int32()

        T.bind(42, var=x)
        T.evaluate(x)

        T.bind(99, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises(ValueError) as exc_info:
        tvm.tirx.analysis.verify_well_formed(func, assert_mode=True)

    error_msg = str(exc_info.value)

    assert "TIR is ill-formed" in error_msg
    assert "multiple nested definitions of variable" in error_msg


def test_error_message_with_previous_definition_location():
    """Test case 2: Error message with 'It was first defined at'

    This tests the scenario where it != end(), so the error message should contain
    both 'TIR is ill-formed, due to multiple definitions of variable' and should also
    contain 'It was first defined at' with the location information.
    """

    @T.prim_func(check_well_formed=False)
    def func():
        x = T.int32()

        T.bind(42, var=x)
        T.bind(99, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises(ValueError) as exc_info:
        tvm.tirx.analysis.verify_well_formed(func, assert_mode=True)

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

    @T.prim_func(check_well_formed=False)
    def func():
        x = T.int32()

        T.bind(1, var=x)
        T.evaluate(x)

        T.bind(2, var=x)  # This should trigger the error
        T.evaluate(x)

    with pytest.raises(ValueError) as exc_info:
        tvm.tirx.analysis.verify_well_formed(func, assert_mode=True)

    error_msg = str(exc_info.value)

    assert "TIR is ill-formed" in error_msg
    assert "multiple nested definitions of variable" in error_msg
    assert "It was first defined at" in error_msg
    assert "was re-defined at" in error_msg


def test_buffer_in_buffer_map_is_well_formed():
    """Buffers defined via function parameter buffer_map are in scope for the body."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.grid(128):
            B[i] = A[i] * 2.0

    tvm.tirx.analysis.verify_well_formed(func)


def test_decl_buffer_is_well_formed():
    """A DeclBuffer statement introduces a buffer into scope for its body."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32")):
        B = T.alloc_buffer((128,), "float32")
        for i in T.grid(128):
            B[i] = A[i] * 2.0

    tvm.tirx.analysis.verify_well_formed(func)


def test_alloc_buffer_in_block_is_well_formed():
    """SBlock::alloc_buffers introduces a buffer into scope for the block body."""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer((128,), "float32")):
            with T.sblock("root"):
                B = T.sblock_alloc_buffer([128], "float32")
                for i in T.grid(128):
                    with T.sblock("write_B"):
                        vi = T.axis.remap("S", [i])
                        B[vi] = A[vi] * 2.0

    tvm.tirx.analysis.verify_well_formed(mod)


def test_match_buffer_in_block_is_well_formed():
    """SBlock::match_buffers introduces a buffer into scope for the block body."""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer((128, 128), "float32")):
            for iters in T.grid(8, 8, 16, 16):
                with T.sblock("compute"):
                    ti, tj, i, j = T.axis.remap("SSSS", iters)
                    A_tile = T.match_buffer(
                        A[ti * 16 : (ti + 1) * 16, tj * 16 : (tj + 1) * 16],
                        dtype="float32",
                    )
                    A_tile[i, j] = A_tile[i, j] * 2.0

    tvm.tirx.analysis.verify_well_formed(mod)


def test_error_undeclared_buffer_in_schedulable_tir():
    """In schedule-level TIR (with SBlock nodes), all buffers must be declared."""
    # Manually construct a BufferStore that uses a buffer without any declaration
    # inside a block context.
    n = tvm.tirx.SizeVar("n", "int32")
    A = tvm.tirx.decl_buffer([n], "float32", name="A")
    i = tvm.tirx.Var("i", "int32")

    # Create an undeclared buffer using an explicit data pointer that is NOT
    # in the buffer_map and NOT wrapped with DeclBuffer.
    B_data = tvm.tirx.Var("B_data", tvm.ir.PointerType(tvm.ir.PrimType("float32")))
    B = tvm.tirx.decl_buffer([n], "float32", name="B", data=B_data)

    # Build a block that writes to B without any declaration of B.
    bi = tvm.tirx.SizeVar("bi", "int32")
    block = tvm.tirx.SBlock(
        iter_vars=[tvm.tirx.IterVar(tvm.ir.Range(0, n), bi, 0)],  # 0 = kDataPar
        reads=[tvm.tirx.BufferRegion(A, [tvm.ir.Range(bi, bi + 1)])],
        writes=[tvm.tirx.BufferRegion(B, [tvm.ir.Range(bi, bi + 1)])],
        body=tvm.tirx.BufferStore(B, tvm.tirx.BufferLoad(A, [bi]), [bi]),
        name_hint="write_B",
    )
    block_realize = tvm.tirx.SBlockRealize(
        iter_values=[i],
        predicate=tvm.tirx.const(True),
        block=block,
    )

    prim_func = tvm.tirx.PrimFunc(
        params=[A.data, B_data],
        body=tvm.tirx.For(i, 0, n, tvm.tirx.ForKind.SERIAL, block_realize),
        buffer_map={A.data: A},
        # Note: B is NOT in buffer_map, so its declaration scope is only
        # within a DeclBuffer node (which we intentionally omit here).
    )

    # B is used in the block but was never declared — should fail.
    with pytest.raises(ValueError, match="buffer B.*without a prior DeclBuffer"):
        tvm.tirx.analysis.verify_well_formed(prim_func)


if __name__ == "__main__":
    tvm.testing.main()
