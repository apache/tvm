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

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I, tir as T


def test_pass_simple():
    @T.prim_func
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                # It's a opaque block , so it can use outside variables
                C[i, j] = B[i, j] * 2.0

    assert tvm.tir.analysis.verify_well_formed(element_wise)
    assert tvm.tir.analysis.verify_well_formed(tvm.IRModule.from_expr(element_wise))


def test_fail_use_out_loop_var():
    @T.prim_func(check_well_formed=False)
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                # we cannot use `i` since it's defined outside the block
                B[vi, vj] = A[i, vj] * 2.0

    assert not tvm.tir.analysis.verify_well_formed(element_wise, assert_mode=False)


def test_error_for_out_of_scope_usage():
    """A variable may not be used after its scope ends"""

    @T.prim_func(check_well_formed=False)
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            T.evaluate(i)
        T.evaluate(i)

    with pytest.raises(
        ValueError, match="Invalid use of undefined variable i at .* no longer in-scope."
    ):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_nested_rebind_usage():
    """A variable may not be re-defined within the initial scope"""

    @T.prim_func(check_well_formed=False)
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            with T.LetStmt(42, var=i):
                T.evaluate(i)

    with pytest.raises(
        ValueError, match="ill-formed, due to multiple nested definitions of variable i"
    ):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_repeated_binding():
    """A variable may not be re-defined after the scope ends"""

    @T.prim_func(check_well_formed=False)
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            T.evaluate(i)
        with T.LetStmt(17, var=i):
            T.evaluate(i)

    with pytest.raises(ValueError, match="multiple definitions of variable i"):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_cross_function_reuse():
    """A variable may not be re-defined in another function"""

    i = tvm.tir.Var("i", "int32")

    @I.ir_module(check_well_formed=False)
    class mod:
        @T.prim_func
        def func1():
            with T.LetStmt(42, var=i):
                T.evaluate(i)

        @T.prim_func
        def func2():
            with T.LetStmt(42, var=i):
                T.evaluate(i)

    with pytest.raises(ValueError, match="multiple definitions of variable i"):
        tvm.tir.analysis.verify_well_formed(mod)


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

    tvm.tir.analysis.verify_well_formed(func)


def test_reuse_of_env_thread_in_function_is_mandatory():
    """An env thread may be reused within a PrimFunc

    Not only are environment threads allowed to have multiple
    definition sites, it is mandatory for them to have multiple
    definition sites.  If a PrimFunc contains more than one
    `"thread_extent"` with the same name, but with different `tir.Var`
    instances, it is ill-formed.
    """

    @T.prim_func
    def func(A: T.Buffer([256], "float32")):
        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 1.0

        with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
            A[threadIdx_x] = A[threadIdx_x] + 2.0

    tvm.tir.analysis.verify_well_formed(func)


def test_reuse_of_env_thread_across_functions_is_ill_formed():
    """An env thread may not be reused across PrimFunc

    However, each function must have its own `tir.Var` representing
    the environment thread, and may not share these variables across
    PrimFuncs.
    """

    threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")

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
        tvm.tir.analysis.verify_well_formed(mod)


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

    tvm.tir.analysis.verify_well_formed(mod)


def test_buffer_bind_scope_defines_buffer_obj():
    """The "buffer_bind_scope" attribute defines a buffer view"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "float32")):

            for tile_i, tile_j in T.grid(16, 16):
                B = T.Buffer([16, 16], "float32")
                T.attr(
                    [B, A],
                    "buffer_bind_scope",
                    T.tvm_tuple(
                        tile_i * 16,
                        16,
                        tile_j * 16,
                        16,
                        dtype="handle",
                    ),
                )
                for i, j in T.grid(16, 16):
                    B[i, j] = 0.0

    tvm.tir.analysis.verify_well_formed(mod)


def test_buffer_bind_scope_defines_symbolic_variables():
    """The "buffer_bind_scope" attribute may define symbolic variables"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "int32")):

            for tile_i, tile_j in T.grid(16, 16):
                elem_offset = T.int32()
                B = T.Buffer([16, 16], "int32", elem_offset=elem_offset)
                T.attr(
                    [B, A],
                    "buffer_bind_scope",
                    T.tvm_tuple(
                        tile_i * 16,
                        16,
                        tile_j * 16,
                        16,
                        dtype="handle",
                    ),
                )
                for i, j in T.grid(16, 16):
                    B[i, j] = elem_offset

    tvm.tir.analysis.verify_well_formed(mod)


def test_block_match_buffer_defines_buffer_obj():
    """In a block, T.match_buffer defines a buffer view"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "float32")):
            for iters in T.grid(16, 16, 16, 16):
                with T.block("compute"):
                    tile_i, tile_j, i, j = T.axis.remap("SSSS", iters)
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                    )
                    B[i, j] = 0.0

    tvm.tir.analysis.verify_well_formed(mod)


def test_block_match_buffer_defines_symbolic_variables():
    """In a block, T.match_buffer may define symbolic variables"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer([256, 256], "int32")):

            for iters in T.grid(16, 16, 16, 16):
                with T.block("compute"):
                    tile_i, tile_j, i, j = T.axis.remap("SSSS", iters)

                    elem_offset = T.int32()
                    B = T.match_buffer(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                        elem_offset=elem_offset,
                    )

                    B[i, j] = elem_offset

    tvm.tir.analysis.verify_well_formed(mod)


def test_buffer_realize_on_external_buffer_is_annotation():
    """A T.realize statement on an existing buffer annotates the region used"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func(A: T.Buffer(256, "int32")):
            T.realize(A[0:16], "global")

            for i in range(16):
                A[i] = 1

    tvm.tir.analysis.verify_well_formed(mod)


def test_buffer_realize_is_allocation():
    """A T.realize statement on an fresh buffer allocates the buffer"""

    @I.ir_module
    class mod:
        @T.prim_func
        def func():
            A = T.Buffer(256, "int32")
            T.realize(A[0:16], "global")

            for i in range(16):
                A[i] = 1

    tvm.tir.analysis.verify_well_formed(mod)


if __name__ == "__main__":
    tvm.testing.main()
