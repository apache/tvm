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
from tvm.script import tir as T
from tvm.ir.instrument import pass_instrument


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
    @T.prim_func
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


def test_pass_buffer_usage():
    @T.prim_func
    def func(a: T.handle):
        # Buffer declaration as part of buffer_map
        A = T.match_buffer(a, 128, "float32")

        # Buffer declaration as part of BlockNode
        B = T.alloc_buffer(128, "float32")
        for i in range(128):
            B[i] = A[i] * 2.0

        # Buffer declaration in a DeclBuffer node
        c_data = T.allocate([128], "float32")
        C = T.decl_buffer(128, "float32", data=c_data)
        for i in range(128):
            C[i] = B[i] * 2.0

    assert tvm.tir.analysis.verify_well_formed(func)


def test_fail_implicit_buffer_alias():
    @T.prim_func
    def func(A: T.Buffer([128, 128], "float32")):
        # Aliased buffer usage without declaration.  The `T.Buffer` in
        # TVMScript does not actually make any TIR node, and does not
        # count as a TIR declaration.
        Alias = T.Buffer(128 * 128, "float32", data=A.data)
        T.evaluate(Alias[0])

    assert not tvm.tir.analysis.verify_well_formed(func, assert_mode=False)


def test_pass_explicit_buffer_alias():
    @T.prim_func
    def func(A: T.Buffer([128, 128], "float32")):
        # Aliased buffer usage with declaration.
        Alias = T.decl_buffer(128 * 128, "float32", data=A.data)
        T.evaluate(Alias[0])

    assert tvm.tir.analysis.verify_well_formed(func)


def matmul():
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])

        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    return func


def launch_env_thread():
    @T.prim_func
    def func(inputs: T.Buffer((64, 2, 4), "float32")) -> None:
        bx = T.launch_thread("blockIdx.x", 64)
        for i, j in T.grid(2, 4):
            T.evaluate(inputs[bx, i, j])

    return func


def copy_using_env_thread():
    shape = (64, 2, 4)

    @T.prim_func
    def func(A: T.Buffer(shape), B: T.Buffer(shape)):
        blocks, M, N = T.meta_var(shape)

        bx = T.launch_thread("blockIdx.x", blocks)
        for i, j in T.grid(M, N):
            B[bx, i, j] = A[bx, i, j]

    return func


@pass_instrument
class InstrumentWellFormed:
    def run_after_pass(self, mod, info):
        for func in mod.functions.values():
            tvm.tir.analysis.verify_well_formed(func)


@pytest.mark.parametrize(
    "generator,target",
    [
        (matmul, "llvm"),
        pytest.param(
            launch_env_thread,
            "cuda",
            marks=tvm.testing.Feature("cuda").marks(support_required="compile-only"),
        ),
        pytest.param(
            copy_using_env_thread,
            "cuda",
            marks=tvm.testing.Feature("cuda").marks(support_required="compile-only"),
        ),
    ],
)
def test_well_formed_all_lowering_steps(generator, target):
    func = generator()

    with tvm.transform.PassContext(instruments=[InstrumentWellFormed()]):
        tvm.build(func, target=target)


if __name__ == "__main__":
    tvm.testing.main()
