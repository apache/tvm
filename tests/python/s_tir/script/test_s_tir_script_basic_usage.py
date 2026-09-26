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
"""S-TIR script basic usage."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import tirx
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm.script import from_source
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.testing import env


@Ts.prim_func
def get_valid_counts(
    data_buf: T.Buffer((1, 2500, 6), "float32"),
    valid_count_buf: T.Buffer((1,), "int32"),
    out_buf: T.Buffer((1, 2500, 6), "float32"),
    out_indices_buf: T.Buffer((1, 2500), "int32"),
    score_threshold: T.float32,
    id_index: T.int32,
    score_index: T.int32,
) -> None:
    with Ts.sblock("init"):
        vi = Ts.axis.S(1, 0)
        valid_count_buf[vi] = T.int32(0)
        for j in range(2500):
            with Ts.sblock("update"):
                vj = Ts.axis.S(2500, j)
                Ts.reads([data_buf[vi, vj, 6]])
                Ts.writes([valid_count_buf[vi], out_indices_buf[vi, vj], out_buf[vi, vj, 6]])
                if (data_buf[vi, vj, score_index] > score_threshold) and (
                    (id_index < 0) or (data_buf[vi, vj, id_index] >= T.float32(0))
                ):
                    for k in T.serial(0, 6):
                        out_buf[vi, valid_count_buf[vi], k] = data_buf[vi, vj, k]
                    out_indices_buf[vi, valid_count_buf[vi]] = vj
                    valid_count_buf[vi] = valid_count_buf[vi] + 1
                if vj >= valid_count_buf[vi]:
                    for k in T.serial(0, 6):
                        out_buf[vi, vj, k] = T.float32(-1)
                    out_indices_buf[vi, vj] = T.int32(-1)


def _check_get_valid_counts_with_numpy(f, dshape, score_threshold, id_index, score_index):
    dtype = "float32"
    ctx = tvm.cpu()
    batch_size, num_anchor, elem_length = dshape
    np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
    np_out1 = np.zeros(shape=(batch_size,), dtype="int32")
    np_out2 = np.zeros(shape=dshape).astype(dtype)
    np_out3 = np.zeros(shape=(batch_size, num_anchor), dtype="int32")
    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor):
            score = np_data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                for k in range(elem_length):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                np_out3[i, inter_idx] = j
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(elem_length):
                    np_out2[i, j, k] = -1.0
                np_out3[i, j] = -1

    in_data = tvm.runtime.tensor(np_data, ctx)
    out1 = tvm.runtime.tensor(np_out1, ctx)
    out2 = tvm.runtime.tensor(np_out2, ctx)
    out3 = tvm.runtime.tensor(np_out3, ctx)
    f(in_data, out1, out2, out3, score_threshold, id_index, score_index)
    tvm.testing.assert_allclose(out1.numpy(), np_out1, rtol=1e-5)
    tvm.testing.assert_allclose(out2.numpy(), np_out2, rtol=1e-5)
    tvm.testing.assert_allclose(out3.numpy(), np_out3, rtol=1e-5)
    print("test get_valid_counts end")


def test_get_valid_counts_script_func():
    device = "llvm"
    # check lowering
    print(get_valid_counts.script())
    mod = tvm.ir.IRModule({"get_valid_counts": get_valid_counts})
    print(mod.script())
    # check building
    f = tvm.compile(mod["get_valid_counts"], target=device)
    _check_get_valid_counts_with_numpy(f, (1, 2500, 6), 0.0, 0, 1)


@Ts.prim_func
def ceildiv_test(A: T.Buffer(16, "int32")):
    for i in range(16):
        A[i] = T.ceildiv(A[i], 4)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_ceildiv():
    f = tvm.compile(ceildiv_test, "llvm")
    a = tvm.runtime.tensor(np.arange(16).astype("int32"))
    f(a)
    ref = (np.arange(16) + 3) // 4
    tvm.testing.assert_allclose(a.numpy(), ref)


def test_tir_func_name():
    @Ts.prim_func
    def matmul(A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert matmul.__name__ == "matmul"
    assert matmul.attrs["global_symbol"] == "matmul"


def test_tir_func_private_attrs():
    @Ts.prim_func(private=True)
    def matmul(A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])) -> None:
        T.func_attr({"attr": "value"})

        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert "global_symbol" not in matmul.attrs


def test_tir_loop_steps():
    N = T.Var("N", "int32")

    @Ts.prim_func(private=True)
    def loop_with_steps(
        A: T.Buffer((N,)), B: T.Buffer((N,)), C: T.Buffer((N,)), tid: T.int32, v: T.int32
    ):
        for i in T.serial(tid, N, step=2):
            C[i] = A[i] + B[i]
        for i in T.unroll(tid, N, step=3):
            C[i] = A[i] + B[i]
        for i in T.vectorized(tid, N, step=4):
            C[i] = A[i] + B[i]
        for i in T.parallel(tid, N, step=5):
            C[i] = A[i] + B[i]
        for i in T.serial(tid, N, step=v):
            C[i] = A[i] + B[i]

    stmts = loop_with_steps.body.seq
    assert stmts[0].step == 2
    assert stmts[1].step == 3
    assert stmts[2].step == 4
    assert stmts[3].step == 5
    assert stmts[4].step.name == "v"


def test_tir_empty_tuple_index():
    @T.inline
    def bar(val):
        T.evaluate(val)

    @Ts.prim_func(private=True)
    def func_with_empty_tuple(A: T.Buffer((), "int32"), B: T.Buffer((), "int32")):
        bar(val=A[()])

    @Ts.prim_func(private=True)
    def expected(A: T.Buffer((), "int32"), B: T.Buffer((), "int32")):
        T.evaluate(A[()])

    tvm.ir.assert_structural_equal(func_with_empty_tuple, expected)


def test_thread_binding_dtype():
    @Ts.prim_func(private=True)
    def func(A: T.Buffer((128, 128)), B: T.Buffer((128, 128))):
        for i in T.thread_binding(T.int64(128), "threadIdx.x"):
            for j in T.thread_binding(128, "threadIdx.y"):
                B[i, j] = A[i, j]

    loop_i = func.body
    loop_j = loop_i.body
    assert loop_i.loop_var.ty.dtype == "int64"
    assert loop_i.thread_binding.var.ty.dtype == "int64"
    assert loop_j.loop_var.ty.dtype == "int32"
    assert loop_j.thread_binding.var.ty.dtype == "int32"


def test_inferred_ty_with_prim_args():
    """A PrimFunc may have inferred Type"""

    @Ts.prim_func
    def func(M: T.int32, N: T.int32) -> T.int32:
        return M * N

    expected = tvm.relax.FuncType(
        [
            tvm.ir.PrimType("int32"),
            tvm.ir.PrimType("int32"),
        ],
        tvm.ir.PrimType("int32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_buffer_args():
    """PrimFunc buffer arguments are inferred as R.Tensor"""

    @Ts.prim_func
    def func(A: T.Buffer([16, 16], "float32"), B: T.Buffer([256], "int32")) -> T.float32:
        return T.float32(42.0)

    expected = tvm.relax.FuncType(
        [
            tvm.relax.TensorType([16, 16], "float32"),
            tvm.relax.TensorType([256], "int32"),
        ],
        tvm.ir.PrimType("float32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_internal_allocation():
    """A pure function may still write to internal allocations.

    Whether a function writes to internal allocations is not a visible
    effect, and does not impact the purity of a function.
    """

    @Ts.prim_func
    def func(A: T.Buffer([16, 16], "float32")) -> T.float32:
        Sum = T.decl_buffer([], "float32")
        Sum[()] = 0.0
        for i, j in T.grid(16, 16):
            Sum[()] = Sum[()] + A[i, j]

        return Sum[()]

    expected = tvm.relax.FuncType(
        [
            tvm.relax.TensorType([16, 16], "float32"),
        ],
        tvm.ir.PrimType("float32"),
        purity=True,
    )
    tvm.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_output_buffer():
    """A pure function may not write to an argument buffer

    If an argument buffer is written to, the function must be impure.
    """

    @Ts.prim_func
    def func(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
        for i in range(16):
            B[i] = A[i]

    expected = tvm.relax.FuncType(
        [
            tvm.relax.TensorType([16], "float32"),
            tvm.relax.TensorType([16], "float32"),
        ],
        tvm.relax.TupleType([]),
        purity=False,
    )
    tvm.ir.assert_structural_equal(func.ty, expected)


def test_reinterpret_nop():
    """Test builtin reinterpret op"""

    @Ts.prim_func
    def func(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with Ts.sblock():
                vi = Ts.axis.remap("S", [i])
                B[vi] = T.reinterpret("float32", A[vi])

    @Ts.prim_func
    def expected(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")) -> None:
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 32):
            with Ts.sblock():
                vi = Ts.axis.remap("S", [i])
                B[vi] = A[vi]

    tvm.ir.assert_structural_equal(func, expected)


def test_launch_thread_i64():
    """Test launching thread with int64"""

    @Ts.prim_func
    def func() -> None:
        blockIdx_x = T.launch_thread("blockIdx.x", T.int64(1))
        if blockIdx_x == T.int64(0):
            T.evaluate(T.int64(0))
        else:
            T.evaluate(T.int64(1))

    assert func.body.node.dom.min.ty.dtype == "int64"
    assert func.body.node.dom.extent.ty.dtype == "int64"


def test_block_annotation_merge():
    def _to_dict(anno: tvm_ffi.container.Map):
        result = {}
        for k, v in anno.items():
            result[k] = _to_dict(v) if isinstance(v, tvm_ffi.container.Map) else v
        return result

    @Ts.prim_func
    def func0():
        with Ts.sblock():
            Ts.sblock_attr({"key1": "block1"})
            Ts.sblock_attr({"key2": "block2"})
            T.evaluate(0)

    assert _to_dict(func0.body.block.annotations) == {"key1": "block1", "key2": "block2"}

    @Ts.prim_func
    def func1():
        with Ts.sblock():
            Ts.sblock_attr({"key": {"key1": "block1"}})
            Ts.sblock_attr({"key": {"key2": "block2"}})
            T.evaluate(0)

    assert _to_dict(func1.body.block.annotations) == {"key": {"key1": "block1", "key2": "block2"}}

    @Ts.prim_func
    def func2():
        with Ts.sblock():
            Ts.sblock_attr({"key1": "block1"})
            Ts.sblock_attr({"key1": "block1"})
            T.evaluate(0)

    assert _to_dict(func2.body.block.annotations) == {"key1": "block1"}

    with pytest.raises(RuntimeError):

        @Ts.prim_func
        def func3():
            with Ts.sblock():
                Ts.sblock_attr({"key1": "block1"})
                Ts.sblock_attr({"key1": "block2"})
                T.evaluate(0)


def test_alloc_inside_block():
    @Ts.prim_func(private=True)
    def func() -> None:
        with Ts.sblock():
            A = Ts.sblock_alloc_buffer([10], "float32")
            for i in T.serial(0, 10):
                B = Ts.sblock_alloc_buffer([10], "float32")
                for j in T.serial(0, 10):
                    B[j] = T.float32(j)
                    A[i] += B[j]

    @Ts.prim_func(private=True)
    def expected() -> None:
        with Ts.sblock():
            A = Ts.sblock_alloc_buffer([10], "float32")
            B = Ts.sblock_alloc_buffer([10], "float32")
            for i, j in T.grid(10, 10):
                B[j] = T.float32(j)
                A[i] += B[j]

    tvm.ir.assert_structural_equal(func, expected)


def test_ifexp():
    @Ts.prim_func(private=True)
    def func(A: T.buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            A[i, j] = i if i < j else j

    @Ts.prim_func(private=True)
    def expected(A: T.buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            A[i, j] = T.if_then_else(i < j, i, j)

    tvm.ir.assert_structural_equal(func, expected)


def test_sequence_compare():
    @Ts.prim_func(private=True)
    def tir_func(A: T.Buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            if 0 < i < 128 and 0 < j < 128:
                A[i, j] = 1
            else:
                A[i, j] = 0

    @Ts.prim_func(private=True)
    def expected(A: T.buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            if (0 < i and i < 128) and (0 < j and j < 128):
                A[i, j] = 1
            else:
                A[i, j] = 0

    tvm.ir.assert_structural_equal(tir_func, expected)


def launch_env_thread():
    @Ts.prim_func
    def main(inputs: T.Buffer((64, 2, 4), "float32")) -> None:
        bx = T.launch_thread("blockIdx.x", 64)
        for i, j in T.grid(2, 4):
            T.evaluate(inputs[bx, i, j])

    return main


def vthread_func():
    @Ts.prim_func
    def vthread_func(A: T.Buffer([256], "float32"), C: T.Buffer([256], "float32")) -> None:
        i0 = T.env_thread("blockIdx.x")
        i1 = T.env_thread("threadIdx.x")
        i2 = T.env_thread("vthread")

        T.launch_thread(i0, 4)
        T.launch_thread(i1, 2)
        T.launch_thread(i2, 2)
        B = T.alloc_buffer((16,), scope="local")
        for j in range(16):
            B[j] = A[i0 * 64 + i1 * 32 + i2 * 16 + j] + T.float32(1)
        for j in range(16):
            C[i0 * 64 + i1 * 32 + i2 * 16 + j] = B[j] * T.float32(2)

    return vthread_func


def for_thread_binding():
    @Ts.prim_func
    def for_thread_binding(
        A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")
    ) -> None:
        for i in T.thread_binding(0, 16, thread="threadIdx.x"):
            for j in T.thread_binding(
                0, 16, thread="threadIdx.y", annotations={"attr_key": "attr_value"}
            ):
                A[i, j] = B[i, j] + T.float32(1)

    return for_thread_binding


def test_for_thread_binding():
    func = for_thread_binding()
    rt_func = tvm.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body, tirx.stmt.For)
    assert rt_func.body.kind == 4
    assert rt_func.body.thread_binding.thread_tag == "threadIdx.x"
    assert isinstance(rt_func.body.body, tirx.stmt.For)
    assert rt_func.body.body.kind == 4
    assert rt_func.body.body.thread_binding.thread_tag == "threadIdx.y"
    assert rt_func.body.body.annotations["attr_key"] == "attr_value"


def while_loop():
    @Ts.prim_func
    def while_loop(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")) -> None:
        i = Ts.sblock_alloc_buffer((), "int32", scope="local")
        for ii in range(16):
            with Ts.sblock():
                vi = Ts.axis.S(16, ii)
                B[vi] = 0
            while i[()] < 10:
                for j in range(16):
                    B[j] += A[j]

    return while_loop


def boolean_argument():
    @Ts.prim_func
    def func(a: T.boolean) -> None:
        T.evaluate(a)

    return func


def bool_argument():
    @Ts.prim_func
    def func(a: T.bool) -> None:
        T.evaluate(a)

    return func


def bool_variable_annotation():
    @Ts.prim_func
    def func() -> None:
        a: T.let[T.bool] = T.call_extern("dummy", dtype="bool")
        T.evaluate(0)

    return func


def return_none():
    @Ts.prim_func
    def func():
        T.evaluate(0)

    return func


def implicit_evaluate():
    @Ts.prim_func
    def func(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    return func


def if_true_else():
    @Ts.prim_func
    def func() -> None:
        if True:
            T.evaluate(0)
        else:
            T.evaluate(1)

    return func


def elif_chain_without_else():
    @Ts.prim_func
    def func(i: T.int32) -> None:
        if i == 0:
            T.evaluate(0)
        elif i == 1:
            T.evaluate(1)
        elif i == 2:
            T.evaluate(2)

    return func


def elif_chain_with_else():
    @Ts.prim_func
    def func(i: T.int32) -> None:
        if i == 0:
            T.evaluate(0)
        elif i == 1:
            T.evaluate(1)
        elif i == 2:
            T.evaluate(2)
        else:
            T.evaluate(3)

    return func


def bind_var():
    @Ts.prim_func
    def func():
        x = T.bind(0)
        y = T.bind(0)
        T.evaluate(0)
        T.evaluate(0)

    return func


def if_then_else_var():
    @Ts.prim_func
    def main(n: T.int32):
        if n == 0:
            x = 5
            T.evaluate(x)
        else:
            x = 10
            T.evaluate(x)

    return main


def ir_module_with_attrs():
    @I.ir_module
    class Module:
        I.module_attrs({"attr": 10})

        @Ts.prim_func
        def tir_func(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
            for i in range(16):
                B[i] = A[i]

    return Module


def subroutine_call():
    """A GlobalVar may reference other functions in the module"""

    @I.ir_module
    class mod:
        @Ts.prim_func
        def main(A: T.Buffer(16, "float32")):
            mod.subroutine(A.data, T.int32(16))

        @Ts.prim_func
        def subroutine(A_data: T.handle("float32"), n: T.int32):
            T.evaluate(0)

    return mod


def subroutine_call_returning_int():
    """An internal function call may return non-void"""

    @I.ir_module
    class mod:
        @Ts.prim_func
        def main(A: T.Buffer(2, "float32")):
            mod.subroutine(A[0]) + mod.subroutine(A[1])

        @Ts.prim_func
        def subroutine(x: T.float32) -> T.float32:
            return x * x

    return mod


def subroutine_call_without_arguments():
    @I.ir_module
    class mod:
        @Ts.prim_func
        def main():
            mod.subroutine()

        @Ts.prim_func
        def subroutine():
            T.evaluate(0)

    return mod


def return_zero():
    @Ts.prim_func
    def func() -> T.int32:
        return 0

    return func


def return_zero_private():
    @Ts.prim_func(private=True)
    def func() -> T.int32:
        return 0

    return func


def return_zero_private_with_attr():
    @Ts.prim_func(private=True)
    def func() -> T.int32:
        T.func_attr({"greeting": "hello"})
        return 0

    return func


def func_with_loop_jumps():
    @Ts.prim_func
    def func(In: T.Buffer((1,), "int32"), Out: T.Buffer((2,), "int32")):
        Out[0] = 0
        Out[1] = 0
        for i in range(1000):
            if i % 13 == 0:
                Out[1] = Out[1] + 1
                continue
            Out[0] = Out[0] + 1
            if Out[0] >= In[0]:
                break

    return func


def func_with_loop_steps():
    @Ts.prim_func
    def func(
        A: T.Buffer((1024,)), B: T.Buffer((1024,)), C: T.Buffer((1024,)), tid: T.int32, v: T.int32
    ):
        for i in T.serial(tid, 1024, step=2):
            C[i] = A[i] + B[i]
        for i in T.unroll(tid, 1024, step=3):
            C[i] = A[i] + B[i]
        for i in T.vectorized(tid, 1024, step=4):
            C[i] = A[i] + B[i]
        for i in T.parallel(tid, 1024, step=5):
            C[i] = A[i] + B[i]
        for i in range(tid, 1024, 6):
            C[i] = A[i] + B[i]

    return func


def test_return_none_no_trailing_type():
    func = return_none()
    script = func.script()
    assert "-> None" not in script


@pytest.mark.parametrize(
    "error_type,message_parts",
    [
        ("RuntimeError", ["x must be positive"]),
        ("ValueError", ["Shape mismatch"]),
        ("TypeError", ["Expected Tensor but got int"]),
        ("TypeError", ["Expected ", "Tensor", " but got ", "int"]),
    ],
    ids=["runtime_error", "value_error", "type_error", "multi_parts"],
)
def test_assert_stmt_roundtrip(error_type, message_parts):
    """Exception types and split messages survive an S-TIR script roundtrip."""

    @Ts.prim_func
    def func(x: T.int32):
        assert x > 0, (error_type, message_parts)

    roundtrip = tvm.script.from_source(
        func.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(func, roundtrip, map_free_vars=True)


@Ts.prim_func
def loop_no_syntax_sugar(A: T.Buffer((128, 128, 128, 128))) -> None:
    for i in T.serial(0, 128):
        for j in T.parallel(0, 128):
            for k in T.vectorized(0, 128):
                for x in T.unroll(0, 128):
                    for y in T.thread_binding(0, 128, thread="threadIdx.x"):
                        for z in T.thread_binding(0, 128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


@Ts.prim_func
def loop_syntax_sugar(A: T.Buffer((128, 128, 128, 128))) -> None:
    for i in T.serial(128):
        for j in T.parallel(128):
            for k in T.vectorized(128):
                for x in T.unroll(128):
                    for y in T.thread_binding(128, "threadIdx.x"):
                        for z in T.thread_binding(128, thread="threadIdx.x"):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


def test_loop_syntax_sugar():
    assert_structural_equal_ignore_global_symbol(loop_no_syntax_sugar, loop_syntax_sugar)


@Ts.prim_func
def elementwise_buffer_default_dtype(
    A: T.Buffer((128, 128, 128, 128)),
    B: T.Buffer((128, 128, 128, 128)),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.prim_func
def elementwise_buffer_kwargs(
    a: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
    b: T.Buffer(shape=(128, 128, 128, 128), dtype="float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


@Ts.prim_func
def elementwise_buffer_no_kwargs(
    a: T.Buffer((128, 128, 128, 128), "float32"),
    b: T.Buffer((128, 128, 128, 128), "float32"),
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):  # noqa: E741
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            b[vi, vj, vk, vl] = a[vi, vj, vk, vl] * 2.0


def test_buffer_signature_syntax_sugar():
    # with kwargs
    assert_structural_equal_ignore_global_symbol(
        elementwise_buffer_default_dtype, elementwise_buffer_kwargs
    )
    # without kwargs
    assert_structural_equal_ignore_global_symbol(
        elementwise_buffer_default_dtype, elementwise_buffer_no_kwargs
    )


def test_buffer_1d():
    @Ts.prim_func
    def func_no_sugar(A: T.Buffer(shape=(16,))):
        for i in T.serial(16):
            A[i] = 0.0

    @Ts.prim_func
    def func_with_sugar(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    assert_structural_equal_ignore_global_symbol(func_no_sugar, func_with_sugar)


def test_bind_bufferload_without_type_annotation():
    # Variable assignment of Expr types uses the dtype of the
    # Expr to determine the variable's dtype.  Parsing of
    # buf[indices] is done by generating a BufferSlice object, which
    # handles both store and load cases.  BufferSlice is not a
    # Expr, and implements BufferSlice.dtype explicitly.

    # Failure occurred during parsing of the tvmscript.
    @Ts.prim_func
    def func_without_type_annotation(A: T.Buffer((1,), "int32")):
        x = A[0]
        T.evaluate(x)


def test_implicit_evaluate_assume():
    @Ts.prim_func
    def explicit(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    @Ts.prim_func
    def implicit(A: T.Buffer(1, "int32")):
        T.assume(A[0] == 5)
        A[0] = 10

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_implicit_evaluate_call_extern():
    @Ts.prim_func
    def explicit(A: T.Buffer(1, "int32")):
        T.evaluate(T.call_extern("extern_func", A.data, dtype="int32"))

    @Ts.prim_func
    def implicit(A: T.Buffer(1, "int32")):
        T.call_extern("extern_func", A.data, dtype="int32")

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_trivial_let_binding():
    """Trivial `T.let[...]` annotations survive the parser as LetStmt and are not inlined.

    In fork, bare `j = i` lowers to a local_scalar (AllocBuffer + BufferStore); the
    LetStmt form is opt-in via `T.let[T.dtype]`. Both the explicit `T.bind(..., var=j)`
    builder API and the `j: T.let[T.dtype]` annotation produce the same LetStmt IR.
    """

    j = T.dynamic("j", "int32")

    @Ts.prim_func
    def explicit(i: T.int32):
        T.bind(i, var=j)
        T.evaluate(j)

    @Ts.prim_func
    def implicit(i: T.int32):
        j: T.let[T.int32] = i
        T.evaluate(j)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_trivial_let_binding_of_value():
    """Same as test_preserve_trivial_let_binding but with a constant RHS."""

    j = T.dynamic("j", "int32")

    @Ts.prim_func
    def explicit(i: T.int32):
        T.bind(42, var=j)
        T.evaluate(j)

    @Ts.prim_func
    def implicit(i: T.int32):
        j: T.let[T.int32] = 42
        T.evaluate(j)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_preserve_parameter_name():
    @Ts.prim_func
    def func(i: T.int32):
        j = i
        T.evaluate(j)

    param_name = func.params[0].name
    assert param_name == "i"


@pytest.mark.parametrize("mutable", [False, True])
def test_preserve_variable_name(mutable):
    """Use variable name when generating tirx::Bind / AllocBuffer"""

    # Bare bindings name the immutable Var; explicit declarations name scalar storage.
    annotation = ": T.int32" if mutable else ""
    func = from_source(
        f"""@Ts.prim_func
def func():
    for i in T.serial(16):
        j{annotation} = i // 4
        T.evaluate(j)
""",
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    )
    binding = func.body.body.seq[0]
    if mutable:
        assert isinstance(binding, tvm.tirx.AllocBuffer)
        var_name = binding.buffer.name
    else:
        assert isinstance(binding, tvm.tirx.Bind)
        var_name = binding.var.name
    assert var_name == "j"


def test_boolean_constant():
    """Python booleans should become T.Bool objects"""

    @Ts.prim_func
    def explicit():
        T.evaluate(T.bool(True))

    @Ts.prim_func
    def implicit():
        T.evaluate(True)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_foldable_boolean_in_assert():
    """Foldable booleans T.Bool objects

    The condition of an assert statement should be a boolean
    expression.  Previously, this test failed because the FFI does not
    distinguish between integer primitives and boolean primitives.
    """

    @Ts.prim_func
    def explicit():
        assert T.bool(False), "Message"
        T.evaluate(0)

    @Ts.prim_func
    def implicit():
        assert 0 == 1, "Message"
        T.evaluate(0)

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_return_statement():
    """A Python `return` statement creates a first-class Return node."""

    @Ts.prim_func
    def explicit():
        T.return_(T.int32(5))

    @Ts.prim_func
    def implicit():
        return 5

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


def test_loop_jump_statement():
    """`break` and `continue` evaluates to TIR intrinsics"""

    @Ts.prim_func
    def explicit():
        for i in range(16):
            if i % 2 == 0:
                T.evaluate(T.continue_loop())
            if i < 15:
                T.evaluate(T.break_loop())

    @Ts.prim_func
    def implicit():
        for i in range(16):
            if i % 2 == 0:
                continue
            if i < 15:
                break

    assert_structural_equal_ignore_global_symbol(implicit, explicit)


@pytest.mark.parametrize(
    "ir_generator",
    [
        launch_env_thread,
        vthread_func,
        while_loop,
        boolean_argument,
        bool_argument,
        bool_variable_annotation,
        return_none,
        implicit_evaluate,
        if_true_else,
        elif_chain_without_else,
        elif_chain_with_else,
        bind_var,
        if_then_else_var,
        ir_module_with_attrs,
        subroutine_call,
        subroutine_call_returning_int,
        subroutine_call_without_arguments,
        return_zero,
        return_zero_private,
        return_zero_private_with_attr,
        func_with_loop_jumps,
        func_with_loop_steps,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_basic_usage(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


# Import-time construction also checks the annotated S-TIR API.
@Ts.prim_func
def element_wise_env_thread_x(
    A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
) -> None:
    j1_0 = T.env_thread("threadIdx.x")
    j0_0 = T.env_thread("threadIdx.x")
    i = T.env_thread("blockIdx.x")

    T.launch_thread(i, 128)
    T.launch_thread(j0_0, 4)
    T.launch_thread(j1_0, 4)

    for blockIdx_x in T.thread_binding(0, 128, "blockIdx.x"):
        for threadIdx_x in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[blockIdx_x, threadIdx_x * 32 + j0_1] = (
                        A[blockIdx_x, threadIdx_x * 32 + j0_1] * 2.0
                    )
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[blockIdx_x, threadIdx_x * 32 + j1_1] = (
                        B[blockIdx_x, threadIdx_x * 32 + j1_1] + 1.0
                    )


if __name__ == "__main__":
    tvm.testing.main()
