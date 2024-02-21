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

from typing import List, Set, Union

import tvm
import tvm.testing
from tvm import tir
from tvm import relax as rx
from tvm.relax.analysis import (
    has_reshape_pattern,
    udchain,
    remove_all_unused,
    name_to_binding,
    all_vars,
    all_global_vars,
    free_vars,
    bound_vars,
)
from tvm.script import relax as R, tir as T


def var_name_set(vars: List[Union[rx.Var, rx.GlobalVar]]) -> Set[str]:
    return set(map(lambda v: v.name_hint, vars))


def test_use_def():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = rx.Var("x", R.Tensor([m, n], "float16"))
    y = rx.Var("y", R.Tensor([n], "float16"))
    ib = rx.BlockBuilder()
    with ib.function("func", [x, y]):
        with ib.dataflow():
            lv0 = ib.emit(rx.op.add(x, y))
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            gv0 = ib.emit_output(lv1)
        ib.emit_func_output(gv0)
    dfb = ib.get()["func"].body.blocks[0]
    udc = udchain(dfb)
    assert set(udc[x]) == {lv0}
    assert set(udc[y]) == {lv0, lv1}
    assert set(udc[lv0]) == {lv1}
    assert set(udc[lv1]) == {gv0}
    assert set(udc[gv0]) == set()


def test_chained_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_dps_packed("my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32"))
                unused1 = R.call_dps_packed(
                    "my_dps_func", (unused0,), R.Tensor((32, 32), dtype="float32")
                )
                R.output(lv0)
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_all_unused():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Since
    bindings within a dataflow block are guaranteed not to have side
    effects, they may be removed if unused.
    """

    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_dps_packed("my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32"))
                unused1 = R.call_dps_packed(
                    "my_dps_func", (unused0,), R.Tensor((32, 32), dtype="float32")
                )
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return z

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_remove_unused_pure_without_dataflow():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Unused
    bindings whose value is a pure operation
    (e.g. `R.call_dps_packed`) may be removed, even if outside of a
    dataflow block.
    """

    @R.function(private=True)
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_dps_packed("my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32"))
        unused1 = R.call_dps_packed("my_dps_func", (unused0,), R.Tensor((32, 32), dtype="float32"))
        return x

    @R.function(private=True)
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        return x

    after = remove_all_unused(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_binding_block_keep_impure_without_dataflow():
    """Remove unused dataflow bindings

    Removal of unused bindings may not remove side effects.  Unused
    bindings whose value is an impure operation (e.g. `R.call_packed`)
    may not be removed, as outside of a dataflow block they may
    contain side effects.
    """

    @R.function(private=True)
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        y = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
        return y

    expected = before

    after = remove_all_unused(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_binding_block_keep_pure_func_used_only_for_impure():
    """Keep bindings that are used for impure functions

    Removal of unused bindings may not result in use of undefined
    variables.  Unused bindings whose value is an impure operation
    (e.g. `R.call_packed`) may not be removed, nor may any of their
    inputs.

    This is a regression test to catch an earlier failure mode, in
    which tracking of unused variables only back-propagated from the
    return value of functions, and did not consider variables that
    were required to execute impure function calls.  In that failure
    mode, the binding of `y` would be removed as unused, even though
    it was required to evaluate the packed function.
    """

    @R.function(private=True)
    def before(x: R.Tensor((32, 32), "int32")):
        y = x * R.const(2)
        z = R.call_packed(
            "function_maybe_with_side_effects", y, sinfo_args=(R.Tensor((32, 32), "int32"))
        )
        return R.tuple()

    expected = before

    after = remove_all_unused(before)
    tvm.ir.assert_structural_equal(expected, after)


def test_binding_block_remove_all_unused_func_without_dataflow():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            lv0 = x

            @R.function
            def internal_unused_func(A: R.Tensor((32, 32), "float32")) -> R.Tensor:
                return A

            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return z

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            lv0 = x
            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return z

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return lv0

    optimized = remove_all_unused(IdentityUnused["main"])

    @tvm.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            # This might bring side effect so cannot be removed.
            z = R.call_packed("vm.builtin.copy", lv0, sinfo_args=(R.Tensor((32, 32), "float32")))
            return lv0

    tvm.ir.assert_structural_equal(optimized, GroundTruth["main"])


def test_edge_binding_block_fake_unused_remove_all_unused():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            z = R.call_packed("vm.builtin.copy", x, sinfo_args=(R.Tensor((32, 32), "float32")))
            return x

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_edge_binding_block_fake_unused_remove_all_unused2():
    @tvm.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor(dtype="int32", ndim=3):
            m = T.int64()
            n = T.int64()
            k = T.int64()
            with R.dataflow():
                lv: R.Shape(ndim=3) = R.call_pure_packed(
                    "vm.builtin.tensor_to_shape", x, sinfo_args=(R.Shape(ndim=3),)
                )
                lv1: R.Shape([m, n, k]) = R.match_cast(lv, R.Shape([m, n, k]))
                gv: R.Tensor((m, n, k), dtype="int32") = R.full(
                    R.shape([m, n, k]), R.const(1, "int32"), dtype="int32"
                )
                R.output(gv)
            return gv

    optimized = remove_all_unused(IdentityUnused["main"])
    tvm.ir.assert_structural_equal(optimized, IdentityUnused["main"])


def test_remove_all_unused_from_dataflow_block():
    """Like test_chained_remove_all_unused, but on a SeqExpr"""

    @R.function
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            unused0 = R.call_dps_packed("my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32"))
            unused1 = R.call_dps_packed(
                "my_dps_func", (unused0,), R.Tensor((32, 32), dtype="float32")
            )
            R.output(lv0)
        return lv0

    @R.function
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0

    after = remove_all_unused(before.body)
    tvm.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_remove_all_unused_from_binding_block():
    """Like test_chained_remove_all_unused, but on a SeqExpr"""

    @R.function
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_dps_packed("my_sigmoid", (x,), R.Tensor((32, 32), dtype="float32"))
        unused1 = R.call_dps_packed("my_dps_func", (unused0,), R.Tensor((32, 32), dtype="float32"))
        return lv0

    @R.function
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        return lv0

    after = remove_all_unused(before.body)
    tvm.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_retain_impure_calls_unused_in_binding_block():
    """An impure call may have side effects, and must be kept"""

    @R.function
    def before(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_packed("my_impure_call", x, sinfo_args=R.Tensor((32, 32), dtype="float32"))
        unused1 = R.call_dps_packed("my_unused_call", (lv0,), R.Tensor((32, 32), dtype="float32"))
        return lv0

    @R.function
    def expected(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        lv0 = x
        unused0 = R.call_packed("my_impure_call", x, sinfo_args=R.Tensor((32, 32), dtype="float32"))
        return lv0

    after = remove_all_unused(before.body)
    tvm.ir.assert_structural_equal(expected.body, after, map_free_vars=True)


def test_name_to_binding_var_shadowing():
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            lv1 = lv0
            R.output(lv1)

        with R.dataflow():
            lv0 = lv1  # shadowing
            lv2 = lv0
            R.output(lv2)
        return lv2

    n2binding = name_to_binding(main)

    assert "lv0" in n2binding
    assert "lv1" in n2binding
    assert "lv2" in n2binding

    assert len(n2binding["lv0"]) == 2


@tvm.script.ir_module
class VarExample:
    @R.function
    def func(a: R.Tensor) -> R.Tensor:
        # normalized into assigning R.add(a, a) to a var and returning it
        return R.add(a, a)

    @R.function
    def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
        cls = VarExample
        z = R.add(x, y)
        # no binding here
        _ = R.match_cast(x, R.Tensor((5, 5)))
        with R.dataflow():
            q = R.add(z, z)
            p = cls.func(q)
            r = R.match_cast(p, R.Tensor((5, 5)))
            s = r
            R.output(s)
        return s


def test_all_vars():
    vars = all_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name_hint == "a"
    # the body of the seq expr in the func body is a var
    assert vars[1] == VarExample["func"].body.body

    var_names = var_name_set(all_vars(VarExample["main"]))
    assert var_names == {"_", "x", "y", "z", "p", "q", "r", "s"}


def test_all_vars_from_expr_using_dataflow():
    """all_vars() should return all Var, including DataflowVar"""
    func = VarExample["main"]
    cls_func_q = func.body.blocks[1].bindings[1].value

    var_names = var_name_set(all_vars(cls_func_q))
    assert var_names == {"q"}


def test_bound_vars():
    vars = bound_vars(VarExample["func"])
    assert len(vars) == 2
    assert vars[0].name_hint == "a"
    # the body of the seq expr in the func body is a bound var
    assert vars[1] == VarExample["func"].body.body

    # all the vars are bound
    var_names = var_name_set(bound_vars(VarExample["main"]))
    assert var_names == {"_", "x", "y", "z", "p", "q", "r", "s"}

    # if we consider only the body, then the function arguments are not bound
    body_names = var_name_set(bound_vars(VarExample["main"].body))
    assert body_names == {"_", "z", "p", "q", "r", "s"}

    # only binding is in the (normalized) body
    simple_body_vars = bound_vars(VarExample["func"].body)
    assert len(simple_body_vars) == 1
    assert simple_body_vars[0] == VarExample["func"].body.body


def test_free_vars():
    # all the vars are bound
    assert len(free_vars(VarExample["func"])) == 0
    assert len(free_vars(VarExample["main"])) == 0

    # the arguments are free if we look only at the bodies
    func_free = var_name_set(free_vars(VarExample["func"].body))
    main_free = var_name_set(free_vars(VarExample["main"].body))
    assert len(func_free) == 1
    assert len(main_free) == 2
    assert "a" in func_free
    assert main_free == {"x", "y"}

    # function that captures vars
    x = rx.Var("x", R.Tensor(ndim=-1))
    y = rx.Var("y", R.Tensor(ndim=-1))
    z = rx.Var("z", R.Tensor(ndim=-1))
    inner = rx.Function(
        [z],
        rx.op.add(x, rx.op.add(y, z)),
        ret_struct_info=R.Tensor(ndim=-1),
    )
    outer = rx.Function(
        [x, y],
        rx.Call(inner, [y]),
        ret_struct_info=R.Tensor(ndim=-1),
    )
    assert len(free_vars(outer)) == 0
    assert var_name_set(free_vars(inner)) == {"x", "y"}


def test_all_global_vars():
    # there is one call to "func"
    global_vars = all_global_vars(VarExample["main"])
    assert len(global_vars) == 1
    assert global_vars[0].name_hint == "func"

    gv1 = rx.GlobalVar("gv1")
    gv2 = rx.GlobalVar("gv2")
    gv3 = rx.GlobalVar("gv3")
    call = rx.Call(gv1, [gv2, gv3])
    call_var_names = var_name_set(all_global_vars(call))
    assert call_var_names == {"gv1", "gv2", "gv3"}


def test_reshape_pattern_reshape():
    @T.prim_func
    def reshape(
        rxplaceholder: T.Buffer((1, 2, 3, 4), "float32"),
        T_reshape: T.Buffer((8, 3), "float32"),
    ):
        for i0, i1 in T.grid(8, 3):
            with T.block("T_reshape"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(
                    rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]
                )
                T.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = rxplaceholder[
                    (ax0 * 3 + ax1) // 24,
                    (ax0 * 3 + ax1) % 24 // 12,
                    (ax0 * 3 + ax1) % 12 // 4,
                    (ax0 * 3 + ax1) % 4,
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_reshape_scheduled():
    @T.prim_func
    def reshape_scheduled(
        rxplaceholder: T.Buffer((1, 2, 3, 4), "float32"),
        T_reshape: T.Buffer((8, 3), "float32"),
    ):
        for i0_i1_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(24, thread="threadIdx.x"):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(8, (i0_i1_fused_0 * 24 + i0_i1_fused_1) // 3)
                    ax1 = T.axis.spatial(3, (i0_i1_fused_0 * 24 + i0_i1_fused_1) % 3)
                    T.reads(
                        rxplaceholder[
                            (ax0 * 3 + ax1) // 24,
                            (ax0 * 3 + ax1) % 24 // 12,
                            (ax0 * 3 + ax1) % 12 // 4,
                            (ax0 * 3 + ax1) % 4,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]

    assert has_reshape_pattern(reshape_scheduled)


def test_reshape_pattern_expand_dims():
    @T.prim_func
    def expand_dims(
        rxplaceholder: T.Buffer((2, 3, 4), "float32"),
        expand_dims: T.Buffer((2, 1, 1, 1, 3, 1, 4, 1), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(2, 1, 1, 1, 3, 1, 4, 1):
            with T.block("expand_dims"):
                i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = T.axis.remap(
                    "SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7]
                )
                T.reads(rxplaceholder[i0_1, i4_1, i6_1])
                T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[
                    i0_1, i4_1, i6_1
                ]

    assert has_reshape_pattern(expand_dims)


def test_reshape_pattern_dyn_1():
    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        n = T.int64()
        A = T.match_buffer(var_A, (n, T.int64(32), T.int64(128)), "float16")
        T_reshape = T.match_buffer(
            var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
        )
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(
                    A[
                        ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1) % n,
                        (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                        v_ax3 % T.int64(128),
                    ]
                )
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1) % n,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_2():
    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
        for ax0 in range(n):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(n, ax0)
                T.reads(A[T.int64(0), v_ax0 % n])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_3():
    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (n, T.int64(4096)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_4():
    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
        T_reshape = T.match_buffer(
            var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
        )
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(
                    A[
                        T.int64(0),
                        ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                        (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                    ]
                )
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                    T.int64(0),
                    ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_dyn_5():
    @T.prim_func
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(
                    A[
                        T.int64(0),
                        (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                        v_ax2 % T.int64(4096) // T.int64(128),
                        v_ax2 % T.int64(128),
                    ]
                )
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[
                    T.int64(0),
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    v_ax2 % T.int64(4096) // T.int64(128),
                    v_ax2 % T.int64(128),
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_with_raggedness():
    @T.prim_func
    def reshape_raggedness(
        A: T.Buffer((100, 768), "float32"),
        src_indptr: T.Buffer((9,), "int32"),
        B: T.Buffer((100, 12, 64), "float32"),
    ):
        for b in T.serial(8):
            with T.block("block0"):
                vb = T.axis.spatial(8, b)
                for i in T.serial(src_indptr[vb + 1] - src_indptr[vb]):
                    for h in T.serial(12):
                        for f in T.serial(64):
                            with T.block("block1"):
                                vi, vh, vf = T.axis.remap("SSS", [i, h, f])
                                B[src_indptr[vb] + vi, vh, vf] = A[
                                    src_indptr[vb] + vi, vh * 64 + vf
                                ]

    assert has_reshape_pattern(reshape_raggedness)


def test_reshape_pattern_reject_seqstmt():
    @T.prim_func
    def identity_bias(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
        C = T.alloc_buffer((128, 128), "float32")
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                C[vi0, vi1] = A[vi0, vi1]
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                B[vi0, vi1] = C[vi0, vi1] + T.float32(1)

    @T.prim_func
    def identity_identity(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
        C = T.alloc_buffer((128, 128), "float32")
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                C[vi0, vi1] = A[vi0, vi1]
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                B[vi0, vi1] = C[vi0, vi1]

    assert not has_reshape_pattern(identity_bias)
    assert not has_reshape_pattern(identity_identity)


def test_reshape_pattern_reject_reduction():
    @T.prim_func
    def reduction(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4,), "float32")):
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SR", [i0, i1])
                with T.init():
                    B[vi0] = T.float32(0)
                B[vi0] = B[vi0] + A[vi0, vi1]

    assert not has_reshape_pattern(reduction)


def test_reshape_pattern_reject_reduction():
    @T.prim_func
    def reduction(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4,), "float32")):
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SR", [i0, i1])
                with T.init():
                    B[vi0] = T.float32(0)
                B[vi0] = B[vi0] + A[vi0, vi1]

    assert not has_reshape_pattern(reduction)


if __name__ == "__main__":
    tvm.testing.main()
