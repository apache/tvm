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

import tvm.testing
from tvm import relax as rx
from tvm import relay, tir
from tvm.relax.analysis import get_var2val
from tvm.relax.dpl import *
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class Module:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.int32()
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for (i0, j0, k0) in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for (i, j) in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        cls = Module
        with R.dataflow():
            lv0 = R.call_tir(cls.tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_tir(cls.tir_relu, (lv0), R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1


main_fn = Module["main"]
bindings = main_fn.body.blocks[0].bindings

## Node-wise Matching
def test_expr_pattern():
    ep = is_expr(rx.Var("x"))
    assert isinstance(ep, ExprPattern)
    assert isinstance(ep.expr, rx.Var)


def test_var_pattern():
    v = is_var("x")
    assert isinstance(v, VarPattern)
    assert v.name == "x"
    assert v.match(rx.Var("x"))
    assert is_var().match(rx.Var("x"))
    assert is_var().match(rx.DataflowVar("x"))  # DataflowVar is also a Var
    assert not v.match(rx.GlobalVar("x"))


def test_dataflow_var_pattern():
    v = is_dfv("x")
    assert isinstance(v, DataflowVarPattern)
    assert v.name == "x"
    assert v.match(rx.DataflowVar("x"))
    assert not v.match(rx.GlobalVar("x"))
    assert is_dfv().match(bindings[0].var)


def test_global_var_pattern():
    assert is_gv("x").match(rx.GlobalVar("x"))
    assert is_gv().match(rx.GlobalVar("x"))
    assert not is_gv("x").match(rx.GlobalVar("y"))
    assert not is_gv("x").match(rx.Var("x"))


def test_constant_pattern():
    c = is_const()
    assert isinstance(c, ConstantPattern)
    assert c.match(rx.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]]))


def test_wildcard_pattern():
    wc = wildcard()
    assert isinstance(wc, WildcardPattern)
    assert wc.match(rx.Var("x"))


def test_call_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("relax.add")(wc1, wc2)
    assert isinstance(c, CallPattern)
    assert isinstance(c.args[0], WildcardPattern)
    assert isinstance(c.args[1], WildcardPattern)
    assert c.match(rx.op.add(rx.Var("x"), rx.Var("y")))


def test_function_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    f = FunctionPattern([wc1, wc2], is_op("relax.add")(wc1, wc2))
    assert isinstance(f, FunctionPattern)
    assert isinstance(f.params[0], WildcardPattern)
    assert isinstance(f.params[1], WildcardPattern)
    assert isinstance(f.body, CallPattern)
    assert isinstance(f.body.args[0], WildcardPattern)
    assert isinstance(f.body.args[1], WildcardPattern)
    x = rx.Var("x", R.Tensor("float32"))
    y = rx.Var("y", R.Tensor("float32"))
    assert f.match(rx.Function([x, y], rx.op.add(x, y), ret_struct_info=R.Tensor("float32")))
    assert not f.match(
        rx.Function([x, y], rx.op.multiply(x, y), ret_struct_info=R.Tensor("float32"))
    )


def test_tuple_pattern():
    wc1 = wildcard()
    wc2 = is_dfv()
    t = is_tuple([wc1, wc2])
    assert isinstance(t, TuplePattern)
    assert isinstance(t.fields[0], WildcardPattern)
    assert isinstance(t.fields[1], DataflowVarPattern)
    assert t.match(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]))
    assert not t.match(rx.Tuple([rx.DataflowVar("x"), rx.GlobalVar("y")]))
    assert not t.match(rx.Tuple([]))
    assert t[0].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0))
    assert t[1].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    # Negative index is also allowed
    assert t[-1].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    # None means any index.
    assert t[None].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0))
    assert t[None].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    with pytest.raises(IndexError):
        t[2]  # index cannot be greater than or equal to the tuple size.


def test_unordered_tuple_pattern():
    t = is_tuple([is_const(), is_dfv()], unordered=True)
    assert isinstance(t, UnorderedTuplePattern)
    assert isinstance(t.fields[0], ConstantPattern)
    assert isinstance(t.fields[1], DataflowVarPattern)
    assert t.match(rx.Tuple([rx.const([]), rx.DataflowVar("x")]))
    assert t.match(rx.Tuple([rx.DataflowVar("x"), rx.const([])]))
    assert not t.match(rx.Tuple([rx.DataflowVar("x"), rx.DataflowVar("y")]))
    assert not t.match(rx.Tuple([]))


def test_tuple_get_item_pattern():
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )


def test_or_pattern():
    dfv_or_gv = is_dfv("x") | is_gv("x")
    assert isinstance(dfv_or_gv, OrPattern)
    assert dfv_or_gv.match(rx.DataflowVar("x"))
    assert dfv_or_gv.match(rx.GlobalVar("x"))
    assert not dfv_or_gv.match(rx.Var("x"))
    assert not dfv_or_gv.match(rx.DataflowVar("y"))
    assert not dfv_or_gv.match(rx.GlobalVar("y"))


def test_and_pattern():
    # float[2, 3, 3]
    f32_233 = wildcard().has_shape((2, 3, 3)) & has_dtype("float32")
    assert isinstance(f32_233, AndPattern)
    assert f32_233.match(rx.Var("x", R.Tensor((2, 3, 3), "float32")))
    assert not f32_233.match(rx.Var("x", R.Tensor((3, 3, 3), "float32")))
    assert not f32_233.match(rx.Var("x", R.Tensor("float32", ndim=3)))


def test_not_pattern():
    no_shape233 = ~wildcard().has_shape((2, 3, 3))
    assert isinstance(no_shape233, NotPattern)
    assert no_shape233.match(rx.Var("x", R.Tensor((3, 3, 3), "float32")))
    assert not no_shape233.match(rx.Var("x", R.Tensor((2, 3, 3), "float32")))


def test_type_pattern():
    assert wildcard().has_type(rx.DynTensorType(2, "float32")).match(bindings[0].var)


def test_dtype_pattern():
    dtype = "float16"
    pattern = has_dtype(dtype)
    assert isinstance(pattern, DataTypePattern)
    assert pattern.dtype == dtype
    assert has_dtype("float32").match(bindings[0].var)


def test_shape_pattern():
    shape = [32, 32]
    pattern = wildcard().has_shape(shape)
    assert isinstance(pattern, ShapePattern)
    tvm.ir.structural_equal(pattern.shape, shape)
    assert pattern.match(bindings[0].var)
    assert wildcard().has_shape([32, 32]).match(bindings[0].var)
    n, m = tir.Var("n", dtype="int64"), tir.Var("m", dtype="int64")
    symsh_var = rx.Var("x", R.Tensor([n, m, n + m], "float32"))
    assert wildcard().has_shape([n, m, n + m]).match(symsh_var)
    assert wildcard().has_shape([n, m, m + n]).match(symsh_var)  # + is commutative.
    assert not wildcard().has_shape([1, 2, 3]).match(symsh_var)
    assert not wildcard().has_shape([m, n, n + m]).match(symsh_var)


def test_prim_arr_pattern():
    """
    The difference between is_shape and has_shape is that:
    1) is_shape directly matches a shape (e.g., as an argument);
    2) has_shape matches a tensor and puts assumptions on the tensor's shape.
    """
    pattern = is_shape([32, 32])
    assert pattern[0] == 32
    assert pattern[1] == 32
    assert isinstance(pattern, PrimArrPattern)
    assert pattern.match(rx.get_shape_of(bindings[0].var))
    n, m = tir.Var("n", dtype="int64"), tir.Var("m", dtype="int64")
    symbolic_shape = rx.ShapeExpr([n, m, n + m])
    assert is_shape([n, m, n + m]).match(symbolic_shape)
    assert not is_shape([n, m, n * m]).match(symbolic_shape)


def test_extern_fn_pattern():
    pattern = ExternFuncPattern("test.blockbuilder.nop")
    assert pattern.match(rx.ExternFunc("test.blockbuilder.nop"))


def test_op_attr():
    x = rx.Var("x", R.Tensor("float32"))
    y = rx.Var("y", R.Tensor("float32"))
    conv2d = relay.nn.conv2d(x, y, kernel_size=(3, 3))
    xp = is_var("x")
    yp = is_var("y")
    # TODO(@yuchen): reenable the assert after figuring out why it fails
    # assert is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size": [3, 3]}).match(conv2d)
    assert not is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size": [4, 3]}).match(conv2d)
    assert not is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size_": [3, 3]}).match(conv2d)


def test_match_call_attr():
    x = rx.Var("x", R.Tensor("float32"))
    y = rx.Var("y", R.Tensor("float32"))
    fn = rx.Function([x, y], rx.op.add(x, y), ret_struct_info=R.Tensor("float32"))
    annotated_fn = fn.with_attr({"Codegen": "test-codegen", "global_symbol": "test-symbol"})
    xp = is_var("x")
    yp = is_var("y")
    root_pattern = FunctionPattern([xp, yp], is_op("relax.add")(xp, yp))
    assert root_pattern.has_attr({"Codegen": "test-codegen", "global_symbol": "test-symbol"}).match(
        annotated_fn
    )

    assert root_pattern.has_attr({"Codegen": "test-codegen"}).match(annotated_fn)
    assert not root_pattern.has_attr({"ping": "pong"}).match(annotated_fn)
    assert root_pattern.has_attr({}).match(annotated_fn)


def test_is_call_tir():
    lv1_val = bindings[1].value
    var2val = get_var2val(Module["main"])
    assert is_call_tir("tir_relu").match(lv1_val)
    assert is_call_tir("tir_relu", [is_call_tir("tir_matmul")]).match(lv1_val, var2val=var2val)
    assert not is_call_tir("tir_relu", [is_call_tir("tir_relu")]).match(lv1_val, var2val=var2val)


@R.function
def simple_call_packed(
    x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
) -> R.Tensor:
    gv0 = R.call_packed("test.vm.mul", x, w, sinfo_args=(R.Tensor(ndim=2, dtype="float32")))
    return gv0


def test_varg_default_wildcard():
    expr = simple_call_packed.body.blocks[0].bindings[0].value
    yes_pattern_explicit = ExternFuncPattern("test.vm.mul")(wildcard(), wildcard())
    yes_pattern_implicit = ExternFuncPattern("test.vm.mul")(varg_default_wildcard=True)
    no_pattern = ExternFuncPattern("test.vm.mul")(wildcard())

    assert yes_pattern_explicit.match(expr)
    assert yes_pattern_implicit.match(expr)
    assert not no_pattern.match(expr)


def test_simple_call_packed():
    expr = simple_call_packed.body.blocks[0].bindings[0].value
    assert is_call_packed("test.vm.mul").match(expr)
    assert is_call_packed("test.vm.mul", [is_var("x"), is_var("w")]).match(expr)


## Graph-wise Matching
def test_simple_used_by():
    with PatternContext() as ctx:
        n0 = is_var("x")  # x is a free var (fn arg)
        n1 = wildcard()
        n0 ^ n1
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == main_fn.params[0]
        assert matched[n1] == dfb.bindings[0].var


def test_simple_call_tir_edge():
    with PatternContext() as ctx:
        n0 = is_call_tir("tir_matmul")
        n1 = is_call_tir("tir_relu")
        n0.used_by(n1)
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == dfb.bindings[0].var
        assert matched[n1] == dfb.bindings[1].var


def test_simple_oub():
    with PatternContext() as ctx:
        n0 = is_call_tir("tir_matmul")
        n1 = is_call_tir("tir_relu")
        n0 >> n1
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == dfb.bindings[0].var
        assert matched[n1] == dfb.bindings[1].var


def test_counter_syntax_match():
    with PatternContext() as ctx:
        n0 = is_call_dps_packed("extern_matmul")
        n1 = is_call_dps_packed("extern_impossible")
        n0 >> n1
        dfb = main_fn.body.blocks[0]
        assert not ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        n0 = is_call_dps_packed("extern_matmul")
        n1 = is_call_dps_packed("extern_impossible")
        n0 ^ n1
        dfb = main_fn.body.blocks[0]
        assert not ctx.match_dfb(dfb)


@tvm.script.ir_module
class Diamond:
    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            #   matmul
            #  /      \
            # relu  sigmoid
            #  \      /
            #    add
            lv0 = R.call_dps_packed("extern_matmul", (x, w), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("extern_relu", (lv0,), R.Tensor((32, 32), dtype="float32"))
            lv2 = R.call_dps_packed("extern_sigmoid", (lv0), R.Tensor((32, 32), dtype="float32"))
            lv3 = R.call_dps_packed("extern_add", (lv1, lv2), R.Tensor((32, 32), dtype="float32"))
            R.output(lv3)
        return lv3


def test_diamond():
    with PatternContext() as ctx:
        n0 = is_call_dps_packed("extern_matmul")
        n1 = is_call_dps_packed("extern_relu")
        n2 = is_call_dps_packed("extern_sigmoid")
        n3 = is_call_dps_packed("extern_add")

        n0 ^ n1
        n0 ^ n2
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]

        assert ctx.match_dfb(dfb)
    # simplify it with fork_to
    with PatternContext() as ctx:
        n1 = is_call_dps_packed("extern_relu")
        n2 = is_call_dps_packed("extern_sigmoid")
        n3 = is_call_dps_packed("extern_add")

        is_call_dps_packed("extern_matmul").fork_to(n1, n2)
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_diamond_counter_oub():
    with PatternContext() as ctx:
        n0 = is_call_dps_packed("extern_matmul")
        n1 = is_call_dps_packed("extern_relu")
        n2 = is_call_dps_packed("extern_sigmoid")
        n3 = is_call_dps_packed("extern_add")

        n0 >> n1
        n0 >> n2
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]
        assert not ctx.match_dfb(dfb)


@tvm.script.ir_module
class SmallDiamond:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            #    relu
            #  /      \
            #  \      /
            #    add
            lv0 = R.call_dps_packed("my_relu", (x,), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("my_add", (lv0, lv0), R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1


@tvm.script.ir_module
class SmallParallel:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # relu   relu
            #   \    /
            #    add
            lv0 = R.call_dps_packed("my_relu", (x,), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("my_relu", (x,), R.Tensor((32, 32), dtype="float32"))
            lv2 = R.call_dps_packed("my_add", (lv0, lv1), R.Tensor((32, 32), dtype="float32"))
            R.output(lv2)
        return lv2


def test_distinguish_diamond_and_parallel():
    # relay pattern lang cannot distinguish the two cases above.
    diamond = SmallDiamond["main"].body.blocks[0]
    parallel = SmallParallel["main"].body.blocks[0]

    with PatternContext() as ctx:
        # describe a diamond pattern
        fork = is_call_dps_packed("my_relu")
        join = is_call_dps_packed("my_add")
        fork.only_used_by(join, index=0)
        fork.only_used_by(join, index=1)

        assert ctx.match_dfb(diamond)
        assert not ctx.match_dfb(parallel)

    with PatternContext() as ctx:
        # describe a parallel pattern
        join = is_call_dps_packed("my_add")
        # Due to one-one matching:
        # is_call_dps_packed("my_relu") creates the 1st relu
        is_call_dps_packed("my_relu") >> join
        # is_call_dps_packed("my_relu")
        # creates the another different relu (obj address is different)
        is_call_dps_packed("my_relu") >> join

        assert ctx.match_dfb(parallel)
        assert not ctx.match_dfb(diamond)


@tvm.script.ir_module
class CBRx2:
    @R.function
    def main(
        x: R.Tensor((32, 32), "float32"),
        w0: R.Tensor((1, 1), "float32"),
        bias0: R.Tensor((32, 32), "float32"),
        w1: R.Tensor((1, 1), "float32"),
        bias1: R.Tensor((32, 32), "float32"),
    ) -> R.Tensor:
        # R.TensorRT's CBR Optimization Pattern
        #     input
        #     /   \
        #  cbr0   cbr1
        #     \   /
        #     concat
        with R.dataflow():
            lv0 = R.call_dps_packed("conv1x1", (x, w0), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("bias_add", (lv0, bias0), R.Tensor((32, 32), dtype="float32"))
            lv2 = R.call_dps_packed("my_relu", (lv1), R.Tensor((32, 32), dtype="float32"))
            lv3 = R.call_dps_packed("conv1x1", (x, w1), R.Tensor((32, 32), dtype="float32"))
            lv4 = R.call_dps_packed("bias_add", (lv3, bias1), R.Tensor((32, 32), dtype="float32"))
            lv5 = R.call_dps_packed("my_relu", (lv4), R.Tensor((32, 32), dtype="float32"))
            lv6 = R.call_dps_packed("concat", (lv2, lv5), R.Tensor((32, 64), dtype="float32"))
            R.output(lv6)
        return lv6


def test_nested_context():
    dfb = CBRx2["main"].body.blocks[0]
    with PatternContext() as ctx0:
        (
            is_call_dps_packed("conv1x1")
            >> is_call_dps_packed("bias_add")
            >> is_call_dps_packed("my_relu")
        )
        with PatternContext() as ctx1:
            is_call_dps_packed("conv1x1") >> is_call_dps_packed("my_relu")  # pattern to miss
            with PatternContext() as ctx2:
                is_call_dps_packed("bias_add") >> is_call_dps_packed("my_relu")
                assert ctx2.match_dfb(dfb)
                assert PatternContext.current() == ctx2
            assert not ctx1.match_dfb(dfb)
            assert PatternContext.current() == ctx1
        assert ctx0.match_dfb(dfb)
        assert PatternContext.current() == ctx0


def test_two_cbr():
    with PatternContext() as ctx:
        cbr0 = (
            is_call_dps_packed("conv1x1")
            >> is_call_dps_packed("bias_add")
            >> is_call_dps_packed("my_relu")
        )
        cbr1 = cbr0.dup()

        assert cbr0.patterns[0] != cbr1.patterns[0]
        assert cbr0.patterns[1] != cbr1.patterns[1]
        assert cbr0.patterns[2] != cbr1.patterns[2]

        is_var("x").fork_to(cbr0, cbr1)
        dfb = CBRx2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        # Deny the pattern
        cbr0 = (
            is_call_dps_packed("conv1x1")
            >> is_call_dps_packed("bias_add")
            >> is_call_dps_packed("my_relu")
        )
        cbr1 = cbr0.dup()

        # input has no fork at y.
        is_var("y").fork_to(cbr0, cbr1)
        dfb = CBRx2["main"].body.blocks[0]
        assert not ctx.match_dfb(dfb)


def test_two_matmul():
    # Same as Figure 2(a) in TASO paper.
    @tvm.script.ir_module
    class MatMul2:
        @R.function
        def main(
            a: R.Tensor((32, 16), "float32"),
            b: R.Tensor((16, 48), "float32"),
            c: R.Tensor((48, 32), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_dps_packed("matmul", (a, b), R.Tensor((32, 48), dtype="float32"))
                lv1 = R.call_dps_packed("matmul", (lv0, c), R.Tensor((32, 32), dtype="float32"))
                R.output(lv1)
            return lv1

    with PatternContext() as ctx:
        is_call_dps_packed("matmul") >> is_call_dps_packed("matmul")
        dfb = MatMul2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        is_call_dps_packed("matmul").has_shape([32, 48]) >> is_call_dps_packed("matmul").has_shape(
            [32, 32]
        )
        dfb = MatMul2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        is_call_dps_packed("matmul") >> is_call_dps_packed("matmul") >> is_call_dps_packed("matmul")
        dfb = MatMul2["main"].body.blocks[0]
        # Three MatMul cannot match
        assert not ctx.match_dfb(dfb)


def test_concat_mm_split():
    # Same as Figure 2(b) in TASO paper.
    @tvm.script.ir_module
    class CMS:
        @R.function
        def main(
            a: R.Tensor((32, 32), "float32"),
            b: R.Tensor((16, 32), "float32"),
            c: R.Tensor((16, 32), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_dps_packed("my_concat", (b, c), R.Tensor((32, 32), dtype="float32"))
                lv1 = R.call_dps_packed("my_matmul", (a, lv0), R.Tensor((32, 32), dtype="float32"))
                lv2 = R.call_dps_packed(
                    "my_split",
                    (lv1,),
                    [R.Tensor((16, 32), dtype="float32"), R.Tensor((16, 32), dtype="float32")],
                )
                lv3 = R.TupleGetItem(lv2, 0)
                lv4 = R.TupleGetItem(lv2, 1)
                lv5 = R.add(lv3, lv4)
                R.output(lv5)
            return lv5

    with PatternContext() as ctx:
        (
            is_call_dps_packed("my_concat")
            >> is_call_dps_packed("my_matmul")
            >> is_call_dps_packed("my_split")
        )
        dfb = CMS["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        split = is_call_dps_packed("my_split")
        lv3 = TupleGetItemPattern(split, 0).has_shape([16, 32])
        lv4 = TupleGetItemPattern(split, 1).has_shape([16, 32])
        split.fork_to(lv3, lv4)
        add = is_op("relax.add")(lv3, lv4)
        # TODO(@ganler): simplify this through implicit graph pattern.
        lv3 >> add
        lv4 >> add

        dfb = CMS["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_self_attention():
    # The example comes from.
    # https://developer.nvidia.com/blog/nlu-with-tensorrt-bert/
    @tvm.script.ir_module
    class SelfAttention:
        @R.function
        def main(
            x: R.Tensor(("b", "s", "n", "h"), "float32"),
            wq: R.Tensor(("h", "h"), "float32"),
            wk: R.Tensor(("h", "h"), "float32"),
            wv: R.Tensor(("h", "h"), "float32"),
        ) -> R.Tensor:
            b, s, n, h = T.int64(), T.int64(), T.int64(), T.int64()
            with R.dataflow():
                fcq = R.call_dps_packed("my_fc", (x, wq), R.Tensor((b, s, n, h), dtype="float32"))
                tpq = R.call_dps_packed(
                    "my_transpose", (fcq,), R.Tensor((b, s, h, n), dtype="float32")
                )

                fck = R.call_dps_packed("my_fc", (x, wk), R.Tensor((b, s, n, h), dtype="float32"))
                tpk = R.call_dps_packed(
                    "my_transpose", (fck,), R.Tensor((b, s, h, n), dtype="float32")
                )

                mul = R.multiply(tpq, tpk)
                scale = R.multiply(mul, R.const(1.1, "float32"))
                softmax = R.call_dps_packed(
                    "softmax", (scale,), R.Tensor((b, s, n, h), dtype="float32")
                )

                fcv = R.call_dps_packed("my_fc", (x, wv), R.Tensor((b, s, n, h), dtype="float32"))
                tpv = R.call_dps_packed(
                    "my_transpose", (fcv,), R.Tensor((b, s, h, n), dtype="float32")
                )

                out = R.multiply(softmax, tpv)
                R.output(out)

            return out

    with PatternContext() as ctx:
        fc_trans_q = is_call_dps_packed("my_fc") >> is_call_dps_packed("my_transpose")
        fc_trans_k = fc_trans_q.dup()
        fc_trans_v = fc_trans_q.dup()

        is_var("x").fork_to(fc_trans_q, fc_trans_k, fc_trans_v)
        dfb = SelfAttention["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_nested_diamond():
    @tvm.script.ir_module
    class DiamondInDiamond:
        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                #   matmul0      matmul1
                #     /    \    /    \
                # sigmoid2  add4  sigmoid3
                #     \    /    \    /
                #      add5      add6
                #          \    /
                #           add7
                lv0 = R.call_dps_packed(
                    "extern_matmul", (x, w), R.Tensor((32, 32), dtype="float32")
                )
                lv1 = R.call_dps_packed(
                    "extern_matmul", (x, w), R.Tensor((32, 32), dtype="float32")
                )
                lv2 = R.call_dps_packed(
                    "extern_sigmoid", (lv0), R.Tensor((32, 32), dtype="float32")
                )
                lv3 = R.call_dps_packed(
                    "extern_sigmoid", (lv1), R.Tensor((32, 32), dtype="float32")
                )
                lv4 = R.call_dps_packed(
                    "extern_add", (lv0, lv1), R.Tensor((32, 32), dtype="float32")
                )
                lv5 = R.call_dps_packed(
                    "extern_add", (lv2, lv4), R.Tensor((32, 32), dtype="float32")
                )
                lv6 = R.call_dps_packed(
                    "extern_add", (lv3, lv4), R.Tensor((32, 32), dtype="float32")
                )
                lv7 = R.call_dps_packed(
                    "extern_add", (lv5, lv6), R.Tensor((32, 32), dtype="float32")
                )
                R.output(lv7)
            return lv7

    # match matmul0 diamond
    with PatternContext() as ctx:
        sigmoid2 = is_call_dps_packed("extern_sigmoid")
        add4 = is_call_dps_packed("extern_add")
        is_call_dps_packed("extern_matmul").fork_to(sigmoid2, add4)
        add5 = is_call_dps_packed("extern_add")
        sigmoid2 >> add5
        add4 ^ add5
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # counter case: mis-match matmul0 diamond
    with PatternContext() as ctx:
        sigmoid2 = is_call_dps_packed("extern_sigmoid")
        add4 = is_call_dps_packed("extern_add")
        is_call_dps_packed("extern_matmul").fork_to(sigmoid2, add4)
        add5 = is_call_dps_packed("extern_add")
        sigmoid2 >> add5
        add4 >> add5  # not only-used-by relation
        assert not ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # match matmul1 diamond
    with PatternContext() as ctx:
        sigmoid3 = is_call_dps_packed("extern_sigmoid")
        add4 = is_call_dps_packed("extern_add")
        is_call_dps_packed("extern_matmul").fork_to(sigmoid3, add4)
        add6 = is_call_dps_packed("extern_add")
        sigmoid3 >> add6
        add4 ^ add6
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # match add-4-5-6-7
    with PatternContext() as ctx:
        add5, add6, add7 = (
            is_call_dps_packed("extern_add"),
            is_call_dps_packed("extern_add"),
            is_call_dps_packed("extern_add"),
        )
        is_call_dps_packed("extern_add").fork_to(add5, add6)  # add4
        add5 >> add7
        add6 >> add7
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])


def test_incremental_solving():
    @R.function
    def simple_chain(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # relu -> sigmoid -> neg
            lv0 = R.call_dps_packed("extern_relu", (x), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("extern_sigmoid", (lv0), R.Tensor((32, 32), dtype="float32"))
            lv2 = R.call_dps_packed("extern_neg", (lv1), R.Tensor((32, 32), dtype="float32"))
            R.output(lv2)
        return lv2

    relu = is_call_dps_packed("extern_relu")
    sigmoid = is_call_dps_packed("extern_sigmoid")
    neg = is_call_dps_packed("extern_neg")

    with PatternContext() as ctx0:
        relu >> sigmoid
        with PatternContext(incremental=True) as ctx1:
            # because we are doing incremental solving
            # relu >> sigmoid is still a constraint in this context.
            # that said the total constraint is:
            # relu >> sigmoid >> neg
            sigmoid >> neg
            assert ctx1.match_dfb(simple_chain.body.blocks[0])

        # match relue -> sigmoid
        assert ctx0.match_dfb(simple_chain.body.blocks[0])


def test_incremental_solving_counter():
    @R.function
    def simple_chain(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # sigmoid -> neg
            lv0 = R.call_dps_packed("extern_sigmoid", (x), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_dps_packed("extern_neg", (lv0), R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1

    relu = is_call_dps_packed("extern_relu")
    sigmoid = is_call_dps_packed("extern_sigmoid")
    neg = is_call_dps_packed("extern_neg")

    with PatternContext() as ctx0:
        relu >> sigmoid  # cannot match

        with PatternContext(incremental=False) as ctx1:
            # total constraint: sigmoid >> neg
            sigmoid >> neg
            assert ctx1.match_dfb(simple_chain.body.blocks[0])

        with PatternContext(incremental=True) as ctx1:
            # total constraint: relu >> sigmoid >> neg
            sigmoid >> neg
            assert not ctx1.match_dfb(simple_chain.body.blocks[0])


def test_rewrite_simple():
    @R.function
    def main(x: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            x2 = R.add(x, x)
            x4 = R.add(x2, x2)
            R.output(x4)
        return x4

    @R.function
    def expected1(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((16, 16), dtype="float32") = R.multiply(x, R.const(2, "float32"))
            x4: R.Tensor((16, 16), dtype="float32") = R.multiply(lv, R.const(2, "float32"))
            R.output(x4)
        return x4

    @R.function
    def expected2(x: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
        with R.dataflow():
            x4: R.Tensor((16, 16), dtype="float32") = R.multiply(x, R.const(4, "float32"))
            R.output(x4)
        return x4

    x = wildcard()
    pattern = is_op("relax.add")(x, x)

    def rewriter(_, matchings):
        return R.multiply(matchings[x], R.const(2, "float32"))

    rewritten = rewrite_call(pattern, rewriter, main)
    tvm.ir.assert_structural_equal(rewritten, expected1.with_attr("global_symbol", "main"))

    add1 = is_op("relax.add")(x, x)
    pattern = is_op("relax.add")(add1, add1)

    def rewriter(_, matchings):
        return R.multiply(matchings[x], R.const(4, "float32"))

    rewritten = rewrite_call(pattern, rewriter, main)
    tvm.ir.assert_structural_equal(rewritten, expected2.with_attr("global_symbol", "main"))

    # No rewriting, return the original call node as is
    def rewriter(orig, _):
        return orig

    rewritten = rewrite_call(pattern, rewriter, main)
    tvm.ir.assert_structural_equal(rewritten, main)


def test_rewrite_attention():
    @R.function
    def main(
        Q: R.Tensor((2, 4096, 8, 40), "float32"),
        K: R.Tensor((2, 4096, 8, 40), "float32"),
        V: R.Tensor((2, 4096, 8, 40), "float32"),
    ) -> R.Tensor((2, 4096, 8, 40), "float32"):
        with R.dataflow():
            lv58 = R.permute_dims(Q, axes=[0, 2, 1, 3])
            lv59 = R.reshape(lv58, R.shape([16, 4096, 40]))

            lv61 = R.permute_dims(K, axes=[0, 2, 1, 3])
            lv62 = R.reshape(lv61, R.shape([16, 4096, 40]))

            lv64 = R.permute_dims(V, axes=[0, 2, 1, 3])
            lv65 = R.reshape(lv64, R.shape([16, 4096, 40]))

            lv62_transposed = R.permute_dims(lv62, axes=[0, 2, 1])
            lv3_1 = R.matmul(lv59, lv62_transposed)
            lv68 = R.multiply(lv3_1, R.const(0.15811388194561005, "float32"))
            lv69 = R.nn.softmax(lv68, axis=-1)
            lv_3 = R.matmul(lv69, lv65)

            lv71 = R.reshape(lv_3, R.shape([2, 8, 4096, 40]))
            lv72 = R.permute_dims(lv71, axes=[0, 2, 1, 3])
            R.output(lv72)

        return lv72

    @R.function
    def expected(
        Q: R.Tensor((2, 4096, 8, 40), dtype="float32"),
        K: R.Tensor((2, 4096, 8, 40), dtype="float32"),
        V: R.Tensor((2, 4096, 8, 40), dtype="float32"),
    ) -> R.Tensor((2, 4096, 8, 40), dtype="float32"):
        with R.dataflow():
            lv72: R.Tensor((2, 4096, 8, 40), dtype="float32") = R.nn.attention(Q, V, K)
            R.output(lv72)
        return lv72

    def BSNH_to_BSH(tensor):
        return is_op("relax.reshape")(is_op("relax.permute_dims")(tensor), wildcard())

    def BSH_to_BSNH(tensor):
        return is_op("relax.permute_dims")(is_op("relax.reshape")(tensor, wildcard()))

    Q = wildcard()
    K = wildcard()
    V = wildcard()

    Q_3D = BSNH_to_BSH(Q)
    V_3D = BSNH_to_BSH(V)
    K_3D = BSNH_to_BSH(K)

    matmul1 = is_op("relax.matmul")(Q_3D, is_op("relax.permute_dims")(V_3D))
    multiply = is_op("relax.multiply")(matmul1, is_const())
    softmax = is_op("relax.nn.softmax")(multiply)
    matmul2 = is_op("relax.matmul")(softmax, K_3D)

    pattern = BSH_to_BSNH(matmul2)

    def rewriter(_, matchings):
        return R.nn.attention(matchings[Q], matchings[K], matchings[V])

    rewritten = rewrite_call(pattern, rewriter, main)
    tvm.ir.assert_structural_equal(rewritten, expected.with_attr("global_symbol", "main"))


def test_attention_qkv():
    @tvm.script.ir_module
    class QKV_proj:
        @R.function
        def main(
            x: R.Tensor((2, 1024, 640), "float32"),
            w0: R.Tensor((640, 640), "float32"),
            w1: R.Tensor((640, 640), "float32"),
            w2: R.Tensor((640, 640), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(x, w1)
                lv2 = R.matmul(x, w2)
                out = (lv0, lv1, lv2)
                R.output(out)
            return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        dfb = QKV_proj["main"].body.blocks[0]
        out = ctx.match_dfb(dfb)

        assert out[Q_weight_pat].name_hint == "w0"
        assert out[K_weight_pat].name_hint == "w1"
        assert out[V_weight_pat].name_hint == "w2"


def test_attention_fake_qkv():
    @tvm.script.ir_module
    class QKV_proj:
        @R.function
        def main(
            x1: R.Tensor((2, 1024, 640), "float32"),
            x2: R.Tensor((2, 1024, 640), "float32"),
            w0: R.Tensor((640, 640), "float32"),
            w1: R.Tensor((640, 640), "float32"),
            w2: R.Tensor((640, 640), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.matmul(x1, w0)
                lv1 = R.matmul(x2, w1)
                lv2 = R.matmul(x2, w2)
                out = (lv0, lv1, lv2)
                R.output(out)
            return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        dfb = QKV_proj["main"].body.blocks[0]
        assert ctx.match_dfb(dfb) is None


def get_qkv_proj_rewriter(
    inp_pat, Q_weight_pat, K_weight_pat, V_weight_pat, matmul1, matmul2, matmul3
):
    def qkv_proj_rewriter(matchings, _):
        inp = matchings[inp_pat]
        Q_weight = matchings[Q_weight_pat]
        K_weight = matchings[K_weight_pat]
        V_weight = matchings[V_weight_pat]
        width = Q_weight.struct_info.shape[1]

        concat = R.concat([Q_weight, K_weight, V_weight], axis=1)
        matmul = R.matmul(inp, concat)
        Q = R.strided_slice(matmul, axes=[2], begin=[0], end=[width])
        K = R.strided_slice(matmul, axes=[2], begin=[width], end=[width * 2])
        V = R.strided_slice(matmul, axes=[2], begin=[width * 2], end=[width * 3])

        return {matchings[matmul1]: Q, matchings[matmul2]: K, matchings[matmul3]: V}

    return qkv_proj_rewriter


def test_combine_matmul_twice():
    @R.function(private=True)
    def qkv_x2(
        x1: R.Tensor((2, 1024, 640), "float32"),
        x2: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
        w3: R.Tensor((640, 640), "float32"),
        w4: R.Tensor((640, 640), "float32"),
        w5: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            lv0 = R.matmul(x1, w0)
            lv1 = R.matmul(x1, w1)
            lv2 = R.matmul(x1, w2)
            lv3 = R.matmul(x2, w3)
            lv4 = R.matmul(x2, w4)
            lv5 = R.matmul(x2, w5)
            out = (lv0, lv1, lv2, lv3, lv4, lv5)
            R.output(out)
        return out

    @R.function(private=True)
    def expected(
        x1: R.Tensor((2, 1024, 640), "float32"),
        x2: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
        w3: R.Tensor((640, 640), "float32"),
        w4: R.Tensor((640, 640), "float32"),
        w5: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            lv = R.concat((w0, w1, w2), axis=1)
            lv1 = R.matmul(x1, lv)
            lv0 = R.strided_slice(lv1, axes=[2], begin=[0], end=[640])
            lv1_1 = R.strided_slice(lv1, axes=[2], begin=[640], end=[1280])
            lv2 = R.strided_slice(lv1, axes=[2], begin=[1280], end=[1920])
            lv2_1 = R.concat((w3, w4, w5), axis=1)
            lv3 = R.matmul(x2, lv2_1, out_dtype="void")
            lv3_1 = R.strided_slice(lv3, axes=[2], begin=[0], end=[640])
            lv4 = R.strided_slice(lv3, axes=[2], begin=[640], end=[1280])
            lv5 = R.strided_slice(lv3, axes=[2], begin=[1280], end=[1920])
            out = lv0, lv1_1, lv2, lv3_1, lv4, lv5
            R.output(out)
        return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        rewriter = get_qkv_proj_rewriter(
            inp_pat, Q_weight_pat, K_weight_pat, V_weight_pat, matmul1, matmul2, matmul3
        )
        rewritten = rewrite_bindings(ctx, rewriter, qkv_x2)
        tvm.ir.assert_structural_equal(rewritten, expected)


def test_combine_matmul_emit_order():
    @R.function(private=True)
    def main(
        x1: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            w0_t = R.permute_dims(w0, axes=None)
            lv0 = R.matmul(x1, w0_t)
            w1_t = R.permute_dims(w1, axes=None)
            w1_t_t = R.permute_dims(w1_t, axes=None)
            lv1 = R.matmul(x1, w1_t_t)
            w2_t = R.permute_dims(w2, axes=None)
            lv2 = R.matmul(x1, w2_t)
            out = (lv0, lv1, lv2)
            R.output(out)
        return out

    @R.function(private=True)
    def expected(
        x1: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, 640), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((640, 640), dtype="float32"),
    ):
        with R.dataflow():
            w0_t = R.permute_dims(w0, axes=None)
            w1_t = R.permute_dims(w1, axes=None)
            w1_t_t = R.permute_dims(w1_t, axes=None)
            w2_t = R.permute_dims(w2, axes=None)
            lv = R.concat((w0_t, w1_t_t, w2_t), axis=1)
            lv1 = R.matmul(x1, lv, out_dtype="void")
            lv0 = R.strided_slice(lv1, axes=[2], begin=[0], end=[640])
            lv1_1 = R.strided_slice(lv1, axes=[2], begin=[640], end=[1280])
            lv2 = R.strided_slice(lv1, axes=[2], begin=[1280], end=[1920])
            out = lv0, lv1_1, lv2
            R.output(out)
        return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        Q_weight_pat = wildcard()
        K_weight_pat = wildcard()
        V_weight_pat = wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        rewriter = get_qkv_proj_rewriter(
            inp_pat, Q_weight_pat, K_weight_pat, V_weight_pat, matmul1, matmul2, matmul3
        )
        rewritten = rewrite_bindings(ctx, rewriter, main)
        tvm.ir.assert_structural_equal(rewritten, expected)

        # make sure it builds
        mod = tvm.IRModule()
        mod["main"] = rewritten
        mod = rx.transform.LegalizeOps()(mod)

        rx.build(mod, target="llvm")


def test_combine_transposed_matmul_twice():
    @R.function(private=True)
    def main(
        x1: R.Tensor((2, 1024, 640), "float32"),
        x2: R.Tensor((2, 1024, 640), "float32"),
        w0: R.Tensor((640, 640), "float32"),
        w1: R.Tensor((640, 640), "float32"),
        w2: R.Tensor((640, 640), "float32"),
        w3: R.Tensor((640, 640), "float32"),
    ):
        with R.dataflow():
            w0_t = R.permute_dims(w0, axes=None)
            lv0 = R.matmul(x1, w0_t)
            w1_t = R.permute_dims(w1, axes=None)
            lv1 = R.matmul(x1, w1_t)
            w2_t = R.permute_dims(w2, axes=None)
            lv2 = R.matmul(x2, w2_t)
            w3_t = R.permute_dims(w3, axes=None)
            lv3 = R.matmul(x2, w3_t)
            out = (lv0, lv1, lv2, lv3)
            R.output(out)
        return out

    @R.function(private=True)
    def expected(
        x1: R.Tensor((2, 1024, 640), dtype="float32"),
        x2: R.Tensor((2, 1024, 640), dtype="float32"),
        w0: R.Tensor((640, 640), dtype="float32"),
        w1: R.Tensor((640, 640), dtype="float32"),
        w2: R.Tensor((640, 640), dtype="float32"),
        w3: R.Tensor((640, 640), dtype="float32"),
    ):
        with R.dataflow():
            lv: R.Tensor((1280, 640), dtype="float32") = R.concat((w0, w1), axis=0)
            lv1: R.Tensor((640, 1280), dtype="float32") = R.permute_dims(lv, axes=None)
            lv2: R.Tensor((2, 1024, 1280), dtype="float32") = R.matmul(x1, lv1, out_dtype="void")
            lv3: R.Tuple(
                R.Tensor((2, 640, 1280), dtype="float32"),
                R.Tensor((2, 384, 1280), dtype="float32"),
                R.Tensor((2, 0, 1280), dtype="float32"),
            ) = R.split(lv2, indices_or_sections=[640, 1280], axis=1)
            lv0: R.Tensor((2, 640, 1280), dtype="float32") = lv3[0]
            lv1_1: R.Tensor((2, 384, 1280), dtype="float32") = lv3[1]
            lv_1: R.Tensor((1280, 640), dtype="float32") = R.concat((w2, w3), axis=0)
            lv1_2: R.Tensor((640, 1280), dtype="float32") = R.permute_dims(lv_1, axes=None)
            lv2_1: R.Tensor((2, 1024, 1280), dtype="float32") = R.matmul(
                x2, lv1_2, out_dtype="void"
            )
            lv3_1: R.Tuple(
                R.Tensor((2, 640, 1280), dtype="float32"),
                R.Tensor((2, 384, 1280), dtype="float32"),
                R.Tensor((2, 0, 1280), dtype="float32"),
            ) = R.split(lv2_1, indices_or_sections=[640, 1280], axis=1)
            lv2_1_1: R.Tensor((2, 640, 1280), dtype="float32") = lv3_1[0]
            lv3_1_1: R.Tensor((2, 384, 1280), dtype="float32") = lv3_1[1]
            out: R.Tuple(
                R.Tensor((2, 640, 1280), dtype="float32"),
                R.Tensor((2, 384, 1280), dtype="float32"),
                R.Tensor((2, 640, 1280), dtype="float32"),
                R.Tensor((2, 384, 1280), dtype="float32"),
            ) = (lv0, lv1_1, lv2_1_1, lv3_1_1)
            R.output(out)
        return out

    with PatternContext() as ctx:
        inp_pat = wildcard()
        w1_pat = wildcard()
        w2_pat = wildcard()
        matmul1 = is_op("relax.matmul")(inp_pat, is_op("relax.permute_dims")(w1_pat))
        matmul2 = is_op("relax.matmul")(inp_pat, is_op("relax.permute_dims")(w2_pat))

        def rewriter(matchings, _):
            inp = matchings[inp_pat]
            w1 = matchings[w1_pat]
            w2 = matchings[w2_pat]

            concat = R.concat([w1, w2], axis=0)
            matmul = R.matmul(inp, R.permute_dims(concat))
            sections = [w1.struct_info.shape[0], w1.struct_info.shape[0] + w2.struct_info.shape[0]]

            chunks = R.split(matmul, sections, 1)

            return {
                matchings[matmul1]: chunks[0],
                matchings[matmul2]: chunks[1],
            }

        rewritten = rewrite_bindings(ctx, rewriter, main)
        tvm.ir.assert_structural_equal(rewritten, expected)

        # make sure it builds
        mod = tvm.IRModule()
        mod["main"] = rewritten
        mod = rx.transform.LegalizeOps()(mod)

        rx.build(mod, target="llvm")


def test_commutative_pattern_match():
    @R.function(private=True)
    def before(
        x: R.Tensor((1024,)),
    ):
        with R.dataflow():
            y = R.add(x, x)
            out = R.add(R.const(1.0), y)
            R.output(out)
        return out

    @R.function(private=True)
    def expected(
        x: R.Tensor((1024,)),
    ):
        with R.dataflow():
            y = R.add(x, x)
            out = R.add(y, R.const(2.0))
            R.output(out)

        return out

    pattern_add = is_op("relax.add")
    pattern_mul = is_op("relax.multiply")
    pattern_op = pattern_add | pattern_mul
    pattern_arg = wildcard()
    pattern_const = is_const()

    pattern = pattern_op(pattern_arg, pattern_const)

    def rewriter(expr, matches):
        op = matches[pattern_op]
        arg = matches[pattern_arg]
        const = matches[pattern_const].data.numpy()
        if const.shape == tuple() and const[()] == 1.0:
            return rx.Call(op, [arg, rx.const(2.0)])
        else:
            return expr

    after = rewrite_call(pattern, rewriter, before)
    tvm.ir.assert_structural_equal(after, expected)


def test_repeated_pattern_match():
    """rewrite_call should iterate until convergence"""

    @R.function(private=True)
    def before(
        x: R.Tensor((1024,)),
        y: R.Tensor((1024,)),
        z: R.Tensor((1024,)),
    ):
        with R.dataflow():
            a = R.add(x, y)
            b = R.add(a, z)
            out = R.multiply(b, R.const(5.0))
            R.output(out)
        return out

    @R.function(private=True)
    def expected(
        x: R.Tensor((1024,)),
        y: R.Tensor((1024,)),
        z: R.Tensor((1024,)),
    ):
        with R.dataflow():
            x = R.multiply(x, R.const(5.0))
            y = R.multiply(y, R.const(5.0))
            a = R.add(x, y)
            z = R.multiply(z, R.const(5.0))
            b = R.add(a, z)
            R.output(b)
        return b

    pattern_add_lhs = wildcard()
    pattern_add_rhs = wildcard()
    pattern_add = is_op("relax.add")(pattern_add_lhs, pattern_add_rhs)

    mul_const = is_const()
    pattern_mul = is_op("relax.multiply")(pattern_add, mul_const)

    pattern = pattern_mul

    def rewriter(_expr, matches):
        const = matches[mul_const]
        return (matches[pattern_add_lhs] * const) + (matches[pattern_add_rhs] * const)

    after = rewrite_call(pattern, rewriter, before)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_without_trivial_binding():
    """rewrite_call should avoid producing trivial "y = x" bindings"""

    @R.function(private=True)
    def before(x: R.Tensor((1024,))):
        with R.dataflow():
            a = R.add(x, x)
            b = R.reshape(a, (1024,))
            R.output(b)
        return b

    @R.function(private=True)
    def expected(x: R.Tensor((1024,))):
        with R.dataflow():
            a = R.add(x, x)
            R.output(a)
        return a

    pattern_arg = wildcard()
    pattern_shape_expr = wildcard()
    pattern = is_op("relax.reshape")(pattern_arg, pattern_shape_expr)

    def rewriter(expr, matches):
        arg = matches[pattern_arg]
        shape_expr = matches[pattern_shape_expr]

        if tvm.ir.structural_equal(arg.struct_info.shape, shape_expr):
            return arg
        else:
            return expr

    after = rewrite_call(pattern, rewriter, before)
    tvm.ir.assert_structural_equal(after, expected)


same_shape_func_type = tvm.testing.parameter(
    "same_static_shape",
    "same_dynamic_shape",
    "different_static_shape",
    "different_dynamic_shape",
)


def test_same_shape_pattern(same_shape_func_type):
    if same_shape_func_type == "same_static_shape":

        @R.function(private=True)
        def func(
            a: R.Tensor((1024, 128), "float32"),
            b: R.Tensor((1024, 128), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                c = R.multiply(a, R.const(2.0))
                d = R.add(b, c)
                out = d
                R.output(out)
            return out

    elif same_shape_func_type == "same_dynamic_shape":

        @R.function(private=True)
        def func(
            a: R.Tensor(("n", 128), "float32"),
            b: R.Tensor(("n", 128), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                c = R.multiply(a, R.const(2.0))
                d = R.add(b, c)
                out = d
                R.output(out)
            return out

    elif same_shape_func_type == "different_static_shape":

        @R.function(private=True)
        def func(
            a: R.Tensor((1024, 128), "float32"),
            b: R.Tensor((1, 128), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                c = R.multiply(a, R.const(2.0))
                d = R.add(b, c)
                out = d
                R.output(out)
            return out

    elif same_shape_func_type == "different_dynamic_shape":

        @R.function(private=True)
        def func(
            a: R.Tensor(("n", 128), "float32"),
            b: R.Tensor(("m", 128), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                c = R.multiply(a, R.const(2.0))
                d = R.add(b, c)
                out = d
                R.output(out)
            return out

    else:
        raise ValueError(f"Unknown value of same_shape_func_type={same_shape_func_type}")

    with PatternContext() as ctx:
        pat_lhs = wildcard()
        pat_rhs = wildcard()
        pat_sum = is_op("relax.add")(pat_lhs, pat_rhs)
        pat_lhs.same_shape_as(pat_rhs)

    block = func.body.blocks[0]
    match = ctx.match_dfb(block)

    if "same" in same_shape_func_type:
        assert match
    else:
        assert match is None


if __name__ == "__main__":
    tvm.testing.main()
