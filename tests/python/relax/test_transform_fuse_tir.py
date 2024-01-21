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
from tvm import relax, topi
from tvm.script import ir as I, relax as R, tir as T


def _check(mod_before, mod_expected):
    mod_after = relax.transform.FuseTIR()(mod_before)
    tvm.ir.assert_structural_equal(mod_expected, mod_after)


def test_simple():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))

        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add_exp_squeeze, [x, p0]))
            bb.emit_func_output(gv)

        return bb.get().with_attrs({"foo": "bar"})

    def expected():
        def fused_add_exp_squeeze(x, p0):
            add = topi.add(x, p0)
            exp = topi.exp(add)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_add_exp_squeeze, x, p0))
            bb.emit_func_output(gv)
        return bb.get().with_attrs({"foo": "bar"})

    _check(before(), expected())


def test_conv2d_fuse():
    def before(dtype):
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w = relax.Var("w", R.Tensor((16, 16, 3, 3), dtype))
        p0 = relax.Var("p0", R.Tensor((), dtype))
        with bb.function("fused_conv2d_add1_add2", [x, w, p0], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    primfunc_name_hint="conv2d",
                )
                lv1 = bb.emit_te(topi.add, p0, lv0, primfunc_name_hint="add1")
                gv = bb.emit_output(bb.call_te(topi.add, lv0, lv1, primfunc_name_hint="add2"))
            bb.emit_func_output(gv)

        # Grouped function 2
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w = relax.Var("w", R.Tensor((16, 16, 1, 1), dtype))
        y = relax.Var("y", R.Tensor((1, 16, 64, 64), dtype))
        with bb.function("fused_conv2d1_add2", [x, w, y], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=0,
                    dilation=1,
                    primfunc_name_hint="conv2d1",
                )
                gv = bb.emit_output(bb.call_te(topi.add, lv0, y, primfunc_name_hint="add2"))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_conv2d_add1_add2 = mod.get_global_var("fused_conv2d_add1_add2")
        fused_conv2d1_add2 = mod.get_global_var("fused_conv2d1_add2")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), dtype))
        w2 = relax.Var("w2", R.Tensor((16, 16, 1, 1), dtype))
        w3 = relax.Var("w3", R.Tensor((16, 16, 3, 3), dtype))
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit(relax.Call(fused_conv2d_add1_add2, [lv0, w1, relax.const(1, dtype)]))
                lv2 = bb.emit_te(
                    topi.nn.conv2d,
                    lv1,
                    w3,
                    strides=1,
                    padding=1,
                    dilation=1,
                )
                gv = bb.emit_output(relax.Call(fused_conv2d1_add2, [lv1, w2, lv2]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dtype):
        def fused_conv2d_add1_add2(x, w, p):
            conv = topi.nn.conv2d(x, w, strides=1, padding=1, dilation=1)
            add = topi.add(p, conv)
            return topi.add(conv, add)

        def fused_conv2d1_add2(x, w, p):
            conv = topi.nn.conv2d(x, w, strides=1, padding=0, dilation=1)
            return topi.add(conv, p)

        bb = relax.BlockBuilder()

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), dtype))
        w2 = relax.Var("w2", R.Tensor((16, 16, 1, 1), dtype))
        w3 = relax.Var("w3", R.Tensor((16, 16, 3, 3), dtype))
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit_te(fused_conv2d_add1_add2, lv0, w1, relax.const(1, dtype))
                lv2 = bb.emit_te(
                    topi.nn.conv2d,
                    lv1,
                    w3,
                    strides=1,
                    padding=1,
                    dilation=1,
                )
                gv = bb.emit_output(bb.call_te(fused_conv2d1_add2, lv1, w2, lv2))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before("float32"), expected("float32"))


def test_two_subfunction():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", R.Tensor([10, 20], "float32"))
        with bb.function("fused_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x]))
                lv2 = bb.emit(relax.Call(func_gv, [lv]))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_squeeze(x):
            exp = topi.exp(x)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.call_te(fused_exp_squeeze, lv)
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_same_primfunc():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", R.Tensor([10, 20], "float32"))
        with bb.function("fused_exp_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv1 = bb.emit_te(topi.exp, x1)
                lv2 = bb.emit_te(topi.exp, lv1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv2))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_exp_squeeze")
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_exp_squeeze(x):
            exp = topi.exp(x)
            exp = topi.exp(exp)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.call_te(fused_exp_exp_squeeze, x)
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_tuple_as_param():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tuple([R.Tensor([10], "float32"), R.Tensor([10], "float32")]))
        with bb.function("fused_exp_add", [x], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv2 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_add")
        x = relax.Var("x", R.Tuple([R.Tensor([10], "float32"), R.Tensor([10], "float32")]))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_add(x1, x2):
            exp = topi.exp(x1)
            return topi.add(exp, x2)

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tuple([R.Tensor([10], "float32"), R.Tensor([10], "float32")]))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                gv = bb.emit_output(bb.call_te(fused_exp_add, lv0, lv1))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_nested_tuple_as_param():
    tuple_struct_info = R.Tuple(
        [R.Tensor([10], "float32"), R.Tuple([R.Tensor([10], "float32"), R.Tensor([10], "float32")])]
    )

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", tuple_struct_info)
        with bb.function("fused_exp_add_add", [x], attrs={"Primitive": True}, private=True):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv0_exp = bb.emit_te(topi.exp, lv0)
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv1_0 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv1_1 = bb.emit(relax.TupleGetItem(lv1, 1))
                lv2 = bb.emit_te(topi.add, lv1_0, lv1_1)
                gv = bb.emit_output(bb.call_te(topi.add, lv0_exp, lv2))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_add_add")
        x = relax.Var("x", tuple_struct_info)
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_add_add(x1, x2, x3):
            exp = topi.exp(x1)
            add = topi.add(x2, x3)
            return topi.add(exp, add)

        bb = relax.BlockBuilder()
        x = relax.Var("x", tuple_struct_info)
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.TupleGetItem(x, 0))
                lv1 = bb.emit(relax.TupleGetItem(x, 1))
                lv2 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv3 = bb.emit(relax.TupleGetItem(lv1, 1))
                gv = bb.emit_output(bb.call_te(fused_exp_add_add, lv0, lv2, lv3))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_call_tir_in_main():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", R.Tensor([10, 20], "float32"))
        with bb.function("fused_exp_squeeze", [x1], attrs={"Primitive": True}):
            with bb.dataflow():
                lv = bb.emit_te(topi.exp, x1)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_exp_squeeze")
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(func_gv, [x]))
                lv1 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                gv = bb.emit_output(lv1)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_exp_squeeze(x):
            exp = topi.exp(x)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit_te(fused_exp_squeeze, x)
                lv2 = bb.call_te(topi.add, lv, relax.const(1, "float32"))
                gv = bb.emit_output(lv2)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_const_in_argument():
    def before():
        bb = relax.BlockBuilder()
        x1 = relax.Var("x1", R.Tensor([10, 20], "float32"))
        x2 = relax.Var("x2", R.Tensor([], "float32"))
        with bb.function("fused_add_exp_squeeze", [x1, x2], attrs={"Primitive": True}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x1, x2)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_add_exp_squeeze")
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.emit(relax.Call(func_gv, [x, relax.const(1, "float32")]))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_add_exp_squeeze(x, y):
            add = topi.add(x, y)
            exp = topi.exp(add)
            squeeze = topi.squeeze(exp)
            return squeeze

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv = bb.call_te(fused_add_exp_squeeze, x, relax.const(1, "float32"))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_tuple_output():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))

        with bb.function("fused_add_exp", [x, p0], attrs={"Primitive": True}):
            with bb.dataflow():
                gv0 = bb.emit_output(bb.call_te(topi.add, x, p0))
                gv1 = bb.emit_output(bb.call_te(topi.exp, gv0))
            bb.emit_func_output(relax.Tuple([gv0, gv1]))
        fused_add_exp = bb.get().get_global_var("fused_add_exp")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add_exp, [x, p0]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        def fused_add_exp(x, p0):
            add = topi.add(x, p0)
            exp = topi.exp(add)
            return add, exp

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor([], "float32"))
        with bb.function("main", [x, p0]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_add_exp, x, p0))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_with_immediate_tuple():
    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        y = relax.Var("y", R.Tensor([10, 20], "float32"))

        with bb.function("fused_add", [x, y], attrs={"Primitive": True}):
            with bb.dataflow():
                lv_tuple = bb.emit(relax.Tuple([x, relax.Tuple([x, y])]))
                lv_x = bb.emit(relax.TupleGetItem(lv_tuple, 0))
                lv0 = bb.emit(relax.TupleGetItem(lv_tuple, 1))
                lv_y = bb.emit(relax.TupleGetItem(lv0, 1))
                gv = bb.emit_output(bb.call_te(topi.add, lv_x, lv_y))
            bb.emit_func_output(gv)
        fused_add = bb.get().get_global_var("fused_add")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        y = relax.Var("y", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x, y]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_add, [x, y]))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        y = relax.Var("y", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x, y]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(topi.add, x, y, primfunc_name_hint="fused_add"))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_fuse_return_partial_result():
    def te_argmax_idx_val(val):
        from tvm import te

        def f_combine(x, y):
            lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
            rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
            return lhs, rhs

        def f_identity(dtype0: tvm.DataType, dtype1: tvm.DataType):
            return tvm.tir.const(-1, dtype0), tvm.te.min_value(dtype1)

        argmax = te.comm_reducer(f_combine, f_identity, name="argmax")
        m, n = val.shape
        k = te.reduce_axis((0, n), "k")
        max_idx, max_val = te.compute(
            (m,), lambda i: argmax((k.var, val[i, k]), axis=k), name="argmax"
        )
        return max_idx, max_val

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        offset = relax.Var("offset", R.Tensor([10], "int32"))
        with bb.function("fused_argmax_add", [x, offset], attrs={"Primitive": True}):
            with bb.dataflow():
                lv = bb.emit_te(te_argmax_idx_val, x)
                idx = bb.emit(relax.TupleGetItem(lv, 0))
                gv = bb.emit_output(bb.call_te(topi.add, idx, offset))
            bb.emit_func_output(gv)
        mod = bb.get()

        func_gv = mod.get_global_var("fused_argmax_add")
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        offset = relax.Var("x", R.Tensor([10], "int32"))
        with bb.function("main", [x, offset]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(func_gv, [x, offset]))
            bb.emit_func_output(gv)
        return bb.get()

    def expected():
        def fused_argmax_add(x, offset):
            idx, value = te_argmax_idx_val(x)
            idx = topi.add(idx, offset)
            return idx

        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        offset = relax.Var("offset", R.Tensor([10], "int32"))
        with bb.function("main", [x, offset]):
            with bb.dataflow():
                gv = bb.emit_output(bb.call_te(fused_argmax_add, x, offset))
            bb.emit_func_output(gv)
        return bb.get()

    _check(before(), expected())


def test_multiple_relax_functions():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": 1}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", R.Tensor([20, 10], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function("fused_add1_exp1_squeeze1", [x, p0], attrs={"Primitive": 1}, private=True):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add1_exp1_squeeze1 = bb.get().get_global_var("fused_add1_exp1_squeeze1")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("func1", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add_exp_squeeze, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        x = relax.Var("x", R.Tensor([20, 10], "float32"))
        with bb.function("func2", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add1_exp1_squeeze1, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        return bb.get()

    @I.ir_module
    class Expected:
        @R.function
        def func1(x: R.Tensor((10, 20), dtype="float32")) -> R.Tensor((10, 20), dtype="float32"):
            with R.dataflow():
                gv2 = R.call_tir(
                    Expected.fused_add_exp_squeeze,
                    (x, R.const(1, "float32")),
                    out_sinfo=R.Tensor((10, 20), dtype="float32"),
                )
                R.output(gv2)
            return gv2

        @R.function
        def func2(x: R.Tensor((20, 10), dtype="float32")) -> R.Tensor((20, 10), dtype="float32"):
            with R.dataflow():
                gv3 = R.call_tir(
                    Expected.fused_add1_exp1_squeeze1,
                    (x, R.const(1, "float32")),
                    out_sinfo=R.Tensor((20, 10), dtype="float32"),
                )
                R.output(gv3)
            return gv3

        @T.prim_func(private=True)
        def fused_add1_exp1_squeeze1(
            x: T.Buffer((T.int64(20), T.int64(10)), "float32"),
            p0: T.Buffer((), "float32"),
            T_squeeze: T.Buffer((T.int64(20), T.int64(10)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            T_add = T.alloc_buffer((T.int64(20), T.int64(10)))
            compute = T.alloc_buffer((T.int64(20), T.int64(10)))
            for ax0, ax1 in T.grid(T.int64(20), T.int64(10)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(x[v_ax0, v_ax1], p0[()])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + p0[()]
            for i0, i1 in T.grid(T.int64(20), T.int64(10)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_add[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.exp(T_add[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(20), T.int64(10)):
                with T.block("T_squeeze"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(compute[v_ax0, v_ax1])
                    T.writes(T_squeeze[v_ax0, v_ax1])
                    T_squeeze[v_ax0, v_ax1] = compute[v_ax0, v_ax1]

        @T.prim_func(private=True)
        def fused_add_exp_squeeze(
            x: T.Buffer((T.int64(10), T.int64(20)), "float32"),
            p0: T.Buffer((), "float32"),
            T_squeeze: T.Buffer((T.int64(10), T.int64(20)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            T_add = T.alloc_buffer((T.int64(10), T.int64(20)))
            compute = T.alloc_buffer((T.int64(10), T.int64(20)))
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(x[v_ax0, v_ax1], p0[()])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = x[v_ax0, v_ax1] + p0[()]
            for i0, i1 in T.grid(T.int64(10), T.int64(20)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_add[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.exp(T_add[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(10), T.int64(20)):
                with T.block("T_squeeze"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(compute[v_ax0, v_ax1])
                    T.writes(T_squeeze[v_ax0, v_ax1])
                    T_squeeze[v_ax0, v_ax1] = compute[v_ax0, v_ax1]

    _check(before(), Expected)


def test_skip_call_dps_packed():
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                y = R.call_dps_packed("func_packed_dps", x, R.Tensor((2, 3), "float32"))
                R.output(y)
            return y

    # FuseTIR should do no change to it.
    _check(Module, Module)


def test_symbolic_shape_aware_fuse():
    @I.ir_module
    class Before:
        @R.function
        def fused_add_exp_squeeze(
            x: R.Tensor(["n", "m"], "float32"), p0: R.Tensor([], "float32")
        ) -> R.Tensor(["n", "m"], dtype="float32"):
            R.func_attr({"Primitive": 1})
            with R.dataflow():
                lv0 = R.emit_te(topi.add, x, p0)
                lv1 = R.emit_te(topi.exp, lv0)
                gv = R.emit_te(topi.squeeze, lv1)
                R.output(gv)
            return gv

        @R.function
        def main(x: R.Tensor(["n", "m"], "float32")) -> R.Tensor(["n", "m"], dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused_add_exp_squeeze(x, R.const(1, "float32"))
                R.output(gv)
            return gv

    def fused_add_exp_squeeze(x, p0):
        return topi.squeeze(topi.exp(topi.add(x, p0)))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(["n", "m"], "float32")) -> R.Tensor(["n", "m"], dtype="float32"):
            with R.dataflow():
                gv = R.emit_te(fused_add_exp_squeeze, x, R.const(1, "float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_fuse_of_dynamic_kernel_with_var_params_and_static_args():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def dynamic_tir_kernel(a: T.handle, b: T.handle):
            m = T.int64()
            n = T.int64()
            A = T.match_buffer(a, [m, n], "float32")
            B = T.match_buffer(b, [m, n], "float32")

            for iters in T.grid(m, n):
                with T.block("compute"):
                    i, j = T.axis.remap("SS", iters)
                    B[i, j] = A[i, j] * i + j

        @R.function(private=True)
        def fused_function(x: R.Tensor([16, 32], "float32")) -> R.Tensor([16, 32], dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                y = R.call_tir(cls.dynamic_tir_kernel, [x], out_sinfo=R.Tensor([16, 32], "float32"))
                z = R.call_tir(cls.dynamic_tir_kernel, [y], out_sinfo=R.Tensor([16, 32], "float32"))
                R.output(z)
            return z

        @R.function
        def main(x: R.Tensor([16, 32], "float32")) -> R.Tensor([16, 32], dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused_function(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_function(
            X: T.Buffer([T.int64(16), T.int64(32)], "float32"),
            Z: T.Buffer([T.int64(16), T.int64(32)], "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            Y = T.alloc_buffer(X.shape, "float32")
            for iters in T.grid(*X.shape):
                with T.block("compute_Y"):
                    i, j = T.axis.remap("SS", iters)
                    Y[i, j] = X[i, j] * i + j

            for iters in T.grid(*X.shape):
                with T.block("compute_Z"):
                    i, j = T.axis.remap("SS", iters)
                    Z[i, j] = Y[i, j] * i + j

        @R.function
        def main(x: R.Tensor([16, 32], "float32")) -> R.Tensor([16, 32], dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.fused_function, [x], out_sinfo=R.Tensor([16, 32], "float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_fuse_of_dynamic_kernel_with_expression_params_and_static_args():
    """Parameters and arguments do not need to match structurally

    Here, the kernel requires arguments (m*n), and is provided
    """

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def dynamic_tir_kernel(a: T.handle, b: T.handle, c: T.handle, d: T.handle):
            m = T.int64()
            n = T.int64()
            A = T.match_buffer(a, [m * n], "float32")
            B = T.match_buffer(b, [m], "float32")
            C = T.match_buffer(c, [n], "float32")
            D = T.match_buffer(d, [m * n], "float32")

            for i, j in T.grid(m, n):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    D[vi * 32 + vj] = A[vi * 32 + vj] * B[vi] + C[vj]

        @R.function(private=True)
        def fused_function(
            x: R.Tensor([16 * 32], "float32"),
            B: R.Tensor([16], "float32"),
            C: R.Tensor([32], "float32"),
        ):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                y = R.call_tir(
                    cls.dynamic_tir_kernel, [x, B, C], out_sinfo=R.Tensor([16 * 32], "float32")
                )
                z = R.call_tir(
                    cls.dynamic_tir_kernel, [y, B, C], out_sinfo=R.Tensor([16 * 32], "float32")
                )
                R.output(z)
            return z

        @R.function
        def main(
            x: R.Tensor([16 * 32], "float32"),
            B: R.Tensor([16], "float32"),
            C: R.Tensor([32], "float32"),
        ):
            cls = Before
            with R.dataflow():
                gv = cls.fused_function(x, B, C)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_function(
            X: T.Buffer(T.int64(512), "float32"),
            B: T.Buffer(T.int64(16), "float32"),
            C: T.Buffer(T.int64(32), "float32"),
            Z: T.Buffer(T.int64(512), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            Y = T.alloc_buffer((T.int64(512),))
            for i, j in T.grid(T.int64(16), T.int64(32)):
                with T.block("compute"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Y[vi * 32 + vj] = X[vi * 32 + vj] * B[vi] + C[vj]

            for i, j in T.grid(T.int64(16), T.int64(32)):
                with T.block("compute_1"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Z[vi * 32 + vj] = Y[vi * 32 + vj] * B[vi] + C[vj]

        @R.function
        def main(
            x: R.Tensor((512,), dtype="float32"),
            B: R.Tensor((16,), dtype="float32"),
            C: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((512,), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_function, (x, B, C), out_sinfo=R.Tensor((512,), dtype="float32")
                )
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_shape_aware_fuse_with_allocation():
    def te_mean(x, axis):
        return topi.divide(topi.sum(x, axis, keepdims=True), 4096)

    @I.ir_module
    class Before:
        @R.function
        def fused_mean_add_tir_sqrt_divide_multiply(
            x: R.Tensor((1, "n", 4096), dtype="float32"),
            y: R.Tensor((1, "n", 4096), dtype="float32"),
            rms_norm_weight: R.Tensor((4096,), dtype="float32"),
        ) -> R.Tensor((1, "n", 4096), dtype="float32"):
            R.func_attr({"Primitive": 1})
            with R.dataflow():
                lv0 = R.emit_te(te_mean, x, axis=2)
                lv1 = R.emit_te(topi.add, lv0, lv0)
                lv2 = R.emit_te(topi.sqrt, lv1)
                lv3 = R.emit_te(topi.divide, y, lv2)
                gv = R.emit_te(topi.multiply, rms_norm_weight, lv3)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, "n", 4096), dtype="float32"),
            y: R.Tensor((1, "n", 4096), dtype="float32"),
            rms_norm_weight: R.Tensor((4096,), dtype="float32"),
        ) -> R.Tensor((1, "n", 4096), dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused_mean_add_tir_sqrt_divide_multiply(x, y, rms_norm_weight)
                R.output(gv)
            return gv

    def fused_mean_add_tir_sqrt_divide_multiply(x, y, rms_norm_weight):
        lv0 = te_mean(x, axis=2)
        lv1 = topi.add(lv0, lv0)
        lv2 = topi.sqrt(lv1)
        lv3 = topi.divide(y, lv2)
        return topi.multiply(rms_norm_weight, lv3)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, "n", 4096), dtype="float32"),
            y: R.Tensor((1, "n", 4096), dtype="float32"),
            rms_norm_weight: R.Tensor((4096,), dtype="float32"),
        ) -> R.Tensor((1, "n", 4096), dtype="float32"):
            with R.dataflow():
                gv = R.emit_te(fused_mean_add_tir_sqrt_divide_multiply, x, y, rms_norm_weight)
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_var_in_call_tir_args():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def foo(
            X: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"),
            Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"),
            rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"),
            m: T.int64,
        ):
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
                with T.block("rotary"):
                    v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    rotary[v0, v1, v2, v3] = Y[m + v1 - 1, v3] * X[v0, v1, v2, v3]

        @R.function
        def fused(
            x: R.Tensor((1, 1, 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, 1, 32, 128), dtype="float32"):
            R.func_attr({"Primitive": 1})
            m = T.int64()
            cls = Before
            with R.dataflow():
                lv1 = R.emit_te(topi.add, x, x)
                gv = R.call_tir(
                    cls.foo,
                    [lv1, y],
                    out_sinfo=R.Tensor((1, 1, 32, 128), dtype="float32"),
                    tir_vars=R.shape([m]),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, 1, 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, 1, 32, 128), dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused(x, y, len)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused(
            X: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"),
            Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"),
            rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float32"),
            m: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            T_add = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        X[v_ax0, v_ax1, v_ax2, v_ax3] + X[v_ax0, v_ax1, v_ax2, v_ax3]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
                with T.block("rotary"):
                    v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    rotary[v0, v1, v2, v3] = Y[m + v1 - T.int64(1), v3] * T_add[v0, v1, v2, v3]

        @R.function
        def main(
            x: R.Tensor((1, 1, 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, 1, 32, 128), dtype="float32"):
            m = T.int64()
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused,
                    (x, y),
                    out_sinfo=R.Tensor([1, 1, 32, 128], "float32"),
                    tir_vars=R.shape([m]),
                )
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_same_buffer_multiple_read():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def concatenate(
            rxplaceholder: T.Buffer((T.int64(1), T.int64(4), T.int64(64), T.int64(64)), "float32"),
            rxplaceholder_1: T.Buffer(
                (T.int64(1), T.int64(4), T.int64(64), T.int64(64)), "float32"
            ),
            T_concat: T.Buffer((T.int64(2), T.int64(4), T.int64(64), T.int64(64)), "float32"),
        ):
            T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(64), T.int64(64)):
                with T.block("T_concat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        rxplaceholder_1[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3],
                        rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3],
                    )
                    T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(
                        T.int64(1) <= v_ax0,
                        rxplaceholder_1[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3],
                        rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3],
                    )

        @T.prim_func(private=True)
        def transpose2(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(64), T.int64(64)), "float32"),
            T_transpose: T.Buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(4)), "float32"),
        ):
            T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(4)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax3, v_ax1, v_ax2])
                    T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[
                        v_ax0, v_ax3, v_ax1, v_ax2
                    ]

        @R.function
        def fused_concatenate_transpose2(
            inp_0: R.Tensor((1, 4, 64, 64), dtype="float32")
        ) -> R.Tensor((2, 64, 64, 4), dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.concatenate,
                    (inp_0, inp_0),
                    out_sinfo=R.Tensor((2, 4, 64, 64), dtype="float32"),
                )
                gv = R.call_tir(
                    cls.transpose2, (lv,), out_sinfo=R.Tensor((2, 64, 64, 4), dtype="float32")
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            inp_0: R.Tensor((1, 4, 64, 64), dtype="float32")
        ) -> R.Tensor((2, 64, 64, 4), dtype="float32"):
            R.func_attr({"num_input": 3})
            cls = Module
            with R.dataflow():
                lv = cls.fused_concatenate_transpose2(inp_0)
                R.output(lv)
            return lv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_concatenate_transpose2(
            inp_0: T.Buffer((T.int64(1), T.int64(4), T.int64(64), T.int64(64)), "float32"),
            T_transpose_handle_intermediate: T.Buffer(
                (T.int64(2), T.int64(64), T.int64(64), T.int64(4)), "float32"
            ),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            T_concat_handle_intermediate = T.alloc_buffer(
                (T.int64(2), T.int64(4), T.int64(64), T.int64(64))
            )
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(64), T.int64(64)):
                with T.block("T_concat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(inp_0[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3])
                    T.writes(T_concat_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_concat_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(
                        T.int64(1) <= v_ax0,
                        inp_0[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3],
                        inp_0[v_ax0, v_ax1, v_ax2, v_ax3],
                    )
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(4)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_concat_handle_intermediate[v_ax0, v_ax3, v_ax1, v_ax2])
                    T.writes(T_transpose_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_handle_intermediate[
                        v_ax0, v_ax1, v_ax2, v_ax3
                    ] = T_concat_handle_intermediate[v_ax0, v_ax3, v_ax1, v_ax2]

        @R.function
        def main(
            inp_0: R.Tensor((1, 4, 64, 64), dtype="float32")
        ) -> R.Tensor((2, 64, 64, 4), dtype="float32"):
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(
                    cls.fused_concatenate_transpose2,
                    (inp_0,),
                    out_sinfo=R.Tensor((2, 64, 64, 4), dtype="float32"),
                )
                R.output(lv)
            return lv

    _check(Module, Expected)


def test_tir_expression_in_shape():
    @I.ir_module
    class Module:
        @R.function
        def fused_transpose_matmul(
            x: R.Tensor((3, 4), dtype="float32"),
            y: R.Tensor(("n - 1", 4), dtype="float32"),
            tir_vars: R.Shape(["n"]),
        ) -> R.Tensor(("n - 1", 3), dtype="float32"):
            R.func_attr({"Primitive": 1})
            with R.dataflow():
                lv = R.emit_te(topi.transpose, x)
                gv = R.emit_te(topi.matmul, y, lv)
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
            y: R.Tensor(("n - 1", 4), dtype="float32"),
            tir_vars: R.Shape(["n"]),
        ) -> R.Tensor(("n - 1", 3), dtype="float32"):
            cls = Module
            with R.dataflow():
                lv = cls.fused_transpose_matmul(x, y, tir_vars)
                R.output(lv)
            return lv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_transpose_matmul(
            x: T.Buffer((T.int64(3), T.int64(4)), "float32"),
            p_y: T.handle,
            p_output0: T.handle,
            n: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            y = T.match_buffer(p_y, (n - T.int64(1), T.int64(4)))
            var_T_matmul_intermediate = T.match_buffer(p_output0, (n - T.int64(1), T.int64(3)))
            var_T_transpose_intermediate = T.alloc_buffer((T.int64(4), T.int64(3)))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(3)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    var_T_transpose_intermediate[v_ax0, v_ax1] = x[v_ax1, v_ax0]
            for ax0, ax1, k in T.grid(n - T.int64(1), T.int64(3), T.int64(4)):
                with T.block("T_matmul"):
                    v_ax0, v_ax1, v_k = T.axis.remap("SSR", [ax0, ax1, k])
                    with T.init():
                        var_T_matmul_intermediate[v_ax0, v_ax1] = T.float32(0)
                    var_T_matmul_intermediate[v_ax0, v_ax1] = (
                        var_T_matmul_intermediate[v_ax0, v_ax1]
                        + y[v_ax0, v_k] * var_T_transpose_intermediate[v_k, v_ax1]
                    )

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
            y: R.Tensor(("n - 1", 4), dtype="float32"),
            tir_vars: R.Shape(["n"]),
        ) -> R.Tensor(("n - 1", 3), dtype="float32"):
            n = T.int64()
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(
                    cls.fused_transpose_matmul,
                    (x, y),
                    out_sinfo=R.Tensor((n - 1, 3), dtype="float32"),
                    tir_vars=R.shape([n]),
                )
                R.output(lv)
            return lv

    _check(Module, Expected)


def test_tuple_input_unused_field():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def reshape(
            A: T.Buffer((T.int64(4), T.int64(8), T.int64(2048)), "float32"),
            T_reshape: T.Buffer((T.int64(4), T.int64(8), T.int64(32), T.int64(64)), "float32"),
        ):
            T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(8), T.int64(32), T.int64(64)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        A[
                            (
                                ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1)
                                // T.int64(8)
                                + v_ax0
                            )
                            % T.int64(4),
                            ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8),
                            (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                        (
                            ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) // T.int64(8)
                            + v_ax0
                        )
                        % T.int64(4),
                        ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8),
                        (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048),
                    ]

        @R.function(private=True)
        def fused_reshape(
            lv: R.Tuple(
                R.Tensor((4, 8, 2048), dtype="float32"), R.Tensor((4, 8, 2048), dtype="float32")
            )
        ) -> R.Tensor((4, 8, 32, 64), dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Module
            with R.dataflow():
                lv1: R.Tensor((4, 8, 2048), dtype="float32") = lv[0]
                gv = R.call_tir(
                    cls.reshape, (lv1,), out_sinfo=R.Tensor((4, 8, 32, 64), dtype="float32")
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            tup: R.Tuple(
                R.Tensor((4, 8, 2048), dtype="float32"), R.Tensor((4, 8, 2048), dtype="float32")
            )
        ) -> R.Tensor((4, 8, 32, 64), dtype="float32"):
            cls = Module
            with R.dataflow():
                lv_1: R.Tensor((4, 8, 32, 64), dtype="float32") = cls.fused_reshape(tup)
                R.output(lv_1)
            return lv_1

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_reshape(
            lv_0: T.Buffer((T.int64(4), T.int64(8), T.int64(2048)), "float32"),
            T_reshape_handle_intermediate: T.Buffer(
                (T.int64(4), T.int64(8), T.int64(32), T.int64(64)), "float32"
            ),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(8), T.int64(32), T.int64(64)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        lv_0[
                            (
                                ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1)
                                // T.int64(8)
                                + v_ax0
                            )
                            % T.int64(4),
                            ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8),
                            (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048),
                        ]
                    )
                    T.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv_0[
                        (
                            ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) // T.int64(8)
                            + v_ax0
                        )
                        % T.int64(4),
                        ((v_ax2 * T.int64(64) + v_ax3) // T.int64(2048) + v_ax1) % T.int64(8),
                        (v_ax2 * T.int64(64) + v_ax3) % T.int64(2048),
                    ]

        @R.function
        def main(
            tup: R.Tuple(
                R.Tensor((4, 8, 2048), dtype="float32"), R.Tensor((4, 8, 2048), dtype="float32")
            )
        ) -> R.Tensor((4, 8, 32, 64), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv: R.Tensor((4, 8, 2048), dtype="float32") = tup[0]
                lv_1 = R.call_tir(
                    cls.fused_reshape, (lv,), out_sinfo=R.Tensor((4, 8, 32, 64), dtype="float32")
                )
                R.output(lv_1)
            return lv_1

    _check(Module, Expected)


def test_unique_duplicated_buffer_allocation():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def add(
            A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        ):
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Out[vi, vj] = A[vi, vj] + T.float16(1.0)

        @T.prim_func(private=True)
        def add1(
            A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        ):
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Out[vi, vj] = A[vi, vj] + T.float16(2.0)

        @R.function
        def main(
            input_embeds: R.Tensor((4096, 4096), dtype="float16"),
        ) -> R.Tensor((4096, 4096), dtype="float16"):
            cls = Module
            with R.dataflow():
                gv: R.Tensor((4096, 4096), dtype="float16") = cls.fused_func(input_embeds)
                R.output(gv)
            return gv

        @R.function(private=True)
        def fused_func(
            input_embeds: R.Tensor((4096, 4096), dtype="float16"),
        ) -> R.Tensor((4096, 4096), dtype="float16"):
            R.func_attr({"Primitive": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.add, (input_embeds,), out_sinfo=R.Tensor((4096, 4096), dtype="float16")
                )
                gv = R.call_tir(cls.add1, (lv,), out_sinfo=R.Tensor((4096, 4096), dtype="float16"))
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused_func(
            input_embeds: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            Out_intermediate_1: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            Out_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(input_embeds[vi, vj])
                    T.writes(Out_intermediate[vi, vj])
                    Out_intermediate[vi, vj] = input_embeds[vi, vj] + T.float16(1)
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add_1"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.reads(Out_intermediate[vi, vj])
                    T.writes(Out_intermediate_1[vi, vj])
                    Out_intermediate_1[vi, vj] = Out_intermediate[vi, vj] + T.float16(2)

        @R.function
        def main(
            input_embeds: R.Tensor((4096, 4096), dtype="float16")
        ) -> R.Tensor((4096, 4096), dtype="float16"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_func,
                    (input_embeds,),
                    out_sinfo=R.Tensor((4096, 4096), dtype="float16"),
                )
                R.output(gv)
            return gv

    _check(Module, Expected)


def test_extern_func():
    bb = relax.BlockBuilder()
    bb.add_func(relax.extern("extern_func"), "extern_func")
    mod = bb.get()
    # FuseTIR should keep the ExternFunc in the IRModule.
    _check(mod, mod)


def test_symbolic_var_in_buffer_shape():
    """A PrimFunc may have dynamic buffer shapes

    Symbolic variables in a PrimFunc may be present in the buffer
    shape without a corresponding parameter.  These symbolic variables
    are inferred from the buffer's shape.  (Or, at runtime, they are
    typically determined from the DLTensor's known shape.)
    """

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def foo(
            X_handle: T.handle,
            Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"),
            rotary_handle: T.handle,
            m: T.int64,
        ):
            sequence_length = T.int64()

            X = T.match_buffer(
                X_handle, [T.int64(1), sequence_length, T.int64(32), T.int64(128)], "float32"
            )
            rotary = T.match_buffer(
                rotary_handle, [T.int64(1), sequence_length, T.int64(32), T.int64(128)], "float32"
            )

            for i0, i1, i2, i3 in T.grid(T.int64(1), sequence_length, T.int64(32), T.int64(128)):
                with T.block("rotary"):
                    v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    rotary[v0, v1, v2, v3] = Y[m + v1 - 1, v3] * X[v0, v1, v2, v3]

        @R.function
        def fused(
            x: R.Tensor((1, "sequence_length", 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, "sequence_length", 32, 128), dtype="float32"):
            R.func_attr({"Primitive": 1})
            sequence_length = T.int64()
            m = T.int64()
            cls = Before
            with R.dataflow():
                lv1 = R.emit_te(topi.add, x, x)
                gv = R.call_tir(
                    cls.foo,
                    [lv1, y],
                    out_sinfo=R.Tensor((1, sequence_length, 32, 128), dtype="float32"),
                    tir_vars=R.shape([m]),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((1, "sequence_length", 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, "sequence_length", 32, 128), dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused(x, y, len)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused(
            X_handle: T.handle,
            Y: T.Buffer((T.int64(2048), T.int64(128)), "float32"),
            rotary_handle: T.handle,
            m: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})

            sequence_length = T.int64()

            X = T.match_buffer(
                X_handle, [T.int64(1), sequence_length, T.int64(32), T.int64(128)], "float32"
            )
            rotary = T.match_buffer(
                rotary_handle, [T.int64(1), sequence_length, T.int64(32), T.int64(128)], "float32"
            )

            T_add = T.alloc_buffer((T.int64(1), sequence_length, T.int64(32), T.int64(128)))
            for ax0, ax1, ax2, ax3 in T.grid(
                T.int64(1), sequence_length, T.int64(32), T.int64(128)
            ):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        X[v_ax0, v_ax1, v_ax2, v_ax3] + X[v_ax0, v_ax1, v_ax2, v_ax3]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), sequence_length, T.int64(32), T.int64(128)):
                with T.block("rotary"):
                    v0, v1, v2, v3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    rotary[v0, v1, v2, v3] = Y[m + v1 - T.int64(1), v3] * T_add[v0, v1, v2, v3]

        @R.function
        def main(
            x: R.Tensor((1, "sequence_length", 32, 128), dtype="float32"),
            y: R.Tensor((2048, 128), dtype="float32"),
            len: R.Shape(["m"]),
        ) -> R.Tensor((1, "sequence_length", 32, 128), dtype="float32"):
            sequence_length = T.int64()
            m = T.int64()
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused,
                    (x, y),
                    out_sinfo=R.Tensor([1, sequence_length, 32, 128], "float32"),
                    tir_vars=R.shape([m]),
                )
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_var_called_with_static_shape():
    """A dynamic PrimFunc may be called with a static shape"""

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def sum_1d(
            X_handle: T.handle,
            Y: T.Buffer([T.int64(1)], "float32"),
        ):
            num_elements = T.int64()

            X = T.match_buffer(X_handle, [num_elements], "float32")

            for i in range(num_elements):
                with T.block("sum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        Y[0] = 0.0
                    Y[0] = Y[0] + X[vi]

        @R.function(private=True)
        def fused(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.sum_1d,
                    [x],
                    out_sinfo=R.Tensor([1], dtype="float32"),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused(
            X: T.Buffer([T.int64(64)], "float32"),
            Y: T.Buffer([T.int64(1)], "float32"),
        ):
            T.func_attr({"tir.noalias": True})

            for i in range(T.int64(64)):
                with T.block("sum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        Y[0] = 0.0
                    Y[0] = Y[0] + X[vi]

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.fused, (x,), out_sinfo=R.Tensor((1,), dtype="float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_var_called_with_multiple_static_shapes():
    """A dynamic PrimFunc may be called with different shapes each time"""

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def sum_1d(
            X_handle: T.handle,
            Sum: T.Buffer([T.int64(1)], "float32"),
        ):
            num_elements = T.int64()

            X = T.match_buffer(X_handle, [num_elements], "float32")

            for i in range(num_elements):
                with T.block("sum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        Sum[0] = 0.0
                    Sum[0] = Sum[0] + X[vi]

        @T.prim_func(private=True)
        def sum_scalar(
            X: T.Buffer([T.int64(1)], "float32"),
            Y: T.Buffer([T.int64(1)], "float32"),
            Sum: T.Buffer([T.int64(1)], "float32"),
        ):
            for i in range(T.int64(1)):
                with T.block("Out"):
                    vi = T.axis.remap("S", [i])
                    Sum[vi] = X[vi] + Y[vi]

        @R.function(private=True)
        def fused(
            x: R.Tensor([64], dtype="float32"),
            y: R.Tensor([16], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                x_sum = R.call_tir(
                    cls.sum_1d,
                    [x],
                    out_sinfo=R.Tensor([1], dtype="float32"),
                )
                y_sum = R.call_tir(
                    cls.sum_1d,
                    [y],
                    out_sinfo=R.Tensor([1], dtype="float32"),
                )
                gv = R.call_tir(
                    cls.sum_scalar,
                    [x_sum, y_sum],
                    out_sinfo=R.Tensor([1], dtype="float32"),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
            y: R.Tensor([16], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused(x, y)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused(
            X: T.Buffer([T.int64(64)], "float32"),
            Y: T.Buffer([T.int64(16)], "float32"),
            Out: T.Buffer([T.int64(1)], "float32"),
        ):
            T.func_attr({"tir.noalias": True})

            XSum = T.alloc_buffer([T.int64(1)], "float32")
            YSum = T.alloc_buffer([T.int64(1)], "float32")

            for i in range(T.int64(64)):
                with T.block("XSum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        XSum[0] = 0.0
                    XSum[0] = XSum[0] + X[vi]

            for i in range(T.int64(16)):
                with T.block("YSum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        YSum[0] = 0.0
                    YSum[0] = YSum[0] + Y[vi]

            for i in range(T.int64(1)):
                with T.block("Out"):
                    vi = T.axis.remap("S", [i])
                    Out[vi] = XSum[vi] + YSum[vi]

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
            y: R.Tensor([16], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.fused, (x, y), out_sinfo=R.Tensor((1,), dtype="float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_symbolic_var_called_with_static_argument():
    """A dynamic PrimFunc may accept a static argument

    The `tir_vars` parameter in `R.call_tir` contains definitions for
    all TIR variables explicitly listed in the function signature, and
    contains the TIR expression to be passed as the argument for for
    each parameter.

    This test is identical to the earlier test named
    "test_symbolic_var_called_with_static_shape", except for the
    explicit parameter in `sum_1d`.
    """

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def sum_1d(
            X_handle: T.handle,
            Y: T.Buffer([T.int64(1)], "float32"),
            num_elements: T.int64,
        ):
            X = T.match_buffer(X_handle, [num_elements], "float32")

            for i in range(num_elements):
                with T.block("sum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        Y[0] = 0.0
                    Y[0] = Y[0] + X[vi]

        @R.function(private=True)
        def fused(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.sum_1d,
                    [x],
                    out_sinfo=R.Tensor([1], dtype="float32"),
                    tir_vars=R.shape([64]),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Before
            with R.dataflow():
                gv = cls.fused(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def fused(
            X: T.Buffer([T.int64(64)], "float32"),
            Y: T.Buffer([T.int64(1)], "float32"),
        ):
            T.func_attr({"tir.noalias": True})

            for i in range(T.int64(64)):
                with T.block("sum"):
                    vi = T.axis.remap("R", [i])
                    with T.init():
                        Y[0] = 0.0
                    Y[0] = Y[0] + X[vi]

        @R.function
        def main(
            x: R.Tensor([64], dtype="float32"),
        ) -> R.Tensor([1], dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.fused, (x,), out_sinfo=R.Tensor((1,), dtype="float32"))
                R.output(gv)
            return gv

    _check(Before, Expected)


def test_gather():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def add(
            A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        ):
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Out[vi, vj] = A[vi, vj] + T.float16(1.0)

        @T.prim_func(private=True)
        def take(
            A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            B: T.Buffer((T.int64(1),), "int32"),
            T_take: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
        ):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(4096)):
                with T.block("T_take"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

        @R.function
        def main(
            input_ids: R.Tensor((1,), dtype="int32"),
            input_embeds: R.Tensor((4096, 4096), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            cls = Before
            with R.dataflow():
                gv: R.Tensor((1, 4096), dtype="float16") = cls.fused_func(input_ids, input_embeds)
                R.output(gv)
            return gv

        @R.function(private=True)
        def fused_func(
            input_ids: R.Tensor((1,), dtype="int32"),
            input_embeds: R.Tensor((4096, 4096), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            R.func_attr({"Primitive": 1})
            cls = Before
            with R.dataflow():
                lv = R.call_tir(
                    cls.add, (input_embeds,), out_sinfo=R.Tensor((4096, 4096), dtype="float16")
                )
                gv = R.call_tir(
                    cls.take, (lv, input_ids), out_sinfo=R.Tensor((1, 4096), dtype="float16")
                )
                R.output(gv)
            return gv

    @I.ir_module
    class After:
        @T.prim_func(private=True)
        def fused_func(
            input_ids: T.Buffer((T.int64(1),), "int32"),
            input_embeds: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
            T_take: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            Out_handle_intermediate = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
            for i, j in T.grid(T.int64(4096), T.int64(4096)):
                with T.block("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    Out_handle_intermediate[vi, vj] = input_embeds[vi, vj] + T.float16(1)
            for ax0, ax1 in T.grid(T.int64(1), T.int64(4096)):
                with T.block("T_take"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T_take[v_ax0, v_ax1] = Out_handle_intermediate[input_ids[v_ax0], v_ax1]

        @R.function
        def main(
            input_ids: R.Tensor((1,), dtype="int32"),
            input_embeds: R.Tensor((4096, 4096), dtype="float16"),
        ) -> R.Tensor((1, 4096), dtype="float16"):
            cls = After
            with R.dataflow():
                gv = R.call_tir(
                    cls.fused_func,
                    (input_ids, input_embeds),
                    out_sinfo=R.Tensor((1, 4096), dtype="float16"),
                )
                R.output(gv)
            return gv

    _check(Before, After)


if __name__ == "__main__":
    tvm.testing.main()
