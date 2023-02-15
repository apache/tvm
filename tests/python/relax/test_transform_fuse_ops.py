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
from tvm.script import relax as R


def _check(mod_actual, mod_expected):
    mod_actual = relax.transform.AnnotateTIROpPattern()(mod_actual)
    mod_actual = relax.transform.FuseOps()(mod_actual)
    mod_expected = relax.transform.AnnotateTIROpPattern()(mod_expected)
    tvm.ir.assert_structural_equal(mod_actual, mod_expected)


def test_fuse_simple():
    """Simple testcase."""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))

        with bb.function("fused_add_exp_squeeze", [x, p0], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.exp, lv0)
                gv = bb.emit_output(bb.call_te(topi.squeeze, lv1))
            bb.emit_func_output(gv)
        fused_add_exp_squeeze = bb.get().get_global_var("fused_add_exp_squeeze")

        x = relax.Var("x", R.Tensor([10, 20], "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(fused_add_exp_squeeze, [x, relax.const(1, "float32")])
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_conv2d_fuse():
    """Test fusion case of conv2d"""

    def before(dtype):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), dtype))
        w2 = relax.Var("w2", R.Tensor((16, 16, 1, 1), dtype))
        w3 = relax.Var("w3", R.Tensor((16, 16, 3, 3), dtype))
        with bb.function("main", [x, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, dtype))
                lv1 = bb.emit_te(topi.nn.conv2d, lv0, w1, strides=1, padding=1, dilation=1)
                # this is the next dominator.
                lv2 = bb.emit_te(topi.add, relax.const(1, dtype), lv1)
                lv3 = bb.emit_te(topi.add, lv1, lv2)
                # second path
                lv4 = bb.emit_te(topi.nn.conv2d, lv3, w2, strides=1, padding=0, dilation=1)
                lv5 = bb.emit_te(topi.nn.conv2d, lv3, w3, strides=1, padding=1, dilation=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv4, lv5))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dtype):
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), dtype))
        w = relax.Var("w", R.Tensor((16, 16, 3, 3), dtype))
        p0 = relax.Var("p0", R.Tensor((), dtype))
        with bb.function("fused_conv2d_add1_add2", [x, w, p0], attrs={"Primitive": 1}):
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
        with bb.function("fused_conv2d1_add2", [x, w, y], attrs={"Primitive": 1}):
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

    _check(before("float32"), expected("float32"))
    _check(before("float16"), expected("float16"))
    _check(before("int8"), expected("int8"))


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv1 = bb.emit_te(topi.nn.upsampling, lv0, scale_h=2.0, scale_w=2.0)
                lv2 = bb.emit_te(topi.concatenate, (lv1, x), axis=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, relax.const(1, "float32")))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((1, 16, 32, 32), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function("fused_upsampling_concatenate_add", [w, x, p0], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.upsampling, w, scale_h=2.0, scale_w=2.0)
                lv1 = bb.emit_te(topi.concatenate, (lv0, x), axis=1)
                gv = bb.emit_output(bb.call_te(topi.add, lv1, p0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_upsampling_concatenate_add = bb.get().get_global_var(
            "fused_upsampling_concatenate_add"
        )

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                gv = bb.emit_output(
                    relax.Call(
                        fused_upsampling_concatenate_add, (lv0, x, relax.const(1, "float32"))
                    )
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""

    def before():
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    x,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv1 = bb.emit_te(topi.nn.upsampling, lv0, scale_h=2.0, scale_w=2.0)
                gv = bb.emit_output((lv1, x))
            bb.emit_func_output(gv)

        return bb.get()

    # The fusion is supposed to make no change.
    _check(before(), before())


def test_fuse_tuple_get_elemwise():
    def before(dim: int):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, dim), "float32"))
        w = relax.Var("w", R.Tensor((3 * dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, w)
                lv1 = bb.emit_te(topi.split, lv0, indices_or_sections=3, axis=1)
                lv2 = bb.emit(relax.TupleGetItem(lv1, 0))
                lv3 = bb.emit_te(topi.sigmoid, lv2)
                lv4 = bb.emit(relax.TupleGetItem(lv1, 1))
                lv5 = bb.emit_te(topi.tanh, lv4)
                lv6 = bb.emit(relax.TupleGetItem(lv1, 2))
                lv7 = bb.emit_te(topi.exp, lv6)
                lv8 = bb.emit_te(topi.multiply, lv5, lv7)
                gv = bb.emit_output(bb.call_te(topi.add, lv3, lv8))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dim: int):
        bb = relax.BlockBuilder()

        # Grouped function
        dense = relax.Var("dense", R.Tensor((1, 3 * dim), "float32"))
        with bb.function(
            "fused_split_sigmoid_tanh_exp_multiply_add", [dense], attrs={"Primitive": 1}
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, dense, indices_or_sections=3, axis=1)
                lv1 = bb.emit(relax.TupleGetItem(lv0, 0))
                lv2 = bb.emit_te(topi.sigmoid, lv1)
                lv3 = bb.emit(relax.TupleGetItem(lv0, 1))
                lv4 = bb.emit_te(topi.tanh, lv3)
                lv5 = bb.emit(relax.TupleGetItem(lv0, 2))
                lv6 = bb.emit_te(topi.exp, lv5)
                lv7 = bb.emit_te(topi.multiply, lv4, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv2, lv7))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_split_sigmoid_tanh_exp_multiply_add = bb.get().get_global_var(
            "fused_split_sigmoid_tanh_exp_multiply_add"
        )

        # Main function
        x = relax.Var("x", R.Tensor((1, dim), "float32"))
        w = relax.Var("w", R.Tensor((3 * dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.dense, x, w)
                gv = bb.emit_output(relax.Call(fused_split_sigmoid_tanh_exp_multiply_add, (lv0,)))
            bb.emit_func_output(gv)

        return bb.get()

    dim = 10
    _check(before(dim), expected(dim))


def test_tuple_get_root():
    def before(dim: int):
        bb = relax.BlockBuilder()
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        w = relax.Var("w", R.Tensor((dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, x, indices_or_sections=3, axis=1)
                lv1 = bb.emit(relax.TupleGetItem(lv0, 0))
                gv = bb.emit_output(bb.call_te(topi.nn.dense, lv1, w))
            bb.emit_func_output(gv)

        return bb.get()

    def expected(dim: int):
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        with bb.function("fused_split", [x], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.split, x, indices_or_sections=3, axis=1)
                gv = bb.emit_output(relax.TupleGetItem(lv0, 0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_split = bb.get().get_global_var("fused_split")

        # Main function
        x = relax.Var("x", R.Tensor((1, 3 * dim), "float32"))
        w = relax.Var("w", R.Tensor((dim, dim), "float32"))
        with bb.function("main", [x, w]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(fused_split, (x,)))
                gv = bb.emit_output(bb.call_te(topi.nn.dense, lv0, w))
            bb.emit_func_output(gv)

        return bb.get()

    dim = 10
    _check(before(dim), expected(dim))


def test_tuple_intermediate():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.squeeze, x)
                lv1 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                lv2 = bb.emit_te(topi.squeeze, lv0)
                lv3 = bb.emit_te(topi.add, lv2, relax.const(1, "float32"))
                lv4 = bb.emit_te(topi.add, lv3, relax.const(1, "float32"))
                lv5 = bb.emit_te(topi.add, lv0, relax.const(1, "float32"))
                lv6 = bb.emit_te(topi.concatenate, (lv1, lv4, lv5), axis=1)
                lv7 = bb.emit_te(topi.squeeze, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv7, relax.const(1, "float32")))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        p1 = relax.Var("p1", R.Tensor((), "float32"))
        p2 = relax.Var("p2", R.Tensor((), "float32"))
        p3 = relax.Var("p3", R.Tensor((), "float32"))
        p4 = relax.Var("p4", R.Tensor((), "float32"))
        with bb.function(
            "fused_squeeze_add_squeeze1_add_add_add_concatenate_squeeze2_add1",
            [x, p0, p1, p2, p3, p4],
            attrs={"Primitive": 1},
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.squeeze, x)
                lv1 = bb.emit_te(topi.add, lv0, p0)
                lv2 = bb.emit_te(topi.squeeze, lv0)
                lv3 = bb.emit_te(topi.add, lv2, p1)
                lv4 = bb.emit_te(topi.add, lv3, p2)
                lv5 = bb.emit_te(topi.add, lv0, p3)
                lv6 = bb.emit_te(topi.concatenate, (lv1, lv4, lv5), axis=1)
                lv7 = bb.emit_te(topi.squeeze, lv6)
                gv = bb.emit_output(bb.call_te(topi.add, lv7, p4))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var(
            "fused_squeeze_add_squeeze1_add_add_add_concatenate_squeeze2_add1"
        )

        # Main func
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(
                    relax.Call(
                        fused_func,
                        (
                            x,
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                        ),
                    )
                )
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_tuple_consecutive():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv1 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv2 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv3 = bb.emit_te(topi.concatenate, (lv0, lv1, lv2), axis=1)
                lv4 = bb.emit_te(topi.add, lv3, relax.const(1, "float32"))
                lv5 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv6 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv7 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv8 = bb.emit_te(topi.concatenate, (lv5, lv6, lv7), axis=1)
                lv9 = bb.emit_te(topi.add, lv8, relax.const(1, "float32"))
                lv10 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv11 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv12 = bb.emit_te(topi.add, x, relax.const(1, "float32"))
                lv13 = bb.emit_te(topi.concatenate, (lv10, lv11, lv12), axis=1)
                lv14 = bb.emit_te(topi.add, lv13, relax.const(1, "float32"))
                lv15 = bb.emit_te(topi.concatenate, (lv4, lv9, lv14), axis=1)
                lv16 = bb.emit_te(
                    topi.nn.pool2d,
                    lv15,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                lv17 = bb.emit_te(topi.add, lv16, relax.const(1, "float32"))
                lv18 = bb.emit_te(topi.add, lv17, relax.const(1, "float32"))
                gv = bb.emit_output((lv17, lv18))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        p1 = relax.Var("p1", R.Tensor((), "float32"))
        p2 = relax.Var("p2", R.Tensor((), "float32"))
        p3 = relax.Var("p3", R.Tensor((), "float32"))
        p4 = relax.Var("p4", R.Tensor((), "float32"))
        p5 = relax.Var("p5", R.Tensor((), "float32"))
        p6 = relax.Var("p6", R.Tensor((), "float32"))
        p7 = relax.Var("p7", R.Tensor((), "float32"))
        p8 = relax.Var("p8", R.Tensor((), "float32"))
        p9 = relax.Var("p9", R.Tensor((), "float32"))
        p10 = relax.Var("p10", R.Tensor((), "float32"))
        p11 = relax.Var("p11", R.Tensor((), "float32"))
        with bb.function(
            "fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1",
            [x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11],
            attrs={"Primitive": 1},
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.add, x, p1)
                lv2 = bb.emit_te(topi.add, x, p2)
                lv3 = bb.emit_te(topi.concatenate, (lv0, lv1, lv2), axis=1)
                lv4 = bb.emit_te(topi.add, lv3, p3)
                lv5 = bb.emit_te(topi.add, x, p4)
                lv6 = bb.emit_te(topi.add, x, p5)
                lv7 = bb.emit_te(topi.add, x, p6)
                lv8 = bb.emit_te(topi.concatenate, (lv5, lv6, lv7), axis=1)
                lv9 = bb.emit_te(topi.add, lv8, p7)
                lv10 = bb.emit_te(topi.add, x, p8)
                lv11 = bb.emit_te(topi.add, x, p9)
                lv12 = bb.emit_te(topi.add, x, p10)
                lv13 = bb.emit_te(topi.concatenate, (lv10, lv11, lv12), axis=1)
                lv14 = bb.emit_te(topi.add, lv13, p11)
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv4, lv9, lv14), axis=1))
            bb.emit_func_output(gv)

        # Grouped function 2
        concat = relax.Var("concat", R.Tensor((1, 144, 64, 64), "float32"))
        p0 = relax.Var("p0", R.Tensor((), "float32"))
        with bb.function("fused_pool2d_add2", [concat, p0], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.pool2d,
                    concat,
                    kernel=(2, 2),
                    stride=(2, 2),
                    dilation=(1, 1),
                    padding=(0, 0, 0, 0),
                    pool_type="max",
                )
                gv = bb.emit_output(bb.call_te(topi.add, lv0, p0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_func1 = mod.get_global_var(
            "fused_add_add_add_concatenate_add1_add_add_add_concatenate_add1_add_add_add_concatenate_add1_concatenate1"
        )
        fused_func2 = mod.get_global_var("fused_pool2d_add2")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit(
                    relax.Call(
                        fused_func1,
                        (
                            x,
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                            relax.const(1, "float32"),
                        ),
                    )
                )
                lv1 = bb.emit(relax.Call(fused_func2, (lv0, relax.const(1, "float32"))))
                lv2 = bb.emit_te(topi.add, lv1, relax.const(1, "float32"))
                gv = bb.emit_output((lv1, lv2))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_inception_like():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w0 = relax.Var("w0", R.Tensor((16, 16, 3, 3), "float32"))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), "float32"))
        w2 = relax.Var("w2", R.Tensor((16, 32, 3, 3), "float32"))
        w3 = relax.Var("w3", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("main", [x, w0, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.conv2d, x, w0, strides=1, padding=1, dilation=1)
                lv1 = bb.emit_te(topi.nn.relu, lv0)
                lv2 = bb.emit_te(topi.nn.conv2d, x, w1, strides=1, padding=1, dilation=1)
                lv3 = bb.emit_te(topi.nn.relu, lv2)
                lv4 = bb.emit_te(topi.concatenate, (lv1, lv3), axis=1)
                lv5 = bb.emit_te(topi.nn.conv2d, lv4, w2, strides=1, padding=1, dilation=1)
                lv6 = bb.emit_te(topi.nn.relu, lv5)
                lv7 = bb.emit_te(topi.nn.conv2d, lv4, w3, strides=1, padding=1, dilation=1)
                lv8 = bb.emit_te(topi.nn.relu, lv7)
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv6, lv8), axis=1))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function 1
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((16, 16, 3, 3), "float32"))
        with bb.function("fused_conv2d_relu", [x, w], attrs={"Primitive": 1}):
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
                gv = bb.emit_output(bb.call_te(topi.nn.relu, lv0))
            bb.emit_func_output(gv)

        # Grouped function 2
        x = relax.Var("x", R.Tensor((1, 32, 64, 64), "float32"))
        w = relax.Var("w", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("fused_conv2d1_relu", [x, w], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(
                    topi.nn.conv2d,
                    x,
                    w,
                    strides=1,
                    padding=1,
                    dilation=1,
                    primfunc_name_hint="conv2d1",
                )
                gv = bb.emit_output(bb.call_te(topi.nn.relu, lv0))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        mod = bb.get()
        fused_conv2d_relu1 = mod.get_global_var("fused_conv2d_relu")
        fused_conv2d_relu2 = mod.get_global_var("fused_conv2d1_relu")

        # Main function
        x = relax.Var("x", R.Tensor((1, 16, 64, 64), "float32"))
        w0 = relax.Var("w0", R.Tensor((16, 16, 3, 3), "float32"))
        w1 = relax.Var("w1", R.Tensor((16, 16, 3, 3), "float32"))
        w2 = relax.Var("w2", R.Tensor((16, 32, 3, 3), "float32"))
        w3 = relax.Var("w3", R.Tensor((16, 32, 3, 3), "float32"))
        with bb.function("main", [x, w0, w1, w2, w3]):
            with bb.dataflow():
                lv0 = bb.emit(relax.Call(fused_conv2d_relu1, (x, w0)))
                lv1 = bb.emit(relax.Call(fused_conv2d_relu1, (x, w1)))
                lv2 = bb.emit_te(topi.concatenate, (lv0, lv1), axis=1)
                lv3 = bb.emit(relax.Call(fused_conv2d_relu2, (lv2, w2)))
                lv4 = bb.emit(relax.Call(fused_conv2d_relu2, (lv2, w3)))
                gv = bb.emit_output(bb.call_te(topi.concatenate, (lv3, lv4), axis=1))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_fuse_parallel_injective():
    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, relax.const(1, "int32"))
                lv1 = bb.emit_te(topi.squeeze, lv0)
                lv2 = bb.emit_te(topi.transpose, lv0, axes=[1, 0])
                lv3 = bb.emit_te(topi.transpose, lv2, axes=[1, 0])
                gv = bb.emit_output(bb.call_te(topi.left_shift, lv1, lv3))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        p0 = relax.Var("p0", R.Tensor((), "int32"))
        with bb.function(
            "fused_add_squeeze_transpose_transpose1_left_shift", [x, p0], attrs={"Primitive": 1}
        ):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.add, x, p0)
                lv1 = bb.emit_te(topi.squeeze, lv0)
                lv2 = bb.emit_te(topi.transpose, lv0, axes=[1, 0])
                lv3 = bb.emit_te(topi.transpose, lv2, axes=[1, 0], primfunc_name_hint="transpose1")
                gv = bb.emit_output(bb.call_te(topi.left_shift, lv1, lv3))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var("fused_add_squeeze_transpose_transpose1_left_shift")

        # Main function
        x = relax.Var("x", R.Tensor((10, 20), "int32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_func, (x, relax.const(1, "int32"))))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


def test_softmax():
    """Test if softmax can be fused with following ops."""

    def before():
        bb = relax.BlockBuilder()

        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.softmax, x)
                gv = bb.emit_output(bb.call_te(topi.cast, lv0, dtype="float16"))
            bb.emit_func_output(gv)

        return bb.get()

    def expected():
        bb = relax.BlockBuilder()

        # Grouped function
        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("fused_softmax_cast", [x], attrs={"Primitive": 1}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.nn.softmax, x)
                gv = bb.emit_output(bb.call_te(topi.cast, lv0, dtype="float16"))
            bb.emit_func_output(gv)

        # Get the global variables of the grouped functions
        fused_func = bb.get().get_global_var("fused_softmax_cast")

        # Main function
        x = relax.Var("x", R.Tensor((16, 16), "float32"))
        with bb.function("main", [x]):
            with bb.dataflow():
                gv = bb.emit_output(relax.Call(fused_func, (x,)))
            bb.emit_func_output(gv)

        return bb.get()

    _check(before(), expected())


if __name__ == "__main__":
    tvm.testing.main()
