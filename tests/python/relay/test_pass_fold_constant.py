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
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload


def annot_expr(e):
    """Returns e wrapped with an on_device annotation."""
    return relay.op.annotation.on_device(e, tvm.cpu(), constrain_result=True)


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_concatenate_const():
    def before():
        data = tvm.nd.array(np.array([1.0, 2.0, 3.0]))
        const = relay.const(data)
        concat = relay.op.concatenate([const, const], axis=0)
        func = relay.Function([], concat)
        return func

    def expected():
        data = tvm.nd.array(np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))
        const = relay.const(data)
        func = relay.Function([], const)
        return func

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_const():
    c_data = np.array([1, 2, 3]).astype("float32")
    t = relay.TensorType([1, 2, 3], "float32")

    def before():
        c = relay.const(c_data)
        x = relay.var("x", t)
        y = relay.add(c, c)
        y = relay.multiply(y, relay.const(2, "float32"))
        y = relay.add(x, y)
        z = relay.add(y, c)
        return relay.Function([x], z)

    def expected():
        x = relay.var("x", t)
        c_folded = (c_data + c_data) * 2
        y = relay.add(x, relay.const(c_folded))
        z = relay.add(y, relay.const(c_data))
        return relay.Function([x], z)

    # the fold constant should work on any context.
    with tvm.target.Target("cuda"):
        zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_const_with_on_device():
    """Make sure on_device annotations don't get in the way of constant folding"""
    c_data = np.array([1, 2, 3]).astype("float32")
    t = relay.TensorType([1, 2, 3], "float32")

    def before():
        c = relay.const(c_data)
        x = relay.var("x", t)
        x.virtual_device_ = tvm.cpu()
        y = relay.add(c, c)
        y = relay.multiply(y, relay.const(2, "float32"))
        y = relay.add(x, y)
        z = relay.add(y, c)
        f = relay.Function([x], z)
        f.virtual_device_ = tvm.cpu()
        return f

    def expected():
        x = relay.var("x", t)
        x.virtual_device_ = tvm.cpu()
        c_folded = (c_data + c_data) * 2
        y = relay.add(x, relay.const(c_folded))
        z = relay.add(y, relay.const(c_data))
        f = relay.Function([x], z)
        f.virtual_device_ = tvm.cpu()
        return f

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_let():
    c_data = np.array(1).astype("float32")
    t = relay.TensorType([1], "float32")

    def before():
        sb = relay.ScopeBuilder()
        x = relay.var("x", t)
        t1 = sb.let("t1", relay.const(c_data))
        t2 = sb.let("t2", relay.add(t1, t1))
        t3 = sb.let("t3", relay.add(t2, x))
        sb.ret(t3)
        return relay.Function([x], sb.get())

    def expected():
        sb = relay.ScopeBuilder()
        x = relay.var("x", t)
        c_folded = c_data + c_data
        t3 = sb.let("t3", relay.add(relay.const(c_folded), x))
        sb.ret(t3)
        return relay.Function([x], sb.get())

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_let_with_on_device():
    """Make sure on_device annotations don't get in the way of constant folding,
    and inlined constants bring their annotations with them."""
    c_data = np.array(1).astype("float32")
    t = relay.TensorType([1], "float32")

    def before():
        sb = relay.ScopeBuilder()
        x = relay.var("x", t)
        x.virtual_device_ = tvm.cpu()
        t1 = sb.let("t1", annot_expr(relay.const(c_data)))
        t2 = sb.let("t2", annot_expr(relay.add(t1, t1)))
        t3 = sb.let("t3", annot_expr(relay.add(t2, x)))
        sb.ret(t3)
        f = relay.Function([x], sb.get())
        f.virtual_device_ = tvm.cpu()
        return f

    def expected():
        sb = relay.ScopeBuilder()
        x = relay.var("x", t)
        x.virtual_device_ = tvm.cpu()
        c_folded = c_data + c_data
        t3 = sb.let("t3", annot_expr(relay.add(annot_expr(relay.const(c_folded)), x)))
        sb.ret(t3)
        f = relay.Function([x], sb.get())
        f.virtual_device_ = tvm.cpu()
        return f

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_tuple():
    c_data = np.array(1).astype("float32")
    t = relay.TensorType([1], "float32")

    def before():
        c = relay.const(c_data)
        x = relay.var("x", t)
        y = relay.Tuple([x, c])
        z = relay.add(y[1], c)
        z = relay.add(z, y[0])
        return relay.Function([x], z)

    def expected():
        c = relay.const(c_data + c_data)
        x = relay.var("x", t)
        z = relay.add(c, x)
        return relay.Function([x], z)

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_concat():
    c_data = np.array([[1, 2, 3]]).astype("float32")

    def before():
        a = relay.const(c_data)
        b = relay.const(c_data)
        y = relay.concatenate((a, b), axis=0)
        return relay.Function([], y)

    def expected():
        y_data = np.concatenate((c_data, c_data), axis=0)
        y = relay.const(y_data)
        return relay.Function([], y)

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_if():
    cond_data = np.array(1).astype("bool")
    x_data = np.array([[1, 2, 3]]).astype("float32")

    def before():
        a = relay.const(cond_data)
        x = relay.const(x_data)
        y = relay.const(x_data)
        iff = relay.If(a, x + y, x - y)
        return relay.Function([], iff)

    def expected():
        y_data = x_data + x_data
        y = relay.const(y_data)
        return relay.Function([], y)

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)

    cond_data = np.array(0).astype("bool")

    def before():
        a = relay.const(cond_data)
        x = relay.const(x_data)
        y = relay.const(x_data)
        iff = relay.If(a, x + y, x - y)
        return relay.Function([], iff)

    def expected():
        y_data = x_data - x_data
        y = relay.const(y_data)
        return relay.Function([], y)

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_shape_of():
    c_shape = (8, 9, 10)

    def before(dtype):
        x = relay.var("x", shape=c_shape, dtype="float32")
        y = relay.var("y", shape=c_shape, dtype="float32")
        z = relay.shape_of(x + y, dtype)
        return relay.Function([x, y], z)

    def expected(dtype):
        x = relay.var("x", shape=c_shape, dtype="float32")
        y = relay.var("y", shape=c_shape, dtype="float32")
        z = relay.const(np.array(c_shape).astype(dtype), dtype=dtype)
        func = relay.Function([x, y], z)
        return func

    for dtype in ["int32", "float32"]:
        zz = run_opt_pass(before(dtype), transform.FoldConstant())
        zexpected = run_opt_pass(expected(dtype), transform.InferType())
        tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_ndarray_size():
    c_shape = (8, 9, 10)

    def before(dtype):
        x = relay.var("x", shape=c_shape, dtype="float32")
        y = relay.var("y", shape=c_shape, dtype="float32")
        z = relay.ndarray_size(x + y, dtype)
        return relay.Function([x, y], z)

    def expected(dtype):
        x = relay.var("x", shape=c_shape, dtype="float32")
        y = relay.var("y", shape=c_shape, dtype="float32")
        z = relay.const(np.size(np.zeros(c_shape)), dtype=dtype)
        func = relay.Function([x, y], z)
        mod = tvm.IRModule.from_expr(func)
        return mod["main"]

    for dtype in ["int32", "float32"]:
        zz = run_opt_pass(before(dtype), transform.FoldConstant())
        zexpected = run_opt_pass(expected(dtype), transform.InferType())
        tvm.ir.assert_structural_equal(zz, zexpected)


def test_fold_batch_norm():
    def expected():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.const(np.zeros((16, 3, 3, 3)))
        bias = relay.const(np.zeros((16, 1, 1)))
        conv = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        add = relay.add(conv, bias)
        return relay.Function(relay.analysis.free_vars(add), add)

    remove_bn_pass = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
        ]
    )

    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    conv = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
    )
    bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mmean, bn_mvar)

    def initializer(_, param):
        param = np.zeros(param.shape)

    mod, params = create_workload(bn_output[0], initializer)
    mod["main"] = bind_params_by_name(mod["main"], params)

    with tvm.transform.PassContext(opt_level=3):
        mod = remove_bn_pass(mod)

    expect = run_infer_type(expected())
    tvm.ir.assert_structural_equal(mod["main"], expect)


def test_fold_dropout():
    def before():
        # A constant graph to fire fold constant
        data = relay.const(np.arange(10).astype(np.float32))
        dropout = relay.nn.dropout(data)
        add = dropout + relay.const(1.0)
        return relay.Function(relay.analysis.free_vars(add), add)

    passes = tvm.transform.Sequential(
        [
            relay.transform.InferType(),
            relay.transform.FoldConstant(),
        ]
    )

    before_mod = tvm.IRModule.from_expr(before())

    with tvm.transform.PassContext(opt_level=3):
        after_mod = passes(before_mod)

    tvm.ir.assert_structural_equal(run_infer_type(before_mod["main"]), after_mod["main"])


def test_fold_qnn_const():
    def before():
        # QNN op with 2 constant arguments.
        add = relay.qnn.op.add(
            relay.const(np.ones((2, 3), dtype="uint8"), dtype="uint8"),
            relay.const(np.ones((2, 3), dtype="uint8"), dtype="uint8"),
            lhs_scale=relay.const(2.0),
            lhs_zero_point=relay.const(0),
            rhs_scale=relay.const(2.0),
            rhs_zero_point=relay.const(0),
            output_scale=relay.const(1.0),
            output_zero_point=relay.const(0),
        )
        # QNN op with 1 constant and 1 non-constant arguments.
        a = relay.var("a", shape=[2, 3], dtype="float32")
        dense = relay.qnn.op.dense(
            relay.qnn.op.quantize(a, relay.const(1.0), relay.const(0)),
            add,
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),
            input_scale=relay.const(2.0),
            kernel_scale=relay.const(2.0),
            units=None,
        )
        # QNN op with 2 non-constant arguments.
        b = relay.var("b", shape=[2], dtype="float32")
        bias = relay.qnn.op.add(
            dense,
            relay.qnn.op.quantize(b, relay.const(1.0), relay.const(0), out_dtype="int32"),
            lhs_scale=relay.const(2.0),
            lhs_zero_point=relay.const(0),
            rhs_scale=relay.const(2.0),
            rhs_zero_point=relay.const(0),
            output_scale=relay.const(1.0),
            output_zero_point=relay.const(0),
        )
        return relay.Function([a, b], bias)

    def expected():
        a = relay.var("a", shape=[2, 3], dtype="float32")
        dense = relay.qnn.op.dense(
            relay.qnn.op.quantize(a, relay.const(1.0), relay.const(0)),
            relay.const(np.array([[4, 4, 4], [4, 4, 4]], dtype="uint8"), dtype="uint8"),
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),
            input_scale=relay.const(2.0),
            kernel_scale=relay.const(2.0),
            units=None,
        )
        b = relay.var("b", shape=[2], dtype="float32")
        bias = relay.qnn.op.add(
            dense,
            relay.qnn.op.quantize(b, relay.const(1.0), relay.const(0), out_dtype="int32"),
            lhs_scale=relay.const(2.0),
            lhs_zero_point=relay.const(0),
            rhs_scale=relay.const(2.0),
            rhs_zero_point=relay.const(0),
            output_scale=relay.const(1.0),
            output_zero_point=relay.const(0),
        )
        return relay.Function([a, b], bias)

    # Nothing changed after applying FoldConstant
    a = run_opt_pass(before(), transform.FoldConstant())
    b = run_opt_pass(before(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)

    # Fold QNN constants
    a = run_opt_pass(before(), transform.FoldConstant(fold_qnn=True))
    b = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)


def test_fold_quantize():
    t = relay.TensorType([1, 2, 3], "int8")

    def before():
        data = tvm.nd.array(np.array([1.0, 2.0, 3.0], dtype="float32"))
        const_fp = relay.const(data, dtype="float32")
        const_i8 = relay.qnn.op.quantize(
            const_fp, output_scale=relay.const(0.5), output_zero_point=relay.const(0)
        )
        x = relay.var("x", t)
        sub = relay.op.subtract(x, const_i8)
        func = relay.Function([x], sub)
        return func

    def expected():
        data = tvm.nd.array(np.array([2, 4, 6], dtype="int8"))
        const_i8 = relay.const(data, dtype="int8")
        x = relay.var("x", t)
        sub = relay.op.subtract(x, const_i8)
        func = relay.Function([x], sub)
        return func

    # Nothing changed after applying FoldConstant
    a = run_opt_pass(before(), transform.FoldConstant())
    b = run_opt_pass(before(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)

    # Fold QNN constants
    a = run_opt_pass(before(), transform.FoldConstant(fold_qnn=True))
    b = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)


def test_fold_qnn_conv2d_qnn_mul():
    def before():
        dtype = "uint8"
        op0 = relay.qnn.op.conv2d(
            relay.const(np.ones((1, 1, 2, 2), dtype=dtype), dtype=dtype),
            relay.const(np.ones((1, 1, 2, 2), dtype=dtype), dtype=dtype),
            input_zero_point=relay.const(0, "int32"),
            kernel_zero_point=relay.const(0, "int32"),
            input_scale=relay.const(1.0, "float32"),
            kernel_scale=relay.const(1.0, "float32"),
            kernel_size=(2, 2),
            channels=1,
        )
        op = relay.qnn.op.mul(
            op0,
            relay.const(np.array([10], dtype="int32"), dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
        )
        func = relay.Function([], op)
        return func

    def expected():
        data = relay.const(np.array([[[[40]]]], dtype="int32"), dtype="int32")
        func = relay.Function([], data)
        return func

    # Nothing changed after applying FoldConstant
    a = run_opt_pass(before(), transform.FoldConstant())
    b = run_opt_pass(before(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)

    # Fold QNN constants
    a = run_opt_pass(before(), transform.FoldConstant(fold_qnn=True))
    b = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)


def test_fold_requantize():
    def before():
        data = tvm.nd.array(np.array([1, 2, 3], dtype="int8"))
        const_i8 = relay.const(data, dtype="int8")
        op = relay.qnn.op.requantize(
            const_i8,
            input_scale=relay.const(2.0, dtype="float32"),
            input_zero_point=relay.const(1, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(1, dtype="int32"),
        )
        x = relay.var("x", relay.TensorType([3], "int8"))
        add = relay.op.add(op, x)
        func = relay.Function([x], add)
        return func

    def expected():
        data = tvm.nd.array(np.array([1, 3, 5], dtype="int8"))
        const_i8 = relay.const(data, dtype="int8")
        x = relay.var("x", relay.TensorType([3], "int8"))
        add = relay.op.add(const_i8, x)
        func = relay.Function([x], add)
        return func

    # Nothing changed after applying FoldConstant
    a = run_opt_pass(before(), transform.FoldConstant())
    b = run_opt_pass(before(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)

    # Fold QNN constants
    a = run_opt_pass(before(), transform.FoldConstant(fold_qnn=True))
    b = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(a, b)


def test_pass_link_params():
    """
    This test checks ensures that proper executor is passed to interpreter instance
    The test will fail if FoldConstant does not override the executor due to "int8"
    is not supported in ScheduleBuilder
    """

    def expr():
        z = relay.const(10, dtype="int8")
        return relay.cast(z, dtype="int32")

    mod = tvm.IRModule.from_expr(expr())
    mod = tvm.relay.transform.InferType()(mod)
    # Add executor with link-params
    mod = mod.with_attr("executor", Executor("aot", {"link-params": True}))
    mod = tvm.relay.transform.FoldConstant()(mod)


if __name__ == "__main__":
    tvm.testing.main()
