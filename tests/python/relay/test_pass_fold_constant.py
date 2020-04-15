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
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


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

    def fail(x):
        raise RuntimeError()

    # the fold constant should work on any context.
    with tvm.target.build_config(add_lower_pass=[(0, fail)]):
        with tvm.target.create("cuda"):
            zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, zexpected)


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
        c_folded = (c_data + c_data)
        t3 = sb.let("t3", relay.add(relay.const(c_folded), x))
        sb.ret(t3)
        return relay.Function([x], sb.get())

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, zexpected)


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
    assert tvm.ir.structural_equal(zz, zexpected)


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
    assert tvm.ir.structural_equal(zz, zexpected)


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
        assert tvm.ir.structural_equal(zz, zexpected)


def test_fold_full():
    c_shape = (8, 9, 10)
    def before():
        dtype = 'float32'
        return relay.full(relay.const(1.0, dtype), c_shape, dtype=dtype)

    def expected():
        # expect no changes
        return before()

    zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, zexpected)


def test_fold_batch_norm():
    def expected():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.const(np.zeros((16, 3, 3, 3)))
        bias = relay.const(np.zeros((16, 1, 1)))
        conv = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                               channels=16, padding=(1, 1))
        add = relay.add(conv, bias)
        return relay.Function(relay.analysis.free_vars(add), add)

    remove_bn_pass = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
    ])

    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    conv = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                           channels=16, padding=(1, 1))
    bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta,
                                    bn_mmean, bn_mvar)
    def initializer(_, param):
        param = np.zeros(param.shape)

    mod, params = create_workload(bn_output[0], initializer)
    mod["main"] = bind_params_by_name(mod["main"], params)

    with relay.build_config(opt_level=3):
        mod = remove_bn_pass(mod)

    expect = run_infer_type(expected())
    assert tvm.ir.structural_equal(mod["main"], expect)


if __name__ == "__main__":
    test_fold_const()
    test_fold_let()
    test_fold_tuple()
    test_fold_concat()
    test_fold_shape_of()
    test_fold_full()
    test_fold_batch_norm()
