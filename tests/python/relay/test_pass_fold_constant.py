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
from tvm import relay
from tvm.relay import transform


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, transform.Pass)

    mod = relay.Module.from_expr(expr)
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
    with tvm.build_config(add_lower_pass=[(0, fail)]):
        with tvm.target.create("cuda"):
            zz = run_opt_pass(before(), transform.FoldConstant())
    zexpected = run_opt_pass(expected(), transform.InferType())
    assert relay.analysis.alpha_equal(zz, zexpected)


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
    assert relay.analysis.graph_equal(zz, zexpected)


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
    assert relay.analysis.graph_equal(zz, zexpected)


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
    assert relay.analysis.graph_equal(zz, zexpected)


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
        assert relay.analysis.graph_equal(zz, zexpected)


if __name__ == "__main__":
    test_fold_const()
    test_fold_let()
    test_fold_tuple()
    test_fold_concat()
    test_fold_shape_of()
