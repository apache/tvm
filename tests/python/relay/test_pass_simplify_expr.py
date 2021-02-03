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
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass


def test_simplify_reshape():
    def before():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    def expected():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(32, 16, 16))
        return relay.Function([x, w], y)

    def symbolic():
        b = tvm.te.size_var("b")
        x = relay.var("x", shape=(b, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    z = before()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)

    z = symbolic()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(symbolic(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_simplify_full_argwhere():
    def verify(x_shape):
        def before():
            x = relay.const(1)
            y = relay.full(x, x_shape, dtype="int64")
            z = relay.argwhere(y)
            return z

        def expected():
            x = relay.const(1)
            full = relay.full(x, x_shape, dtype="int64")
            start = relay.const(0)
            end = relay.take(relay.shape_of(full, "int32"), relay.const(0), 0)
            step = relay.const(1)
            y = relay.arange(start, end, step, dtype="int32")
            z = relay.reshape(y, [-1, 1])
            return z

        z = before()
        zz = run_opt_pass(z, transform.SimplifyExpr())
        after = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(zz, after)

        mod1 = tvm.IRModule.from_expr(z)
        mod2 = tvm.IRModule.from_expr(zz)

        with tvm.transform.PassContext(disabled_pass="SimplifyExpr"):
            ex1 = relay.create_executor("vm", mod=mod1, ctx=tvm.cpu(), target="llvm")
        ex2 = relay.create_executor("vm", mod=mod2, ctx=tvm.cpu(), target="llvm")

        result1 = ex1.evaluate()()
        result2 = ex2.evaluate()()

        tvm.testing.assert_allclose(result1.asnumpy(), result2.asnumpy())

    verify([128])
    verify(relay.const([128], dtype="int64"))

if __name__ == "__main__":
    test_simplify_reshape()
    test_simplify_full_argwhere()
