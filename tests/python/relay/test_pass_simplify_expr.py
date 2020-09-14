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


if __name__ == "__main__":
    test_simplify_reshape()
