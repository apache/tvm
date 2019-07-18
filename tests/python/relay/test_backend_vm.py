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
from tvm.relay.op import add
from tvm.relay.module import Module
from tvm.relay.testing.config import ctx_list

def check_result(expr, args, expected_result, mod=None):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on Relay VM.

    Parameters
    ----------
    expr:
        The expression to evaluate

    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    for target, ctx in ctx_list():
        print("Testing {} {}\n".format(target, ctx))
        vm = relay.create_executor('vm', ctx=ctx, target=target, mod=mod)

        rts_result = vm.evaluate(expr)(*args)
        tvm.testing.assert_allclose(expected_result, rts_result.asnumpy())

def test_add_op_scalar():
    """
    test_add_op_scalar:
        fn (x, y) {
            return x + y;
        }
    """
    mod = relay.Module()
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    func = relay.Function([x, y], add(x, y))
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')
    mod["main"] = func
    check_result(func, [x_data, y_data], x_data + y_data, mod=mod)

def test_add_op_tensor():
    """
    test_add_op_tensor:
        fn (x, y) {
            return x + y;
        }
    """
    mod = relay.Module()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(10, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    mod["main"] = func
    check_result(func, [x_data, y_data], x_data + y_data, mod=mod)

def test_add_op_broadcast():
    """
    test_add_op_broadcast:
        fn (x, y) {
            return x + y;
        }
    """
    mod = relay.Module()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(1, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(1, 5).astype('float32')
    mod["main"] = func
    check_result(func, [x_data, y_data], x_data + y_data, mod=mod)

if __name__ == "__main__":
    test_add_op_scalar()
    test_add_op_tensor()
    test_add_op_broadcast()
