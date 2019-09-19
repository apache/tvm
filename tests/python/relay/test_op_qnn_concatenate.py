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
import numpy as np
from tvm import relay
from tvm.contrib import graph_runtime
import topi.testing

def test_same_io_qnn_params():
    data_dtype = 'int32'
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)
    x_scale = (62 + 64) / (np.power(2, 32) - 1.0)
    y_scale = (62 + 64) / (np.power(2, 32) - 1.0)

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.op.concatenate((x, y),
                                 input_scales=[x_scale, y_scale],
                                 input_zero_points=[0, 0],
                                 output_scale=y_scale,
                                 output_zero_point=0,
                                 axis=axis)

    func = relay.Function([x, y], z)
    mod = relay.Module.from_expr(func)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data, y_data), axis=axis)

    intrp = relay.create_executor("graph", ctx=tvm.cpu(0), target="llvm")
    op_res = intrp.evaluate(func)(x_data, y_data)
    np.testing.assert_equal(op_res.asnumpy(), golden_output)

def test_different_io_qnn_params():
    data_dtype = 'int32'
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)
    x_scale = (62 + 64) / (np.power(2, 32) - 1.0)
    y_scale = (62 + 64) / (np.power(2, 32) - 1.0)

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.op.concatenate((x, y),
                                 input_scales=[x_scale, y_scale],
                                 input_zero_points=[3, 4],
                                 output_scale=y_scale,
                                 output_zero_point=1,
                                 axis=axis)

    func = relay.Function([x, y], z)
    mod = relay.Module.from_expr(func)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data - 2, y_data - 3), axis=axis)

    intrp = relay.create_executor("graph", ctx=tvm.cpu(0), target="llvm")
    op_res = intrp.evaluate(func)(x_data, y_data)
    np.testing.assert_equal(op_res.asnumpy(), golden_output)

def test_few_same_io_qnn_params():
    data_dtype = 'int32'
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)
    x_scale = (62 + 64) / (np.power(2, 32) - 1.0)
    y_scale = (62 + 64) / (np.power(2, 32) - 1.0)

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.op.concatenate((x, y),
                                 input_scales=[x_scale, y_scale],
                                 input_zero_points=[0, 1],
                                 output_scale=y_scale,
                                 output_zero_point=1,
                                 axis=axis)

    func = relay.Function([x, y], z)
    mod = relay.Module.from_expr(func)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data + 1, y_data), axis=axis)

    intrp = relay.create_executor("graph", ctx=tvm.cpu(0), target="llvm")
    op_res = intrp.evaluate(func)(x_data, y_data)
    np.testing.assert_equal(op_res.asnumpy(), golden_output)

def test_same_i_qnn_params():
    data_dtype = 'int32'
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)
    x_scale = (62 + 64) / (np.power(2, 32) - 1.0)
    y_scale = (62 + 64) / (np.power(2, 32) - 1.0)

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.op.concatenate((x, y),
                                 input_scales=[x_scale, y_scale],
                                 input_zero_points=[0, 0],
                                 output_scale=y_scale,
                                 output_zero_point=1,
                                 axis=axis)

    func = relay.Function([x, y], z)
    mod = relay.Module.from_expr(func)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data + 1, y_data + 1), axis=axis)

    intrp = relay.create_executor("graph", ctx=tvm.cpu(0), target="llvm")
    op_res = intrp.evaluate(func)(x_data, y_data)
    np.testing.assert_equal(op_res.asnumpy(), golden_output)

if __name__ == '__main__':
    test_same_io_qnn_params()
    test_different_io_qnn_params()
    test_few_same_io_qnn_params()
    test_same_i_qnn_params()
