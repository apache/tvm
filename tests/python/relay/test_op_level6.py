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
""" Support level6 operator test cases.
"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list

def test_argsort():
    def verify_argsort(shape, axis, is_ascend, dtype):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.argsort(x, axis=axis, is_ascend=is_ascend, dtype=dtype)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=shape).astype("float32")
        if is_ascend:
            ref_res = np.argsort(x_data, axis=axis)
        else:
            ref_res = np.argsort(-x_data, axis=axis)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.astype(dtype), rtol=1e-5)
    for dtype in ["int32", "int64", "float32", "float64"]:
        verify_argsort((2, 3, 4), axis=0, is_ascend=False, dtype=dtype)
        verify_argsort((1, 4, 6), axis=1, is_ascend=True, dtype=dtype)
        verify_argsort((3, 5, 6), axis=-1, is_ascend=False, dtype=dtype)


def test_topk():
    def verify_topk(k, axis, ret_type, is_ascend, dtype):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        out = relay.topk(x, k, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x], out)
        np_data = np.random.uniform(size=shape).astype("float32")
        if is_ascend:
            np_indices = np.argsort(np_data, axis=axis)
        else:
            np_indices = np.argsort(-np_data, axis=axis)
        kk = k if k >= 1 else shape[axis]
        if axis == 0:
            np_indices = np_indices[:kk, :]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[1]):
                np_values[:, i] = np_data[np_indices[:, i], i]
        else:
            np_indices = np_indices[:, :kk]
            np_values = np.zeros(np_indices.shape).astype("float32")
            for i in range(shape[0]):
                np_values[i, :] = np_data[i, np_indices[i, :]]
        np_indices = np_indices.astype(dtype)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(np_data)
                if ret_type == "both":
                    tvm.testing.assert_allclose(op_res[0].asnumpy(), np_values)
                    tvm.testing.assert_allclose(op_res[1].asnumpy(), np_indices)
                elif ret_type == "values":
                    tvm.testing.assert_allclose(op_res.asnumpy(), np_values)
                else:
                    tvm.testing.assert_allclose(op_res.asnumpy(), np_indices)
    np.random.seed(0)
    for k in [0, 1, 5]:
        for axis in [0, -1, 1]:
            for ret_type in ["both", "values", "indices"]:
                verify_topk(k, axis, ret_type, True, "int64")
                verify_topk(k, axis, ret_type, False, "float32")


if __name__ == "__main__":
    test_argsort()
    test_topk()
