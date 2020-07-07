
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
from tvm import te
from tvm import relay
from tvm.relay.testing import ctx_list

def test_dynamic_topk():
    def verify_topk(k, axis, ret_type, is_ascend, dtype):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        k_var = relay.var("x", relay.TensorType((1,), "float32"))
        out = relay.topk(x, k_var, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x, k_var], out)

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
            if "llvm" not in target: continue
            for kind in ["vm", "debug"]:
                mod = tvm.ir.IRModule.from_expr(func)
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(np_data, np.array([k]).astype("float32"))
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
    test_topk()
