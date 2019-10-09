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
"""Test code for vision package"""
from __future__ import print_function
import numpy as np
import tvm
import topi
import topi.testing

def test_argsort():
    dshape = (20, 100)
    data = tvm.placeholder(dshape, name="data", dtype="float32")
    np_data = np.random.rand(dshape[0], dshape[1]).astype(data.dtype)
    np_result = np.argsort(-np_data)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.argsort(data, axis=-1, is_ascend=False)
            s = topi.generic.schedule_argsort(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype="float32"), ctx)
        f = tvm.build(s, [data, out], device)
        f(tvm_data, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result.astype("float32"), rtol=1e0)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)

def verify_topk(k, axis, ret_type, is_ascend, dtype):
    shape = (20, 100)
    data_dtype = "float32"
    data = tvm.placeholder(shape, name="data", dtype=data_dtype)

    np_data = np.random.uniform(size=shape).astype(data_dtype)
    if is_ascend:
        np_indices = np.argsort(np_data, axis=axis)
    else:
        np_indices = np.argsort(-np_data, axis=axis)
    kk = k if k >= 1 else shape[axis]
    if axis == 0:
        np_indices = np_indices[:kk, :]
        np_values = np.zeros(np_indices.shape).astype(data_dtype)
        for i in range(shape[1]):
            np_values[:, i] = np_data[np_indices[:, i], i]
    else:
        np_indices = np_indices[:, :kk]
        np_values = np.zeros(np_indices.shape).astype(data_dtype)
        for i in range(shape[0]):
            np_values[i, :] = np_data[i, np_indices[i, :]]
    np_indices = np_indices.astype(dtype)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            outs = topi.topk(data, k, axis, ret_type, is_ascend, dtype)
            outs = outs if isinstance(outs, list) else [outs]
            s = topi.generic.schedule_topk(outs)
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_res = []
        for t in outs:
            tvm_res.append(tvm.nd.empty(t.shape, dtype=t.dtype, ctx=ctx))
        f = tvm.build(s, [data] + outs, device)
        f(tvm_data, *tvm_res)
        if ret_type == "both":
            tvm.testing.assert_allclose(tvm_res[0].asnumpy(), np_values)
            tvm.testing.assert_allclose(tvm_res[1].asnumpy(), np_indices)
        elif ret_type == "values":
            tvm.testing.assert_allclose(tvm_res[0].asnumpy(), np_values)
        else:
            tvm.testing.assert_allclose(tvm_res[0].asnumpy(), np_indices)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)

def test_topk():
    np.random.seed(0)
    for k in [0, 1, 5]:
        for axis in [0, -1, 1]:
            for ret_type in ["both", "values", "indices"]:
                verify_topk(k, axis, ret_type, True, "int64")
                verify_topk(k, axis, ret_type, False, "float32")


if __name__ == "__main__":
    test_argsort()
    test_topk()
