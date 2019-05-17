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
    dshape = (1, 8)
    valid_count_shape = (2,)
    data = tvm.placeholder(dshape, name="data", dtype="float32")
    valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
    np_data = np.random.rand(dshape[0], dshape[1]).astype(data.dtype)
    np_valid_count = np.array([4]).astype(valid_count.dtype)
    np_result = np.argsort(-np_data)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.argsort(data, valid_count, axis=-1, is_ascend=False, flag=False)
            s = topi.generic.schedule_argsort(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype="float32"), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result.astype("float32"), rtol=1e0)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)


def verify_topk(k, axis, ret_type, dtype):
    shape = (3, 4)
    data_dtype = "float32"
    data = tvm.placeholder(shape, name="data", dtype=data_dtype)
    outs = topi.topk(data, k, axis, ret_type, dtype=dtype)
    # print(outs)
    # return
    np_data = np.random.uniform(size=shape).astype(data_dtype)
    np_indices = np.argsort(-np_data, axis=axis)
    if axis == 0:
        np_indices = np_indices[:k, :].astype(dtype)
        np_values = np.zeros(np_indices.shape).astype(data_dtype)
        for i in range(shape[1]):
            np_values[:, i] = np_data[np_indices[:, i], i]
        #real_indices = np.array(list(zip(np_indices, np.arange(shape[1]))))
        #np_values = np_data[np_indices, np.arange(shape[0])]
    else:
        np_indices = np_indices[:, :k]
        np_values = np.zeros(np_indices.shape).astype(data_dtype)
        print(np_values.shape)
        for i in range(shape[0]):
            np_values[i, :] = np_data[i, np_indices[:, i]]
        #np_values = np_data[np.arange(shape[0]), np_indices]
    #np_values = np_data[np_indices]
    print(np_data)
    print(np_indices)
    #print(real_indices)
    print(np_values)



def test_topk():
    verify_topk(2, -1, "both", "int64")
    return
    dshape = (1, 8)
    #np_result = np.argsort(-np_data)
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.topk(data, axis=-1, is_ascend=False, flag=False)
            s = topi.generic.schedule_argsort(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype="float32"), ctx)
        f = tvm.build(s, [data, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result.astype("float32"), rtol=1e0)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)


if __name__ == "__main__":
    #test_argsort()
    test_topk()
