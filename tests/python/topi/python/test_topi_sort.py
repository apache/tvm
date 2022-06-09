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
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi

_sort_implement = {
    "generic": (topi.sort, topi.generic.schedule_sort),
    "gpu": (topi.cuda.sort, topi.cuda.schedule_sort),
}

_argsort_implement = {
    "generic": (topi.argsort, topi.generic.schedule_argsort),
    "gpu": (topi.cuda.argsort, topi.cuda.schedule_argsort),
}

_topk_implement = {
    "generic": (topi.topk, topi.generic.schedule_topk),
    "gpu": (topi.cuda.topk, topi.cuda.schedule_topk),
}

axis = tvm.testing.parameter(0, -1, 1)
is_ascend = tvm.testing.parameter(True, False, ids=["is_ascend", "not_ascend"])
dtype = tvm.testing.parameter("int64", "float32")

topk = tvm.testing.parameter(0, 1, 5)
topk_ret_type = tvm.testing.parameter("values", "indices", "both")


def test_sort(target, dev, axis, is_ascend):
    np.random.seed(0)

    dshape = (20, 100)
    data_dtype = "float32"
    data = te.placeholder(dshape, name="data", dtype=data_dtype)

    perm = np.arange(dshape[0] * dshape[1], dtype=data_dtype)
    np.random.shuffle(perm)
    np_data = perm.reshape(dshape)

    if is_ascend:
        np_sort = np.sort(np_data, axis=axis)
    else:
        np_sort = -np.sort(-np_data, axis=axis)

    if axis == 0:
        np_sort = np_sort[: dshape[axis], :]
    else:
        np_sort = np_sort[:, : dshape[axis]]

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _sort_implement)
        out = fcompute(data, axis=axis, is_ascend=is_ascend)
        s = fschedule(out)

    tvm_data = tvm.nd.array(np_data, dev)
    tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data_dtype), dev)
    f = tvm.build(s, [data, out], target)
    f(tvm_data, tvm_out)
    tvm.testing.assert_allclose(tvm_out.numpy(), np_sort, rtol=1e0)


def test_argsort(target, dev, axis, is_ascend):
    dshape = (20, 100)
    data_dtype = "float32"
    data = te.placeholder(dshape, name="data", dtype=data_dtype)

    perm = np.arange(dshape[0] * dshape[1], dtype=data_dtype)
    np.random.shuffle(perm)
    np_data = perm.reshape(dshape)

    if is_ascend:
        np_indices = np.argsort(np_data, axis=axis)
    else:
        np_indices = np.argsort(-np_data, axis=axis)

    if axis == 0:
        np_indices = np_indices[: dshape[axis], :]
    else:
        np_indices = np_indices[:, : dshape[axis]]

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _argsort_implement)
        out = fcompute(data, axis=axis, is_ascend=is_ascend)
        s = fschedule(out)

    tvm_data = tvm.nd.array(np_data, dev)
    tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data_dtype), dev)
    f = tvm.build(s, [data, out], target)
    f(tvm_data, tvm_out)
    tvm.testing.assert_allclose(tvm_out.numpy(), np_indices.astype(data_dtype), rtol=1e0)


def test_topk(target, dev, topk, axis, topk_ret_type, is_ascend, dtype):
    np.random.seed(0)

    shape = (20, 100)
    data_dtype = "float32"
    data = te.placeholder(shape, name="data", dtype=data_dtype)

    np_data = np.random.uniform(size=shape).astype(data_dtype)
    if is_ascend:
        np_indices = np.argsort(np_data, axis=axis)
    else:
        np_indices = np.argsort(-np_data, axis=axis)
    kk = topk if topk >= 1 else shape[axis]
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

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _topk_implement)
        outs = fcompute(data, topk, axis, topk_ret_type, is_ascend, dtype)
        outs = outs if isinstance(outs, list) else [outs]
        s = fschedule(outs)
    tvm_data = tvm.nd.array(np_data, dev)
    tvm_res = []
    for t in outs:
        tvm_res.append(tvm.nd.empty(t.shape, dtype=t.dtype, device=dev))
    f = tvm.build(s, [data] + outs, target)
    f(tvm_data, *tvm_res)
    if topk_ret_type == "both":
        tvm.testing.assert_allclose(tvm_res[0].numpy(), np_values)
        tvm.testing.assert_allclose(tvm_res[1].numpy(), np_indices)
    elif topk_ret_type == "values":
        tvm.testing.assert_allclose(tvm_res[0].numpy(), np_values)
    else:
        tvm.testing.assert_allclose(tvm_res[0].numpy(), np_indices)


if __name__ == "__main__":
    tvm.testing.main()
