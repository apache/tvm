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
"""Test code for reduce."""
import os
import numpy as np
import tvm
import topi

from common import get_all_backend

def _my_npy_argmax(arr, axis, keepdims):
    if not keepdims:
        return arr.argmax(axis=axis)
    else:
        if axis is not None:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
        else:
            out_shape = [1 for _ in range(len(arr.shape))]
        return arr.argmax(axis=axis).reshape(out_shape)


def _my_npy_argmin(arr, axis, keepdims):
    if not keepdims:
        return arr.argmin(axis=axis)
    else:
        out_shape = list(arr.shape)
        out_shape[axis] = 1
        return arr.argmin(axis=axis).reshape(out_shape)


def verify_reduce_map_ele(in_shape, axis, keepdims, type="sum", dtype="float32"):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A", dtype=dtype)
    A1 = topi.sqrt(topi.exp(A))
    out_dtype = dtype
    if type == "sum":
        B = topi.sum(A1, axis=axis, keepdims=keepdims)
    elif type == "all":
        B = topi.all(A, axis=axis, keepdims=keepdims)
    elif type == "max":
        B = topi.max(A1, axis=axis, keepdims=keepdims)
    elif type == "min":
        B = topi.min(A1, axis=axis, keepdims=keepdims)
    elif type == "argmax":
        B = topi.argmax(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    elif type == "argmin":
        B = topi.argmin(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    else:
        raise NotImplementedError

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_reduce(B)

        foo = tvm.build(s, [A, B], device, name=type)
        # Test
        if dtype == 'bool':
            in_npy_map = in_npy = np.random.choice([True, False], size=in_shape)
        else:
            in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
            in_npy_map = np.sqrt(np.exp(in_npy)).astype(dtype)

        if type == "sum":
            out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
        elif type == "all" and dtype == 'bool':
            out_npy = in_npy_map.all(axis=axis, keepdims=keepdims)
        elif type == "max":
            out_npy = in_npy_map.max(axis=axis, keepdims=keepdims)
        elif type == "min":
            out_npy = in_npy_map.min(axis=axis, keepdims=keepdims)
        elif type == "argmax":
            out_npy = _my_npy_argmax(in_npy_map, axis=axis, keepdims=keepdims)
        elif type == "argmin":
            out_npy = _my_npy_argmin(in_npy_map, axis=axis, keepdims=keepdims)
        else:
            raise NotImplementedError
        data_tvm = tvm.nd.array(in_npy, ctx=ctx)
        out_tvm = tvm.nd.empty(shape=out_npy.shape, ctx=ctx, dtype=out_dtype)
        for _ in range(1):
            foo(data_tvm, out_tvm)
        if type == "argmax" or type == "argmin":
            out_tvm_indices = out_tvm.asnumpy()
            if keepdims:
                out_tvm_indices = np.take(out_tvm_indices, indices=0, axis=axis)
            if axis is None:
                out_tvm_val = in_npy_map.ravel()[out_tvm_indices]
            else:
                other_indices = tuple(np.indices(in_shape[0:axis] + in_shape[(axis+1):]))
                sel_indices = other_indices[0:axis] + (out_tvm_indices,) + other_indices[axis:]
                out_tvm_val = in_npy_map[sel_indices]
            if type == "argmax":
                tvm.testing.assert_allclose(out_tvm_val, in_npy_map.max(axis=axis), 1E-3, 1E-3)
            elif type == "argmin":
                tvm.testing.assert_allclose(out_tvm_val, in_npy_map.min(axis=axis), 1E-3, 1E-3)
        else:
            tvm.testing.assert_allclose(out_tvm.asnumpy(), out_npy, 1E-3, 1E-3)
    for device in get_all_backend():
        check_device(device)


def test_reduce_map():

    verify_reduce_map_ele(in_shape=(32,),
                          axis=0,
                          keepdims=False,
                          type="argmax")
    verify_reduce_map_ele(in_shape=(128, 24, 128, 24),
                          axis=(1, 2, 3),
                          keepdims=True,
                          type="sum")
    verify_reduce_map_ele(in_shape=(2, 3),
                          axis=None,
                          keepdims=True,
                          type="all",
                          dtype='bool')
    verify_reduce_map_ele(in_shape=(128, 24 * 128 * 24),
                          axis=(1,),
                          keepdims=False,
                          type="max")
    verify_reduce_map_ele(in_shape=(32, 128, 24),
                          axis=None,
                          keepdims=True,
                          type="sum")
    verify_reduce_map_ele(in_shape=(32, 128, 24),
                          axis=None,
                          keepdims=True,
                          dtype='bool',
                          type="all")
    verify_reduce_map_ele(in_shape=(128, 24, 128, 24),
                          axis=(0, 2),
                          keepdims=False,
                          type="min")
    verify_reduce_map_ele(in_shape=(32, 128),
                          axis=1,
                          keepdims=True,
                          type="argmax")
    verify_reduce_map_ele(in_shape=(32, 24, 32, 24),
                          axis=2,
                          keepdims=False,
                          type="argmin")
    verify_reduce_map_ele(in_shape=(31, 21, 15),
                          axis=None,
                          keepdims=True,
                          type="argmax")
    verify_reduce_map_ele(in_shape=(31, 21, 15),
                          axis=None,
                          keepdims=False,
                          type="sum")
    verify_reduce_map_ele(in_shape=(128, 24, 128, 24),
                          axis=(1, 2, 3),
                          keepdims=True,
                          type="sum",
                          dtype="float64")

if __name__ == "__main__":
    test_reduce_map()
