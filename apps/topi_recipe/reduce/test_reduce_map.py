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
import os

import numpy as np
import tvm
from tvm import te, topi
from tvm.contrib import nvcc

TASK = "reduce_map"
USE_MANUAL_CODE = False


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code, target):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_reduce_map(in_shape, axis, keepdims, type="sum", test_id=0):
    global TASK
    # Build the logic and compile the function
    A = te.placeholder(shape=in_shape, name="A")
    if type == "sum":
        TASK = "sum_map_id%d" % test_id
        B = topi.sum(A, axis=axis, keepdims=keepdims)
    elif type == "max":
        TASK = "max_map_id%d" % test_id
        B = topi.max(A, axis=axis, keepdims=keepdims)
    elif type == "min":
        TASK = "min_map_id%d" % test_id
        B = topi.min(A, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError
    s = topi.cuda.schedule_reduce(B)
    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_step": 16,
            }
        }
    ):
        fcuda = tvm.build(s, [A, B], "cuda", name="sum")

    # Test
    in_npy = np.random.normal(size=in_shape).astype(np.float32)
    if type == "sum":
        out_npy = in_npy.sum(axis=axis, keepdims=keepdims)
    elif type == "max":
        out_npy = in_npy.max(axis=axis, keepdims=keepdims)
    elif type == "min":
        out_npy = in_npy.min(axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError

    data_tvm = tvm.nd.array(in_npy, device=tvm.cuda())
    out_tvm = tvm.nd.empty(shape=out_npy.shape, device=tvm.cuda())

    for _ in range(2):
        fcuda(data_tvm, out_tvm)
    tvm.testing.assert_allclose(out_tvm.numpy(), out_npy, rtol=4e-4, atol=4e-4)


if __name__ == "__main__":
    test_reduce_map(
        in_shape=(128, 24, 128, 24), axis=(1, 2, 3), keepdims=True, type="sum", test_id=0
    )
    test_reduce_map(in_shape=(128, 24 * 128 * 24), axis=(1,), keepdims=False, type="max", test_id=1)
    test_reduce_map(in_shape=(32, 128, 24), axis=None, keepdims=True, type="sum", test_id=2)
    test_reduce_map(in_shape=(128, 24, 128, 24), axis=(0, 2), keepdims=False, type="min", test_id=3)
