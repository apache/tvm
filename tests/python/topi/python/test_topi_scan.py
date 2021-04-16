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
from typing import Callable

import numpy as np
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import topi

topi_funcs = {
    "cumsum": {"generic": topi.cumsum, "cuda": topi.cuda.cumsum},
    "cumprod": {"generic": topi.cumprod, "cuda": topi.cuda.cumprod},
}

identity_value = {"cumsum": 0, "cumprod": 1}


def get_implementations(name, axis, dtype, exclusive):
    topi_func_generic = topi_funcs[name]["generic"]
    topi_func_cuda = topi_funcs[name]["cuda"]

    return {
        "generic": (
            lambda x: topi_func_generic(x, axis, dtype, exclusive=exclusive),
            topi.generic.schedule_extern,
        ),
        "cuda": (
            lambda x: topi_func_cuda(x, axis, dtype, exclusive=exclusive),
            topi.cuda.schedule_scan,
        ),
        "nvptx": (
            lambda x: topi_func_cuda(x, axis, dtype, exclusive=exclusive),
            topi.cuda.schedule_scan,
        ),
        "vulkan": (
            lambda x: topi_func_cuda(x, axis, dtype, exclusive=exclusive),
            topi.cuda.schedule_scan,
        ),
        "metal": (
            lambda x: topi_func_cuda(x, axis, dtype, exclusive=exclusive),
            topi.cuda.schedule_scan,
        ),
    }


def _run_tests(
    dev,
    target,
    op_name: str = "cumsum",
    gt_func: Callable[..., np.array] = np.cumsum,
):
    def check_scan(np_ref, data, axis=None, dtype=None, exclusive=False):
        implementations = get_implementations(op_name, axis, dtype, exclusive)
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
        tvm.topi.testing.compare_numpy_tvm([data], np_ref, target, dev, fcompute, fschedule)

    data = np.array([2, 3, 0])
    check_scan(gt_func(data), data)

    data = np.random.rand(10) > 0.5
    data = data.astype(np.int32)
    check_scan(gt_func(data, dtype=np.int32), data)
    check_scan(gt_func(data), data, dtype="int64")

    data = np.random.rand(10) > 0.5
    check_scan(gt_func(data, dtype=np.int32), data, dtype="int32")

    for in_dtype in ["float32", "float64"]:
        if target == "metal" and in_dtype == "float64":
            # float64 is not supported in metal
            continue
        data = np.random.randn(10, 10).astype(in_dtype)
        check_scan(gt_func(data), data)
        check_scan(gt_func(data, axis=0), data, axis=0)
        check_scan(gt_func(data, axis=1), data, axis=1)

        data = np.random.randn(10, 5, 10).astype(in_dtype)
        check_scan(gt_func(data), data)
        check_scan(gt_func(data, axis=0), data, axis=0)
        check_scan(gt_func(data, axis=1), data, axis=1)
        check_scan(gt_func(data, axis=-1), data, axis=-1)

    for in_dtype in ["int32", "int64"]:
        data = np.random.randint(-100, 100, size=(100, 100)).astype(in_dtype)
        check_scan(gt_func(data, dtype=in_dtype), data)
        check_scan(gt_func(data), data, dtype="int64")
        check_scan(gt_func(data, axis=0, dtype=in_dtype), data, axis=0)
        check_scan(gt_func(data, axis=1, dtype=in_dtype), data, axis=1)

        data = np.random.randint(1 << 30, (1 << 31) - 1, size=(100)).astype(in_dtype)
        check_scan(gt_func(data), data, dtype="int64")

    data = np.random.randint(-100, 100, size=(100, 100)).astype("int64")

    expected_result = np.roll(gt_func(data), 1)
    expected_result[0] = identity_value[op_name]
    check_scan(expected_result, data, dtype="int64", exclusive=True)

    expected_result = np.roll(gt_func(data, axis=0, dtype=in_dtype), 1, axis=0)
    expected_result[0, :] = identity_value[op_name]
    check_scan(expected_result, data, axis=0, exclusive=True)

    expected_result = np.roll(gt_func(data, axis=1, dtype=in_dtype), 1, axis=1)
    expected_result[:, 0] = identity_value[op_name]
    check_scan(gt_func(data, axis=1, dtype=in_dtype), data, axis=1)


@tvm.testing.parametrize_targets
def test_cumsum(dev, target):
    _run_tests(dev, target, op_name="cumsum", gt_func=np.cumsum)


@tvm.testing.parametrize_targets
def test_cumprod(dev, target):
    _run_tests(dev, target, op_name="cumprod", gt_func=np.cumprod)


if __name__ == "__main__":
    test_cumsum(tvm.device("cpu"), tvm.target.Target("llvm"))
    test_cumsum(tvm.device("cuda"), tvm.target.Target("cuda"))
    test_cumsum(tvm.device("nvptx"), tvm.target.Target("nvptx"))
    test_cumsum(tvm.device("vulkan"), tvm.target.Target("vulkan"))
    test_cumsum(tvm.device("metal"), tvm.target.Target("metal"))

    test_cumprod(tvm.device("cpu"), tvm.target.Target("llvm"))
    test_cumprod(tvm.device("cuda"), tvm.target.Target("cuda"))
    test_cumprod(tvm.device("nvptx"), tvm.target.Target("nvptx"))
    test_cumprod(tvm.device("vulkan"), tvm.target.Target("vulkan"))
    test_cumprod(tvm.device("metal"), tvm.target.Target("metal"))
