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
import tvm.testing
import tvm.topi.testing
from tvm import topi


@tvm.testing.parametrize_targets
def test_cumprod(ctx, target):
    def check_cumprod(np_ref, data, axis=None, dtype=None, exclusive=False):
        implementations = {
            "generic": (
                lambda x: topi.cumprod(x, axis, dtype, exclusive=exclusive),
                topi.generic.schedule_extern,
            ),
            "cuda": (
                lambda x: topi.cuda.cumprod(x, axis, dtype, exclusive=exclusive),
                topi.cuda.schedule_scan,
            ),
            "nvptx": (
                lambda x: topi.cuda.cumprod(x, axis, dtype, exclusive=exclusive),
                topi.cuda.schedule_scan,
            ),
            "vulkan": (
                lambda x: topi.cuda.cumprod(x, axis, dtype, exclusive=exclusive),
                topi.cuda.schedule_scan,
            ),
            "metal": (
                lambda x: topi.cuda.cumprod(x, axis, dtype, exclusive=exclusive),
                topi.cuda.schedule_scan,
            ),
        }
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
        tvm.topi.testing.compare_numpy_tvm([data], np_ref, target, ctx, fcompute, fschedule)

    data = np.array([2, 3, 0])
    check_cumprod(np.cumprod(data), data)

    data = np.random.rand(10) > 0.5
    data = data.astype(np.int32)
    check_cumprod(np.cumprod(data, dtype=np.int32), data)
    check_cumprod(np.cumprod(data), data, dtype="int64")

    data = np.random.rand(10) > 0.5
    check_cumprod(np.cumprod(data, dtype=np.int32), data, dtype="int32")

    for in_dtype in ["float32", "float64"]:
        if target == "metal" and in_dtype == "float64":
            # float64 is not supported in metal
            continue
        data = np.random.randn(10, 10).astype(in_dtype)
        check_cumprod(np.cumprod(data), data)
        check_cumprod(np.cumprod(data, axis=0), data, axis=0)
        check_cumprod(np.cumprod(data, axis=1), data, axis=1)

        data = np.random.randn(10, 5, 10).astype(in_dtype)
        check_cumprod(np.cumprod(data), data)
        check_cumprod(np.cumprod(data, axis=0), data, axis=0)
        check_cumprod(np.cumprod(data, axis=1), data, axis=1)
        check_cumprod(np.cumprod(data, axis=-1), data, axis=-1)

    for in_dtype in ["int32", "int64"]:
        data = np.random.randint(-100, 100, size=(100, 100)).astype(in_dtype)
        check_cumprod(np.cumprod(data, dtype=in_dtype), data)
        check_cumprod(np.cumprod(data), data, dtype="int64")
        check_cumprod(np.cumprod(data, axis=0, dtype=in_dtype), data, axis=0)
        check_cumprod(np.cumprod(data, axis=1, dtype=in_dtype), data, axis=1)

        data = np.random.randint(1 << 30, (1 << 31) - 1, size=(100)).astype(in_dtype)
        check_cumprod(np.cumprod(data), data, dtype="int64")

    data = np.random.randint(-100, 100, size=(100, 100)).astype("int64")

    expected_result = np.roll(np.cumprod(data), 1)
    expected_result[0] = 1
    check_cumprod(expected_result, data, dtype="int64", exclusive=True)

    expected_result = np.roll(np.cumprod(data, axis=0, dtype=in_dtype), 1, axis=0)
    expected_result[0, :] = 1
    check_cumprod(expected_result, data, axis=0, exclusive=True)

    expected_result = np.roll(np.cumprod(data, axis=1, dtype=in_dtype), 1, axis=1)
    expected_result[:, 0] = 1
    check_cumprod(np.cumprod(data, axis=1, dtype=in_dtype), data, axis=1)


if __name__ == "__main__":
    test_cumprod(tvm.context("cpu"), tvm.target.Target("llvm"))
    test_cumprod(tvm.context("cuda"), tvm.target.Target("cuda"))
    test_cumprod(tvm.context("nvptx"), tvm.target.Target("nvptx"))
    test_cumprod(tvm.context("vulkan"), tvm.target.Target("vulkan"))
    test_cumprod(tvm.context("metal"), tvm.target.Target("metal"))
