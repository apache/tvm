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
from tvm import topi
import tvm.topi.testing


@tvm.testing.parametrize_targets
def test_cumsum(ctx, target):
    def check_cumsum(np_ref, data, axis=None, dtype=None):
        implementations = {
            "generic": (lambda x: topi.cumsum(x, axis, dtype), topi.generic.schedule_extern),
            "cuda": (lambda x: topi.cuda.cumsum(x, axis, dtype), topi.cuda.schedule_scan),
            "nvptx": (lambda x: topi.cuda.cumsum(x, axis, dtype), topi.cuda.schedule_scan),
        }
        fcompute, fschedule = tvm.topi.testing.dispatch(target, implementations)
        tvm.topi.testing.compare_numpy_tvm([data], np_ref, target, ctx, fcompute, fschedule)

    data = np.array([2, 3, 0])
    check_cumsum(np.cumsum(data), data)

    data = np.random.rand(10) > 0.5
    data = data.astype(np.int32)
    check_cumsum(np.cumsum(data, dtype=np.int32), data)
    check_cumsum(np.cumsum(data), data, dtype="int64")

    data = np.random.rand(10) > 0.5
    check_cumsum(np.cumsum(data, dtype=np.int32), data, dtype="int32")

    for in_dtype in ["float32", "float64"]:
        data = np.random.randn(10, 10).astype(in_dtype)
        check_cumsum(np.cumsum(data), data)
        check_cumsum(np.cumsum(data, axis=0), data, axis=0)
        check_cumsum(np.cumsum(data, axis=1), data, axis=1)

        data = np.random.randn(10, 5, 10).astype(in_dtype)
        check_cumsum(np.cumsum(data), data)
        check_cumsum(np.cumsum(data, axis=0), data, axis=0)
        check_cumsum(np.cumsum(data, axis=1), data, axis=1)
        check_cumsum(np.cumsum(data, axis=-1), data, axis=-1)

    for in_dtype in ["int32", "int64"]:
        data = np.random.randint(-100, 100, size=(100, 100)).astype(in_dtype)
        check_cumsum(np.cumsum(data, dtype=in_dtype), data)
        check_cumsum(np.cumsum(data), data, dtype="int64")
        check_cumsum(np.cumsum(data, axis=0, dtype=in_dtype), data, axis=0)
        check_cumsum(np.cumsum(data, axis=1, dtype=in_dtype), data, axis=1)

        data = np.random.randint(1 << 30, (1 << 31) - 1, size=(100)).astype(in_dtype)
        check_cumsum(np.cumsum(data), data, dtype="int64")


if __name__ == "__main__":
    test_cumsum(tvm.context("cpu"), tvm.target.Target("llvm"))
    test_cumsum(tvm.context("cuda"), tvm.target.Target("cuda"))
    test_cumsum(tvm.context("nvptx"), tvm.target.Target("nvptx"))
