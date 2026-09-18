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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T

CONSTRAINTS = {
    "max_shared_memory_per_block": 49152,
    "max_local_memory_per_block": 2147483647,
    "max_threads_per_block": 1024,
}


def _shared_memory_kernel(shape, dtype):
    @T.prim_func(s_tir=True)
    def main(a: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        A = T.match_buffer(a, [64], dtype=dtype)
        T.launch_thread(blockIdx_x, 1)
        T.launch_thread(threadIdx_x, 64)
        shared = T.decl_buffer(shape, dtype, scope="shared")
        shared[0, 0] = A[threadIdx_x]
        A[threadIdx_x] = shared[0, 0]

    return main


def test_shared_memory_within_limit():
    assert tvm.s_tir.analysis.verify_gpu_code(_shared_memory_kernel((8, 8), "float32"), CONSTRAINTS)


def test_shared_memory_over_limit():
    assert not tvm.s_tir.analysis.verify_gpu_code(
        _shared_memory_kernel((256, 256), "float32"), CONSTRAINTS
    )


@pytest.mark.parametrize(
    "shape,dtype",
    [
        # 2**32 * 2**32 elements: the element count alone does not fit.
        ((2**32, 2**32), "int8"),
        # 2**62 elements of 4 bytes: the element count fits, the byte count does not.
        ((2**31, 2**31), "float32"),
    ],
)
def test_shared_memory_size_that_does_not_fit(shape, dtype):
    """A size that wraps around must not be counted as zero shared memory."""
    assert not tvm.s_tir.analysis.verify_gpu_code(_shared_memory_kernel(shape, dtype), CONSTRAINTS)


if __name__ == "__main__":
    tvm.testing.main()
