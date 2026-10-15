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
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@T.function
def ptx_griddepcontrol(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")) -> None:
    T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
    T.launch_thread("blockIdx.x", 1)
    tx = T.launch_thread("threadIdx.x", 32)
    T.ptx.griddepcontrol.wait()
    B[tx] = A[tx]
    T.ptx.griddepcontrol.launch_dependents()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_griddepcontrol():
    f = ptx_griddepcontrol
    mod = tvm_tirx.tirx.compile(f, target="cuda")
    A_np = np.random.default_rng(0).standard_normal(32).astype("float32")
    B_np = np.zeros((32,), dtype="float32")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        A_nd = tvm_tirx.runtime.tensor(A_np, device=dev)
        B_nd = tvm_tirx.runtime.tensor(B_np, device=dev)
        mod(A_nd, B_nd)
        tvm_tirx.testing.assert_allclose(B_nd.numpy(), A_np, rtol=0, atol=0)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    test_ptx_griddepcontrol()
