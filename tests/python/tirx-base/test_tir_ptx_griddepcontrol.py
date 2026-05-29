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
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def ptx_griddepcontrol(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")) -> None:
    T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
    bx = T.env_thread("blockIdx.x")
    tx = T.env_thread("threadIdx.x")
    T.launch_thread(bx, 1)
    T.launch_thread(tx, 32)
    with T.sblock():
        T.reads(A[0:32])
        T.writes(B[0:32])
        T.evaluate(T.ptx.griddepcontrol.wait(dtype=""))
        B[tx] = A[tx]
        T.evaluate(T.ptx.griddepcontrol.launch_dependents(dtype=""))


@tvm.testing.requires_cuda_compute_version(9)
def test_ptx_griddepcontrol():
    f = ptx_griddepcontrol
    mod = tvm.compile(f, target="cuda")
    A_np = np.random.default_rng(0).standard_normal(32).astype("float32")
    B_np = np.zeros((32,), dtype="float32")
    dev = tvm.cuda(0)
    A_nd = tvm.runtime.tensor(A_np, device=dev)
    B_nd = tvm.runtime.tensor(B_np, device=dev)
    mod(A_nd, B_nd)
    tvm.testing.assert_allclose(B_nd.numpy(), A_np, rtol=0, atol=0)


if __name__ == "__main__":
    test_ptx_griddepcontrol()
