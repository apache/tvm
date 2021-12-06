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

import tvm
import tvm.testing
from tvm import te
import tvm.contrib.hexagon.hexagon as hexagon
from tvm.contrib import utils
import numpy as np

from ..conftest import requires_hexagon_toolchain, requires_rpc_tracker_and_android_key


@requires_rpc_tracker_and_android_key
@requires_hexagon_toolchain
class TestMatMul:
    M = tvm.testing.parameter(32)
    N = tvm.testing.parameter(32)
    K = tvm.testing.parameter(32)

    def test_matmul(self, M, N, K, rpc_sess, remote_path):
        X = te.placeholder((M, K), dtype="float32")
        Y = te.placeholder((K, N), dtype="float32")
        k1 = te.reduce_axis((0, K), name="k1")
        Z = te.compute((M, N), lambda i, j: te.sum(X[i, k1] * Y[k1, j], axis=[k1]))
        schedule = te.create_schedule(Z.op)

        target_hexagon = tvm.target.hexagon("v68", link_params=True)
        mod = tvm.build(schedule, [X, Y, Z], target=target_hexagon, target_host=target_hexagon)

        temp = utils.tempdir()
        dso_binary_path = temp.relpath(os.path.basename(remote_path))
        mod.save(dso_binary_path)

        rpc_sess.upload(dso_binary_path, target=remote_path)

        mod = rpc_sess.load_module(remote_path)

        x = np.random.uniform(size=[i.value for i in X.shape]).astype(X.dtype)
        y = np.random.uniform(size=[i.value for i in Y.shape]).astype(Y.dtype)
        z = np.zeros([i.value for i in Z.shape], dtype=Z.dtype)

        dev = rpc_sess.hexagon(0)
        xt = tvm.nd.array(x, device=dev)
        yt = tvm.nd.array(y, device=dev)
        zt = tvm.nd.array(z, device=dev)
        mod(xt, yt, zt)

        target_llvm = tvm.target.Target("llvm")
        mod = tvm.build(schedule, [X, Y, Z], target=target_llvm, target_host=target_llvm)
        device = tvm.cpu(0)
        xtcpu = tvm.nd.array(x, device)
        ytcpu = tvm.nd.array(y, device)
        ztcpu = tvm.nd.array(z, device)
        mod(xtcpu, ytcpu, ztcpu)

        tvm.testing.assert_allclose(zt.asnumpy(), ztcpu.asnumpy(), rtol=1e-4)
