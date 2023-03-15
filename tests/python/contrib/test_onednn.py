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

import sys

import pytest

import tvm
import tvm.testing
from tvm import te
from tvm import relay
from tvm.contrib import onednn
from tvm.contrib.nvcc import have_fp16
from tvm.contrib import graph_executor
import numpy as np
import tvm.topi.testing
import tvm.testing


requires_onednn = pytest.mark.skipif(
    tvm.get_global_func("tvm.contrib.onednn.conv2d.forward", True) is None,
    reason="OneDNN is not enabled",
)


def verify_softmax(shape, axis, dtype="float32", log_softmax=False):
    onednn_op = onednn.log_softmax if log_softmax else onednn.softmax
    testing_op = (
        tvm.topi.testing.log_softmax_python if log_softmax else tvm.topi.testing.softmax_python
    )

    A = te.placeholder(shape, dtype=dtype, name="A")
    B = onednn_op(A, axis)
    s = te.create_schedule([B.op])

    dev = tvm.opencl()
    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = testing_op(a_np)
    print("### alloc array a")
    a = tvm.nd.array(a_np, dev)
    print("### alloc array b")
    b = tvm.nd.array(b_np, dev)
    print("tvm.build")
    f = tvm.build(s, [A, B], target="opencl --host=llvm", name="softmax")
    print("f run")
    f(a, b)
    print("testing assert_allclose")
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@requires_onednn
def test_softmax():
    verify_softmax((32, 10), 1)

if __name__ == "__main__":
    #tvm.testing.main()
    test_softmax()
