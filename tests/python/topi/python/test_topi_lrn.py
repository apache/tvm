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
"""Test code for local response normalization"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple
import tvm.topi.testing
import tvm.testing

_lrn_schedule = {
    "generic": topi.generic.schedule_lrn,
    "gpu": topi.cuda.schedule_lrn,
    "opencl": topi.cuda.schedule_lrn,
    "metal": topi.cuda.schedule_lrn,
    "rocm": topi.cuda.schedule_lrn,
    "vulkan": topi.cuda.schedule_lrn,
    "nvptx": topi.cuda.schedule_lrn,
}


def verify_lrn(shape, size, axis, bias, alpha, beta, dtype="float32", rtol=1e-5, atol=1e-5):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = topi.nn.lrn(A, size, axis, alpha, beta, bias)

    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = tvm.topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)

    def check_device(device):
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            s_func = tvm.topi.testing.dispatch(device, _lrn_schedule)
            s = s_func([B])
        dev = tvm.device(device, 0)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), dev)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=rtol, atol=atol)

    for device in ["llvm", "cuda", "opencl", "metal", "rocm", "vulkan", "nvptx"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_lrn():
    verify_lrn((1, 3, 5, 5), 3, 1, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5)
    verify_lrn((1, 3, 20, 20), 3, 1, 2.0, 1.0, 0.75)
    verify_lrn((1, 3, 5, 5), 3, 3, 1.0, 1.0, 0.5, dtype="float16", rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_lrn()
