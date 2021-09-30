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
from tvm import te, topi

topi_funcs = {"searchsorted": {"generic": topi.searchsorted}}


def get_implementations(name, axis, dtype, exclusive):
    topi_func_generic = topi_funcs[name]["generic"]
    # topi_func_cuda = topi_funcs[name]["cuda"]

    return {
        "generic": (
            lambda x: topi_func_generic(x, axis, dtype, exclusive=exclusive),
            topi.generic.schedule_extern,
        ),
    }


@tvm.testing.parametrize_targets
def test_cumsum(dev, target):
    n = 1024
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = topi.searchsorted(A, B)
    s = te.create_schedule(C.op)

    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B, C], target)

    dev = tvm.device(target, 0)
    a_np = np.random.uniform(size=n).astype(A.dtype)
    b_np = np.random.uniform(size=n).astype(B.dtype)
    a_np = np.sort(a_np)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    func(a, b, c)
    ref = np.searchsorted(a_np, b_np)
    tvm.testing.assert_allclose(c.numpy(), ref)
    print("ok")


if __name__ == "__main__":
    target = "llvm"
    test_cumsum(tvm.device(target, 0), target)
