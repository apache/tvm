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
import tvm
import numpy as np

def test_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.placeholder((m, n), name="X")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    res = tvm.scan(s_init, s_update, s_state)

    # schedule
    s = tvm.create_schedule(res.op)
    num_thread = 256
    block_x = tvm.thread_axis(None, "blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
    s[s_init].bind(xo, block_x)
    s[s_init].bind(xi, thread_x)
    xo, xi = s[s_update].split(s_update.op.axis[1], factor=num_thread)
    s[s_update].bind(xo, block_x)
    s[s_update].bind(xi, thread_x)

    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return
        fscan = tvm.build(s, [X, res],
                          device,
                          name="myscan")
        # launch the kernel.
        n = 1024
        m = 10
        a_np = np.random.uniform(size=(m, n)).astype(res.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros((m, n), dtype=res.dtype), ctx)
        fscan(a, b)
        tvm.testing.assert_allclose(
            b.asnumpy(), np.cumsum(a_np, axis=0))

    check_device("vulkan")
    check_device("cuda")
    check_device("metal")
    check_device("opencl")


if __name__ == "__main__":
    test_scan()
