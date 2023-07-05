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
"""Test scheduling adn running scan operators."""
import numpy as np

import tvm
import tvm.testing
from tvm import te


@tvm.testing.requires_gpu
def test_scan():
    """Test scan operators."""
    size_var_m = te.size_var("m")
    size_var_n = te.size_var("n")
    placeholder_x = te.placeholder((size_var_m, size_var_n), name="X")
    s_state = te.placeholder((size_var_m, size_var_n))
    s_init = te.compute((1, size_var_n), lambda _, i: placeholder_x[0, i])
    s_update = te.compute(
        (size_var_m, size_var_n), lambda t, i: s_state[t - 1, i] + placeholder_x[t, i]
    )
    scan = tvm.te.scan(s_init, s_update, s_state)
    # test scan + compute case
    res = te.compute((size_var_m, size_var_n), lambda i, j: scan[i, j])

    # schedule
    schedule = te.create_schedule(res.op)
    num_thread = 256
    block_x = te.thread_axis(None, "blockIdx.x")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    axis_xo, axis_xi = schedule[s_init].split(s_init.op.axis[1], factor=num_thread)
    schedule[s_init].bind(axis_xo, block_x)
    schedule[s_init].bind(axis_xi, thread_x)
    axis_xo, axis_xi = schedule[s_update].split(s_update.op.axis[1], factor=num_thread)
    schedule[s_update].bind(axis_xo, block_x)
    schedule[s_update].bind(axis_xi, thread_x)
    axis_xo, axis_xi = schedule[res].split(res.op.axis[1], factor=num_thread)
    schedule[res].bind(axis_xo, block_x)
    schedule[res].bind(axis_xi, thread_x)

    # one line to build the function.
    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fscan = tvm.build(schedule, [placeholder_x, res], device, name="myscan")
        # launch the kernel.
        num_n = 1024
        num_m = 10
        a_np = np.random.uniform(size=(num_m, num_n)).astype(res.dtype)
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(np.zeros((num_m, num_n), dtype=res.dtype), dev)
        fscan(buff_a, buff_b)
        tvm.testing.assert_allclose(buff_b.numpy(), np.cumsum(a_np, axis=0))

    check_device("vulkan")
    check_device("cuda")
    check_device("metal")
    check_device("opencl")


if __name__ == "__main__":
    test_scan()
