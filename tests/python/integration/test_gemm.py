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
"""Test scheduling and running a gemm!"""
import numpy as np

import tvm
import tvm.testing
from tvm import te


@tvm.testing.requires_gpu
def test_gemm():
    """Test the gemm!"""
    # graph
    dim1_length = 1024
    dim_n = tvm.runtime.convert(dim1_length)
    dim_m = dim_n
    dim_l = dim_n
    placeholder_a = te.placeholder((dim_n, dim_l), name="A")
    placeholder_b = te.placeholder((dim_m, dim_l), name="B")
    axis_k = te.reduce_axis((0, dim_l), name="k")
    result_c = te.compute(
        (dim_n, dim_m),
        lambda ii, jj: te.sum(placeholder_a[ii, axis_k] * placeholder_b[jj, axis_k], axis=axis_k),
        name="CC",
    )
    # schedule
    schedule = te.create_schedule(result_c.op)
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis("threadIdx.y")

    cache_write = schedule.cache_write(result_c, "local")
    cache_read_a = schedule.cache_read(placeholder_a, "shared", [cache_write])
    cache_read_b = schedule.cache_read(placeholder_b, "shared", [cache_write])
    axis_by, axis_yi = schedule[result_c].split(result_c.op.axis[0], factor=block_factor)
    axis_bx, axis_xi = schedule[result_c].split(result_c.op.axis[1], factor=block_factor)
    schedule[result_c].reorder(axis_by, axis_bx, axis_yi, axis_xi)
    schedule[result_c].bind(axis_by, block_y)
    schedule[result_c].bind(axis_bx, block_x)
    axis_ty, axis_yi = schedule[result_c].split(axis_yi, nparts=num_thread)
    axis_tx, axis_xi = schedule[result_c].split(axis_xi, nparts=num_thread)
    schedule[result_c].reorder(axis_ty, axis_tx, axis_yi, axis_xi)
    schedule[result_c].bind(axis_ty, thread_y)
    schedule[result_c].bind(axis_tx, thread_x)
    axis_yo, axis_xo = cache_write.op.axis
    schedule[cache_write].reorder(axis_k, axis_yo, axis_xo)

    schedule[cache_write].compute_at(schedule[result_c], axis_tx)
    schedule[cache_read_a].compute_at(schedule[cache_write], axis_k)
    schedule[cache_read_b].compute_at(schedule[cache_write], axis_k)
    schedule[cache_read_a].double_buffer()
    schedule[cache_read_b].double_buffer()
    axis_ty, axis_xi = schedule[cache_read_a].split(
        schedule[cache_read_a].op.axis[0], nparts=num_thread
    )
    axis_tx, axis_xi = schedule[cache_read_a].split(axis_xi, nparts=num_thread)
    schedule[cache_read_a].bind(axis_ty, thread_y)
    schedule[cache_read_a].bind(axis_tx, thread_x)

    axis_ty, axis_xi = schedule[cache_read_b].split(
        schedule[cache_read_b].op.axis[0], nparts=num_thread
    )
    axis_tx, axis_xi = schedule[cache_read_b].split(axis_xi, nparts=num_thread)
    schedule[cache_read_b].bind(axis_ty, thread_y)
    schedule[cache_read_b].bind(axis_tx, thread_x)

    # lowering test
    schedule = schedule.normalize()

    # one line to build the function.
    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return

        with tvm.target.Target(device):
            f = tvm.build(schedule, [placeholder_a, placeholder_b, result_c])

        # launch the kernel.
        num_n = dim1_length
        num_m = num_n
        num_l = num_n
        a_np = np.random.uniform(size=(num_n, num_l)).astype(placeholder_a.dtype)
        b_np = np.random.uniform(size=(num_m, num_l)).astype(placeholder_b.dtype)
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros((num_n, num_m), dtype=result_c.dtype), dev)
        ftimer = f.time_evaluator(f.entry_name, dev, number=1)
        tcost = ftimer(buff_a, buff_b, buff_c).mean
        print("%s: exec=%g sec/op" % (dev, tcost))
        tvm.testing.assert_allclose(buff_c.numpy(), np.dot(a_np, b_np.T), rtol=1e-5)

    check_device("vulkan")
    check_device("nvptx -mcpu=sm_20")
    check_device("rocm")
    check_device("metal")
    check_device("opencl")
    check_device("cuda")


if __name__ == "__main__":
    test_gemm()
