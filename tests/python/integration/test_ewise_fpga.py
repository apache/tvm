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
"""Test elementwise ops on fpga."""
import os

import numpy as np
import tvm
import tvm.testing
from tvm import te

os.environ["XCL_EMULATION_MODE"] = "1"
os.environ["CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA"] = "1"


@tvm.register_func
def tvm_callback_vhls_postproc(code, _):
    """Hook to inspect the Vivado HLS code before actually run it"""
    print(code)
    return code


def test_exp():
    """Test scheduling and running exp function."""
    # graph
    arr_length = 1024
    arr_length_tvm = tvm.runtime.convert(arr_length)
    placeholder_b = te.placeholder((arr_length_tvm,), name="A")
    result_b = te.compute(placeholder_b.shape, lambda *i: te.exp(placeholder_b(*i)), name="B")
    schedule = te.create_schedule(result_b.op)
    # create iter var and assign them tags.
    axis1, _ = schedule[result_b].split(result_b.op.axis[0], nparts=1)
    schedule[result_b].bind(axis1, te.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.testing.device_enabled(device):
            return
        dev = tvm.device(device, 0)
        fexp = tvm.build(schedule, [placeholder_b, result_b], device, host, name="myexp")
        dev = tvm.device(device, 0)
        # launch the kernel.
        buff_a = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_b.dtype), dev)
        buff_b = tvm.nd.array(np.zeros(arr_length, dtype=result_b.dtype), dev)
        fexp(buff_a, buff_b)
        tvm.testing.assert_allclose(buff_b.numpy(), np.exp(buff_a.numpy()), rtol=1e-5)

    check_device("sdaccel")
    if "AWS_PLATFORM" in os.environ:
        check_device("sdaccel -device=" + os.environ.get("AWS_PLATFORM"))

    check_device("aocl_sw_emu")


def test_multi_kernel():
    """Test scheduling with multiple computes."""
    # graph
    arr_length = 1024
    arr_length_tvm = tvm.runtime.convert(arr_length)
    placeholder_a = te.placeholder((arr_length_tvm,), name="A")
    placeholder_b = te.placeholder((arr_length_tvm,), name="B")
    result_c = te.compute(
        placeholder_a.shape, lambda *i: placeholder_a(*i) + placeholder_b(*i), name="C"
    )
    result_d = te.compute(
        placeholder_a.shape, lambda *i: placeholder_a(*i) + result_c(*i), name="D"
    )
    schedule = te.create_schedule(result_d.op)
    # create iter var and assign them tags.
    axis1, _ = schedule[result_c].split(result_c.op.axis[0], nparts=1)
    schedule[result_c].bind(axis1, te.thread_axis("pipeline"))
    axis1, _ = schedule[result_d].split(result_d.op.axis[0], nparts=1)
    schedule[result_d].bind(axis1, te.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.testing.device_enabled(device):
            return
        dev = tvm.device(device, 0)
        fadd = tvm.build(
            schedule, [placeholder_a, placeholder_b, result_c, result_d], device, host, name="myadd"
        )
        dev = tvm.device(device, 0)
        # launch the kernel.
        buff_a = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_a.dtype), dev)
        buff_b = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_b.dtype), dev)
        buff_c = tvm.nd.array(np.random.uniform(size=arr_length).astype(result_c.dtype), dev)
        buff_d = tvm.nd.array(np.random.uniform(size=arr_length).astype(result_d.dtype), dev)
        fadd(buff_a, buff_b, buff_c, buff_d)
        tvm.testing.assert_allclose(buff_d.numpy(), buff_a.numpy() * 2 + buff_b.numpy(), rtol=1e-5)

    check_device("sdaccel")
    check_device("aocl_sw_emu")


if __name__ == "__main__":
    test_exp()
    test_multi_kernel()
