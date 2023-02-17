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
"""LLVM enablement tests."""
import numpy as np
import tvm
import tvm.testing
from tvm import te


@tvm.testing.requires_llvm
def test_llvm_add_pipeline():
    """all-platform-minimal-test: Check LLVM enablement."""
    arr_size = 1024
    tvm_arr_size = tvm.runtime.convert(arr_size)
    placeholder_a = te.placeholder((tvm_arr_size,), name="A")
    placeholder_b = te.placeholder((tvm_arr_size,), name="B")
    result_aa = te.compute((tvm_arr_size,), placeholder_a, name="A")
    result_bb = te.compute((tvm_arr_size,), placeholder_b, name="B")
    result_t = te.compute(placeholder_a.shape, lambda *i: result_aa(*i) + result_bb(*i), name="T")
    result_c = te.compute(placeholder_a.shape, result_t, name="C")
    schedule = te.create_schedule(result_c.op)
    x_o, x_i = schedule[result_c].split(result_c.op.axis[0], factor=4)
    x_o1, x_o2 = schedule[result_c].split(x_o, factor=13)
    schedule[result_c].parallel(x_o2)
    schedule[result_c].pragma(x_o1, "parallel_launch_point")
    schedule[result_c].pragma(x_o2, "parallel_stride_pattern")
    schedule[result_c].pragma(x_o2, "parallel_barrier_when_finish")
    schedule[result_c].vectorize(x_i)

    def check_llvm():
        # Specifically allow offset to test codepath when offset is available
        decl_buffer_ab = tvm.tir.decl_buffer(
            placeholder_a.shape,
            placeholder_a.dtype,
            elem_offset=te.size_var("Aoffset"),
            offset_factor=8,
            name="A",
        )
        binds = {placeholder_a: decl_buffer_ab}
        # BUILD and invoke the kernel.
        func = tvm.build(schedule, [placeholder_a, placeholder_b, result_c], "llvm", binds=binds)
        dev = tvm.cpu(0)
        # launch the kernel.
        np_arr_a = tvm.nd.array(np.random.uniform(size=arr_size).astype(placeholder_a.dtype), dev)
        np_arr_b = tvm.nd.array(np.random.uniform(size=arr_size).astype(placeholder_b.dtype), dev)
        np_arr_c = tvm.nd.array(np.zeros(arr_size, dtype=result_c.dtype), dev)
        func(np_arr_a, np_arr_b, np_arr_c)
        tvm.testing.assert_allclose(np_arr_c.numpy(), np_arr_a.numpy() + np_arr_b.numpy())

    check_llvm()
