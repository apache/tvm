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
from tvm import te
import numpy as np
from tvm import topi
import unittest
from tvm.contrib.nvcc import have_fp16, have_int8, have_bf16
from tvm.contrib import nvcc
import tvm.testing

tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_metal_inf_nan():
    target = "metal"

    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float16")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float16")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_metal
def test_metal_erf():
    target = "metal"

    def check_erf(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        C = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], tx)
        fun = tvm.build(s, [A, C], target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float16")


if __name__ == "__main__":
    test_metal_inf_nan()
    test_metal_erf()
