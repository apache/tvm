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
import tvm.testing
import re

target = "opencl"


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_ternary_expression():
    def check_if_then_else(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        true_value = tvm.tir.const(1, dtype=dtype)
        false_value = tvm.tir.const(3, dtype=dtype)
        max_lhs = tvm.tir.const(2, dtype=dtype)
        max_rhs = tvm.tir.if_then_else(A[0] > 0, true_value, false_value)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    def check_select(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        true_value = tvm.tir.const(1, dtype=dtype)
        false_value = tvm.tir.const(3, dtype=dtype)
        max_lhs = tvm.tir.const(2, dtype=dtype)
        max_rhs = tvm.tir.Select(A[0] > 0, true_value, false_value)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_if_then_else(dev, 1, "int8")
    check_if_then_else(dev, 1, "uint8")
    check_if_then_else(dev, 1, "int16")
    check_if_then_else(dev, 1, "uint16")
    check_select(dev, 1, "int8")
    check_select(dev, 1, "uint8")
    check_select(dev, 1, "int16")
    check_select(dev, 1, "uint16")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_inf_nan():
    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)
        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_max():
    def check_max(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        max_lhs = A[0] + tvm.tir.const(1, dtype=dtype)
        max_rhs = tvm.tir.const(0, dtype=dtype)
        C = te.compute((n,), lambda i: tvm.te.max(max_lhs, max_rhs), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_max(dev, 1, "int8")
    check_max(dev, 1, "uint8")
    check_max(dev, 1, "int16")
    check_max(dev, 1, "uint16")
    check_max(dev, 1, "float32")
    check_max(dev, 1, "float64")


def test_opencl_erf():
    def check_erf(dev, n, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        C = te.compute(A.shape, lambda *i: te.erf(A(*i)), name="C")
        s = te.create_schedule(C.op)
        s[C].bind(s[C].op.axis[0], te.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, C], target)
        source_str = fun.imported_modules[0].get_source()
        matches = re.findall("erf", source_str)
        error_matches = re.findall("erff", source_str)
        assert len(matches) == 1 and len(error_matches) == 0

    dev = tvm.device(target, 0)

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float64")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_type_casting():
    def check_type_casting(ctx, n, dtype):
        block_size = 4
        C = te.compute(
            (n,),
            lambda i: tvm.tir.Select(
                tvm.tir.all(
                    *[
                        i // block_size == tvm.tir.const(3, "int32"),
                        i % block_size == tvm.tir.const(3, "int32"),
                    ]
                ),
                tvm.tir.const(1, dtype),
                tvm.tir.const(0, dtype),
            ),
            name="C",
        )
        s = te.create_schedule(C.op)
        (tx, vx) = s[C].split(s[C].op.axis[0], factor=block_size)
        s[C].vectorize(vx)
        thrx = te.thread_axis("threadIdx.x")

        s[C].bind(tx, thrx)
        fun = tvm.build(s, [C], target)

        c = tvm.nd.empty((n,), dtype, ctx)
        assembly = fun.imported_modules[0].get_source()
        false_branch = "((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f))"
        true_branch = "((float4)(1.000000e+00f, 1.000000e+00f, 1.000000e+00f, 1.000000e+00f))"
        lcond = "(convert_uint4(((uint4)((((int)get_local_id(0)) == 3), (((int)get_local_id(0)) == 3), (((int)get_local_id(0)) == 3), (((int)get_local_id(0)) == 3)))))"
        rcond = "(convert_uint4((((int4)((0)+(1*0), (0)+(1*1), (0)+(1*2), (0)+(1*3))) == ((int4)(3, 3, 3, 3)))))"
        cond = "({} && {})".format(lcond, rcond)
        select = "select({}, {}, {})".format(false_branch, true_branch, cond)
        count = assembly.count(select)
        assert count == 1

        fun(c)

    dev = tvm.device(target, 0)

    check_type_casting(dev, 16, "float32")


if __name__ == "__main__":
    tvm.testing.main()
