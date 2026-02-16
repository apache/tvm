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
import re

import tvm
import tvm.testing
from tvm.script import tir as T, ir as I

target = "opencl"


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_ternary_expression():
    def check_if_then_else(dev, n, dtype):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(1, i)
                        T.reads(A[0])
                        T.writes(C[v_i])
                        C[v_i] = T.max(
                            T.Cast(dtype, 2),
                            T.if_then_else(
                                0 < T.Cast("int32", A[0]),
                                T.Cast(dtype, 1),
                                T.Cast(dtype, 3),
                            ),
                        )

        fun = tvm.tir.build(Module, target=target)
        a = tvm.runtime.empty((n,), dtype, dev)
        c = tvm.runtime.empty((n,), dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    def check_select(dev, n, dtype):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(1, i)
                        T.reads(A[0])
                        T.writes(C[v_i])
                        C[v_i] = T.max(
                            T.Cast(dtype, 2),
                            T.Select(
                                0 < T.Cast("int32", A[0]),
                                T.Cast(dtype, 1),
                                T.Cast(dtype, 3),
                            ),
                        )

        fun = tvm.tir.build(Module, target=target)
        a = tvm.runtime.empty((n,), dtype, dev)
        c = tvm.runtime.empty((n,), dtype, dev)
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
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(1, i)
                        T.reads()
                        T.writes(C[v_i])
                        C[v_i] = T.Cast(dtype, value)

        fun = tvm.tir.build(Module, target=target)
        a = tvm.runtime.empty((n,), dtype, dev)
        c = tvm.runtime.empty((n,), dtype, dev)
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
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(1, i)
                        T.reads(A[0])
                        T.writes(C[v_i])
                        C[v_i] = T.max(A[0] + T.Cast(dtype, 1), T.Cast(dtype, 0))

        fun = tvm.tir.build(Module, target=target)
        a = tvm.runtime.empty((n,), dtype, dev)
        c = tvm.runtime.empty((n,), dtype, dev)
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
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i0 in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i0 = T.axis.spatial(1, i0)
                        T.reads(A[v_i0])
                        T.writes(C[v_i0])
                        C[v_i0] = T.erf(A[v_i0])

        fun = tvm.tir.build(Module, target=target)

        source_str = fun.imports[0].inspect_source()
        matches = re.findall("erf", source_str)
        error_matches = re.findall("erff", source_str)
        assert len(matches) == 1 and len(error_matches) == 0

    dev = tvm.device(target, 0)

    check_erf(dev, 1, "float32")
    check_erf(dev, 1, "float64")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
def test_opencl_type_casting():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(C: T.Buffer((32,), "float32")):
            T.func_attr({"tir.noalias": True})
            for i_0 in T.thread_binding(8, thread="threadIdx.x"):
                for i_1 in T.vectorized(4):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(32, i_0 * 4 + i_1)
                        T.reads()
                        T.writes(C[v_i])
                        C[v_i] = T.Select(
                            v_i // 4 == 3 and v_i % 3 == 1, T.float32(1.0), T.float32(0.0)
                        )

    def check_type_casting(ctx, n, dtype):
        fun = tvm.tir.build(Module, target=target)
        c = tvm.runtime.empty((n,), dtype, ctx)
        assembly = fun.imports[0].inspect_source()
        lcond = "convert_int4(((convert_uint4(((uint4)(((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3)))))"
        rcond = "(convert_uint4(((((int4)(((convert_int(get_local_id(0))))+(1*0), ((convert_int(get_local_id(0))))+(1*1), ((convert_int(get_local_id(0))))+(1*2), ((convert_int(get_local_id(0))))+(1*3))) % ((int4)(3, 3, 3, 3))) == ((int4)(1, 1, 1, 1))))))))"
        pattern_cond = "({} && {})".format(lcond, rcond)
        assert assembly.count(pattern_cond) != 0
        fun(c)

    dev = tvm.device(target, 0)

    check_type_casting(dev, 32, "float32")
    # fp16 is not yet supported in ci
    # check_type_casting(dev, 16, "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl", {"kind": "opencl", "device": "adreno"})
def test_opencl_ceil_log2(target):
    def _check(target, n, dtype):
        target_obj = tvm.target.Target(target)
        is_adreno = "adreno" in target_obj.attrs.get("device", "")
        inter_dtype = "float32" if is_adreno else "float64"

        @I.ir_module
        class Module:
            @T.prim_func
            def main(C: T.Buffer((n,), "int32")):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(n, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(n, i)
                        T.reads()
                        T.writes(C[v_i])
                        C[v_i] = T.Cast("int32", T.ceil(T.log2(T.Cast(inter_dtype, v_i))))

        fun = tvm.tir.build(Module, target=target)
        assembly = fun.imports[0].inspect_source()
        if is_adreno:
            pattern = "convert_float"
        else:
            pattern = "convert_double"
        assert assembly.count(pattern) != 0

    _check(target, 32, "float32")


def _get_maximum_kernel_args(source):
    def get_kernel_args(source):
        import re

        p = re.tir.build(r"__kernel void .+\((.*)\)")
        args = p.findall(source)
        return args

    args = get_kernel_args(source)
    max_args = len(args[0].split(","))
    for arg_line in args:
        max_args = max(max_args, len(arg_line.split(",")))
    return max_args


if __name__ == "__main__":
    tvm.testing.main()
