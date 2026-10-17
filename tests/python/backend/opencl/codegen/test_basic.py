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
# ruff: noqa: E501
import re

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

target = "opencl"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_opencl_ternary_expression():
    def check_if_then_else(n, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", 1) as i:
                    C[i] = T.max(
                        T.Cast(dtype, 2),
                        T.if_then_else(
                            0 < T.Cast("int32", A[0]), T.Cast(dtype, 1), T.Cast(dtype, 3)
                        ),
                    )

        fun = tvm_tirx.tirx.compile(Module, target=target).mod

        def run_and_check():
            dev = tvm_tirx.opencl(0)
            a = tvm_tirx.runtime.empty((n,), dtype, dev)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    def check_select(n, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", 1) as i:
                    C[i] = T.max(
                        T.Cast(dtype, 2),
                        T.Select(0 < T.Cast("int32", A[0]), T.Cast(dtype, 1), T.Cast(dtype, 3)),
                    )

        fun = tvm_tirx.tirx.compile(Module, target=target).mod

        def run_and_check():
            dev = tvm_tirx.opencl(0)
            a = tvm_tirx.runtime.empty((n,), dtype, dev)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_if_then_else(1, "int8")
    check_if_then_else(1, "uint8")
    check_if_then_else(1, "int16")
    check_if_then_else(1, "uint16")
    check_select(1, "int8")
    check_select(1, "uint8")
    check_select(1, "int16")
    check_select(1, "uint16")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_opencl_inf_nan():
    def check_inf_nan(n, value, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", 1) as i:
                    C[i] = T.Cast(dtype, value)

        fun = tvm_tirx.tirx.compile(Module, target=target).mod

        def run_and_check():
            dev = tvm_tirx.opencl(0)
            a = tvm_tirx.runtime.empty((n,), dtype, dev)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_inf_nan(1, -float("inf"), "float32")
    check_inf_nan(1, -float("inf"), "float64")
    check_inf_nan(1, float("inf"), "float32")
    check_inf_nan(1, float("inf"), "float64")
    check_inf_nan(1, float("nan"), "float32")
    check_inf_nan(1, float("nan"), "float64")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_opencl_max():
    def check_max(n, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", 1) as i:
                    C[i] = T.max(A[0] + T.Cast(dtype, 1), T.Cast(dtype, 0))

        fun = tvm_tirx.tirx.compile(Module, target=target).mod

        def run_and_check():
            dev = tvm_tirx.opencl(0)
            a = tvm_tirx.runtime.empty((n,), dtype, dev)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_max(1, "int8")
    check_max(1, "uint8")
    check_max(1, "int16")
    check_max(1, "uint16")
    check_max(1, "float32")
    check_max(1, "float64")


def test_opencl_erf():
    def check_erf(n, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", 1) as i0:
                    C[i0] = T.erf(A[i0])

        fun = tvm_tirx.tirx.compile(Module, target=target).mod

        source_str = fun.imports[0].inspect_source()
        matches = re.findall("erf", source_str)
        error_matches = re.findall("erff", source_str)
        assert len(matches) == 1 and len(error_matches) == 0

    check_erf(1, "float32")
    check_erf(1, "float64")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
def test_opencl_type_casting():
    @I.ir_module
    class Module:
        @T.function
        def main(C: T.Tensor((32,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("threadIdx.x", 8) as i_0:
                for i_1 in T.vectorized(4):
                    C[i_0 * 4 + i_1] = T.Select(
                        (i_0 * 4 + i_1) // 4 == 3 and (i_0 * 4 + i_1) % 3 == 1,
                        T.float32(1.0),
                        T.float32(0.0),
                    )

    def check_type_casting(n, dtype):
        fun = tvm_tirx.tirx.compile(Module, target=target).mod
        assembly = fun.imports[0].inspect_source()
        lcond = "convert_int4(((convert_uint4(((uint4)(((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3), ((convert_int(get_local_id(0))) == 3)))))"
        rcond = "(convert_uint4(((((int4)(((convert_int(get_local_id(0))))+(1*0), ((convert_int(get_local_id(0))))+(1*1), ((convert_int(get_local_id(0))))+(1*2), ((convert_int(get_local_id(0))))+(1*3))) % ((int4)(3, 3, 3, 3))) == ((int4)(1, 1, 1, 1))))))))"
        pattern_cond = f"({lcond} && {rcond})"
        assert assembly.count(pattern_cond) != 0

        def run_and_check():
            dev = tvm_tirx.opencl(0)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_type_casting(32, "float32")
    # fp16 is not yet supported in ci
    # check_type_casting(dev, 16, "float16")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_opencl(), reason="need opencl")
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("opencl", marks=pytest.mark.gpu),
        pytest.param({"kind": "opencl", "device": "adreno"}, marks=pytest.mark.gpu),
    ],
)
def test_opencl_ceil_log2(target):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    def _check(target, n, dtype):
        target_obj = tvm_tirx.target.Target(target)
        is_adreno = "adreno" in target_obj.attrs.get("device", "")
        inter_dtype = "float32" if is_adreno else "float64"

        @I.ir_module
        class Module:
            @T.function
            def main(C: T.Tensor((n,), "int32")):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", n) as i:
                    C[i] = T.Cast("int32", T.ceil(T.log2(T.Cast(inter_dtype, i))))

        fun = tvm_tirx.tirx.compile(Module, target=target).mod
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

        p = re.compile(r"__kernel void .+\((.*)\)")
        args = p.findall(source)
        return args

    args = get_kernel_args(source)
    max_args = len(args[0].split(","))
    for arg_line in args:
        max_args = max(max_args, len(arg_line.split(",")))
    return max_args


if __name__ == "__main__":
    tvm_tirx.testing.main()
