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

import numpy as np

import tvm
import tvm.testing
from tvm.script import tirx as T


def generate_random_data(shape, dtype):
    np.random.seed(0)
    return np.random.randn(*shape).astype(dtype)


def create_tvm_arrays(data_np, device):
    return [tvm.runtime.tensor(data, device=device) for data in data_np]


def build_and_run_tvm_func(sch, target, *args):
    func = tvm.compile(sch.mod, target=target)
    func(*args)
    return func, args[-1]


def from_source(code):
    return tvm.script.from_source(code, s_tir=True)


def verify_result(C_tvm, C_np):
    tvm.testing.assert_allclose(C_tvm.numpy(), C_np, rtol=1e-5)


def verify_tir_code(code):
    assert from_source(code).script() == code


def verify_cuda_code_array(func, dim_num, dtype, *dims):
    generated_code = func.mod.imports[0].inspect_source()

    match = re.search(r"// print_buffer starts(.*?)// print_buffer ends", generated_code, re.DOTALL)
    if not match:
        raise AssertionError("print_buffer section not found in generated code")

    print_buffer_section = match.group(1).strip()
    loop_pattern = re.compile(r"for \(int i(\d+) = 0; i\1 < (\d+); \+\+i\1\)")
    loops = loop_pattern.findall(print_buffer_section)
    if len(loops) != dim_num:
        raise AssertionError(f"Expected {dim_num} nested loops, but found {len(loops)}")

    loop_limits = [int(limit) for _, limit in loops]
    if loop_limits != list(dims):
        raise AssertionError(f"Expected loop limits {dims}, but found {loop_limits}")

    dtype_to_printf = {"float32": "%f", "float16": "%f", "int32": "%d", "uint32": "%u"}
    expected_printf_specifier = dtype_to_printf.get(dtype)
    if not expected_printf_specifier:
        raise AssertionError(f"Unsupported dtype {dtype}")
    variable_access_pattern = r"\w+\[.*\]"

    if dtype == "float16":
        # Look for `printf("%f", static_cast<float>(C[...]))`
        printf_pattern = re.compile(
            r'printf\s*\(\s*"'
            + re.escape(expected_printf_specifier)
            + r'"\s*,\s*static_cast<float>\('
            + variable_access_pattern
            + r"\)\s*\)"
        )
    else:
        # Look for `printf("%f", C[...])`
        printf_pattern = re.compile(
            r'printf\s*\(\s*"'
            + re.escape(expected_printf_specifier)
            + r'"\s*,\s*'
            + variable_access_pattern
            + r"\s*\)"
        )

    if not printf_pattern.search(print_buffer_section):
        raise AssertionError(
            f'Expected element printf statement with format "{expected_printf_specifier}" and a buffer access, but not found'  # noqa: E501
        )


def verify_cuda_code_scalar(func, dtype, expected_value_or_varname):
    generated_code = func.mod.imports[0].inspect_source()

    all_print_blocks = re.findall(
        r"// print_buffer starts(.*?)// print_buffer ends", generated_code, re.DOTALL
    )
    if not all_print_blocks:
        raise AssertionError("No print_buffer sections found in generated code")

    dtype_to_printf = {"float32": "%f", "float16": "%f", "int32": "%d", "uint32": "%u"}
    expected_printf = dtype_to_printf.get(dtype)
    if not expected_printf:
        raise AssertionError(f"Unsupported dtype for scalar verification: {dtype}")

    value_pattern = ""
    if isinstance(expected_value_or_varname, int | float):
        if "float" in dtype:
            value_pattern = re.escape(str(float(expected_value_or_varname))) + "f?"
        else:
            value_pattern = re.escape(str(int(expected_value_or_varname)))
    elif isinstance(expected_value_or_varname, str):
        value_pattern = re.escape(expected_value_or_varname)
    else:
        raise TypeError(
            "expected_value_or_varname must be a number (for literals) or a string (for variables)"
        )

    if dtype == "float16":
        printf_pattern = re.compile(
            r'printf\s*\(\s*".*?'
            + re.escape(expected_printf)
            + r'.*?",\s*static_cast<float>\(\s*'
            + value_pattern
            + r"\s*\)\s*\)"
        )
    else:
        printf_pattern = re.compile(
            r'printf\s*\(\s*".*?'
            + re.escape(expected_printf)
            + r'.*?",\s*'
            + value_pattern
            + r"\s*\)"
        )

    for block in all_print_blocks:
        if printf_pattern.search(block):
            return

    raise AssertionError(
        f'Could not find a scalar printf with format "{expected_printf}" and value/variable '
        f'"{expected_value_or_varname}" in any print_buffer block.'
    )


def verify_cuda_code_string(func, expected_var_name, expected_string_literal):
    generated_code = func.mod.imports[0].inspect_source()

    all_print_blocks = re.findall(
        r"// print_buffer starts(.*?)// print_buffer ends", generated_code, re.DOTALL
    )
    if not all_print_blocks:
        raise AssertionError("No print_buffer sections found in generated code")

    var_printf_pattern = re.compile(
        r'printf\s*\(\s*".*?%s.*?",\s*\(char\*\)' + re.escape(expected_var_name) + r"\s*\)"
    )
    literal_printf_pattern = re.compile(
        r'printf\s*\(\s*".*?%s.*?",\s*\(char\*\)\s*"'
        + re.escape(expected_string_literal)
        + r'"\s*\)'
    )

    for block in all_print_blocks:
        if var_printf_pattern.search(block) or literal_printf_pattern.search(block):
            return

    raise AssertionError(
        f'Could not find a string printf using variable "{expected_var_name}" or '
        f'string literal "{expected_string_literal}" in any print_buffer block.'
    )


@tvm.testing.requires_cuda
def test_print():
    DEV = tvm.cuda()
    target = tvm.target.Target("cuda")

    def test_vector_add_1D(dtype, dtype_str):
        M = 6
        M_BLK = 6
        dim_num = 1
        A_np, B_np = generate_random_data((M,), dtype), generate_random_data((M,), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func(s_tir=True)
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M,), dtype_str)
            B = T.match_buffer(B_ptr, (M,), dtype_str)
            C = T.match_buffer(C_ptr, (M,), dtype_str)

            for i in T.grid(M):
                with T.sblock("C"):
                    vi = T.axis.spatial(M, i)
                    C[vi] = A[vi] + B[vi]
                T.print_buffer(C.data, dtype_str, False, False, dim_num, (M,))

        sch = tvm.s_tir.Schedule(add_func)
        blk = sch.get_sblock("C")
        i = sch.get_loops(blk)[0]

        i0, i1 = sch.split(i, factors=[None, M_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(i1, "threadIdx.x")

        C_np_tmp = np.zeros((M,), dtype=dtype)
        C_tvm = tvm.runtime.tensor(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code_array(func, dim_num, dtype_str, M)

    def test_vector_add_2D(dtype, dtype_str):
        M, N = 6, 6
        M_BLK, N_BLK = 6, 6
        dim_num = 2
        A_np, B_np = generate_random_data((M, N), dtype), generate_random_data((M, N), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func(s_tir=True)
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), dtype_str)
            B = T.match_buffer(B_ptr, (M, N), dtype_str)
            C = T.match_buffer(C_ptr, (M, N), dtype_str)

            for i, j in T.grid(M, N):
                with T.sblock("C"):
                    vi = T.axis.spatial(M, i)
                    vj = T.axis.spatial(N, j)
                    C[vi, vj] = A[vi, vj] + B[vi, vj]
                T.print_buffer(C.data, C.dtype, False, False, dim_num, (M, N))

        sch = tvm.s_tir.Schedule(add_func)
        blk = sch.get_sblock("C")
        i, j = sch.get_loops(blk)

        i0, i1 = sch.split(i, factors=[None, M_BLK])
        j0, j1 = sch.split(j, factors=[None, N_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(j0, "blockIdx.y")
        sch.bind(i1, "threadIdx.x")
        sch.bind(j1, "threadIdx.y")

        C_np_tmp = np.zeros((M, N), dtype=dtype)
        C_tvm = tvm.runtime.tensor(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code_array(func, dim_num, dtype_str, M, N)

    def test_vector_add_3D(dtype, dtype_str):
        M, N, K = 6, 6, 6
        M_BLK, N_BLK, K_BLK = 6, 6, 6
        dim_num = 3
        A_np, B_np = generate_random_data((M, N, K), dtype), generate_random_data((M, N, K), dtype)
        C_np = A_np + B_np

        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func(s_tir=True)
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N, K), dtype_str)
            B = T.match_buffer(B_ptr, (M, N, K), dtype_str)
            C = T.match_buffer(C_ptr, (M, N, K), dtype_str)

            for i, j, k in T.grid(M, N, K):
                with T.sblock("C"):
                    vi = T.axis.spatial(M, i)
                    vj = T.axis.spatial(N, j)
                    vk = T.axis.spatial(K, k)
                    C[vi, vj, vk] = A[vi, vj, vk] + B[vi, vj, vk]
                T.print_buffer(C.data, C.dtype, False, False, dim_num, (M, N, K))

        sch = tvm.s_tir.Schedule(add_func)
        blk = sch.get_sblock("C")
        i, j, k = sch.get_loops(blk)

        i0, i1 = sch.split(i, factors=[None, M_BLK])
        j0, j1 = sch.split(j, factors=[None, N_BLK])
        k0, k1 = sch.split(k, factors=[None, K_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(j0, "blockIdx.y")
        sch.bind(k0, "blockIdx.z")
        sch.bind(i1, "threadIdx.x")
        sch.bind(j1, "threadIdx.y")
        sch.bind(k1, "threadIdx.z")

        C_np_tmp = np.zeros((M, N, K), dtype=dtype)
        C_tvm = tvm.runtime.tensor(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code_array(func, dim_num, dtype_str, M, N, K)

    def test_const_scalar(dtype, dtype_str):
        M = 6
        M_BLK = 6
        dim_num = 1
        A_np, B_np = generate_random_data((M,), dtype), generate_random_data((M,), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func(s_tir=True)
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M,), dtype_str)
            B = T.match_buffer(B_ptr, (M,), dtype_str)
            C = T.match_buffer(C_ptr, (M,), dtype_str)
            Ten: T.let = T.IntImm(dtype_str, 10)

            for i in T.grid(M):
                with T.sblock("C"):
                    vi = T.axis.spatial(M, i)
                    C[vi] = A[vi] + B[vi]
                T.print_buffer(Ten, "int32", False, True, dim_num, ())

        sch = tvm.s_tir.Schedule(add_func)
        blk = sch.get_sblock("C")
        i = sch.get_loops(blk)[0]

        i0, i1 = sch.split(i, factors=[None, M_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(i1, "threadIdx.x")

        C_np_tmp = np.zeros((M,), dtype=dtype)
        C_tvm = tvm.runtime.tensor(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code_scalar(func, dtype_str, 10)

    def test_string(dtype, dtype_str, test_string):
        M = 6
        M_BLK = 6
        dim_num = 1
        A_np, B_np = generate_random_data((M,), dtype), generate_random_data((M,), dtype)
        C_np = A_np + B_np
        A_tvm, B_tvm = create_tvm_arrays([A_np, B_np], DEV)

        @T.prim_func(s_tir=True)
        def add_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M,), dtype_str)
            B = T.match_buffer(B_ptr, (M,), dtype_str)
            C = T.match_buffer(C_ptr, (M,), dtype_str)
            string_var = T.StringImm(test_string)

            for i in T.grid(M):
                with T.sblock("C"):
                    vi = T.axis.spatial(M, i)
                    C[vi] = A[vi] + B[vi]
                T.print_buffer(string_var, "int8", True, False, dim_num, ())

        sch = tvm.s_tir.Schedule(add_func)
        blk = sch.get_sblock("C")
        i = sch.get_loops(blk)[0]

        i0, i1 = sch.split(i, factors=[None, M_BLK])

        sch.bind(i0, "blockIdx.x")
        sch.bind(i1, "threadIdx.x")

        C_np_tmp = np.zeros((M,), dtype=dtype)
        C_tvm = tvm.runtime.tensor(C_np_tmp, device=DEV)
        func, C_tvm = build_and_run_tvm_func(sch, target, A_tvm, B_tvm, C_tvm)
        verify_result(C_tvm, C_np)
        verify_tir_code(add_func.script())
        verify_cuda_code_string(func, "string_var", test_string)

    test_vector_add_1D(np.float32, "float32")
    test_vector_add_2D(np.int32, "int32")
    test_vector_add_2D(np.float16, "float16")
    test_vector_add_3D(np.uint32, "uint32")
    test_string(np.float32, "float32", "hello tirx!")
    test_const_scalar(np.int32, "int32")


if __name__ == "__main__":
    test_print()
