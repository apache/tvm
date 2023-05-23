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

""" Test parallelism for multiple different scalar workloads. """

import numpy as np

import tvm
from tvm.script import tir as T

from .infrastructure import get_hexagon_target

TEST_OUTPUT_TEMPLATE = (
    "Test {} with {} operations... \n"
    "    -Single Thread: {} ms \n"
    "    -Parallel: {} ms\n    -Speedup: {}x\n"
)


def get_add_operator(operations):
    """Generate add operator."""

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations], dtype="float64")
        b_buffer = T.match_buffer(b, [operations], dtype="float64")
        c_buffer = T.match_buffer(c, [operations], dtype="float64")
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_buffer[vn_ind] = a_buffer[vn_ind] + b_buffer[vn_ind]

    return operator


def get_multiply_operator(operations):
    """Generate multiply operator."""

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations], dtype="float64")
        b_buffer = T.match_buffer(b, [operations], dtype="float64")
        c_buffer = T.match_buffer(c, [operations], dtype="float64")
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_buffer[vn_ind] = a_buffer[vn_ind] * b_buffer[vn_ind]

    return operator


def get_sub_operator(operations):
    """Generate subtract operator."""

    @T.prim_func
    def operator(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        a_buffer = T.match_buffer(a, [operations], dtype="float64")
        b_buffer = T.match_buffer(b, [operations], dtype="float64")
        c_buffer = T.match_buffer(c, [operations], dtype="float64")
        for n in T.grid(operations):
            with T.block("c_buffer"):
                vn_ind = T.axis.remap("S", [n])
                c_buffer[vn_ind] = a_buffer[vn_ind] - b_buffer[vn_ind]

    return operator


def evaluate(hexagon_session, operations, expected, sch):
    """Evalute schedule."""
    shape = operations
    dtype = "float64"

    func_tir = tvm.build(sch.mod["main"], target=get_hexagon_target("v68"))
    module = hexagon_session.load_module(func_tir)

    # np.random.random returns float64 by default, but make the cast explicit
    # to make it easier to switch when necessary.
    a = np.random.random(shape).astype(dtype)
    b = np.random.random(shape).astype(dtype)
    c = np.zeros(shape, dtype=dtype)

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
    b_hexagon = tvm.runtime.ndarray.array(b, device=hexagon_session.device)
    c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device)

    # These are reduced for CI but number=100 and repeat=10 does a good job of removing noise.
    number = 1
    repeat = 1

    timer = module.time_evaluator(
        "__tvm_main__", hexagon_session.device, number=number, repeat=repeat
    )
    runtime = timer(a_hexagon, b_hexagon, c_hexagon)

    tvm.testing.assert_allclose(c_hexagon.asnumpy(), expected(a, b))

    return round(runtime.mean * 1000, 6)


class TestMatMulVec:
    """MatMul test class."""

    (operation_name, operator_producer, expected_output_producer,) = tvm.testing.parameters(
        ("add", get_add_operator, (lambda a, b: a + b)),
        ("mul", get_multiply_operator, (lambda a, b: a * b)),
        ("sub", get_sub_operator, (lambda a, b: a - b)),
    )

    # Removed most of these to speedup CI.
    operations = tvm.testing.parameter(
        128,
        # 256,
        # 512,
        # Single thread runs faster since L2 cache can handle the entire request quickly
        # 1024,
        # 2048,
        # Significant performance degredation once the inputs and outputs cannot all fit in L2
        # 4096,
        # 8192,
        # 16384,
    )

    split_factor = tvm.testing.parameter(4)

    @tvm.testing.requires_hexagon
    def test_add(
        self,
        hexagon_session,
        operation_name,
        operator_producer,
        expected_output_producer,
        operations,
        split_factor,
    ):
        """Test Add operator."""

        sch = tvm.tir.Schedule(operator_producer(operations))
        single_thread_runtime = evaluate(hexagon_session, operations, expected_output_producer, sch)

        sch = tvm.tir.Schedule(operator_producer(operations))
        block = sch.get_block("c_buffer")
        b = sch.get_loops(block)
        b_output, _ = sch.split(b[0], factors=[split_factor, None])
        sch.parallel(b_output)
        parallel_runtime = evaluate(hexagon_session, operations, expected_output_producer, sch)

        speedup = round(single_thread_runtime / parallel_runtime, 2)
        print(
            TEST_OUTPUT_TEMPLATE.format(
                operation_name, operations, single_thread_runtime, parallel_runtime, speedup
            )
        )


if __name__ == "__main__":
    tvm.testing.main()
