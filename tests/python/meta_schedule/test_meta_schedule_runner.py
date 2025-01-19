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
""" Test Meta Schedule Runner """

import itertools
import sys
import time
from typing import Any, List

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm._ffi import register_func
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    LocalRunner,
    PyRunner,
    RPCConfig,
    RPCRunner,
    RunnerFuture,
    RunnerInput,
)
from tvm.meta_schedule.runner.local_runner import (
    default_alloc_argument as local_default_alloc_argument,
)
from tvm.meta_schedule.runner.rpc_runner import (
    T_ARG_INFO_JSON_OBJ_LIST,
    T_ARGUMENT_LIST,
)
from tvm.meta_schedule.runner.rpc_runner import (
    default_alloc_argument as rpc_default_alloc_argument,
)
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.meta_schedule.utils import (
    derived_object,
    get_global_func_with_default_on_worker,
)
from tvm.rpc import RPCSession
from tvm.runtime import Device, Module
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import FloatImm

MATMUL_N = 16
MATMUL_M = 32

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring,unbalanced-tuple-unpacking


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulReluModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, d: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        D = T.match_buffer(d, (16, 16), "float32")
        C = T.alloc_buffer((16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(16, 16):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


@tvm.script.ir_module
class BatchMatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 32, 32])
        B = T.match_buffer(b, [16, 32, 32])
        C = T.match_buffer(c, [16, 32, 32])
        for n, i, j, k in T.grid(16, 32, 32, 32):
            with T.block("update"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    C[vn, vi, vj] = 0.0
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.ir_module
class AddModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [32], "float32")
        B = T.match_buffer(b, [32], "float32")
        C = T.match_buffer(c, [32], "float32")
        for i in range(32):
            with T.block("add"):
                vi = T.axis.S(32, i)
                C[vi] = A[vi] + B[vi]


# A huge matmul that must cause timeout in the timeout test below.
@tvm.script.ir_module
class MatmulHugeModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (4096, 4096), "float32")
        B = T.match_buffer(b, (4096, 4096), "float32")
        C = T.match_buffer(c, (4096, 4096), "float32")
        for i, j, k in T.grid(4096, 4096, 4096):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def _clean_build(artifact_path: str) -> None:
    f_clean_build = get_global_func_with_default_on_worker("meta_schedule.remove_build_dir", None)
    if f_clean_build is not None:
        f_clean_build(artifact_path)
    else:
        raise RuntimeError("Unable to find remove_build_dir function.")


def test_meta_schedule_rpc_single_run():
    """Test meta schedule rpc runner for a single run"""
    # Build the module
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(rpc_config, evaluator_config)
        # Run the module
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_local_single_run():
    """Test meta schedule local runner for a single run"""
    # Build the module
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )
    runner = LocalRunner(timeout_sec=100, evaluator_config=evaluator_config)
    # Run the module
    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_rpc_multiple_runs():
    """Test meta schedule rpc runner for multiple runs"""
    # Build the module
    mods = [
        MatmulModule,
        MatmulReluModule,
        BatchMatmulModule,
    ]
    builder = LocalBuilder()
    builder_inputs = [BuilderInput(mod, Target("llvm")) for mod in mods]
    builder_results = builder.build(builder_inputs)
    for builder_result in builder_results:
        assert builder_result.artifact_path is not None
        assert builder_result.error_msg is None

    args_infos = [
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
        [
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
        ],
    ]

    runner_inputs = [
        RunnerInput(builder_results[i].artifact_path, "llvm", args_infos[i])
        for i in range(len(mods))
    ]

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(rpc_config, evaluator_config)
        # Run the module
        runner_futures = runner.run(runner_inputs)
        runner_results = [runner_future.result() for runner_future in runner_futures]

    for runner_result in runner_results:
        assert runner_result.error_msg is None
        for result in runner_result.run_secs:
            if isinstance(result, FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0

    for builder_result in builder_results:
        _clean_build(builder_result.artifact_path)


def test_meta_schedule_local_multiple_runs():
    """Test meta schedule local runner for multiple runs"""
    # Build the module
    mods = [
        MatmulModule,
        MatmulReluModule,
        BatchMatmulModule,
    ]
    builder = LocalBuilder()
    builder_inputs = [BuilderInput(mod, Target("llvm")) for mod in mods]
    builder_results = builder.build(builder_inputs)
    for builder_result in builder_results:
        assert builder_result.artifact_path is not None
        assert builder_result.error_msg is None

    args_infos = [
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
        [
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
            TensorInfo("float32", [16, MATMUL_M, MATMUL_M]),
        ],
    ]

    runner_inputs = [
        RunnerInput(builder_results[i].artifact_path, "llvm", args_infos[i])
        for i in range(len(mods))
    ]

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )

    runner = LocalRunner(timeout_sec=100, evaluator_config=evaluator_config)

    # Run the module
    runner_futures = runner.run(runner_inputs)
    runner_results = [runner_future.result() for runner_future in runner_futures]

    for runner_result in runner_results:
        assert runner_result.error_msg is None
        for result in runner_result.run_secs:
            if isinstance(result, FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0

    for builder_result in builder_results:
        _clean_build(builder_result.artifact_path)


def test_meta_schedule_py_runner():
    """Test meta schedule PyRunner"""

    @derived_object
    class TestRunner(PyRunner):
        def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
            raise ValueError("TestRunner")

    runner = TestRunner()
    with pytest.raises(ValueError, match="TestRunner"):
        runner.run([])


def test_meta_schedule_rpc_runner_time_out():
    """Test meta schedule RPC Runner time out by using a super large workload"""

    builder = LocalBuilder()
    builder_inputs = [BuilderInput(MatmulHugeModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    builder_results[0].artifact_path

    runner_input = RunnerInput(
        builder_results[0].artifact_path,
        "llvm",
        [
            TensorInfo("float32", (4096, 4096)),
            TensorInfo("float32", (4096, 4096)),
            TensorInfo("float32", (4096, 4096)),
        ],
    )

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=1,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(
            rpc_config,
            evaluator_config,
        )
        # Run the module
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "RPCRunner: An exception occurred"
    )
    assert runner_result.run_secs is None


def test_meta_schedule_local_runner_time_out():
    """Test meta schedule Local Runner time out"""
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    def initializer():
        @register_func("meta_schedule.runner.test_time_out")
        def timeout_session_creator(  # pylint: disable=unused-variable
            device: Device,  # pylint: disable=unused-argument
            args_info: T_ARG_INFO_JSON_OBJ_LIST,  # pylint: disable=unused-argument
            alloc_repeat: int,  # pylint: disable=unused-argument
        ) -> RPCSession:
            time.sleep(2)

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )

    runner = LocalRunner(
        timeout_sec=1,
        evaluator_config=evaluator_config,
        initializer=initializer,
        f_alloc_argument="meta_schedule.runner.test_time_out",
    )

    # Run the module
    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "LocalRunner: Timeout, killed after"
    )
    assert runner_result.run_secs is None
    _clean_build(builder_result.artifact_path)


@pytest.mark.skip("Disable this test to unblock CI.")
def test_meta_schedule_rpc_runner_exception():
    """Test meta schedule RPC Runner exception"""

    def initializer():
        @register_func("meta_schedule.runner.test_exception")
        def exception_session_creator(  # pylint: disable=unused-variable
            rpc_config: RPCConfig,  # pylint: disable=unused-argument
        ) -> RPCSession:
            raise Exception("Test")

    runner_input = RunnerInput(
        "test",
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(
            rpc_config,
            evaluator_config,
            initializer=initializer,
            f_create_session="meta_schedule.runner.test_exception",
        )
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "RPCRunner: An exception occurred\n"
    )
    assert runner_result.run_secs is None


def test_meta_schedule_local_runner_exception():
    """Test meta schedule Local Runner exception"""
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    def initializer():
        @register_func("meta_schedule.runner.test_exception")
        def timeout_session_creator(  # pylint: disable=unused-variable
            device: Device,  # pylint: disable=unused-argument
            args_info: T_ARG_INFO_JSON_OBJ_LIST,  # pylint: disable=unused-argument
            alloc_repeat: int,  # pylint: disable=unused-argument
        ) -> RPCSession:
            raise Exception("Test")

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )

    runner = LocalRunner(
        evaluator_config=evaluator_config,
        initializer=initializer,
        f_alloc_argument="meta_schedule.runner.test_exception",
    )

    # Run the module
    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "LocalRunner: An exception occurred\n"
    )
    assert runner_result.run_secs is None
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_runner_matmul_test():
    """Test meta schedule runner with add module"""

    def _check_correct_matmul(
        args_before: List[np.ndarray],
        args_after: List[np.ndarray],
    ) -> None:
        a_before, b_before, c_before = args_before
        a_after, b_after, c_after = args_after
        c_before = np.matmul(a_before, b_before)
        assert (a_before == a_after).all()
        assert (b_before == b_after).all()
        tvm.testing.assert_allclose(c_before, c_after, rtol=1e-5)

    def test_alloc_argument(
        session: RPCSession,
        device: Device,
        args_info: Any,
        alloc_repeat: int,
    ) -> List[Any]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_before = []  # type: ignore
        repeated_args = rpc_default_alloc_argument(session, device, args_info, alloc_repeat)
        for args in repeated_args:
            repeated_args_before.append([arg.numpy() for arg in args])  # type: ignore
        return repeated_args

    def test_run_evaluator(
        session: RPCSession,  # pylint: disable=unused-argument
        rt_mod: Module,
        device: Device,
        evaluator_config: EvaluatorConfig,
        repeated_args: List[Any],
    ) -> List[float]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_after = []
        evaluator = rt_mod.time_evaluator(
            func_name=rt_mod.entry_name,
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
            repeated_args_after.append([arg.numpy() for arg in args])
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        for args_before, args_after in zip(
            repeated_args_before,  # type: ignore
            repeated_args_after,
        ):
            _check_correct_matmul(args_before, args_after)
        del repeated_args_before  # type: ignore
        return costs

    # Build the module
    mod = MatmulModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(
            rpc_config,
            evaluator_config,
            f_alloc_argument=test_alloc_argument,
            f_run_evaluator=test_run_evaluator,
        )
        # Run the module
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_runner_add_test():
    """Test meta schedule runner with add module"""

    def _check_correct_add(args_before: List[np.ndarray], args_after: List[np.ndarray]) -> None:
        a_before, b_before, c_before = args_before
        a_after, b_after, c_after = args_after
        c_before = a_before + b_before
        assert (a_before == a_after).all()
        assert (b_before == b_after).all()
        assert (c_before == c_after).all()

    def test_alloc_argument(
        session: RPCSession,
        device: Device,
        args_info: Any,
        alloc_repeat: int,
    ) -> List[Any]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_before = []  # type: ignore
        repeated_args = rpc_default_alloc_argument(
            session,
            device,
            args_info,
            alloc_repeat,
        )
        for args in repeated_args:
            repeated_args_before.append([arg.numpy() for arg in args])  # type: ignore
        return repeated_args

    def test_run_evaluator(
        session: RPCSession,  # pylint: disable=unused-argument
        rt_mod: Module,
        device: Device,
        evaluator_config: EvaluatorConfig,
        repeated_args: List[Any],
    ) -> List[float]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_after = []
        evaluator = rt_mod.time_evaluator(
            func_name=rt_mod.entry_name,
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
            repeated_args_after.append([arg.numpy() for arg in args])
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        for args_before, args_after in zip(
            repeated_args_before,  # type: ignore
            repeated_args_after,
        ):
            _check_correct_add(args_before, args_after)
        del repeated_args_before  # type: ignore
        return costs

    # Build the module
    mod = AddModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", [MATMUL_M]),
            TensorInfo("float32", [MATMUL_M]),
            TensorInfo("float32", [MATMUL_M]),
        ],
    )

    with LocalRPC() as rpc:
        rpc_config = RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        evaluator_config = EvaluatorConfig(
            number=1,
            repeat=1,
            min_repeat_ms=0,
            enable_cpu_cache_flush=False,
        )
        runner = RPCRunner(
            rpc_config,
            evaluator_config,
            f_alloc_argument=test_alloc_argument,
            f_run_evaluator=test_run_evaluator,
        )
        # Run the module
        (runner_future,) = runner.run([runner_input])
        runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_local_runner_add_test():
    """Test meta schedule local runner with add module"""

    def _check_correct_add(args_before: List[np.array], args_after: List[np.array]) -> None:
        a_before, b_before, c_before = args_before
        a_after, b_after, c_after = args_after
        c_before = a_before + b_before
        assert (a_before == a_after).all()
        assert (b_before == b_after).all()
        assert (c_before == c_after).all()

    def test_alloc_argument(
        device: Device,
        args_info: T_ARG_INFO_JSON_OBJ_LIST,  # pylint: disable=unused-argument
        alloc_repeat: int,
    ) -> List[T_ARGUMENT_LIST]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_before = []
        repeated_args = local_default_alloc_argument(device, args_info, alloc_repeat)
        for args in repeated_args:
            repeated_args_before.append([arg.asnumpy() for arg in args])
        return repeated_args

    def test_run_evaluator(
        rt_mod: Module,
        device: Device,
        evaluator_config: EvaluatorConfig,
        repeated_args: List[Any],
    ) -> List[float]:
        global repeated_args_before  # pylint: disable=global-variable-undefined, invalid-name
        repeated_args_after = []
        evaluator = rt_mod.time_evaluator(
            func_name=rt_mod.entry_name,
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
            repeated_args_after.append([arg.asnumpy() for arg in args])
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        for args_before, args_after in zip(repeated_args_before, repeated_args_after):
            _check_correct_add(args_before, args_after)
        del repeated_args_before
        return costs

    # Build the module
    mod = AddModule
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", [MATMUL_M]),
            TensorInfo("float32", [MATMUL_M]),
            TensorInfo("float32", [MATMUL_M]),
        ],
    )

    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )
    runner = LocalRunner(
        timeout_sec=100,
        evaluator_config=evaluator_config,
        f_alloc_argument=test_alloc_argument,
        f_run_evaluator=test_run_evaluator,
    )
    # Run the module
    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


if __name__ == "__main__":
    tvm.testing.main()
