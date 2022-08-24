import os
import tempfile
from tvm.script import tir as T
from typing import Callable, List, Optional
from tvm.contrib.popen_pool import PopenPoolExecutor
from tvm.meta_schedule.utils import cpu_count, derived_object
from tvm.meta_schedule.builder import LocalBuilder
from tvm.contrib.hexagon.session import export_module
from tvm.meta_schedule.runner import (
    EvaluatorConfig,
    RunnerInput,
    RunnerFuture,
    PyRunner,
)
from tvm.meta_schedule.runner.rpc_runner import (
    default_alloc_argument,
    default_run_evaluator,
    RPCRunnerFuture,
)


@derived_object
class HexagonRPCRunner(PyRunner):
    def __init__(
        self,
        hexagon_launcher,
        evaluator_config: Optional[EvaluatorConfig] = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        max_workers: Optional[int] = None,
        initializer: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.hexagon_launcher = hexagon_launcher
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        if max_workers is None:
            max_workers = cpu_count(logical=True)
        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=100,
            initializer=initializer,
        )

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results = []
        for runner_input in runner_inputs:
            future = RPCRunnerFuture(
                future=self.pool.submit(
                    _worker_func,
                    self.hexagon_launcher,
                    self.evaluator_config,
                    self.alloc_repeat,
                    str(runner_input.artifact_path),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                ),
                timeout_sec=100,
            )
            results.append(future)
        return results


def _worker_func(hexagon_launcher, evaluator_config, alloc_repeat, artifact_path, args_info):
    with hexagon_launcher.start_session() as session:
        device = session.device
        _, remote_path = os.path.split(artifact_path)
        uploaded = session.upload(artifact_path, remote_path)
        rt_mod = session.load_module(uploaded)
        repeated_args = default_alloc_argument(
            session,
            device,
            args_info,
            alloc_repeat,
        )
        costs = default_run_evaluator(
            session,
            rt_mod,
            device,
            evaluator_config,
            repeated_args,
        )
    return costs


def get_hexagon_local_builder():
    def export_func(mod):
        binary_path, _ = export_module(mod, tempfile.mkdtemp())
        return str(binary_path)

    return LocalBuilder(f_export=export_func)


def get_hexagon_rpc_runner(hexagon_launcher):
    evaluator_config = EvaluatorConfig(
        number=1,
        repeat=1,
        min_repeat_ms=0,
        enable_cpu_cache_flush=False,
    )

    return HexagonRPCRunner(
        hexagon_launcher,
        evaluator_config,
    )
