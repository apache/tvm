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
"""RPC Runner Micro"""

from contextlib import contextmanager
from typing import Callable, List, Optional
from collections import namedtuple
import signal

from tvm import micro
from tvm import nd
from tvm.contrib.popen_pool import PopenPoolExecutor
from tvm.rpc.server import Server
from tvm.rpc.tracker import Tracker
from tvm.meta_schedule.logging import get_logger
from tvm.meta_schedule.utils import cpu_count, derived_object
from tvm.meta_schedule.runner.config import EvaluatorConfig, RPCConfig
from tvm.meta_schedule.runner import PyRunner, RunnerFuture, RunnerInput
from tvm.meta_schedule.runner.rpc_runner import RPCRunnerFuture
from tvm.meta_schedule.runner.utils import T_ARG_INFO_JSON_OBJ_LIST

logger = get_logger(__name__)  # pylint: disable=invalid-name


@derived_object
class RPCRunnerMicro(PyRunner):
    """RPC based runner for tuning micro models."""

    def __init__(
        self,
        platform: str = "crt",
        project_options: Optional[dict] = None,
        rpc_config: Optional[RPCConfig] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
        max_workers: Optional[int] = None,
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        platform: str
            The platform used for project generation.
        project_options: dict
            The options for the generated micro project.
        rpc_config: RPCConfig
            The rpc configuration.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        max_workers: Optional[int] = None
            The maximum number of connections. Defaults to number of logical CPU cores.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        """
        super().__init__()
        self.platform = platform
        if project_options is None:
            project_options = {}
        self.project_options = project_options
        self.rpc_config = RPCConfig._normalized(rpc_config)
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)

        if max_workers is None:
            max_workers = cpu_count(logical=True)
        logger.info("RPCRunner: max_workers = %d", max_workers)
        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=rpc_config.session_timeout_sec,
            initializer=initializer,
        )

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []

        for runner_input in runner_inputs:
            future = RPCRunnerFuture(
                future=self.pool.submit(
                    _worker_func,
                    self.platform,
                    self.project_options or {},
                    self.rpc_config,
                    self.evaluator_config,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.rpc_config.session_timeout_sec,
            )
            results.append(future)  # type: ignore
        return results


def _worker_func(
    platform: str,
    project_options: dict,
    rpc_config: RPCConfig,
    evaluator_config: EvaluatorConfig,
    artifact_path: str,
    device_type: str,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
) -> List[float]:

    module_loader = micro.AutoTvmModuleLoader(
        template_project_dir=micro.get_microtvm_template_projects(platform),
        project_options=project_options,
    )

    remote_kw = {
        "device_key": rpc_config.tracker_key,
        "host": rpc_config.tracker_host,
        "port": rpc_config.tracker_port,
        "priority": 0,
        "timeout": 100,
    }
    build_result = namedtuple("BuildResult", ["filename"])(artifact_path)

    with module_loader(remote_kw, build_result) as (remote, mod):
        dev = remote.device(device_type, 0)
        f_prepare = ""
        if evaluator_config.enable_cpu_cache_flush:
            f_prepare = "cache_flush_cpu_non_first_arg"
        time_f = mod.time_evaluator(
            mod.entry_name,
            dev,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc=f_prepare,
        )

        random_fill = remote.get_function("tvm.contrib.random.random_fill")
        args = [nd.empty(x[2], x[1], dev) for x in args_info]
        for arg in args:
            random_fill(arg)
        dev.sync()

        costs = time_f(*args).results
    return costs


@contextmanager
def get_rpc_runner_micro(
    platform,
    options,
    rpc_config: RPCConfig = None,
    evaluator_config: EvaluatorConfig = None,
    session_timeout_sec=300,
):
    """Parameters
    ----------
    platform: str
        The platform used for project generation.
    project_options: dict
        The options for the generated micro project.
    rpc_config: RPCConfig
        The rpc configuration.
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    session_timeout_sec: int
        The session timeout. if the number of candidates sent to runner is larger
        than the runner workers, increase the timeout.
    """
    if rpc_config is None:
        tracker_host = "127.0.0.1"
        tracker_port = 9000
        tracker_key = "$local$device$%d" % tracker_port
        rpc_config = RPCConfig(
            tracker_host=tracker_host,
            tracker_port=tracker_port,
            tracker_key=tracker_key,
            session_priority=0,
            session_timeout_sec=session_timeout_sec,
        )
    tracker_port_end = rpc_config.tracker_port + 1000

    if evaluator_config is None:
        evaluator_config = EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )

    tracker = Tracker(
        port=rpc_config.tracker_port,
        port_end=tracker_port_end,
        silent=True,
        reuse_addr=True,
        timeout=60,
    )
    server = Server(
        port=rpc_config.tracker_port,
        port_end=tracker_port_end,
        key=rpc_config.tracker_key,
        silent=True,
        tracker_addr=(rpc_config.tracker_host, rpc_config.tracker_port),
        reuse_addr=True,
        timeout=60,
    )

    def terminate():
        tracker.terminate()
        server.terminate()

    def handle_SIGINT(signal, frame):
        terminate()
        raise KeyboardInterrupt("Received SIGINT")

    signal.signal(signal.SIGINT, handle_SIGINT)

    try:
        yield RPCRunnerMicro(
            platform=platform,
            project_options=options,
            rpc_config=rpc_config,
            evaluator_config=evaluator_config,
        )
    finally:
        terminate()
