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
from typing import Callable, List, Optional, Union
from collections import namedtuple
import signal
import random

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
        rpc_configs: Optional[List[RPCConfig]] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
        max_workers: Optional[int] = None,
        initializer: Optional[Callable[[], None]] = None,
        session_timeout_sec: int = 300,
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
        session_timeout_sec: int
            The session timeout, including the pending time. if the number of candidates sent to runner is larger
            than the runner workers, increase the timeout.
        """
        super().__init__()
        self.platform = platform
        if project_options is None:
            project_options = {}
        self.project_options = project_options
        self.rpc_configs = rpc_configs
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.session_timeout_sec = session_timeout_sec

        if max_workers is None:
            max_workers = cpu_count(logical=True)
        logger.info("RPCRunner: max_workers = %d", max_workers)
        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=session_timeout_sec,
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
                    self.rpc_configs,
                    self.evaluator_config,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.session_timeout_sec,
            )
            results.append(future)  # type: ignore
        return results


def _worker_func(
    platform: str,
    project_options: dict,
    rpc_configs: List[RPCConfig],
    evaluator_config: EvaluatorConfig,
    artifact_path: str,
    device_type: str,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
) -> List[float]:

    module_loader = micro.AutoTvmModuleLoader(
        template_project_dir=micro.get_microtvm_template_projects(platform),
        project_options=project_options,
    )

    rpc_config = random.choice(rpc_configs)
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
    evaluator_config: EvaluatorConfig = None,
    tracker_host: Optional[str] = None,
    tracker_port: Union[None, int, str] = None,
    session_timeout_sec: int = 300,
    rpc_timeout_sec: int = 10,
    serial_numbers: List[str] = None,
):
    """Parameters
    ----------
    platform: str
        The platform used for project generation.
    options: dict
        The options for the generated micro project.
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    tracker_host: Optional[str]
        The host url of the rpc server.
    tracker_port: Union[None, int, str]
        The TCP port to bind to
    session_timeout_sec: int
        The session timeout. if the number of candidates sent to runner is larger
        than the runner workers, increase the timeout.
    rpc_timeout_sec:
        The rpc session timeout.
    serial_numbers:
        List of board serial numbers to be used during tuning.
        For "CRT" and "QEMU" platforms the serial numners are not used,
        but the length of the list determines the number of runner instances.
    """

    if evaluator_config is None:
        evaluator_config = EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )

    if tracker_host is None:
        tracker_host = "127.0.0.1"

    if tracker_port is None:
        tracker_port = 9000
    else:
        tracker_port = int(tracker_port)
    tracker_port_end = tracker_port + 1000

    if not (serial_numbers):
        serial_numbers = ["$local$device"]

    tracker = Tracker(
        port=tracker_port,
        port_end=tracker_port_end,
        silent=True,
        reuse_addr=True,
        timeout=60,
    )

    servers = []
    rpc_configs = []
    for serial_number in serial_numbers:
        key = serial_number
        rpc_config = RPCConfig(
            tracker_host=tracker_host,
            tracker_port=tracker_port,
            tracker_key=key,
            session_priority=0,
            session_timeout_sec=rpc_timeout_sec,
        )
        rpc_configs.append(rpc_config)

        server = Server(
            port=tracker_port,
            port_end=tracker_port_end,
            key=key,
            silent=True,
            tracker_addr=(tracker_host, tracker_port),
            reuse_addr=True,
            timeout=60,
        )
        servers.append(server)

    def terminate():
        tracker.terminate()
        for server in servers:
            server.terminate()

    def handle_SIGINT(signal, frame):
        terminate()
        raise KeyboardInterrupt("Received SIGINT")

    signal.signal(signal.SIGINT, handle_SIGINT)

    try:
        yield RPCRunnerMicro(
            platform=platform,
            project_options=options,
            rpc_configs=rpc_configs,
            evaluator_config=evaluator_config,
            session_timeout_sec=session_timeout_sec,
        )
    finally:
        terminate()
