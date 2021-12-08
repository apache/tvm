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
"""Local Runner"""
from contextlib import contextmanager
from typing import Callable, List, Optional, Union
import tvm

from ...contrib.popen_pool import PopenPoolExecutor
from ...runtime import Device, Module
from ..utils import get_global_func_with_default_on_worker
from .config import EvaluatorConfig
from .runner import PyRunner, RunnerFuture, RunnerInput, RunnerResult
from .utils import (
    T_ARG_INFO_JSON_OBJ_LIST,
    T_ARGUMENT_LIST,
    alloc_argument_common,
    run_evaluator_common,
)


class LocalRunnerFuture(RunnerFuture):
    """Local based runner future

    Parameters
    ----------
    res: Optional[List[float]]
        The optional result as a list of float.
    error_message: Optional[str]
        The optional error message.

    Note
    ----
    Only one of the parameters should be None upon the creation
    of LocalRunnerFuture object
    """

    res: Optional[List[float]]
    error_message: Optional[str]

    def __init__(
        self, res: Optional[List[float]] = None, error_message: Optional[str] = None
    ) -> None:
        """Constructor

        Parameters
        ----------
        res: Optional[List[float]]
            The result of this LocalRunnerFuture
        error_message: Optional[str]
            The stringfied error message of any exception during execution

        """
        super().__init__()
        self.res = res
        self.error_message = error_message

        # sanity check upon the creation of LocalRunnerFuture object
        if (res is None and error_message is None) or (
            res is not None and error_message is not None
        ):
            raise AttributeError(
                "Only one of the two parameters should be None upon the creation"
                "of LocalRunnerFuture object."
            )

    def done(self) -> bool:
        return True

    def result(self) -> RunnerResult:
        return RunnerResult(self.res, self.error_message)


class LocalRunner(PyRunner):
    """Local runner

    Parameters
    ----------
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    cooldown_sec: float
        The cooldown in seconds.
    alloc_repeat: int
        The number of times to repeat the allocation.
    f_alloc_argument: Optional[str, Callable]
        The function name to allocate the arguments or the function itself.
    f_run_evaluator: Optional[str, Callable]
        The function name to run the evaluator or the function itself.
    f_cleanup: Optional[str, Callable]
        The function name to cleanup the session or the function itself.
    pool: PopenPoolExecutor
        The popen pool executor.

    Attributes
    ----------
    T_ALLOC_ARGUMENT : typing._GenericAlias
        The signature of the function `f_alloc_argument`, which is:

        .. code-block:: python

        def default_alloc_argument(
            device: Device,
            args_info: T_ARG_INFO_JSON_OBJ_LIST,
            alloc_repeat: int,
        ) -> List[T_ARGUMENT_LIST]:
            ...

    T_RUN_EVALUATOR : typing._GenericAlias
        The signature of the function `f_run_evaluator`, which is:

        .. code-block:: python

        def default_run_evaluator(
            rt_mod: Module,
            device: Device,
            evaluator_config: EvaluatorConfig,
            repeated_args: List[T_ARGUMENT_LIST],
        ) -> List[float]:
            ...

    T_CLEANUP : typing._GenericAlias
        The signature of the function `f_cleanup`, which is:

        .. code-block:: python

        def default_cleanup() -> None:
            ...
    """

    T_ALLOC_ARGUMENT = Callable[
        [
            Device,  # The device on the remote
            T_ARG_INFO_JSON_OBJ_LIST,  # The metadata information of the arguments to be allocated
            int,  # The number of repeated allocations to be done
        ],
        List[T_ARGUMENT_LIST],  # A list of argument lists
    ]
    T_RUN_EVALUATOR = Callable[
        [
            Module,  # The Module opened on the remote
            Device,  # The device on the remote
            EvaluatorConfig,  # The evaluator configuration
            List[T_ARGUMENT_LIST],  # A list of argument lists
        ],
        List[float],  # A list of running time
    ]
    T_CLEANUP = Callable[
        [],
        None,
    ]

    timeout_sec: float
    evaluator_config: EvaluatorConfig
    cooldown_sec: float
    alloc_repeat: int

    f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None]
    f_run_evaluator: Union[T_RUN_EVALUATOR, str, None]
    f_cleanup: Union[T_CLEANUP, str, None]

    pool: PopenPoolExecutor

    def __init__(
        self,
        timeout_sec: float = 30,
        evaluator_config: Optional[EvaluatorConfig] = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None] = None,
        f_run_evaluator: Union[T_RUN_EVALUATOR, str, None] = None,
        f_cleanup: Union[T_CLEANUP, str, None] = None,
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        timeout_sec: float
            The timeout setting.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        cooldown_sec: float
            The cooldown in seconds.
        alloc_repeat: int
            The number of times to random fill the allocation.
        f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None]
            The function name to allocate the arguments or the function itself.
        f_run_evaluator: Union[T_RUN_EVALUATOR, str, None]
            The function name to run the evaluator or the function itself.
        f_cleanup: Union[T_CLEANUP, str, None]
            The function name to cleanup the session or the function itself.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        """
        super().__init__()
        self.timeout_sec = timeout_sec
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        self.f_alloc_argument = f_alloc_argument
        self.f_run_evaluator = f_run_evaluator
        self.f_cleanup = f_cleanup

        self.pool = PopenPoolExecutor(
            max_workers=1,  # one local worker
            timeout=timeout_sec,
            initializer=initializer,
        )
        self._sanity_check()

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []
        for runner_input in runner_inputs:
            future = self.pool.submit(
                LocalRunner._worker_func,
                self.f_alloc_argument,
                self.f_run_evaluator,
                self.f_cleanup,
                self.evaluator_config,
                self.alloc_repeat,
                str(runner_input.artifact_path),
                str(runner_input.device_type),
                tuple(arg_info.as_json() for arg_info in runner_input.args_info),
            )
            try:
                result: List[float] = future.result()
                error_message: str = None
            except TimeoutError as exception:
                result = None
                error_message = f"LocalRunner: Timeout, killed after {self.timeout_sec} seconds\n"
            except Exception as exception:  # pylint: disable=broad-except
                result = None
                error_message = "LocalRunner: An exception occurred\n" + str(exception)
            local_future = LocalRunnerFuture(res=result, error_message=error_message)
            results.append(local_future)
        return results

    def _sanity_check(self) -> None:
        def _check(
            f_alloc_argument,
            f_run_evaluator,
            f_cleanup,
        ) -> None:
            get_global_func_with_default_on_worker(name=f_alloc_argument, default=None)
            get_global_func_with_default_on_worker(name=f_run_evaluator, default=None)
            get_global_func_with_default_on_worker(name=f_cleanup, default=None)
            get_global_func_with_default_on_worker(
                name="tvm.contrib.random.random_fill", default=None
            )

        value = self.pool.submit(
            _check,
            self.f_alloc_argument,
            self.f_run_evaluator,
            self.f_cleanup,
        )
        value.result()

    @staticmethod
    def _worker_func(
        _f_alloc_argument: Optional[str],
        _f_run_evaluator: Optional[str],
        _f_cleanup: Optional[str],
        evaluator_config: EvaluatorConfig,
        alloc_repeat: int,
        artifact_path: str,
        device_type: str,
        args_info: T_ARG_INFO_JSON_OBJ_LIST,
    ) -> List[float]:
        f_alloc_argument: LocalRunner.T_ALLOC_ARGUMENT = get_global_func_with_default_on_worker(
            _f_alloc_argument, default_alloc_argument
        )
        f_run_evaluator: LocalRunner.T_RUN_EVALUATOR = get_global_func_with_default_on_worker(
            _f_run_evaluator, default_run_evaluator
        )
        f_cleanup: LocalRunner.T_CLEANUP = get_global_func_with_default_on_worker(
            _f_cleanup, default_cleanup
        )

        @contextmanager
        def resource_handler():
            try:
                yield
            finally:
                # Final step. Always clean up
                f_cleanup()

        with resource_handler():
            # Step 1: create the local runtime module
            rt_mod = tvm.runtime.load_module(artifact_path)
            # Step 2: create the local device
            device = tvm.runtime.device(dev_type=device_type, dev_id=0)
            # Step 3: Allocate input arguments
            repeated_args: List[T_ARGUMENT_LIST] = f_alloc_argument(
                device,
                args_info,
                alloc_repeat,
            )
            # Step 4: Run time_evaluator
            costs: List[float] = f_run_evaluator(
                rt_mod,
                device,
                evaluator_config,
                repeated_args,
            )
        return costs


def default_alloc_argument(
    device: Device,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
    alloc_repeat: int,
) -> List[T_ARGUMENT_LIST]:
    """Default function to allocate the arguments

    Parameters
    ----------
    device: Device
        The device to allocate the arguments
    args_info: T_ARG_INFO_JSON_OBJ_LIST
        The arguments info
    alloc_repeat: int
        The number of times to repeat the allocation

    Returns
    -------
    repeated_args: List[T_ARGUMENT_LIST]
        The allocation args
    """
    f_random_fill = get_global_func_with_default_on_worker(
        name="tvm.contrib.random.random_fill", default=None
    )
    return alloc_argument_common(f_random_fill, device, args_info, alloc_repeat)


def default_run_evaluator(
    rt_mod: Module,
    device: Device,
    evaluator_config: EvaluatorConfig,
    repeated_args: List[T_ARGUMENT_LIST],
) -> List[float]:
    """Default function to run the evaluator

    Parameters
    ----------
    rt_mod: Module
        The runtime module
    device: Device
        The device to run the evaluator
    evaluator_config: EvaluatorConfig
        The evaluator config
    repeated_args: List[T_ARGUMENT_LIST]
        The repeated arguments

    Returns
    -------
    costs: List[float]
        The evaluator results
    """
    return run_evaluator_common(rt_mod, device, evaluator_config, repeated_args)


def default_cleanup() -> None:
    """Default function to clean up the session"""
    pass  # pylint: disable=unnecessary-pass
