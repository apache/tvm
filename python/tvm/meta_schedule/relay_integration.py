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
"""MetaSchedule-Relay integration"""
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on
import numpy as np  # type: ignore
from tvm import nd
from tvm._ffi import get_global_func
from tvm.ir import IRModule, transform
from tvm.runtime import NDArray
from tvm.target import Target

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .extracted_task import ExtractedTask
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .profiler import Profiler
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext
from .utils import fork_seed

if TYPE_CHECKING:
    from tvm import relay

_extract_task = get_global_func(  # pylint: disable=invalid-name
    "relay.backend.MetaScheduleExtractTask",
    allow_missing=True,
)


@contextmanager
def _autotvm_silencer():
    """A context manager that silences autotvm warnings."""
    from tvm import autotvm  # pylint: disable=import-outside-toplevel

    silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True
    try:
        yield
    finally:
        autotvm.GLOBAL_SCOPE.silent = silent


def _normalize_params(
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    pass_config: Mapping[str, Any],
    executor: Optional["relay.backend.Executor"],
) -> Tuple[
    IRModule,
    Target,
    Dict[str, NDArray],
    Dict[str, Any],
    Optional["relay.backend.Executor"],
]:
    from tvm import relay  # pylint: disable=import-outside-toplevel

    if isinstance(mod, relay.Function):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)
    if params is None:
        params = {}
    relay_params = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = nd.array(param)
        relay_params[name] = param

    if executor is None:
        executor = relay.backend.Executor("graph")

    if mod.get_attr("executor") is None:
        mod = mod.with_attr("executor", executor)
    else:
        executor = mod.get_attr("executor")

    pass_config = dict(pass_config)
    return mod, target, relay_params, pass_config, executor


def extract_tasks(
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    *,
    opt_level: int = 3,
    pass_config: Mapping[str, Any] = MappingProxyType(
        {
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "default",
        }
    ),
    executor: Optional["relay.backend.Executor"] = None,
    module_equality: str = "structural",
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod : IRModule
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    opt_level : int
        The optimization level of the compilation
    pass_config : Mapping[str, Any]
        The pass configuration
    executor : Optional[relay.backend.Executor]
        The executor to use
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-ndarray" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tir/analysis/analysis.py.

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """
    # pylint: disable=import-outside-toplevel
    from tvm import autotvm

    # pylint: enable=import-outside-toplevel
    mod, target, params, pass_config, _ = _normalize_params(
        mod, target, params, pass_config, executor
    )
    if target.kind.name != "cuda" and isinstance(
        autotvm.DispatchContext.current, autotvm.FallbackContext
    ):
        tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.utils.EmptyContext()
    with Profiler.timeit("TaskExtraction"):
        with target, _autotvm_silencer(), tophub_context:
            with transform.PassContext(
                opt_level=opt_level,
                config=pass_config,
            ):
                return list(_extract_task(mod, target, params, module_equality))


def extracted_tasks_to_tune_contexts(
    extracted_tasks: List[ExtractedTask],
    work_dir: str,
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_threads: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
) -> Tuple[List[TuneContext], List[float]]:
    """Convert ExtractedTask to TuneContext.

    Parameters
    ----------
    tasks : List[ExtractedTask]
        The tasks to be converted
    work_dir : str
        The working directory to store logs and databases
    space : SpaceGenerator.SpaceGeneratorType
        The space generator to use.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy to use.
    num_threads : Union[Literal["physical", "logical"], int]
        The number of threads to use in multi-threaded search algorithm.
    seed : Optional[int]
        The random seed to use.

    Returns
    -------
    tasks : List[TuneContext]
        The converted tasks
    task_weights : List[float]
        The weights of the tasks
    """
    tasks: List[TuneContext] = []
    task_weights: List[float] = []
    for task, logger, rand_state in zip(
        extracted_tasks,
        get_loggers_from_work_dir(work_dir, [t.task_name for t in extracted_tasks]),
        fork_seed(seed, n=len(extracted_tasks)),
    ):
        tasks.append(
            TuneContext(
                mod=task.dispatched[0],
                target=task.target,
                space_generator=space,
                search_strategy=strategy,
                task_name=task.task_name,
                logger=logger,
                rand_state=rand_state,
                num_threads=num_threads,
            ).clone()
        )
        task_weights.append(task.weight)
    return tasks, task_weights


def tune_relay(
    mod: IRModule,
    params: Dict[str, NDArray],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    *,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    seed: Optional[int] = None,
    module_equality: str = "structural",
) -> Database:
    """Tune a Relay program.

    Parameters
    ----------
    mod : Union[IRModule, tir.PrimFunc]
        The module or function to tune
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    target : Union[Target, str]
        The compilation target
    work_dir : str
        The working directory to store the tuning records
    max_trials_global : int
        The maximum number of trials to run
    max_trials_per_task : Optional[int]
        The maximum number of trials to run for each task
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : BuilderType
        The builder to use
    runner : RunnerType
        The runner to use
    database : DatabaseType
        The database to use
    cost_model : CostModelType
        The cost model to use
    measure_callbacks : CallbackListType
        The measure callbacks to use
    task_scheduler : TaskSchedulerType
        The task scheduler to use
    space : SpaceGeneratorType
        The space generator to use
    strategy : SearchStrategyType
        The search strategy to use
    seed : Optional[int]
        The random seed
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-ndarray" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tir/analysis/analysis.py.

    Returns
    -------
    database : Database
        The database that contains the tuning records
    """
    tasks, task_weights = extracted_tasks_to_tune_contexts(
        extracted_tasks=extract_tasks(mod, target, params, module_equality=module_equality),
        work_dir=work_dir,
        space=space,
        strategy=strategy,
        seed=seed,
    )
    return tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        module_equality=module_equality,
    )


def compile_relay(
    database: Database,
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    *,
    backend: Literal["graph", "vm"] = "graph",
    opt_level: int = 3,
    pass_config: Mapping[str, Any] = MappingProxyType(
        {
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "default",
        }
    ),
    executor: Optional["relay.backend.Executor"] = None,
):
    """Compile a relay program with a MetaSchedule database.

    Parameters
    ----------
    database : Database
        The database to use
    mod : IRModule
        The Relay program to be compiled
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    backend : str
        The backend to use. Builtin backends:
            - "graph"
            - "vm"
    opt_level : int
        The optimization level of the compilation
    pass_config : Mapping[str, Any]
        The pass configuration
    executor : Optional[relay.backend.Executor]
        The executor to use in relay.build. It is not supported by RelayVM.

    Returns
    -------
    lib : Union[Module, tvm.runtime.vm.Executable]
        The built runtime module or vm Executable for the given relay workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    # pylint: enable=import-outside-toplevel
    mod, target, params, pass_config, executor = _normalize_params(
        mod, target, params, pass_config, executor
    )
    pass_config.setdefault("relay.backend.use_meta_schedule_dispatch", True)
    with Profiler.timeit("PostTuningCompilation"):
        with target, _autotvm_silencer(), database:
            with transform.PassContext(
                opt_level=opt_level,
                config=pass_config,
            ):
                if backend == "graph":
                    return relay.build(mod, target=target, params=params, executor=executor)
                elif backend == "vm":
                    return relay.vm.compile(mod, target=target, params=params)
                else:
                    raise ValueError(f"Unknown backend: {backend}")


def is_meta_schedule_enabled() -> bool:
    """Return whether the meta-schedule is enabled.

    Returns
    -------
    enabled: bool
        Whether the meta schedule is enabled
    """
    return transform.PassContext.current().config.get(
        "relay.backend.use_meta_schedule",
        False,
    )
