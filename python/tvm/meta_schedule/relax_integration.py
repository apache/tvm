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
"""Meta schedule integration with high-level IR"""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import warnings

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import get_global_func, register_func
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.tir.expr import IntImm

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .extracted_task import ExtractedTask
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext
from .utils import fork_seed

if TYPE_CHECKING:
    from tvm import relax

_extract_task_func = get_global_func(  # pylint: disable=invalid-name
    "relax.backend.MetaScheduleExtractTask",
    allow_missing=False,
)


def extract_tasks(
    mod: Union[IRModule, "relax.Function"],
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    module_equality: str = "structural",
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
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
        The tasks extracted from this module
    """
    # pylint: disable=import-outside-toplevel
    from tvm.relax.expr import Function as RelaxFunc
    from tvm.relax.transform import BindParams

    # pylint: enable=import-outside-toplevel
    if isinstance(mod, RelaxFunc):
        mod = IRModule({"main": mod})
    if not isinstance(target, Target):
        target = Target(target)
    if params:
        mod = BindParams("main", params)(mod)
    return list(_extract_task_func(mod, target, module_equality))


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
        if task.mod.attrs is not None and task.mod.attrs.get("tir.is_scheduled", False):
            warnings.warn("The task {task.task_name} is already scheduled, skipping it.")
            continue
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


def tune_relax(
    mod: Union[IRModule, "relax.Function"],
    params: Dict[str, NDArray],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: Optional[int] = None,
    op_names: Optional[List[str]] = None,
    *,
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
    """Tune a Relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
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
    op_names: Optional[List[str]]
        A list of operator names to specify which op to tune. When it is None, all operators
        are tuned.
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
    all_tasks = extract_tasks(mod, target, params, module_equality=module_equality)

    if not op_names:
        selected_tasks = all_tasks
    else:
        selected_tasks = []

        for task in all_tasks:
            for op_name in op_names:
                if op_name in task.task_name:
                    selected_tasks.append(task)

    tasks, task_weights = extracted_tasks_to_tune_contexts(
        extracted_tasks=selected_tasks,
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


@register_func("tvm.meta_schedule.tune_relax")
def _tune_relax(
    mod: Union[IRModule, "relax.Function"],
    params: Dict[str, NDArray],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: Optional[int] = None,
    op_names: Optional[List[str]] = None,
    *,
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
    """Interface with tuning api to tune a Relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
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
    op_names: Optional[List[str]]
        A list of operator names to specify which op to tune. When it is None, all operators
        are tuned.
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
    ret_mod : IRModule
        IRModule
    """
    if isinstance(max_trials_global, IntImm):
        max_trials_global = int(max_trials_global)
    if isinstance(max_trials_per_task, IntImm):
        max_trials_per_task = int(max_trials_per_task)

    tune_relax(
        mod,
        params,
        target,
        work_dir,
        max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        op_names=op_names,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        space=space,
        strategy=strategy,
        seed=seed,
        module_equality=module_equality,
    )
    # Return original IRModule
    # This pass only makes optimization decision
    return mod


def compile_relax(
    database: Database,
    mod: IRModule,
    target: Union[Target, str],
    params: Optional[Dict[str, NDArray]],
    enable_warning: bool = False,
) -> "relax.Executable":
    """Compile a relax program with a MetaSchedule database.

    Parameters
    ----------
    database : Database
        The database to use
    mod : IRModule
        The Relax program to be compiled
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    enable_warning : bool
        A boolean value indicating if to print warnings for TIR functions not
        showing up in the database. By default we don't print warning.

    Returns
    -------
    lib : relax.Executable
        The built runtime module or vm Executable for the given relax workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.relax.transform import BindParams, MetaScheduleApplyDatabase
    from tvm.relax import build as relax_build

    # pylint: enable=import-outside-toplevel
    if not isinstance(target, Target):
        target = Target(target)
    if params:
        mod = BindParams("main", params)(mod)

    with target, database, PassContext(opt_level=3):
        relax_mod = MetaScheduleApplyDatabase(enable_warning=enable_warning)(mod)
        relax_ex = relax_build(relax_mod, target=target)
    return relax_ex
