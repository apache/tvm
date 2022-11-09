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
"""MetaSchedule-TIR integration"""
from typing import Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on
from tvm import ir, tir
from tvm.target import Target

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext, _normalize_mod
from .utils import fork_seed


def tune_tir(
    mod: Union[ir.IRModule, tir.PrimFunc],
    target: Union[str, Target],
    work_dir: str,
    max_trials_global: int,
    *,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "round-robin",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    task_name: str = "main",
    num_threads: Union[Literal["physical", "logical"], int] = "physical",
    seed: Optional[int] = None,
) -> Database:
    """Tune a TIR function.

    Parameters
    ----------
    mod : Union[ir.IRModule, tir.PrimFunc]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy.
    task_name : str
        The name of the task.
    num_threads : Union[Literal["physical", "logical"], int]
        The number of threads to use.
    seed : Optional[int]
        The seed for the random number generator.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    (logger,) = get_loggers_from_work_dir(work_dir, [task_name])
    (seed,) = fork_seed(seed, n=1)
    return tune_tasks(
        tasks=[
            TuneContext(
                mod=mod,
                target=target,
                space_generator=space,
                search_strategy=strategy,
                task_name=task_name,
                logger=logger,
                rand_state=seed,
                num_threads=num_threads,
            ).clone()
        ],
        task_weights=[1.0],
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
    )


def compile_tir(
    database: Database,
    mod: Union[ir.IRModule, tir.PrimFunc],
    target: Union[Target, str],
) -> tir.Schedule:
    """Compile a TIR to tir.Schedule, according to the records in the database.

    Parameters
    ----------
    database : Database
        The database of tuning records.
    mod : Union[ir.IRModule, tir.PrimFunc]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.

    Returns
    -------
    sch : tir.Schedule
        The best schedule found in the database.
    """
    mod = _normalize_mod(mod)
    if not isinstance(target, Target):
        target = Target(target)
    return database.query_schedule(mod, target, workload_name="main")
