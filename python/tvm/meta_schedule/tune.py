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
"""The core tuning API"""
from typing import List, Optional

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .measure_callback import MeasureCallback
from .runner import Runner
from .task_scheduler import TaskScheduler
from .tune_context import TuneContext


def tune_tasks(
    *,
    tasks: List[TuneContext],
    task_weights: List[float],
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    module_equality: str = "structural",
) -> Database:
    """Tune a list of tasks. Using a task scheduler.

    Parameters
    ----------
    tasks : List[TuneContext]
        The list of tasks to tune.
    task_weights : List[float]
        The weight of each task.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
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
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    if len(tasks) != len(task_weights):
        raise ValueError(
            f"Length of tasks ({len(tasks)}) and task_weights ({len(task_weights)}) do not match."
        )
    if max_trials_per_task is None:
        max_trials_per_task = max_trials_global
    if not isinstance(builder, Builder):
        builder = Builder.create(builder)
    if not isinstance(runner, Runner):
        runner = Runner.create(runner)
    if database == "json":
        database = Database.create(database, work_dir=work_dir, module_equality=module_equality)
    elif not isinstance(database, Database):
        database = Database.create(database, module_equality=module_equality)
    if not isinstance(cost_model, CostModel):
        cost_model = CostModel.create(cost_model)
    if isinstance(measure_callbacks, MeasureCallback):
        measure_callbacks = [measure_callbacks]
    elif measure_callbacks == "default":
        measure_callbacks = MeasureCallback.create(measure_callbacks)
    if not isinstance(task_scheduler, TaskScheduler):
        task_scheduler = TaskScheduler.create(task_scheduler)
    task_scheduler.tune(
        tasks=tasks,
        task_weights=task_weights,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        measure_callbacks=measure_callbacks,
        database=database,
        cost_model=cost_model,
    )
    return database
