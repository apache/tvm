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
"""User-facing Tuning API"""
# pylint: disable=import-outside-toplevel
import logging
import logging.config
import os
from os import path as osp
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

from tvm.ir import IRModule
from tvm.ir.transform import PassContext
from tvm.runtime import Module, NDArray, vm
from tvm.target import Target
from tvm.te import Tensor, create_prim_func
from tvm.tir import PrimFunc, Schedule

from . import default_config
from .builder import Builder
from .cost_model import CostModel
from .database import Database, TuningRecord
from .extracted_task import ExtractedTask
from .measure_callback import MeasureCallback
from .mutator import Mutator
from .postproc import Postproc
from .profiler import Profiler
from .runner import Runner
from .schedule_rule import ScheduleRule
from .search_strategy import EvolutionarySearch, ReplayFunc, ReplayTrace
from .space_generator import PostOrderApply, SpaceGenerator
from .task_scheduler import GradientBased, RoundRobin
from .tune_context import TuneContext
from .utils import autotvm_silencer, batch_parameterize_config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

FnSpaceGenerator = Callable[[], SpaceGenerator]
FnScheduleRule = Callable[[], List[ScheduleRule]]
FnPostproc = Callable[[], List[Postproc]]
FnMutatorProb = Callable[[], Dict[Mutator, float]]


class TuneConfig(NamedTuple):
    """Configuration for tuning

    Parameters
    ----------
    max_trials_global: int
        Maximum number of trials to run.
    num_trials_per_iter: int
        Number of trials to run per iteration.
    max_trials_per_task: Optional[int]
        Maximum number of trials to run per task. If None, use `max_trials_global`.
    task_scheduler: str = "gradient"
        Task scheduler to use.
        Valid options are: round_robin, gradient.
    strategy: str = "evolutionary"
        Search strategy to use.
        Valid options are: evolutionary, replay_func, replay_trace.
    task_scheduler_config: Optional[Dict[str, Any]] = None
        Configuration for task scheduler.
    search_strategy_config: Optional[Dict[str, Any]] = None
        Configuration for search strategy.
    logger_config: Optional[Dict[str, Any]] = None
        Configuration for logger.
    adaptive_training: Optional[bool] = None
        Whether adpative training is enabled for cost model.
    """

    max_trials_global: int
    num_trials_per_iter: int
    max_trials_per_task: Optional[int] = None
    task_scheduler: str = "gradient"
    strategy: str = "evolutionary"
    task_scheduler_config: Optional[Dict[str, Any]] = None
    search_strategy_config: Optional[Dict[str, Any]] = None
    logger_config: Optional[Dict[str, Any]] = None
    adaptive_training: Optional[bool] = None

    def create_strategy(self):
        """Create search strategy from configuration"""
        cls_tbl = {
            "evolutionary": EvolutionarySearch,
            "replay_func": ReplayFunc,
            "replay_trace": ReplayTrace,
        }
        if self.strategy not in cls_tbl:
            raise ValueError(
                f"Invalid search strategy: {self.strategy}. "
                "Valid options are: {}".format(", ".join(cls_tbl.keys()))
            )
        # `max_trials_per_task` defaults to `max_trials_global`
        max_trials_per_task = self.max_trials_per_task
        if max_trials_per_task is None:
            max_trials_per_task = self.max_trials_global
        # `search_strategy_config` defaults to empty dict
        config = self.search_strategy_config
        if config is None:
            config = {}
        return cls_tbl[self.strategy](
            num_trials_per_iter=self.num_trials_per_iter,
            max_trials_per_task=max_trials_per_task,
            **config,
        )

    def create_task_scheduler(self, **kwargs):
        """Create task scheduler from configuration"""
        cls_tbl = {
            "round_robin": RoundRobin,
            "gradient": GradientBased,
        }
        if self.task_scheduler not in cls_tbl:
            raise ValueError(
                f"Invalid task scheduler: {self.task_scheduler}. "
                "Valid options are: {}".format(", ".join(cls_tbl.keys()))
            )
        # `task_scheduler_config` defaults to empty dict
        config = self.task_scheduler_config
        if config is None:
            config = {}
        return cls_tbl[self.task_scheduler](
            max_trials=self.max_trials_global,
            **kwargs,
            **config,
        )

    def create_loggers(
        self,
        log_dir: str,
        params: List[Dict[str, Any]],
        disable_existing_loggers: bool = False,
    ):
        """Create loggers from configuration"""
        if self.logger_config is None:
            config = {}
        else:
            config = self.logger_config

        config.setdefault("loggers", {})
        config.setdefault("handlers", {})
        config.setdefault("formatters", {})

        global_logger_name = "tvm.meta_schedule"
        global_logger = logging.getLogger(global_logger_name)
        if global_logger.level is logging.NOTSET:
            global_logger.setLevel(logging.INFO)

        config["loggers"].setdefault(
            global_logger_name,
            {
                "level": logging._levelToName[  # pylint: disable=protected-access
                    global_logger.level
                ],
                "handlers": [handler.get_name() for handler in global_logger.handlers]
                + [global_logger_name + ".console", global_logger_name + ".file"],
                "propagate": False,
            },
        )
        config["loggers"].setdefault(
            "{logger_name}",
            {
                "level": "INFO",
                "handlers": [
                    "{logger_name}.file",
                ],
                "propagate": False,
            },
        )
        config["handlers"].setdefault(
            global_logger_name + ".console",
            {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["handlers"].setdefault(
            global_logger_name + ".file",
            {
                "class": "logging.FileHandler",
                "filename": "{log_dir}/" + __name__ + ".task_scheduler.log",
                "mode": "a",
                "level": "INFO",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["handlers"].setdefault(
            "{logger_name}.file",
            {
                "class": "logging.FileHandler",
                "filename": "{log_dir}/{logger_name}.log",
                "mode": "a",
                "level": "INFO",
                "formatter": "tvm.meta_schedule.standard_formatter",
            },
        )
        config["formatters"].setdefault(
            "tvm.meta_schedule.standard_formatter",
            {
                "format": "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        )

        # set up dictConfig loggers
        p_config = {"version": 1, "disable_existing_loggers": disable_existing_loggers}
        for k, v in config.items():
            if k in ["formatters", "handlers", "loggers"]:
                p_config[k] = batch_parameterize_config(v, params)  # type: ignore
            else:
                p_config[k] = v
        logging.config.dictConfig(p_config)

        # check global logger
        if global_logger.level not in [logging.DEBUG, logging.INFO]:
            global_logger.warning(
                "Logging level set to %s, please set to logging.INFO"
                " or logging.DEBUG to view full log.",
                logging._levelToName[global_logger.level],  # pylint: disable=protected-access
            )
        global_logger.info("Logging directory: %s", log_dir)


def tune_extracted_tasks(
    extracted_tasks: List[ExtractedTask],
    config: TuneConfig,
    work_dir: str,
    *,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    cost_model: Optional[CostModel] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    space: Optional[FnSpaceGenerator] = None,
    sch_rules: Optional[FnScheduleRule] = None,
    postprocs: Optional[FnPostproc] = None,
    mutator_probs: Optional[FnMutatorProb] = None,
    num_threads: Optional[int] = None,
) -> Database:
    """Tune extracted tasks with a given target.

    Parameters
    ----------
    extracted_tasks : List[ExtractedTask]
        The list of extracted tasks.
    config : TuneConfig
        The search strategy config.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    cost_model : Optional[CostModel]
        The cost model to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    task_scheduler : Optional[TaskScheduler]
        The task scheduler to use.
    space : Optional[FnSpaceGenerator]
        The space generator to use.
    sch_rules : Optional[FnScheduleRule]
        The search rules to use.
    postprocs : Optional[FnPostproc]
        The postprocessors to use.
    mutator_probs : Optional[FnMutatorProb]
        The probability distribution to use different mutators.
    num_threads : Optional[int]
        The number of threads to use.

    Returns
    -------
    database : Database
        The database containing all the tuning results.

    """
    # pylint: disable=protected-access
    # logging directory is set to `work_dir/logs` by default
    log_dir = osp.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    max_width = len(str(len(extracted_tasks) - 1))
    logger_name_pattern = __name__ + ".task_{task_id:0" + f"{max_width}" + "d}_{task_name}"

    config.create_loggers(
        log_dir=log_dir,
        params=[
            {
                "log_dir": log_dir,
                "logger_name": logger_name_pattern.format(task_id=i, task_name=task.task_name),
            }
            for i, task in enumerate(extracted_tasks)
        ],
    )

    logger.info("Working directory: %s", work_dir)
    database = default_config.database(database, work_dir)
    builder = default_config.builder(builder)
    runner = default_config.runner(runner)
    cost_model = default_config.cost_model(cost_model, config.adaptive_training)
    measure_callbacks = default_config.callbacks(measure_callbacks)
    # parse the tuning contexts
    tune_contexts = []
    for i, task in enumerate(extracted_tasks):
        assert len(task.dispatched) == 1, "Only size 1 dispatched task list is supported for now"
        tune_contexts.append(
            TuneContext(
                mod=default_config.mod(task.dispatched[0]),
                target=task.target,
                space_generator=default_config.space_generator(space),
                search_strategy=config.create_strategy(),
                sch_rules=default_config.schedule_rules(sch_rules, task.target),
                postprocs=default_config.postproc(postprocs, task.target),
                mutator_probs=default_config.mutator_probs(mutator_probs, task.target),
                task_name=task.task_name,
                logger=logging.getLogger(
                    logger_name_pattern.format(task_id=i, task_name=task.task_name)
                ),
                num_threads=num_threads,
            )
        )
    # parse the task scheduler
    # pylint: enable=protected-access
    task_scheduler = config.create_task_scheduler(
        tasks=tune_contexts,
        task_weights=[float(t.weight) for t in extracted_tasks],
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
    )
    if config.max_trials_global > 0:
        task_scheduler.tune()
        cost_model.save(osp.join(work_dir, "cost_model.xgb"))
    return database


def tune_tir(
    mod: Union[IRModule, PrimFunc],
    target: Union[str, Target],
    config: TuneConfig,
    work_dir: str,
    *,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    cost_model: Optional[CostModel] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    space: Optional[FnSpaceGenerator] = None,
    blocks: Optional[List[str]] = None,
    sch_rules: Optional[FnScheduleRule] = None,
    postprocs: Optional[FnPostproc] = None,
    mutator_probs: Optional[FnMutatorProb] = None,
    task_name: str = "main",
    num_threads: Optional[int] = None,
) -> Optional[Schedule]:
    """Tune a TIR IRModule with a given target.

    Parameters
    ----------
    mod : Union[IRModule, PrimFunc]
        The module to tune.
    target : Union[str, Target]
        The target to tune for.
    config : TuneConfig
        The search strategy config.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    cost_model : Optional[CostModel]
        The cost model to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    space : Optional[FnSpaceGenerator]
        The space generator to use.
    blocks : Optional[List[str]]
        A list of block names specifying blocks to be tuned. Note that if
        the list is not None, blocks outside this list will not be tuned.
        Only one of this argument and space may be provided.
    sch_rules : Optional[FnScheduleRule]
        The search rules to use.
    postprocs : Optional[FnPostproc]
        The postprocessors to use.
    mutator_probs : Optional[FnMutatorProb]
        The probability distribution to use different mutators.
    task_name : str
        The name of the function to extract schedules from.
    num_threads : Optional[int]
        The number of threads to use

    Returns
    -------
    sch : Optional[Schedule]
        The tuned schedule.
    """
    # logging directory is set to `work_dir/logs` by default
    log_dir = osp.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    config.create_loggers(
        log_dir=log_dir,
        params=[{"log_dir": log_dir, "logger_name": __name__ + f".task_{task_name}"}],
    )

    if blocks is not None:
        assert space is None, "Can not specify blocks to tune when a search space is given."
        # Create a filter function to identify named blocks.
        def _f_block_filter(block, target_names) -> bool:
            return block.name_hint in target_names

        # Create a space generator that targets specific blocks.
        space = PostOrderApply(f_block_filter=lambda block: _f_block_filter(block, blocks))

    # pylint: disable=protected-access
    mod = default_config.mod(mod)
    target = default_config.target(target)
    # pylint: enable=protected-access
    database = tune_extracted_tasks(
        extracted_tasks=[
            ExtractedTask(
                task_name=task_name,
                mod=mod,
                dispatched=[mod],
                target=target,
                weight=1,
            ),
        ],
        config=config,
        work_dir=work_dir,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        space=space,
        sch_rules=sch_rules,
        postprocs=postprocs,
        mutator_probs=mutator_probs,
        num_threads=num_threads,
    )
    with Profiler.timeit("PostTuningCompilation"):
        bests: List[TuningRecord] = database.get_top_k(database.commit_workload(mod), top_k=1)
        if not bests:
            return None
        assert len(bests) == 1
        sch = Schedule(mod)
        bests[0].trace.apply_to_schedule(sch, remove_postproc=False)
    return sch


def tune_te(
    tensors: List[Tensor],
    target: Union[str, Target],
    config: TuneConfig,
    work_dir: str,
    *,
    task_name: str = "main",
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    cost_model: Optional[CostModel] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    space: Optional[FnSpaceGenerator] = None,
    sch_rules: Optional[FnScheduleRule] = None,
    postprocs: Optional[FnPostproc] = None,
    mutator_probs: Optional[FnMutatorProb] = None,
    num_threads: Optional[int] = None,
) -> Optional[Schedule]:
    """Tune a TE compute DAG with a given target.

    Parameters
    ----------
    tensor : List[Tensor]
        The list of input/output tensors of the TE compute DAG.
    target : Union[str, Target]
        The target to tune for.
    config : TuneConfig
        The search strategy config.
    task_name : str
        The name of the task.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.

    Returns
    -------
    sch : Optional[Schedule]
        The tuned schedule.
    """
    with Profiler.timeit("CreatePrimFunc"):
        func = create_prim_func(tensors)
    return tune_tir(
        mod=func,
        target=target,
        config=config,
        work_dir=work_dir,
        task_name=task_name,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        space=space,
        sch_rules=sch_rules,
        postprocs=postprocs,
        mutator_probs=mutator_probs,
        num_threads=num_threads,
    )


def tune_relay(
    mod: IRModule,
    target: Union[str, Target],
    config: TuneConfig,
    work_dir: str,
    *,
    backend: str = "graph",
    params: Optional[Dict[str, NDArray]] = None,
    builder: Optional[Builder] = None,
    runner: Optional[Runner] = None,
    database: Optional[Database] = None,
    cost_model: Optional[CostModel] = None,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    space: Optional[FnSpaceGenerator] = None,
    sch_rules: Optional[FnScheduleRule] = None,
    postprocs: Optional[FnPostproc] = None,
    mutator_probs: Optional[FnMutatorProb] = None,
    num_threads: Optional[int] = None,
) -> Union[Module, vm.Executable]:
    """Tune a Relay IRModule with a given target.

    Parameters
    ----------
    mod : IRModule
        The module to tune.
    target : Union[str, Target]
        The target to tune for.
    config : TuneConfig
        The search strategy config.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    task_name : str
        The name of the task.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    builder : Optional[Builder]
        The builder to use.
    runner : Optional[Runner]
        The runner to use.
    database : Optional[Database]
        The database to use.
    measure_callbacks : Optional[List[MeasureCallback]]
        The callbacks used during tuning.
    backend : str = "graph"
        The backend to use for relay compilation(graph / vm).

    Returns
    -------
    lib : Union[Module, tvm.runtime.vm.Executable]
        The built runtime module or vm Executable for the given relay workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    from .relay_integration import extract_task_from_relay

    # pylint: disable=protected-access, enable=import-outside-toplevel
    target = default_config.target(target)
    # pylint: enable=protected-access,
    # parse the tuning contexts
    with Profiler.timeit("TaskExtraction"):
        extracted_tasks = extract_task_from_relay(mod, target, params)
    database = tune_extracted_tasks(
        extracted_tasks,
        config,
        work_dir,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        space=space,
        sch_rules=sch_rules,
        postprocs=postprocs,
        mutator_probs=mutator_probs,
        num_threads=num_threads,
    )
    relay_build = {"graph": relay.build, "vm": relay.vm.compile}[backend]
    with Profiler.timeit("PostTuningCompilation"):
        with target, autotvm_silencer(), database:
            with PassContext(
                opt_level=3,
                config={
                    "relay.backend.use_meta_schedule": True,
                    "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
                    "relay.backend.tir_converter": "default",
                },
            ):
                return relay_build(mod, target=target, params=params)
