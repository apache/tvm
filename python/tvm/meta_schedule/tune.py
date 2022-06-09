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

from tvm._ffi.registry import register_func
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
from tvm.runtime import Module, NDArray
from tvm.target import Target
from tvm.te import Tensor, create_prim_func
from tvm.tir import PrimFunc, Schedule

from .apply_history_best import ApplyHistoryBest
from .builder import Builder, LocalBuilder
from .cost_model import CostModel, XGBModel
from .database import Database, JSONDatabase, TuningRecord
from .extracted_task import ExtractedTask
from .feature_extractor import PerStoreFeature
from .measure_callback import MeasureCallback
from .mutator import Mutator
from .postproc import Postproc
from .runner import LocalRunner, Runner
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


class DefaultLLVM:
    """Default tuning configuration for LLVM."""

    @staticmethod
    def _sch_rules() -> List[ScheduleRule]:
        from tvm.meta_schedule import schedule_rule as M

        return [
            M.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            M.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
            M.MultiLevelTiling(
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=M.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=64,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
            M.RandomComputeLocation(),
        ]

    @staticmethod
    def _postproc() -> List[Postproc]:
        from tvm.meta_schedule import postproc as M

        return [
            M.DisallowDynamicLoop(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
        ]

    @staticmethod
    def _mutator_probs() -> Dict[Mutator, float]:
        from tvm.meta_schedule import mutator as M

        return {
            M.MutateTileSize(): 0.9,
            M.MutateComputeLocation(): 0.05,
            M.MutateUnroll(): 0.03,
            M.MutateParallel(max_jobs_per_core=16): 0.02,
        }


class DefaultCUDA:
    """Default tuning configuration for CUDA."""

    @staticmethod
    def _sch_rules() -> List[ScheduleRule]:
        from tvm.meta_schedule import schedule_rule as M

        return [
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=M.ReuseType(
                    req="must",
                    levels=[3],
                    scope="local",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
            M.AutoBind(
                max_threadblocks=256,
                thread_extents=[32, 64, 128, 256, 512, 1024],
            ),
        ]

    @staticmethod
    def _postproc() -> List[Postproc]:
        from tvm.meta_schedule import postproc as M

        return [
            M.DisallowDynamicLoop(),
            M.RewriteCooperativeFetch(),
            M.RewriteUnboundBlock(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.VerifyGPUCode(),
        ]

    @staticmethod
    def _mutator_probs() -> Dict[Mutator, float]:
        from tvm.meta_schedule import mutator as M

        return {
            M.MutateTileSize(): 0.9,
            M.MutateUnroll(): 0.08,
            M.MutateThreadBinding(): 0.02,
        }


class Parse:
    """Parse tuning configuration from user inputs."""

    @staticmethod
    @register_func("tvm.meta_schedule.tune.parse_mod")  # for use in ApplyHistoryBest
    def _mod(mod: Union[PrimFunc, IRModule]) -> IRModule:
        if isinstance(mod, PrimFunc):
            mod = mod.with_attr("global_symbol", "main")
            mod = mod.with_attr("tir.noalias", True)
            mod = IRModule({"main": mod})
        if not isinstance(mod, IRModule):
            raise TypeError(f"Expected `mod` to be PrimFunc or IRModule, but gets: {mod}")
        # in order to make sure the mod can be found in ApplyHistoryBest
        # different func name can cause structural unequal
        func_names = mod.get_global_vars()
        (func_name,) = func_names
        if len(func_names) == 1 and func_name != "main":
            mod = IRModule({"main": mod[func_name]})
        return mod

    @staticmethod
    def _target(target: Union[str, Target]) -> Target:
        if isinstance(target, str):
            target = Target(target)
        if not isinstance(target, Target):
            raise TypeError(f"Expected `target` to be str or Target, but gets: {target}")
        return target

    @staticmethod
    def _builder(builder: Optional[Builder]) -> Builder:
        if builder is None:
            builder = LocalBuilder()  # type: ignore
        if not isinstance(builder, Builder):
            raise TypeError(f"Expected `builder` to be Builder, but gets: {builder}")
        return builder

    @staticmethod
    def _runner(runner: Optional[Runner]) -> Runner:
        if runner is None:
            runner = LocalRunner()  # type: ignore
        if not isinstance(runner, Runner):
            raise TypeError(f"Expected `runner` to be Runner, but gets: {runner}")
        return runner

    @staticmethod
    def _database(database: Union[None, Database], path: str) -> Database:
        if database is None:
            path_workload = osp.join(path, "database_workload.json")
            path_tuning_record = osp.join(path, "database_tuning_record.json")
            logger.info(
                "Creating JSONDatabase. Workload at: %s. Tuning records at: %s",
                path_workload,
                path_tuning_record,
            )
            database = JSONDatabase(
                path_workload=path_workload,
                path_tuning_record=path_tuning_record,
            )
        if not isinstance(database, Database):
            raise TypeError(f"Expected `database` to be Database, but gets: {database}")
        return database

    @staticmethod
    def _callbacks(
        measure_callbacks: Optional[List[MeasureCallback]],
    ) -> List[MeasureCallback]:
        if measure_callbacks is None:
            from tvm.meta_schedule import measure_callback as M

            return [
                M.AddToDatabase(),
                M.RemoveBuildArtifact(),
                M.EchoStatistics(),
                M.UpdateCostModel(),
            ]
        if not isinstance(measure_callbacks, (list, tuple)):
            raise TypeError(
                f"Expected `measure_callbacks` to be List[MeasureCallback], "
                f"but gets: {measure_callbacks}"
            )
        measure_callbacks = list(measure_callbacks)
        for i, callback in enumerate(measure_callbacks):
            if not isinstance(callback, MeasureCallback):
                raise TypeError(
                    f"Expected `measure_callbacks` to be List[MeasureCallback], "
                    f"but measure_callbacks[{i}] is: {callback}"
                )
        return measure_callbacks

    @staticmethod
    def _cost_model(cost_model: Optional[CostModel]) -> CostModel:
        if cost_model is None:
            return XGBModel(extractor=PerStoreFeature())  # type: ignore
        if not isinstance(cost_model, CostModel):
            raise TypeError(f"Expected `cost_model` to be CostModel, but gets: {cost_model}")
        return cost_model

    @staticmethod
    def _space_generator(space_generator: Optional[FnSpaceGenerator]) -> SpaceGenerator:
        if space_generator is None:
            return PostOrderApply()
        if callable(space_generator):
            space_generator = space_generator()
        if not isinstance(space_generator, SpaceGenerator):
            raise TypeError(
                f"Expected `space_generator` to return SpaceGenerator, "
                f"but gets: {space_generator}"
            )
        return space_generator

    @staticmethod
    def _sch_rules(sch_rules: Optional[FnScheduleRule], target: Target) -> List[ScheduleRule]:
        if callable(sch_rules):
            return sch_rules()
        if sch_rules is not None:
            raise TypeError(f"Expected `sch_rules` to be None or callable, but gets: {sch_rules}")
        # pylint: disable=protected-access
        if target.kind.name == "llvm":
            return DefaultLLVM._sch_rules()
        if target.kind.name in ["cuda", "rocm", "vulkan"]:
            return DefaultCUDA._sch_rules()
        # pylint: enable=protected-access
        raise ValueError(f"Unsupported target: {target}")

    @staticmethod
    def _postproc(postproc: Optional[FnPostproc], target: Target) -> List[Postproc]:
        if callable(postproc):
            return postproc()
        if postproc is not None:
            raise TypeError(f"Expected `postproc` to be None or callable, but gets: {postproc}")
        # pylint: disable=protected-access
        if target.kind.name == "llvm":
            return DefaultLLVM._postproc()
        if target.kind.name in ["cuda", "rocm", "vulkan"]:
            return DefaultCUDA._postproc()
        # pylint: enable=protected-access
        raise ValueError(f"Unsupported target: {target}")

    @staticmethod
    def _mutator_probs(
        mutator_probs: Optional[FnMutatorProb],
        target: Target,
    ) -> Dict[Mutator, float]:
        if callable(mutator_probs):
            return mutator_probs()
        if mutator_probs is not None:
            raise TypeError(
                f"Expected `mutator_probs` to be None or callable, but gets: {mutator_probs}"
            )
        # pylint: disable=protected-access
        if target.kind.name == "llvm":
            return DefaultLLVM._mutator_probs()
        if target.kind.name in ["cuda", "rocm", "vulkan"]:
            return DefaultCUDA._mutator_probs()
        # pylint: enable=protected-access
        raise ValueError(f"Unsupported target: {target}")


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
    """

    max_trials_global: int
    num_trials_per_iter: int
    max_trials_per_task: Optional[int] = None
    task_scheduler: str = "gradient"
    strategy: str = "evolutionary"
    task_scheduler_config: Optional[Dict[str, Any]] = None
    search_strategy_config: Optional[Dict[str, Any]] = None
    logger_config: Optional[Dict[str, Any]] = None

    def create_strategy(self, **kwargs):
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
            **kwargs,
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
        The list of extraced tasks.
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
    database = Parse._database(database, work_dir)
    builder = Parse._builder(builder)
    runner = Parse._runner(runner)
    cost_model = Parse._cost_model(cost_model)
    measure_callbacks = Parse._callbacks(measure_callbacks)
    # parse the tuning contexts
    tune_contexts = []
    for i, task in enumerate(extracted_tasks):
        assert len(task.dispatched) == 1, "Only size 1 dispatched task list is supported for now"
        tune_contexts.append(
            TuneContext(
                mod=Parse._mod(task.dispatched[0]),
                target=task.target,
                space_generator=Parse._space_generator(space),
                search_strategy=config.create_strategy(),
                sch_rules=Parse._sch_rules(sch_rules, task.target),
                postprocs=Parse._postproc(postprocs, task.target),
                mutator_probs=Parse._mutator_probs(mutator_probs, task.target),
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

    # pylint: disable=protected-access
    mod = Parse._mod(mod)
    target = Parse._target(target)
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
    bests: List[TuningRecord] = database.get_top_k(
        database.commit_workload(mod),
        top_k=1,
    )
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
    return tune_tir(
        mod=create_prim_func(tensors),
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
) -> Module:
    """Tune a TIR IRModule with a given target.

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

    Returns
    -------
    lib : Module
        The built runtime module for the given relay workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.relay import build as relay_build

    from .relay_integration import extract_task_from_relay

    # pylint: disable=protected-access, enable=import-outside-toplevel
    target = Parse._target(target)
    # pylint: enable=protected-access,
    # parse the tuning contexts
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
    with target, autotvm_silencer(), ApplyHistoryBest(database):
        with PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            return relay_build(mod, target=target, params=params)
