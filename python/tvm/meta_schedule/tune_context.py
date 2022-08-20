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
"""Meta Schedule tuning context."""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from tvm import IRModule
from tvm._ffi import register_object
from tvm.meta_schedule.utils import cpu_count, make_logging_func
from tvm.runtime import Object
from tvm.target import Target
from tvm.tir import PrimFunc, Schedule

from . import _ffi_api

if TYPE_CHECKING:
    from .cost_model import CostModel
    from .database import Database
    from .mutator import Mutator
    from .postproc import Postproc
    from .runner import RunnerResult
    from .schedule_rule import ScheduleRule
    from .search_strategy import MeasureCandidate, SearchStrategy
    from .space_generator import ScheduleFn, ScheduleFnType, SpaceGenerator
    from .tune import TuneConfig


@register_object("meta_schedule.TuneContext")
class TuneContext(Object):
    """
    The tune context class is designed to contain all resources for a tuning task.

    Different tuning tasks are separated in different TuneContext classes, but different classes in
    the same task can interact with each other through tune context. Most classes have a function
    to initialize with a tune context.

    Parameters
    ----------
    mod : Optional[IRModule] = None
        The workload to be optimized.
    target : Optional[Target] = None
        The target to be optimized for.
    space_generator : Union[None, ScheduleFnType, SpaceGenerator] = None
        The design space generator.
    search_strategy : Union[None, TuneConfig, SearchStrategy] = None
        The search strategy.
        if None, the strategy is left blank.
        If TuneConfig, the strategy is initialized with the TuneConfig.create_strategy().
    sch_rules: Union[None, str, List[ScheduleRule]] = None,
        The schedule rules.
        If None, use an empty list of rules.
        if "default", use target-default rules.
    postprocs: Union[None, str, List[Postproc"]] = None,
        The postprocessors.
        If None, use an empty list of rules.
        if "default", use target-default rules.
    mutator_probs: Union[None, str, Dict[Mutator, float]]
        Mutators and their probability mass.
        If None, use an empty list of rules.
        if "default", use target-default rules.
    task_name : Optional[str] = None
        The name of the tuning task.
    logger : logging.Logger
        The logger for the tuning task.
    rand_state : int = -1
        The random state.
        Need to be in integer in [1, 2^31-1], -1 means using random number.
    num_threads : int = None
        The number of threads to be used, None means using the logical cpu count.

    Note
    ----
    In most cases, mod and target should be available in the tuning context. They are "Optional"
    because we allow the user to customize the tuning context, along with other classes, sometimes
    without mod and target. E.g., we can have a stand alone search strategy that generates measure
    candidates without initializing with the tune context.
    """

    mod: Optional[IRModule]
    target: Optional[Target]
    space_generator: Optional["SpaceGenerator"]
    search_strategy: Optional["SearchStrategy"]
    sch_rules: List["ScheduleRule"]
    postprocs: List["Postproc"]
    mutator_probs: Optional[Dict["Mutator", float]]
    task_name: str
    logger: Optional[logging.Logger]
    rand_state: int
    num_threads: int

    def __init__(
        self,
        mod: Optional[IRModule] = None,
        *,
        target: Optional[Target] = None,
        space_generator: Union[None, "ScheduleFnType", "ScheduleFn", "SpaceGenerator"] = None,
        search_strategy: Union[None, "SearchStrategy", "TuneConfig"] = None,
        sch_rules: Union[None, str, List["ScheduleRule"]] = None,
        postprocs: Union[None, str, List["Postproc"]] = None,
        mutator_probs: Union[None, str, Dict["Mutator", float]] = None,
        task_name: str = "main",
        logger: Optional[logging.Logger] = None,
        rand_state: int = -1,
        num_threads: Optional[int] = None,
    ):
        # pylint: disable=import-outside-toplevel
        from . import default_config
        from .space_generator import ScheduleFn
        from .tune import TuneConfig

        # pylint: enable=import-outside-toplevel
        if isinstance(mod, PrimFunc):
            mod = IRModule.from_expr(mod)
        if callable(space_generator):
            space_generator = ScheduleFn(space_generator)
        if isinstance(search_strategy, TuneConfig):
            search_strategy = search_strategy.create_strategy()
        if isinstance(sch_rules, str):
            if sch_rules == "default":
                if target is None:
                    raise ValueError("target is required when sch_rules is 'default'")
                sch_rules = default_config.schedule_rules(None, target)
            else:
                raise ValueError("sch_rules should be a list of ScheduleRule or 'default'")
        if isinstance(postprocs, str):
            if postprocs == "default":
                if target is None:
                    raise ValueError("target is required when postprocs is 'default'")
                postprocs = default_config.postproc(None, target)
            else:
                raise ValueError("postprocs should be a list of Postproc or 'default'")
        if isinstance(mutator_probs, str):
            if mutator_probs == "default":
                if target is None:
                    raise ValueError("target is required when mutator_probs is 'default'")
                mutator_probs = default_config.mutator_probs(None, target)
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        if num_threads is None:
            num_threads = cpu_count(logical=False)
        self.__init_handle_by_constructor__(
            _ffi_api.TuneContext,  # type: ignore # pylint: disable=no-member
            mod,
            target,
            space_generator,
            search_strategy,
            sch_rules,
            postprocs,
            mutator_probs,
            task_name,
            make_logging_func(logger),
            rand_state,
            num_threads,
        )
        _ffi_api.TuneContextInitialize(self)  # type: ignore # pylint: disable=no-member

    def _set_measure_candidates(self, candidates):
        """Set candidates in a tuning context.

        Parameters
        ----------
        candidates : List[MeasureCandidate]
            A list of measure candidates for the tuning context.
        """
        _ffi_api.TuneContextSetMeasureCandidates(self, candidates)  # type: ignore # pylint: disable=no-member

    def _send_to_builder(self, builder):
        """Send candidates to builder.

        Parameters
        ----------
        builder : Builder
            The builder for building the candidates.
        """
        _ffi_api.TuneContextSendToBuilder(self, builder)  # type: ignore # pylint: disable=no-member

    def _send_to_runner(self, runner):
        """Send candidates to runner.

        Parameters
        ----------
        runner : Runner
            The runner for running the candidates.
        """
        _ffi_api.TuneContextSendToRunner(self, runner)  # type: ignore # pylint: disable=no-member

    def _join(self):
        """Join the runner processes.

        Returns
        -------
        result : List[RunnerResult]
            The runner results.
        """
        return _ffi_api.TuneContextJoin(self)  # type: ignore # pylint: disable=no-member

    def _clear_measure_state(self):
        """Clear the measure states."""
        _ffi_api.TuneContextClearMeasureState(self)  # type: ignore # pylint: disable=no-member

    def generate_design_space(self) -> List[Schedule]:
        """Generate design spaces given a module.

        Delegated to self.space_generator.generate_design_space with self.mod

        Returns
        -------
        design_spaces : List[Schedule]
            The generated design spaces, i.e., schedules.
        """
        if self.mod is None:
            raise ValueError("`mod` is not provided. Please construct TuneContext with `mod`")
        if self.space_generator is None:
            raise ValueError(
                "space_generator is not provided."
                "Please construct TuneContext with space_generator"
            )
        return self.space_generator.generate_design_space(self.mod)

    def pre_tuning(
        self,
        design_spaces: Optional[List[Schedule]] = None,
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """A method to be called for SearchStrategy to do necessary preparation before tuning.

        Delegated to self.search_strategy.pre_tuning.

        Parameters
        ----------
        design_spaces : Optional[List[Schedule]]
            The design spaces used during tuning process.
            If None, use the outcome of `self.generate_design_space()`.
        database : Optional[Database] = None
            The database used during tuning process.
            If None, and the search strategy is `EvolutionarySearch`,
            then use `tvm.meta_schedule.database.MemoryDatabase`.
        cost_model : Optional[CostModel] = None
            The cost model used during tuning process.
            If None, and the search strategy is `EvolutionarySearch`,
            then use `tvm.meta_schedule.cost_model.RandomModel`.
        """
        # pylint: disable=import-outside-toplevel
        from .cost_model import RandomModel
        from .database import MemoryDatabase
        from .search_strategy import EvolutionarySearch

        # pylint: enable=import-outside-toplevel

        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided."
                "Please construct TuneContext with search_strategy"
            )
        if design_spaces is None:
            design_spaces = self.generate_design_space()
        if database is None:
            if isinstance(self.search_strategy, EvolutionarySearch):
                database = MemoryDatabase()  # type: ignore
        if cost_model is None:
            if isinstance(self.search_strategy, EvolutionarySearch):
                cost_model = RandomModel()  # type: ignore
        return self.search_strategy.pre_tuning(design_spaces, database, cost_model)

    def post_tuning(self) -> None:
        """A method to be called for SearchStrategy to do necessary cleanup after tuning.

        Delegated to self.search_strategy.post_tuning.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided."
                "Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.post_tuning()

    def generate_measure_candidates(self) -> Optional[List["MeasureCandidate"]]:
        """Generate a batch of measure candidates from design spaces for measurement.

        Delegated to self.search_strategy.generate_measure_candidates.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if search is finished.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided."
                "Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List["MeasureCandidate"],
        results: List["RunnerResult"],
    ) -> None:
        """Update the state in SearchStrategy with profiling results.

        Delegated to self.search_strategy.notify_runner_results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided."
                "Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.notify_runner_results(measure_candidates, results)
