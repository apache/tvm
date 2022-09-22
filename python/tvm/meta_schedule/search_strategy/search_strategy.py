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
"""
Meta Schedule search strategy that generates the measure
candidates for measurement.
"""
from typing import TYPE_CHECKING, Callable, List, Optional

# isort: off
from typing_extensions import Literal

# isort: on
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..runner import RunnerResult

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext


@register_object("meta_schedule.MeasureCandidate")
class MeasureCandidate(Object):
    """Measure candidate class.

    Parameters
    ----------
    sch : Schedule
        The schedule to be measured.
    args_info : List[ArgInfo]
        The argument information.
    """

    sch: Schedule
    args_info: List[ArgInfo]

    def __init__(
        self,
        sch: Schedule,
        args_info: List[ArgInfo],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        sch : Schedule
            The schedule to be measured.
        args_info : List[ArgInfo]
            The argument information.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureCandidate,  # type: ignore # pylint: disable=no-member
            sch,
            args_info,
        )


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """
    Search strategy is the class that generates the measure candidates. It has to be pre-tuned
    before usage and post-tuned after usage.
    """

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        _ffi_api.SearchStrategyInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def pre_tuning(
        self,
        design_spaces: List[Schedule],
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        design_spaces : List[Schedule]
            The design spaces used during tuning process.
        database : Optional[Database] = None
            The database used during tuning process.
        cost_model : Optional[CostModel] = None
            The cost model used during tuning process.
        """
        _ffi_api.SearchStrategyPreTuning(  # type: ignore # pylint: disable=no-member
            self,
            design_spaces,
            database,
            cost_model,
        )

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        _ffi_api.SearchStrategyPostTuning(self)  # type: ignore # pylint: disable=no-member

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        return _ffi_api.SearchStrategyGenerateMeasureCandidates(self)  # type: ignore # pylint: disable=no-member

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the search strategy with profiling results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        _ffi_api.SearchStrategyNotifyRunnerResults(  # type: ignore # pylint: disable=no-member
            self,
            measure_candidates,
            results,
        )

    def clone(self) -> "SearchStrategy":
        """Clone the search strategy.

        Returns
        -------
        cloned : SearchStrategy
            The cloned search strategy.
        """
        return _ffi_api.SearchStrategyClone(self)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PySearchStrategy")
class _PySearchStrategy(SearchStrategy):
    """
    A TVM object search strategy to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PySearchStrategy
    """

    def __init__(
        self,
        f_initialize_with_tune_context: Callable = None,
        f_pre_tuning: Callable = None,
        f_post_tuning: Callable = None,
        f_generate_measure_candidates: Callable = None,
        f_notify_runner_results: Callable = None,
        f_clone: Callable = None,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyPySearchStrategy,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_pre_tuning,
            f_post_tuning,
            f_generate_measure_candidates,
            f_notify_runner_results,
            f_clone,
        )


class PySearchStrategy:
    """
    An abstract search strategy with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PySearchStrategy,
        "methods": [
            "_initialize_with_tune_context",
            "pre_tuning",
            "post_tuning",
            "generate_measure_candidates",
            "notify_runner_results",
            "clone",
        ],
    }

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        raise NotImplementedError

    def pre_tuning(self, design_spaces: List[Schedule]) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        design_spaces : List[Schedule]
            The design spaces for pre-tuning.
        """
        raise NotImplementedError

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        raise NotImplementedError

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        raise NotImplementedError

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the search strategy with profiling results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        raise NotImplementedError

    def clone(self) -> SearchStrategy:
        """Clone the search strategy.

        Returns
        -------
        strategy : SearchStrategy
            The cloned search strategy.
        """
        raise NotImplementedError


def create(  # pylint: disable=keyword-arg-before-vararg
    kind: Literal[
        "evolutionary",
        "replay_trace",
        "replay_func",
    ] = "evolutionary",
    *args,
    **kwargs,
) -> SearchStrategy:
    """Create a search strategy."""
    from . import (  # pylint: disable=import-outside-toplevel
        EvolutionarySearch,
        ReplayFunc,
        ReplayTrace,
    )

    if kind == "evolutionary":
        return EvolutionarySearch(*args, **kwargs)
    if kind == "replay_trace":
        return ReplayTrace(*args, **kwargs)
    if kind == "replay_func":
        return ReplayFunc(*args, **kwargs)
    raise ValueError(f"Unknown SearchStrategy: {kind}")
