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
"""Search Strategy"""

from typing import List, Optional, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..runner import RunnerResult

if TYPE_CHECKING:
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

    def __init__(self, sch: Schedule, args_info: List[ArgInfo]) -> None:
        """Constructor.

        Parameters
        ----------
        sch : Schedule
            The schedule to be measured.
        args_info : List[ArgInfo]
            The argument information.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureCandidate,  # pylint: disable=no-member
            sch,
            args_info,
        )


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """
    Search strategy is the class that generates the measure candidates. It has to be pre-tuned
    before usage and post-tuned after usage.
    """

    def initialize_with_tune_context(
        self,
        tune_context: "TuneContext",
    ) -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initialization.
        """
        _ffi_api.SearchStrategyInitializeWithTuneContext(  # pylint: disable=no-member
            self, tune_context
        )

    def pre_tuning(self, design_spaces: List[Schedule]) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        design_spaces : List[Schedule]
            The design spaces for pre-tuning.
        """
        _ffi_api.SearchStrategyPreTuning(self, design_spaces)  # pylint: disable=no-member

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        _ffi_api.SearchStrategyPostTuning(self)  # pylint: disable=no-member

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        return _ffi_api.SearchStrategyGenerateMeasureCandidates(self)  # pylint: disable=no-member

    def notify_runner_results(self, results: List[RunnerResult]) -> None:
        """Update the search strategy with profiling results.

        Parameters
        ----------
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        _ffi_api.SearchStrategyNotifyRunnerResults(self, results)  # pylint: disable=no-member


@register_object("meta_schedule.PySearchStrategy")
class PySearchStrategy(SearchStrategy):
    """An abstract search strategy with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_initialize_with_tune_context(context: "TuneContext") -> None:
            self.initialize_with_tune_context(context)

        def f_pre_tuning(design_spaces: List[Schedule]) -> None:
            self.pre_tuning(design_spaces)

        def f_post_tuning() -> None:
            self.post_tuning()

        def f_generate_measure_candidates() -> List[MeasureCandidate]:
            return self.generate_measure_candidates()

        def f_notify_runner_results(results: List["RunnerResult"]) -> None:
            self.notify_runner_results(results)

        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyPySearchStrategy,  # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_pre_tuning,
            f_post_tuning,
            f_generate_measure_candidates,
            f_notify_runner_results,
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        raise NotImplementedError

    def pre_tuning(self, design_spaces: List[Schedule]) -> None:
        raise NotImplementedError

    def post_tuning(self) -> None:
        raise NotImplementedError

    def generate_measure_candidates(self) -> List[MeasureCandidate]:
        raise NotImplementedError

    def notify_runner_results(self, results: List["RunnerResult"]) -> None:
        raise NotImplementedError
