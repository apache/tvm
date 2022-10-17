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
"""Meta Schedule MeasureCallback."""

from typing import TYPE_CHECKING, Callable, List, Union

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..builder import BuilderResult
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..utils import _get_default_str

if TYPE_CHECKING:
    from ..task_scheduler import TaskScheduler


@register_object("meta_schedule.MeasureCallback")
class MeasureCallback(Object):
    """Rules to apply after measure results is available."""

    CallbackListType = Union[List["MeasureCallback"], "MeasureCallback", Literal["default"]]

    def apply(
        self,
        task_scheduler: "TaskScheduler",
        task_id: int,
        measure_candidates: List[MeasureCandidate],
        builder_results: List[BuilderResult],
        runner_results: List[RunnerResult],
    ) -> None:
        """Apply a measure callback to the given schedule.

        Parameters
        ----------
        task_scheduler: TaskScheduler
            The task scheduler.
        task_id: int
            The task id.
        measure_candidates: List[MeasureCandidate]
            The measure candidates.
        builder_results: List[BuilderResult]
            The builder results by building the measure candidates.
        runner_results: List[RunnerResult]
            The runner results by running the built measure candidates.
        """
        return _ffi_api.MeasureCallbackApply(  # type: ignore # pylint: disable=no-member
            self,
            task_scheduler,
            task_id,
            measure_candidates,
            builder_results,
            runner_results,
        )

    @staticmethod
    def create(kind: Literal["default"]) -> List["MeasureCallback"]:
        """Create a list of measure callbacks."""
        if kind == "default":
            return _ffi_api.MeasureCallbackDefault()  # type: ignore # pylint: disable=no-member
        raise ValueError(f"Unknown kind of MeasureCallback list: {kind}")


@register_object("meta_schedule.PyMeasureCallback")
class _PyMeasureCallback(MeasureCallback):
    """
    A TVM object measure callback to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyMeasureCallback
    """

    def __init__(self, f_apply: Callable, f_as_string: Callable = None):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MeasureCallbackPyMeasureCallback,  # type: ignore # pylint: disable=no-member
            f_apply,
            f_as_string,
        )


class PyMeasureCallback:
    """
    An abstract measure callback with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyMeasureCallback,
        "methods": ["apply", "__str__"],
    }

    def apply(
        self,
        task_scheduler: "TaskScheduler",
        task_id: int,
        measure_candidates: List[MeasureCandidate],
        builder_results: List[BuilderResult],
        runner_results: List[RunnerResult],
    ) -> None:
        """Apply a measure callback to the given schedule.

        Parameters
        ----------
        task_scheduler: TaskScheduler
            The task scheduler.
        task_id: int
            The task id.
        measure_candidates: List[MeasureCandidate]
            The measure candidates.
        builder_results: List[BuilderResult]
            The builder results by building the measure candidates.
        runner_results: List[RunnerResult]
            The runner results by running the built measure candidates.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return _get_default_str(self)
