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
Meta Schedule schedule rules are used for modification of
blocks in a schedule. See also PostOrderApply.
"""
from typing import TYPE_CHECKING, Callable, List

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import BlockRV, Schedule

from .. import _ffi_api
from ..utils import _get_default_str

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.ScheduleRule")
class ScheduleRule(Object):
    """Rules to modify a block in a schedule."""

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the schedule rule with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the schedule rule.
        """
        _ffi_api.ScheduleRuleInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        """Apply a schedule rule to the specific block in the given schedule.

        Parameters
        ----------
        sch : tvm.tir.Schedule
            The schedule to be modified.
        block : BlockRV
            The specific block to apply the schedule rule.

        Returns
        -------
        design_spaces : List[tvm.tir.Schedule]
            The list of schedules generated by applying the schedule rule.
        """
        return _ffi_api.ScheduleRuleApply(  #  type: ignore # pylint: disable=no-member
            self, sch, block
        )

    def clone(self) -> "ScheduleRule":
        """Deep clone the schedule rule.

        Returns
        -------
        cloned_rule : ScheduleRule
            The cloned schedule rule.
        """
        return _ffi_api.ScheduleRuleClone(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(kind: Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"]) -> List["ScheduleRule"]:
        """Create a list of schedule rules for the given kind.

        Parameters
        ----------
        kind : Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"]
            The kind of the schedule rules.

        Returns
        -------
        rules : List[ScheduleRule]
            The list of schedule rules.
        """
        funcs = {
            # pylint: disable=no-member
            "llvm": _ffi_api.ScheduleRuleDefaultLLVM,  # type: ignore
            "cuda": _ffi_api.ScheduleRuleDefaultCUDA,  # type: ignore
            "cuda-tensorcore": _ffi_api.ScheduleRuleDefaultCUDATensorCore,  # type: ignore
            "hexagon": _ffi_api.ScheduleRuleDefaultHexagon,  # type: ignore
            # pylint: enable=no-member
        }
        for k, v in funcs.items():
            if k == kind:
                return v()
        raise ValueError(f"Unsupported kind {kind} for schedule rule creation.")


create = ScheduleRule.create  # pylint: disable=invalid-name


@register_object("meta_schedule.PyScheduleRule")
class _PyScheduleRule(ScheduleRule):
    """
    A TVM object schedule rule to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyScheduleRule
    """

    def __init__(
        self,
        f_initialize_with_tune_context: Callable = None,
        f_apply: Callable = None,
        f_clone: Callable = None,
        f_as_string: Callable = None,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRulePyScheduleRule,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_clone,
            f_as_string,
        )


class PyScheduleRule:
    """
    An abstract schedule rule with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyScheduleRule,
        "methods": ["_initialize_with_tune_context", "apply", "clone", "__str__"],
    }

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the schedule rule with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the schedule rule.
        """
        raise NotImplementedError

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        """Apply a schedule rule to the specific block in the given schedule.

        Parameters
        ----------
        sch : Schedule
            The schedule to be modified.
        block : BlockRV
            The specific block to apply the schedule rule.

        Returns
        -------
        design_spaces : List[Schedule]
            The list of schedules generated by applying the schedule rule.
        """
        raise NotImplementedError

    def clone(self) -> ScheduleRule:
        """Deep clone the schedule rule.

        Returns
        -------
        cloned_rule : ScheduleRule
            The cloned schedule rule.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Get the schedule rule as string with name.

        Return
        ------
        result : str
            Get the schedule rule as string with name.
        """
        return _get_default_str(self)
