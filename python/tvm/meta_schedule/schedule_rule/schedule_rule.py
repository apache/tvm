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
from typing import TYPE_CHECKING, List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule, BlockRV

from ..utils import _get_hex_address
from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.ScheduleRule")
class ScheduleRule(Object):
    """Rules to modify a block in a schedule."""

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the schedule rule with a tune context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.ScheduleRuleInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, tune_context
        )

    def apply(self, schedule: Schedule, block: BlockRV) -> List[Schedule]:
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
        return _ffi_api.ScheduleRuleApply(self, schedule, block)


@register_object("meta_schedule.PyScheduleRule")
class PyScheduleRule(ScheduleRule):
    """An abstract schedule rule with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_initialize_with_tune_context(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        def f_apply(sch: Schedule, block: BlockRV) -> List[Schedule]:
            return self.apply(sch, block)

        def f_as_string() -> str:
            return self.__str__()

        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRulePyScheduleRule,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_as_string,
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        raise NotImplementedError

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"PyScheduleRule({_get_hex_address(self.handle)})"
