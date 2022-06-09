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
Meta schedule design space generators that generates design
space via a schedule function.
"""
from typing import TYPE_CHECKING, Callable, List, Union

from tvm.ir import IRModule
from tvm.ir.container import Array
from tvm.meta_schedule.utils import derived_object
from tvm.tir.schedule import Schedule

from .space_generator import PySpaceGenerator

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@derived_object
class ScheduleFn(PySpaceGenerator):
    """A design space generator with design spaces specified by a schedule function."""

    # Multiple cases of schedule functions supported
    SCH_FN_TYPE = Union[
        Callable[[Schedule], None],  # No output
        Callable[[Schedule], Schedule],  # Single output
        Callable[[Schedule], List[Schedule]],  # Multiple outputs
    ]

    def __init__(self, sch_fn: SCH_FN_TYPE):
        """Constructor.

        Parameters
        ----------
        sch_fn : SCH_FN_TYPE
            The schedule function.
        """
        super().__init__()
        self.sch_fn = sch_fn

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the design space generator.
        """

    def generate_design_space(self, mod: IRModule) -> List[Schedule]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : List[Schedule]
            The generated design spaces, i.e., schedules.
        """
        sch = Schedule(mod)  # Make sure the schedule is traced
        result = self.sch_fn(sch)  # Call the schedule function
        if result is None:  # Case 1. No output
            return [sch]
        if isinstance(result, Schedule):  # Case 2. Single output
            return [result]
        if isinstance(result, (list, tuple, Array)):  # Case 3. Multiple outputs
            for ret in result:  # enumerate the outputs
                if not isinstance(ret, Schedule):
                    raise TypeError(
                        "Wrong type of element in the list, expected Schedule got "
                        + f"'{type(ret)}': {ret}"
                    )
            return result
        raise TypeError(f"Unexpected return type {type(result)}: {result}")
