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
"""A lightweight wrapper on an arbitrary function that can be used to schedule a TIR PrimFunc."""
from typing import Callable, List, Union

from tvm import tir
from tvm.target import Target


class ScheduleRule:  # pylint: disable=too-few-public-methods
    """A thin wrapper on an arbitrary function that can be used to schedule a TIR PrimFunc.

    Given a PrimFunc, a target, and a tunable flag, the apply method of a ScheduleRule
    returns either a Schedule, a list of Schedules, or None, where None means that the rule
    is not applicable to the given PrimFunc. If the tunable flag is True, the ScheduleRule is
    allowed to return either a Schedule or a list of Schedules, and the Schedules are allowed to
    contain tunable instructions. If the tunable flag is False, the ScheduleRule is only allowed to
    return a Schedule, and the Schedule is not allowed to contain tunable instructions.
    """

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        tunable: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        """Apply the ScheduleRule to the given PrimFunc.

        Parameters
        ----------
        func : tir.PrimFunc
            The PrimFunc to apply the ScheduleRule to.
        target : Target
            The compilation target the schedule is supposed to be built for.
        tunable : bool
            Whether the schedule is allowed to contain tunable instructions.

        Returns
        -------
        results : Union[None, tir.Schedule, List[tir.Schedule]]
            Either a Schedule, a list of Schedules, or None, where None means that the rule
            is not applicable to the given PrimFunc.
        """
        raise NotImplementedError

    @staticmethod
    def from_callable(
        name,
    ) -> Callable[
        [
            Callable[
                [tir.PrimFunc, Target, bool],
                Union[None, tir.Schedule, List[tir.Schedule]],
            ],
        ],
        "ScheduleRule",
    ]:
        """Create a ScheduleRule from a callable.

        Parameters
        ----------
        name : str

        Returns
        -------
        decorator : Callable
            A decorator that takes a callable and returns a ScheduleRule.

        Examples
        --------
        .. code-block:: python

            @ScheduleRule.from_callable("MyRule")
            def my_rule(func: tir.PrimFunc, target: Target, tunable: bool) -> Union[None, Schedule]
                # Do something with func and target
        """

        def decorator(f) -> "ScheduleRule":  # pylint: disable=invalid-name
            class _Rule(ScheduleRule):
                def apply(
                    self,
                    func: tir.PrimFunc,
                    target: Target,
                    tunable: bool,
                ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
                    return f(func, target, tunable)

            _Rule.__name__ = name
            return _Rule()

        return decorator

    def is_target_available(self, target: Target) -> bool:  # pylint: disable=unused-argument
        """Check whether the rule is available for the given target.

        Parameters
        ----------
        target : Target
            The compilation target the schedule is supposed to be built for.

        Returns
        -------
        available : bool
            Whether the rule is available for the given target.
        """
        return True
