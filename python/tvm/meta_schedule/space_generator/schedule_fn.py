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
"""Union of meta Schedule design space generators."""
from tvm._ffi import register_object

from .. import _ffi_api
from .space_generator import (
    MutatorProbType,
    PostprocType,
    ScheduleRuleType,
    SpaceGenerator,
    _normalize_rules,
)


@register_object("meta_schedule.ScheduleFn")
class ScheduleFn(SpaceGenerator):
    """Create a design space generator with customized schedule function.
    The schedule function can have the following signatures:
    - 1) [Schedule] -> None
    - 2) [Schedule] -> Schedule
    - 3) [Schedule] -> List[Schedule]
    """

    def __init__(
        self,
        sch_fn: SpaceGenerator.ScheduleFnType,
        sch_rules: ScheduleRuleType = "from-target",
        postprocs: PostprocType = "from-target",
        mutator_probs: MutatorProbType = "from-target",
    ):
        """Constructor.

        Parameters
        ----------
        sch_fn : SpaceGenerator.ScheduleFnType
            The schedule function, which can have the following signatures:
            - 1) [Schedule] -> None
            - 2) [Schedule] -> Schedule
            - 3) [Schedule] -> List[Schedule]
        """
        sch_rules, postprocs, mutator_probs = _normalize_rules(sch_rules, postprocs, mutator_probs)
        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorScheduleFn,  # type: ignore # pylint: disable=no-member
            sch_fn,
            sch_rules,
            postprocs,
            mutator_probs,
        )
