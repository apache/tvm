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
"""Post Order Apply Space Generator."""
from tvm._ffi import register_object

from .. import _ffi_api
from .space_generator import (
    MutatorProbType,
    PostprocType,
    ScheduleRuleType,
    SpaceGenerator,
    _normalize_rules,
)


@register_object("meta_schedule.PostOrderApply")
class PostOrderApply(SpaceGenerator):
    """
    PostOrderApply is the design space generator that generates design spaces by applying schedule
    rules to blocks in post-DFS order.

    Parameters
    ----------
    f_block_filter : Optional[function]
        An optional callback function that is used to filter which blocks have schedules generated
        for them. The function should take in a block and return True if a schedule should
        be generated or False if that block should be skipped. If no function is provided
        all blocks will have schedules generated.
    """

    def __init__(
        self,
        f_block_filter=None,
        sch_rules: ScheduleRuleType = "from-target",
        postprocs: PostprocType = "from-target",
        mutator_probs: MutatorProbType = "from-target",
    ):
        """Constructor"""
        sch_rules, postprocs, mutator_probs = _normalize_rules(sch_rules, postprocs, mutator_probs)
        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorPostOrderApply,  # type: ignore # pylint: disable=no-member
            f_block_filter,
            sch_rules,
            postprocs,
            mutator_probs,
        )
