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
"""Rules which apply cross-thread reduction to some reduction blocks correspondingly when needed"""
from typing import List

from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


@register_object("meta_schedule.CrossThreadReduction")
class CrossThreadReduction(ScheduleRule):
    """A schedule rule which applies cross-thread reduction to some reduction blocks
    correspondingly when needed

    Parameters
    ----------
    thread_extents: List[int]
        Candidates of thread axis extent (values are required to be positive).
    """

    def __init__(self, thread_extents: List[int]) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleCrossThreadReduction,  # type: ignore # pylint: disable=no-member
            thread_extents,
        )
