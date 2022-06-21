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
"""Auto-bind Rule that binds blocks to threads if needed"""
from typing import List, Optional

from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


@register_object("meta_schedule.AutoBind")
class AutoBind(ScheduleRule):
    """Auto bind loops around the block to BlockIdx and ThreadIdx

    Parameters
    ----------
    max_threadblocks: int
        The maximum number of threadblock on GPU.
    thread_extents: Optional[List[int]]
        Candidates of thread axis extent.
    """

    def __init__(
        self,
        max_threadblocks: int = 256,
        thread_extents: Optional[List[int]] = None,
    ) -> None:
        if thread_extents is None:
            thread_extents = [32, 64, 128, 256, 512, 1024]
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleAutoBind,  # type: ignore # pylint: disable=no-member
            max_threadblocks,
            thread_extents,
        )
