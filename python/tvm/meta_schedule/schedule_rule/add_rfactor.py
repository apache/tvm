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
"""Add-rfactor Rule that add-rfactor to some blocks if needed"""
from typing import Optional

from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


@register_object("meta_schedule.AddRFactor")
class AddRFactor(ScheduleRule):
    """Rules for add-rfactor to some blocks if needed.

    Parameters
    ----------
    max_jobs_per_core: int
        The maximum number of jobs to be launched per CPU core. It sets the uplimit of CPU
        parallelism, i.e. `num_cores * max_jobs_per_core`.
        Use -1 to disable parallelism.
    max_innermost_factor: Optional[int] = None
        The maximum size of the innermost factor. None means no limit.
    """

    def __init__(
        self,
        max_jobs_per_core: int = 16,
        max_innermost_factor: Optional[int] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleAddRFactor,  # type: ignore # pylint: disable=no-member
            max_jobs_per_core,
            max_innermost_factor,
        )
