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
"""A database for injecting handcrafted schedule functions."""
from typing import Callable

from tvm._ffi import register_object
from tvm.tir import Schedule

from .. import _ffi_api
from .database import Database


@register_object("meta_schedule.ScheduleFnDatabase")
class ScheduleFnDatabase(Database):
    """A database for injecting handcrafted schedule functions."""

    def __init__(
        self,
        schedule_fn: Callable[[Schedule], bool],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseScheduleFnDatabase,  # type: ignore # pylint: disable=no-member
            schedule_fn,
        )
