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
"""Round Robin Task Scheduler"""

from typing import List, TYPE_CHECKING

from tvm._ffi import register_object

from ..builder import Builder
from ..runner import Runner
from ..database import Database
from .task_scheduler import TaskScheduler

from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.RoundRobin")
class RoundRobin(TaskScheduler):
    """Round Robin Task Scheduler"""

    def __init__(
        self,
        tasks: List["TuneContext"],
        builder: Builder,
        runner: Runner,
        database: Database,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        tasks : List[TuneContext]
            List of tasks to schedule.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        database : Database
            The database.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerRoundRobin,  # type: ignore # pylint: disable=no-member
            tasks,
            builder,
            runner,
            database,
        )
