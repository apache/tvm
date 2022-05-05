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

import logging
from typing import TYPE_CHECKING, List, Optional

from tvm._ffi import register_object
from tvm.meta_schedule.measure_callback.measure_callback import MeasureCallback

from .. import _ffi_api
from ..builder import Builder
from ..cost_model import CostModel
from ..database import Database
from ..runner import Runner
from ..utils import make_logging_func
from .task_scheduler import TaskScheduler

if TYPE_CHECKING:
    from ..tune_context import TuneContext

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@register_object("meta_schedule.RoundRobin")
class RoundRobin(TaskScheduler):
    """Round Robin Task Scheduler

    Parameters
    ----------
    tasks: List[TuneContext]
        The list of tune context to process.
    builder: Builder
        The builder of the scheduler.
    runner: Runner
        The runner of the scheduler.
    database: Database
        The database of the scheduler.
    measure_callbacks: Optional[List[MeasureCallback]] = None
        The list of measure callbacks of the scheduler.
    """

    def __init__(
        self,
        tasks: List["TuneContext"],
        task_weights: List[float],
        builder: Builder,
        runner: Runner,
        database: Database,
        max_trials: int,
        *,
        cost_model: Optional[CostModel] = None,
        measure_callbacks: Optional[List[MeasureCallback]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        tasks : List[TuneContext]
            List of tasks to schedule.
        task_weights : List[float]
            List of weights for each task. Not used in round robin.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        database : Database
            The database.
        max_trials : int
            The maximum number of trials.
        cost_model : Optional[CostModel]
            The cost model.
        measure_callbacks: Optional[List[MeasureCallback]]
            The list of measure callbacks of the scheduler.
        """
        del task_weights
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerRoundRobin,  # type: ignore # pylint: disable=no-member
            tasks,
            builder,
            runner,
            database,
            max_trials,
            cost_model,
            measure_callbacks,
            make_logging_func(logger),
        )
