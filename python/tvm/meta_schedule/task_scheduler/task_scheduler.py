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
"""Auto-tuning Task Scheduler"""

from typing import List

from tvm._ffi import register_object
from tvm.runtime import Object

from ..runner import Runner
from ..builder import Builder
from ..database import Database
from ..tune_context import TuneContext
from .. import _ffi_api
from ..utils import check_override


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """The abstract task scheduler interface.

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
    """

    tasks: List[TuneContext]
    builder: Builder
    runner: Runner
    database: Database

    def tune(self) -> None:
        """Auto-tuning."""
        _ffi_api.TaskSchedulerTune(self)  # type: ignore # pylint: disable=no-member

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # type: ignore # pylint: disable=no-member

    def _initialize_task(self, task_id: int) -> None:
        """Initialize modules of the given task.

        Parameters
        ----------
        task_id : int
            The task id to be initialized.
        """
        _ffi_api.TaskSchedulerInitializeTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def _set_task_stopped(self, task_id: int) -> None:
        """Set specific task to be stopped.

        Parameters
        ----------
        task_id : int
            The task id to be stopped.
        """
        _ffi_api.TaskSchedulerSetTaskStopped(self, task_id)  # type: ignore # pylint: disable=no-member

    def _is_task_running(self, task_id: int) -> bool:
        """Check whether the task is running.

        Parameters
        ----------
        task_id : int
            The task id to be checked.

        Returns
        -------
        bool
            Whether the task is running.
        """
        return _ffi_api.TaskSchedulerIsTaskRunning(self, task_id)  # type: ignore # pylint: disable=no-member

    def _join_running_task(self, task_id: int) -> None:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.
        """
        _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyTaskScheduler")
class PyTaskScheduler(TaskScheduler):
    """An abstract task scheduler with customized methods on the python-side."""

    def __init__(
        self,
        tasks: List[TuneContext],
        builder: Builder,
        runner: Runner,
        database: Database,
    ):
        """Constructor.

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
        """

        @check_override(self.__class__, TaskScheduler, required=False)
        def f_tune() -> None:
            self.tune()

        @check_override(self.__class__, TaskScheduler)
        def f_next_task_id() -> int:
            return self.next_task_id()

        @check_override(
            PyTaskScheduler, TaskScheduler, required=False, func_name="_initialize_task"
        )
        def f_initialize_task(task_id: int) -> None:
            self._initialize_task(task_id)

        @check_override(
            PyTaskScheduler, TaskScheduler, required=False, func_name="_set_task_stopped"
        )
        def f_set_task_stopped(task_id: int) -> None:
            self._set_task_stopped(task_id)

        @check_override(
            PyTaskScheduler, TaskScheduler, required=False, func_name="_is_task_running"
        )
        def f_is_task_running(task_id: int) -> bool:
            return self._is_task_running(task_id)

        @check_override(
            PyTaskScheduler, TaskScheduler, required=False, func_name="_join_running_task"
        )
        def f_join_running_task(task_id: int) -> None:
            self._join_running_task(task_id)

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerPyTaskScheduler,  # type: ignore # pylint: disable=no-member
            tasks,
            builder,
            runner,
            database,
            f_tune,
            f_initialize_task,
            f_set_task_stopped,
            f_is_task_running,
            f_join_running_task,
            f_next_task_id,
        )
