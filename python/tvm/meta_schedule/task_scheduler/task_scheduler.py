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
from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """The abstract task scheduler interface."""

    def tune(self) -> None:
        """Auto-tuning."""
        _ffi_api.TaskSchedulerTune(self)  # pylint: disable=no-member

    def _set_task_stopped(self, task_id: int) -> None:
        """Set specific task to be stopped.

        Parameters
        ----------
        task_id : int
            The task id to be stopped.
        """
        _ffi_api.TaskSchedulerSetTaskStopped(self, task_id)  # pylint: disable=no-member

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
        return _ffi_api.TaskSchedulerIsTaskRunning(self, task_id)  # pylint: disable=no-member

    def _join_running_task(self, task_id: int) -> None:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.
        """
        _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # pylint: disable=no-member

    def _next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # pylint: disable=no-member


@register_object("meta_schedule.PyTaskScheduler")
class PyTaskScheduler(TaskScheduler):
    """An abstract task scheduler with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_tune() -> None:
            self.tune()

        def f_set_task_stopped(task_id: int) -> None:
            self._set_task_stopped(task_id)

        def f_is_task_running(task_id: int) -> bool:
            return self._is_task_running(task_id)

        def f_join_running_task(task_id: int) -> None:
            self._join_running_task(task_id)

        def f_next_task_id() -> int:
            return self._next_task_id()

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerPyTaskScheduler,  # pylint: disable=no-member
            f_tune,
            f_set_task_stopped,
            f_is_task_running,
            f_join_running_task,
            f_next_task_id,
        )

    def tune(self) -> None:
        raise NotImplementedError()

    def _set_task_stopped(self, task_id: int) -> None:
        _ffi_api.TaskSchedulerSetTaskStopped(self, task_id)  # pylint: disable=no-member

    def _is_task_running(self, task_id: int) -> bool:
        return _ffi_api.TaskSchedulerIsTaskRunning(self, task_id)  # pylint: disable=no-member

    def _join_running_task(self, task_id: int) -> None:
        _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # pylint: disable=no-member

    def _next_task_id(self) -> int:
        return _ffi_api.TaskSchedulerNextTaskId(self)  # pylint: disable=no-member
