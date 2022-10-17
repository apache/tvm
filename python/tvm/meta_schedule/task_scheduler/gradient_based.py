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
"""Gradient Based Task Scheduler"""
from tvm._ffi import register_object

from .. import _ffi_api
from ..logging import get_logger, get_logging_func
from .task_scheduler import TaskScheduler

logger = get_logger(__name__)  # pylint: disable=invalid-name


@register_object("meta_schedule.GradientBased")
class GradientBased(TaskScheduler):
    """Gradient Based Task Scheduler"""

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        window_size: int = 3,
        seed: int = -1,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        alpha : float = 0.2
            The parameter alpha in gradient computation.
        window_size : int = 3
            The parameter to control backward window size in gradient computation.
        seed : int = -1
            The random seed.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerGradientBased,  # type: ignore # pylint: disable=no-member
            get_logging_func(logger),
            alpha,
            window_size,
            seed,
        )
