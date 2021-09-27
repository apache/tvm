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
"""Runners"""
from typing import List, Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api


@register_object("meta_schedule.RunnerResult")
class RunnerResult(Object):
    """The runner's result

    Parameters
    ----------
    run_secs : Optional[List[float]]
        The run time in seconds.
    error_msg : Optional[str]
        The error message, if any.
    """

    run_secs: Optional[List[float]]
    error_msg: Optional[str]

    def __init__(
        self,
        run_secs: Optional[List[float]],
        error_msg: Optional[str],
    ) -> None:
        """Constructor

        Parameters
        ----------
        run_secs : Optional[List[float]]
            The run time in seconds.
        error_msg : Optional[str]
            The error message, if any.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerResult,  # type: ignore # pylint: disable=no-member
            run_secs,
            error_msg,
        )
