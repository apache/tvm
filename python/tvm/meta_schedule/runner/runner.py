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
from ..arg_info import ArgInfo
from ..utils import check_override


@register_object("meta_schedule.RunnerInput")
class RunnerInput(Object):
    """The runner's input

    Parameters
    ----------
    artifact_path : str
        The path to the built artifact.
    device_type : str
        The device type.
    args_info : List[ArgInfo]
        The argument information.
    """

    artifact_path: str
    device_type: str
    args_info: List[ArgInfo]

    def __init__(
        self,
        artifact_path: str,
        device_type: str,
        args_info: List[ArgInfo],
    ) -> None:
        """Constructor

        Parameters
        ----------
        artifact_path : str
            The path to the built artifact.
        device_type : str
            The device type.
        args_info : List[ArgInfo]
            The argument information.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerInput,  # type: ignore # pylint: disable=no-member
            artifact_path,
            device_type,
            args_info,
        )


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


@register_object("meta_schedule.RunnerFuture")
class RunnerFuture(Object):
    """A class to fetch asynchronous runner's output."""

    def __init__(self) -> None:
        """Constructor"""

        def f_done():
            return self.done()

        def f_result():
            return self.result()

        self.__init_handle_by_constructor__(
            _ffi_api.RunnerFuture,  # type: ignore # pylint: disable=no-member
            f_done,
            f_result,
        )

    def done(self) -> bool:
        """Check whether the runner has finished."""
        raise NotImplementedError

    def result(self) -> RunnerResult:
        """Fetch the runner's output if it is ready."""
        raise NotImplementedError


@register_object("meta_schedule.Runner")
class Runner(Object):
    """The abstract runner interface"""

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        """Run the built artifact and get runner futures.

        Parameters
        ----------
        runner_inputs : List[RunnerInput]
            The inputs to the runner.

        Returns
        -------
        runner_futures: List[RunnerFuture]
            The runner futures.
        """
        return _ffi_api.RunnerRun(self, runner_inputs)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyRunner")
class PyRunner(Runner):
    """An abstract runner with customized build method on the python-side."""

    def __init__(self) -> None:
        """Constructor"""

        @check_override(self.__class__, Runner)
        def f_run(runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
            return self.run(runner_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.RunnerPyRunner,  # type: ignore # pylint: disable=no-member
            f_run,
        )
