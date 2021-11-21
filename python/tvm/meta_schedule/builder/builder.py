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
"""Meta Schedule builders that translate IRModule to runtime.Module, and then export"""
from typing import List, Optional

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.target import Target

from .. import _ffi_api
from ..utils import check_override


@register_object("meta_schedule.BuilderInput")
class BuilderInput(Object):
    """The builder's input.

    Parameters
    ----------
    mod : IRModule
        The IRModule to be built.
    target : Target
        The target to be built for.
    """

    mod: IRModule
    target: Target

    def __init__(self, mod: IRModule, target: Target) -> None:
        """Constructor.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be built.
        target : Target
            The target to be built for.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.BuilderInput,  # type: ignore # pylint: disable=no-member
            mod,
            target,
        )


@register_object("meta_schedule.BuilderResult")
class BuilderResult(Object):
    """The builder's result.

    Parameters
    ----------
    artifact_path : Optional[str]
        The path to the artifact.
    error_msg : Optional[str]
        The error message.
    """

    artifact_path: Optional[str]
    error_msg: Optional[str]

    def __init__(
        self,
        artifact_path: Optional[str],
        error_msg: Optional[str],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        artifact_path : Optional[str]
            The path to the artifact.
        error_msg : Optional[str]
            The error message.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.BuilderResult,  # type: ignore # pylint: disable=no-member
            artifact_path,
            error_msg,
        )


@register_object("meta_schedule.Builder")
class Builder(Object):
    """The abstract builder interface."""

    def build(self, build_inputs: List[BuilderInput]) -> List[BuilderResult]:
        """Build the given inputs.

        Parameters
        ----------
        build_inputs : List[BuilderInput]
            The inputs to be built.
        Returns
        -------
        build_results : List[BuilderResult]
            The results of building the given inputs.
        """
        return _ffi_api.BuilderBuild(self, build_inputs)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyBuilder")
class PyBuilder(Builder):
    """An abstract builder with customized build method on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, Builder)
        def f_build(build_inputs: List[BuilderInput]) -> List[BuilderResult]:
            return self.build(build_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.BuilderPyBuilder,  # type: ignore # pylint: disable=no-member
            f_build,
        )
