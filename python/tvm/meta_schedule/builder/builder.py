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
from typing import Callable, Dict, List, Optional

# isort: off
from typing_extensions import Literal

# isort: on
from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import NDArray, Object
from tvm.target import Target

from .. import _ffi_api


@register_object("meta_schedule.BuilderInput")
class BuilderInput(Object):
    """The builder's input.

    Parameters
    ----------
    mod : IRModule
        The IRModule to be built.
    target : Target
        The target to be built for.
    params: Optional[Dict[str, NDArray]]
        The parameters for Relay build module
    """

    mod: IRModule
    target: Target
    params: Optional[Dict[str, NDArray]]

    def __init__(
        self,
        mod: IRModule,
        target: Target,
        params: Optional[Dict[str, NDArray]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be built.
        target : Target
            The target to be built for.
        params: Optional[Dict[str, NDArray]]
            The parameters for Relay build module
        """
        self.__init_handle_by_constructor__(
            _ffi_api.BuilderInput,  # type: ignore # pylint: disable=no-member
            mod,
            target,
            params,
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
class _PyBuilder(Builder):
    """
    A TVM object builder to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyBuilder
    """

    def __init__(self, f_build: Callable = None):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.BuilderPyBuilder,  # type: ignore # pylint: disable=no-member
            f_build,
        )


class PyBuilder:
    """
    An abstract builder with customized build method on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {"cls": _PyBuilder, "methods": ["build"]}

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
        raise NotImplementedError


def create(  # pylint: disable=keyword-arg-before-vararg
    kind: Literal["local"] = "local",
    *args,
    **kwargs,
) -> Builder:
    """Create a Builder."""
    from . import LocalBuilder  # pylint: disable=import-outside-toplevel

    if kind == "local":
        return LocalBuilder(*args, **kwargs)  # type: ignore
    raise ValueError(f"Unknown Builder: {kind}")
