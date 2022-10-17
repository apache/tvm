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
"""Meta Schedule Postproc."""
from typing import TYPE_CHECKING, Callable, List

# isort: off
from typing_extensions import Literal

# isort: on

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api
from ..utils import _get_default_str

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.Postproc")
class Postproc(Object):
    """Rules to apply a postprocessor to a schedule."""

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the postprocessor with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the postprocessor.
        """
        _ffi_api.PostprocInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def apply(self, sch: Schedule) -> bool:
        """Apply a postprocessor to the given schedule.

        Parameters
        ----------
        sch : Schedule
            The schedule to be post processed.

        Returns
        -------
        result : bool
            Whether the postprocessor was successfully applied.
        """
        return _ffi_api.PostprocApply(self, sch)  # type: ignore # pylint: disable=no-member

    def clone(self) -> "Postproc":
        """Clone the postprocessor.

        Returns
        -------
        cloned_postproc : Postproc
            The cloned postprocessor.
        """
        return _ffi_api.PostprocClone(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(kind: Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"]) -> List["Postproc"]:
        """Create a list of default postprocessors.

        Parameters
        ----------
        kind : Literal["llvm", "cuda", "cuda-tensorcore", "hexagon"]
            The kind of the postprocessors.

        Returns
        -------
        postprocs : List[Mutator]
            The list of postprocessors.
        """
        funcs = {
            # pylint: disable=no-member
            "llvm": _ffi_api.PostprocDefaultLLVM,  # type: ignore
            "cuda": _ffi_api.PostprocDefaultCUDA,  # type: ignore
            "cuda-tensorcore": _ffi_api.PostprocDefaultCUDATensorCore,  # type: ignore
            "hexagon": _ffi_api.PostprocDefaultHexagon,  # type: ignore
            # pylint: enable=no-member
        }
        for k, v in funcs.items():
            if k == kind:
                return v()
        raise ValueError(f"Unsupported kind {kind} for postproc creation.")


create = Postproc.create  # pylint: disable=invalid-name


@register_object("meta_schedule.PyPostproc")
class _PyPostproc(Postproc):
    """
    A TVM object post processor to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyPostproc
    """

    def __init__(
        self,
        f_initialize_with_tune_context: Callable = None,
        f_apply: Callable = None,
        f_clone: Callable = None,
        f_as_string: Callable = None,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.PostprocPyPostproc,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_clone,
            f_as_string,
        )


class PyPostproc:
    """
    An abstract post processor with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyPostproc,
        "methods": ["_initialize_with_tune_context", "apply", "clone", "__str__"],
    }

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the postprocessor with a tune context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the postprocessor.
        """
        raise NotImplementedError

    def apply(self, sch: Schedule) -> bool:
        """Apply a postprocessor to the given schedule.

        Parameters
        ----------
        sch : Schedule
            The schedule to be post processed.

        Returns
        -------
        result : bool
            Whether the postprocessor was successfully applied.
        """
        raise NotImplementedError

    def clone(self) -> Postproc:
        """Clone the postprocessor.

        Returns
        -------
        cloned_postproc : Postproc
            The cloned postprocessor.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Get the post processor as string with name.

        Return
        ------
        result : str
            Get the post processor as string with name.
        """
        return _get_default_str(self)
