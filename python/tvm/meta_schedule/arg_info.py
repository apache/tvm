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
"""The argument information"""
from typing import Any, List, Union

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import DataType, Object, ShapeTuple
from tvm.tir import PrimFunc

from . import _ffi_api
from .utils import _json_de_tvm


@register_object("meta_schedule.ArgInfo")
class ArgInfo(Object):
    """Argument information"""

    def as_json(self) -> Any:
        """Converts the ArgInfo to its corresponding JSON representation."""
        return _json_de_tvm(_ffi_api.ArgInfoAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any) -> "ArgInfo":
        """Parse the argument information from a JSON object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.

        Returns
        -------
        parsed : ArgInfo
            The argument information parsed.
        """
        return _ffi_api.ArgInfoFromJSON(json_obj)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_prim_func(func: PrimFunc) -> List["ArgInfo"]:
        """Extract a list of the argument information from PrimFunc.

        Parameters
        ----------
        func : PrimFunc
            The PrimFunc to get argument information from.

        Returns
        -------
        extracted : List[ArgInfo]
            An array of the argument information derived.
        """
        return _ffi_api.ArgInfoFromPrimFunc(func)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_entry_func(mod: IRModule, remove_preproc: bool = True) -> List["ArgInfo"]:
        """Extract a list of the argument information from the entry func of an IRModule.

        Parameters
        ----------
        mod : IRModule
            The IRModule to get argument information from.
        remove_preproc : bool
            Whether to remove the preprocessing blocks.

        Returns
        -------
        extracted : List[ArgInfo]
            An array of the argument information derived.
        """
        return _ffi_api.ArgInfoFromEntryFunc(mod, remove_preproc)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.TensorInfo")
class TensorInfo(ArgInfo):
    """Tensor argument information

    Parameters
    ----------
    dtype : DataType
        The data type of the tensor.
    shape : ShapeTuple
        The shape of the tensor.
    """

    dtype: DataType
    shape: ShapeTuple

    def __init__(
        self,
        dtype: DataType,
        shape: Union[ShapeTuple, List[int]],
    ) -> None:
        """Constructor

        Parameters
        ----------
        dtype : DataType
            The data type of the tensor.
        shape : ShapeTuple
            The shape of the tensor.
        """
        if isinstance(shape, ShapeTuple):
            shape_tuple = shape
        else:
            shape_tuple = ShapeTuple(shape)
        self.__init_handle_by_constructor__(
            _ffi_api.TensorInfo,  # type: ignore # pylint: disable=no-member
            dtype,
            shape_tuple,
        )
