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
# pylint: disable=invalid-name
"""Function definitions."""
from typing import Union, Dict

from enum import IntEnum
import tvm.runtime
from tvm.runtime.object import Object
from .expr import RelayExpr
from .attrs import DictAttrs
from . import _ffi_api


class CallingConv(IntEnum):
    """Possible kinds of calling conventions."""

    DEFAULT = 0
    C_PACKED_FUNC = 1
    DEVICE_KERNEL_LAUNCH = 2


class BaseFunc(RelayExpr):
    """Base class of all functions."""

    @property
    def attrs(self):
        """Return the attrs member of the function."""
        return _ffi_api.BaseFunc_Attrs(self)

    def with_attr(self, attr_key_or_dict, attr_value=None) -> "BaseFunc":
        """Create a new copy of the function and update the attribute.

        Parameters
        ----------
        attr_key_or_dict : Union[str, dict]
            The attribute key to use or a dict containing multiple key value pairs.

        attr_value : Object
            The new attribute value.

        Returns
        -------
        func : BaseFunc
            A new copy of the function
        """
        # make sure we first copy so that we can safely do copy on write
        # for multiple updates.
        res = _ffi_api.BaseFuncCopy(self)

        if isinstance(attr_key_or_dict, dict):
            for key, val in attr_key_or_dict.items():
                res = _ffi_api.BaseFuncWithAttr(res._move(), key, tvm.runtime.convert(val))
            return res

        return _ffi_api.BaseFuncWithAttr(
            res._move(), attr_key_or_dict, tvm.runtime.convert(attr_value)
        )

    def with_attrs(self, attr_map: Union[DictAttrs, Dict[str, Object]]) -> "BaseFunc":
        """Copy the IRModule and add the given attribute map to it.
        Parameters
        ----------
        attr_map: Union[DictAttrs, Dict[str, Object]]
            The attribute map
        Returns
        -------
        func : BaseFunc
            A new copy of the function
        """
        if isinstance(attr_map, tvm.ir.DictAttrs):
            attr_map = attr_map._dict()

        return _ffi_api.BaseFuncWithAttrs(self, attr_map)

    def without_attr(self, attr_key: str) -> "BaseFunc":
        """Create a new copy of the function with an attribute without provided key.

        Parameters
        ----------
        attr_key : str
            The attribute key to delete from the attrubte pairs.


        Returns
        -------
        func : BaseFunc
            A new copy of the function
        """
        return _ffi_api.BaseFuncWithoutAttr(self, attr_key)
