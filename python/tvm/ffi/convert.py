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
"""Conversion utilities to bring python objects into ffi values."""
from numbers import Number
from typing import Any
from . import core
from . import container


def convert(value: Any) -> Any:
    """Convert a python object to ffi values.

    Parameters
    ----------
    value : Any
        The python object to be converted.

    Returns
    -------
    ffi_obj : Any
        The converted TVM FFI object.
    """
    if isinstance(value, core.Object):
        return value
    elif isinstance(value, core.PyNativeObject):
        return value
    elif isinstance(value, (bool, Number)):
        return value
    elif isinstance(value, (list, tuple)):
        return container.Array(value)
    elif isinstance(value, dict):
        return container.Map(value)
    elif isinstance(value, str):
        return core.String(value)
    elif isinstance(value, (bytes, bytearray)):
        return core.Bytes(value)
    elif isinstance(value, core.ObjectGeneric):
        return value.asobject()
    elif callable(value):
        return core._convert_to_ffi_func(value)
    elif value is None:
        return None
    elif hasattr(value, "__dlpack__"):
        return core.from_dlpack(
            value,
            required_alignment=core.__dlpack_auto_import_required_alignment__,
        )
    elif isinstance(value, Exception):
        return core._convert_to_ffi_error(value)
    else:
        raise TypeError(f"don't know how to convert type {type(value)} to object")


core._set_func_convert_to_object(convert)
