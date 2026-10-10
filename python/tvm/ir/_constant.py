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
"""Construction of primitive IR constants."""

from tvm.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api


def _scalar_type_inference(value):
    if hasattr(value, "dtype"):
        return str(value.dtype)
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, float):
        # We intentionally prefer convert the float to float32 since it's more common in DL.
        if -3.40282347e38 <= value <= 3.40282347e38:
            return "float32"
        else:
            return "float64"
    elif isinstance(value, int):
        # We intentionally prefer convert the python int to int32 since it's more common in DL.
        if -2147483648 <= value <= 2147483647:
            return "int32"
        else:
            return "int64"
    else:
        raise NotImplementedError(f"Cannot automatically inference the type. value={value}")


def const(value, dtype=None, loc: Location = UNKNOWN_LOC):
    """construct a constant

    Parameters
    ----------
    value : number
        The content of the constant number.

    dtype : str or None, optional
        The data type.

    loc : Location, optional
        The location of the constant value in the source.

    Returns
    -------
    const_val: tvm.Expr
        The result expression.
    """
    if dtype is None:
        dtype = _scalar_type_inference(value)
    return _ffi_api._const(value, dtype, loc)
