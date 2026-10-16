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

from typing import TYPE_CHECKING

import numpy as np

from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api

if TYPE_CHECKING:
    from .expr import Expr


def const(value, ty=None, loc: Location = UNKNOWN_LOC) -> "Expr":
    """Construct a constant with an explicit or inferred primitive type.

    When ``ty`` is omitted or None, boolean, integer and floating-point
    scalars use ``bool``, ``int32`` and ``float32``, respectively. The same
    defaults apply to NumPy scalars, regardless of their original width.
    Values outside the inferred type's range raise an error; the type never
    widens automatically. NaN and infinity are accepted floating-point values.
    Specify ``ty`` explicitly to select another width or a vector type.

    Parameters
    ----------
    value : number
        The content of the constant number.

    ty : str, DataType or tvm_tirx.ir.PrimType, optional
        The primitive type. None uses the scalar-kind defaults above.

    loc : Location, optional
        The location of the constant value in the source.

    Returns
    -------
    const_val : tvm_tirx.ir.Expr
        The result expression.

    Examples
    --------
    .. code-block:: python

        ir.const(True)                 # bool
        ir.const(1)                    # int32
        ir.const(1.0)                  # float32
        ir.const(1 << 31)              # raises ValueError: exceeds int32 range
        ir.const(1 << 31, ty="int64")  # explicit wider type
    """
    if ty is None:
        if isinstance(value, bool | np.bool_):
            ty = "bool"
        elif isinstance(value, int | np.integer):
            ty = "int32"
        elif isinstance(value, float | np.floating):
            ty = "float32"
        else:
            raise TypeError(f"Cannot infer a constant type from {type(value).__name__}")
    return _ffi_api._const(value, ty, loc)
