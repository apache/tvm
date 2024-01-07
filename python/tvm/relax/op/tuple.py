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
"""Tuple operators."""
from typing import Union

import tvm
from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr, PrimValue


def tuple_get_item(tuple_expr: Expr, index: Union[int, PrimExpr, Expr]) -> Expr:
    """Perform tuple access

    Use of this method is recommended, rather than constructing a
    `relax.TupleGetItem` directly.

    1. May resolve to the tuple's contents, avoiding the intermediate
       `TupleGetItem`.

    2. Handles access of a tuple at a dynamic index, where
       `TupleGetItem` requires a statically-known index.

    Parameters
    ----------
    tuple_expr: Expr

        The tuple to be accessed.  The tuple is not required to be an
        in-line `relax.Tuple`, but must have `TupleStructInfo`

    index: Union[int, PrimExpr, Expr]

        The index at which the tuple is accessed.  The index may be
        static or dynamic.

    Returns
    -------
    Expr

        An expression representing the item in the tuple.
    """

    if not isinstance(index, Expr):
        index = PrimValue(index)

    return _ffi_api.tuple_get_item(tuple_expr, index)  # type: ignore


def tuple_get_item_dyn(tuple_expr: Expr, index: Union[int, PrimExpr, Expr]) -> Expr:
    """Explicitly generate a call to tuple_get_item_dyn

    This method is not recommended for general use, and is provided to
    ensure round-trip consistency in TVMScript.  In most cases, the
    `tuple_get_item` method should be used, which will delegate to the
    dynamic builtin for cases where the index is dynamic.

    Parameters
    ----------
    tuple_expr: Expr

        The tuple to be accessed.  The tuple is not required to be an
        in-line `relax.Tuple`, but must have `TupleStructInfo`

    index: Union[int, PrimExpr, Expr]

        The index at which the tuple is accessed.  The index may be
        static or dynamic.

    Returns
    -------
    Expr

        An expression representing the item in the tuple.

    """
    if not isinstance(index, Expr):
        index = PrimValue(index)
    return tvm.relax.Call(tvm.ir.Op.get("relax.tuple_get_item_dyn"), [tuple_expr, index])
