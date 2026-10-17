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
"""Vector expression nodes shared by TVM IR dialects."""

import tvm_ffi

from ..expr import Expr, ExprWithOp
from ..location import UNKNOWN_LOC, Location
from . import _ffi_api as _prim_ffi_api


@tvm_ffi.register_object("prim.Ramp")
class Ramp(ExprWithOp):
    """Ramp node.

    Parameters
    ----------
    base : Expr
        The base expression.

    stride : Expr
        The stride of the ramp.

    lanes : Expr
        The lanes of the expression.

    loc : Location, optional
        The location of this expression in the source code.
    """

    base: Expr
    stride: Expr
    lanes: Expr

    def __init__(self, base: Expr, stride: Expr, lanes: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Ramp,
            base,
            stride,
            lanes,
            loc,  # type: ignore
        )


@tvm_ffi.register_object("prim.Broadcast")
class Broadcast(ExprWithOp):
    """Broadcast node.

    Parameters
    ----------
    value : Expr
        The value of the expression.

    lanes : Expr
        The lanes of the expression.

    loc : Location, optional
        The location of this expression in the source code.
    """

    value: Expr
    lanes: Expr

    def __init__(self, value: Expr, lanes: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_prim_ffi_api.Broadcast, value, lanes, loc)  # type: ignore


@tvm_ffi.register_object("prim.Shuffle")
class Shuffle(ExprWithOp):
    """Shuffle node.

    Parameters
    ----------
    vectors : List[Expr]
        The vectors

    indices : List[Expr]
        The indices

    loc : Location, optional
        The location of this expression in the source code.
    """

    vectors: list[Expr]
    indices: list[Expr]

    def __init__(
        self, vectors: list[Expr], indices: list[Expr], loc: Location = UNKNOWN_LOC
    ) -> None:
        self.__init_handle_by_constructor__(
            _prim_ffi_api.Shuffle,
            vectors,
            indices,
            loc,  # type: ignore
        )
