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
""" Iterator (quasi)affine mapping patterns."""
import tvm._ffi
from tvm.runtime import Object
from tvm.ir import PrimExpr
from . import _ffi_api


class IterMapExpr(PrimExpr):
    """Base class of all IterMap expressions."""


@tvm._ffi.register_object("arith.IterMark")
class IterMark(Object):
    """Mark the source as an iterator in [0, extent).

    Parameters
    ----------
    source : PrimExpr.
        The source expression.

    extent : PrimExpr
        The extent of the iterator.
    """

    def __init__(self, source, extent):
        self.__init_handle_by_constructor__(_ffi_api.IterMark, source, extent)


@tvm._ffi.register_object("arith.IterSplitExpr")
class IterSplitExpr(IterMapExpr):
    """Split of an iterator.

    result = floormod(floordiv(source, lower_factor), extent) * scale

    Parameters
    ----------
    source : IterMark
        The source marked iterator.

    lower_factor : PrimExpr
        The lower factor to split the domain.

    extent : PrimExpr
        The extent of the split.

    scale : PrimExpr
        Additional scale to the split.
    """

    def __init__(self, source, lower_factor, extent, scale):
        self.__init_handle_by_constructor__(
            _ffi_api.IterSplitExpr, source, lower_factor, extent, scale
        )


@tvm._ffi.register_object("arith.IterSumExpr")
class IterSumExpr(IterMapExpr):
    """Fuse multiple iterators by summing them with scaling.

    result = sum(args) + base

    Parameters
    ----------
    args : List[IterSplitExpr]
        The input to the sum expression.

    base : PrimExpr
        The base offset.
    """

    def __init__(self, args, base):
        self.__init_handle_by_constructor__(_ffi_api.IterSumExpr, args, base)


def detect_iter_map(indices, input_iters):
    """Detect if indices can be written mapped iters from input_iters.

    Parameters
    ----------
    indices : List[PrimExpr]
        The input indices.

    input_iters : Map[Var, Range]
        The domain of each input iterators.

    Returns
    -------
    results : List[IterSumExpr]
        The iter map matching result.
        Empty array if no match can be found.
    """
    return _ffi_api.DetectIterMap(indices, input_iters)
