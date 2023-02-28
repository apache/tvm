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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name

"""IRBuilder for distributed Relax dialect"""
from typing import Union, List, Tuple, Optional

from tvm.ir import PrimExpr
from tvm.relax.expr import Expr, ShapeExpr, Call, ExternFunc
from tvm.relax.expr import Tuple as RxTuple
from tvm.relax.distributed import DTensorStructInfo
from . import _ffi_api
from tvm.relax.utils import args_converter


@args_converter.auto
def call_tir(
    func: Union[str, Expr],
    args: Expr,
    out_sinfo: Union[DTensorStructInfo, List[DTensorStructInfo]],
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """Distributed version of call_tir

    Parameters:
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc or PrimFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[DTensorStructInfo, List[DTensorStructInfo]]
        The structure info of the call_tir output.
        It should be a single or a list of DTensorStructInfo. Each one denotes the
        structure info of a returned distributed tensor.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir_dist(func, args, out_sinfo, tir_vars)  # type: ignore
