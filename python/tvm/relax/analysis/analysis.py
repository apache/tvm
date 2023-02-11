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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains the set of passes for Relax, which exposes an interface for
configuring the passes and scripting them in Python.
"""

from typing import Dict
from enum import IntEnum

from tvm import tir
from tvm.relax.ty import Type
from tvm.relax.struct_info import StructInfo, FuncStructInfo
from tvm.relax.expr import Var, Expr, Call
from . import _ffi_api


def get_static_type(sinfo: StructInfo) -> Type:
    """Get the corresponding static type from a StructInfo.

    Parameters
    ----------
    sinfo : StructInfo
        The input struct info.

    Returns
    -------
    ret : Type
        The corresponding static type.
    """
    return _ffi_api.GetStaticType(sinfo)  # type: ignore


def erase_to_well_defined(
    sinfo: StructInfo,
    shape_var_map: Dict[tir.Var, tir.PrimExpr] = None,
    var_map: Dict[Var, Expr] = None,
) -> StructInfo:
    """Erase sinfo into a well defined form.

    This function removes the StructInfo's dependencies on shape and vars that
    are not defined in given maps.

    Parameters
    ----------
    sinfo : StructInfo
        The input struct info.

    shape_var_map : Dict[tir.Var, tir.PrimExpr]
        Specifies the defined shape vars and the values they should map to.

    var_map : Dict[Var, Expr]
        Specifies the defined vars and the values they should map to.

    Returns
    -------
    ret : StructInfo
        The corresponding erased struct info.
    """
    shape_var_map = {} if shape_var_map is None else shape_var_map
    var_map = {} if var_map is None else var_map

    return _ffi_api.EraseToWellDefined(sinfo, shape_var_map, var_map)  # type: ignore


class BaseCheckResult(IntEnum):
    """Return result of fine-grained base check.

    Note
    ----
    Base check comes with fine-grained fail levels.

    - FAIL_L0: The lhs and rhs have no intersection at all.
    - FAIL_L1: We get the failure by looking at static information.
    - FAIL_L2: We get the failure due to unknown symbolic variable relations.
    """

    FAIL_L0 = 0
    FAIL_L1 = 1
    FAIL_L2 = 2
    PASS = 3


def struct_info_base_check(base: StructInfo, derived: StructInfo) -> BaseCheckResult:
    """Run a base check to see if base subsumes derived.

    Parameters
    ----------
    base: StructInfo
        The base struct info.

    derived: StructInfo
        The derived struct info.

    Returns
    -------
    ret : StructInfo
        The derived return value struct info.
    """
    return _ffi_api.StructInfoBaseCheck(base, derived)  # type: ignore


def derive_call_ret_struct_info(
    func_sinfo: FuncStructInfo, call: Call, ctx: "tvm.relax.BlockBuilder"
) -> StructInfo:
    """Derive the call's ret value struct info from inputs.

    Parameters
    ----------
    func_sinfo: FuncStructInfo
        The call's function signature.

    call: Call
        The call expression

    ctx: tvm.relax.BlockBuilder
        The context block builder.

    Returns
    -------
    ret : StructInfo
        The derived return value struct info.

    Note
    ----
    This is an internal derivation function, call.op field is
    ignored in this case and the derivation only depends on func_sinfo.
    """
    return _ffi_api.DeriveCallRetStructInfo(func_sinfo, call, ctx)  # type: ignore


def struct_info_lca(lhs: StructInfo, rhs: StructInfo) -> StructInfo:
    """Unify the two struct info to their least common ancestor.

    Parameters
    ----------
    lhs: StructInfo
        The left operand.

    rhs: StructInfo
        The right operand.

    Returns
    -------
    ret : StructInfo
        The corresponding lca result.
    """
    return _ffi_api.StructInfoLCA(lhs, rhs)  # type: ignore


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ffi_api.post_order_visit(expr, fvisit)  # type: ignore


def has_reshape_pattern(func: tir.PrimFunc) -> bool:
    """Check if the given PrimFunc is essentially doing a reshape operation.
    The reshape operation also includes expand_dims, squeeze, flatten, etc.

    Here the allowed reshape pattern is: for example, assume the operation is
    `B[l_0, l_1, ..., l_b] = A[r_0, r_1, ..., r_a]`, we check if we can prove
    that the flattened index of l_0, ..., l_b under buffer B equals to the
    flattened index of r_0, ..., r_a under buffer A.

    Parameters
    ----------
    func : tir.PrimFunc
        The function to be examined.

    Returns
    -------
    ret : bool
        A boolean indicating if the given PrimFunc is doing a reshape.

    Notes
    -----
    According to the description above, the returned result can only be
    false-negative and cannot be false-positive, since whenever we cannot
    prove the equality, we return false. This property guarantees the safety
    of this function.
    """
    return _ffi_api.has_reshape_pattern(func)  # type: ignore
