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
"""Statement functor utilities for IR transformations"""
from . import _ffi_api


def ir_transform(stmt, preorder, postorder, only_enable=None):
    """Recursively visit and transform ir nodes in post DFS order.

    Parameters
    ----------
    stmt : tvm.tir.Stmt
        The input to be transformed.

    preorder: function
        The function called in before recursive mutation
        If preorder returns None, then the transform will proceed to recursive call.
        If preorder returns a not None tvm.tir.Stmt/Expr, the transformer will simply return it and
        won't do further recursion.

    postorder : function
        The function called after recursive mutation.

    only_enable : Optional[List[str]]
        List of types that we only enable.

    Returns
    -------
    result : tvm.tir.Stmt
        The result.
    """
    return _ffi_api.IRTransform(stmt, preorder, postorder, only_enable)  # type: ignore


def post_order_visit(stmt, fvisit):
    """Recursively visit the ir in post DFS order node, apply fvisit
       Each node is guaranteed to be visited only once.

    Parameters
    ----------
    fvisit: function
        The visitor function.
    """
    return _ffi_api.PostOrderVisit(stmt, fvisit)  # type: ignore


def substitute(node, vmap):
    """Substitute the var specified by vmap.

    Parameters
    ----------
    node: ObjectRef
        The input.

    vmap : Dict[Var, PrimExpr]
        The variable mapping.

    Returns
    -------
    result : tvm.tir.Stmt
        The result.
    """
    return _ffi_api.Substitute(node, vmap)  # type: ignore
