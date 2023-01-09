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
"""Detect common patterns."""

from typing import Dict

from tvm.tir import PrimExpr
from . import _ffi_api


def detect_linear_equation(expr, var_list):
    """Match `expr = sum_{i=0}^{n-1} var[i] * coeff[i] + coeff[n]`

    Where coeff[i] and base are invariant of var[j] for all i and j.

    Parameters
    ----------
    expr : PrimExpr
        The expression to be matched.

    var_list : List[tvm.tir.Var]
        A list of variables.

    Returns
    -------
    coeff : List[PrimExpr]
        A list of co-efficients if the match is successful.
        An empty list if the match failed.
    """
    return _ffi_api.DetectLinearEquation(expr, var_list)


def detect_clip_bound(expr, var_list):
    """Detect if expression corresponds to clip bound of the vars

    Parameters
    ----------
    expr : PrimExpr
        The expression to be matched.

    var_list : List[tvm.tir.Var]
        A list of variables.

    Returns
    -------
    coeff : List[PrimExpr]
        `concat([min_value[i], max_value[i]] for i, v in enumerate(var_list))`
        An empty list if the match failed.
    """
    return _ffi_api.DetectClipBound(expr, var_list)


def detect_common_subexpr(expr: PrimExpr, threshold: int) -> Dict[PrimExpr, int]:
    """Detect common sub expression which shows up more than a threshold times

    Parameters
    ----------
    expr : PrimExpr
        The expression to be analyzed.

    threshold : int
        The threshold of repeat times that determines a common sub expression

    Returns
    -------
    cse_dict : Dict[PrimExpr, int]
        The detected common sub expression dict, with sub expression and repeat times
    """
    return _ffi_api.DetectCommonSubExpr(expr, threshold)
