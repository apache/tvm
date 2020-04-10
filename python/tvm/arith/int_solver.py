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
"""integer constraints data structures and solvers"""
import tvm._ffi
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("arith.IntConstraints")
class IntConstraints(Object):
    """Represent a set of integer constraints including variables, their ranges and
       the relations between them (either equations or inequalities)

    Parameters
    ----------
    variables : List[tvm.tir.Var]
        The variables in the constraints. Must be integers
    ranges    : Map[tvm.tir.Var, tvm.ir.Range]
        The ranges of the variables.
    relations : List[tvm.ir.PrimExpr]
        The relations between the variables (either equations or inequalities)
    """
    def __init__(self, variables, ranges, relations):
        self.__init_handle_by_constructor__(
            _ffi_api.IntConstraints, variables, ranges, relations)


@tvm._ffi.register_object("arith.IntConstraintsTransform")
class IntConstraintsTransform(Object):
    """We can have different set of variables to represent the same integer constraints.
       For example, the following two constrains are equivalent,
       {a + b = 0 | a >= 0, b >= 0} and
       {m - n = 0 | m >= 0, n <= 0}
       This data structure represents the transformation
       between two equivalent integer constraints.
       In the above example,
       src        : {a + b = 0 | a >= 0, b >= 0}
       dst        : {m - n = 0 | m >= 0, n <= 0}
       src_to_dst : {a -> m, b -> -n}
       dst_to_src : {m -> a, n -> -b}

    Parameters
    ----------
    src : arith.IntConstraints
        source integer constraints, e.g., {a + b = 0 | a >= 0, b >= 0}
    dst : arith.IntConstraints
        integer constraints equivalent to the source, e.g., {m - n = 0 | m >= 0, n <= 0}
    src_to_dst : Map[tvm.tir.Var, tvm.ir.PrimExpr]
        mapping from variables in the src to the variables in the dst,
                e.g., {a -> m, b -> -n}
    dst_to_src : Map[tvm.tir.Var, tvm.ir.PrimExpr]
        mapping from variables in the dst to the variables in the src,
        e.g., {m -> a, n -> -b}
    """
    def __init__(self, src, dst, src_to_dst, dst_to_src):
        self.__init_handle_by_constructor__(
            _ffi_api.IntConstraintsTransform, src, dst, src_to_dst, dst_to_src)


def solve_linear_equations(equations, variables=None, ranges=None):
    """Solve linear equations.

    Parameters
    ----------
    equations: List[tvm.ir.PrimExpr] or IntConstraints
        The equations of the variables
    variables : Optional[List[tvm.tir.Var]]
        The variables in the system.
    ranges    : Optional[Map[tvm.tir.Var, tvm.ir.Range]]
        The ranges of the variables.

    Returns
    -------
    int_constraints_transform : IntConstraintsTransform
        New integer constraints, with less variables (if the problem is NOT of full rank),
        or no variable (if the problem is of full rank),
        or an empty integer constraints (if the problem is unsolvable).
        It also provides the ranges of the variables in the new system,
        as well as inequalities inferred from the problem.
        You can get the mapping from the original variables to the solution via
        int_constraints_transform.src_to_dst.
    """
    if isinstance(equations, IntConstraints):
        return _ffi_api.SolveLinearEquations(equations)
    return _ffi_api.SolveLinearEquations(variables, ranges, equations)
