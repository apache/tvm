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


@tvm._ffi.register_object("arith.IntGroupBounds")
class IntGroupBounds(Object):
    """Represent integer grouped bounds which are classified into
       lower bounds (include), upper bounds (include) and equalities.

    Parameters
    ----------
    coef : tvm.ir.PrimExpr
        The coefficient. Must be integer type.
        coef * var >= lower
        coef * var == equal
        coef * var >= upper
    lower : List[tvm.ir.PrimExpr]
        the lower bounds (include)
    equal : List[tvm.ir.PrimExpr]
        equalities
    upper : List[tvm.ir.PrimExpr]
        the upper bounds (include)
    """

    def __init__(self, coef, lower, equal, upper):
        self.__init_handle_by_constructor__(_ffi_api.IntGroupBounds, coef, lower, equal, upper)

    @staticmethod
    def from_range(rng):
        """Construct a IntGroupedBounds by Range.

        Parameters
        ----------
        rng : tvm.ir.Range


        Returns
        -------
        ret : Range
            The constructed range.
        """
        return _ffi_api.IntGroupBounds_from_range(rng)

    def find_best_range(self):
        """Return the best range from the grouped bounds.
        None if (-inf, +inf).
        """
        return _ffi_api.IntGroupBounds_FindBestRange(self)


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
        self.__init_handle_by_constructor__(_ffi_api.IntConstraints, variables, ranges, relations)


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
            _ffi_api.IntConstraintsTransform, src, dst, src_to_dst, dst_to_src
        )


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


def solve_linear_inequalities(equations, variables=None, ranges=None, deskew_range=False):
    """Solve linear inequalities.

    Parameters
    ----------
    equations   : List[tvm.ir.PrimExpr] or IntConstraints
        The inequalities of the variables
    variables   : Optional[List[tvm.tir.Var]]
        The variables in the system.
    ranges      : Optional[Map[tvm.tir.Var, tvm.ir.Range]]
        The ranges of the variables.
    deskew_range: Optional[bool]
        Whether deskew the result ranges to be started from zero.
        Default false.

    Returns
    -------
    ret_ranges: IntConstraints or IntConstraintsTransform
        The result ranges for each variables.
        Constrains that cannot be transformed to Range will be stored in IntConstraints.relations.
        If deskew_range is set (=True), the result ranges will be deskewed to be started from zero.
        New variables are created accordingly therefore IntConstraintsTransform is returned.
    """
    solver = (
        _ffi_api.SolveInequalitiesDeskewRange if deskew_range else _ffi_api.SolveInequalitiesToRange
    )
    if isinstance(equations, IntConstraints):
        assert variables is None
        assert ranges is None
        return solver(equations)
    return solver(variables, ranges, equations)
