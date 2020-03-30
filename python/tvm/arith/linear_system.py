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
"""Linear system data structures and solvers"""
import tvm._ffi
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("arith.LinearSystem")
class LinearSystem(Object):
    """Represent a linear system including variables, their ranges and
       the linear relations between them (either equations or inequalities)

    Parameters
    ----------
    variables : List[tvm.tir.Var]
        The variables in the system.
    ranges    : Map[tvm.tir.Var, tvm.ir.Range]
        The ranges of the variables.
    relations : List[tvm.ir.PrimExpr]
        The linear relations between the variables (either equations or inequalities)
    """
    def __init__(self, variables, ranges, relations):
        self.__init_handle_by_constructor__(
            _ffi_api.LinearSystem, variables, ranges, relations)


@tvm._ffi.register_object("arith.LinearSystemTransform")
class LinearSystemTransform(Object):
    """We can have different set of variables to represent the same linear system.
       For example, the following two systems are equivalent,
       {a + b = 0 | a >= 0, b >= 0} and
       {m - n = 0 | m >= 0, n <= 0}
       This data structure represents the transformation
       between two equivalent linear systems.
       In the above example,
       src        : {a + b = 0 | a >= 0, b >= 0}
       dst        : {m - n = 0 | m >= 0, n <= 0}
       src_to_dst : {a -> m, b -> -n}
       dst_to_src : {m -> a, n -> -b}

    Parameters
    ----------
    src : arith.LinearSystem
        source linear system, e.g., {a + b = 0 | a >= 0, b >= 0}
    dst : arith.LinearSystem
        linear system equivalent to the source, e.g., {m - n = 0 | m >= 0, n <= 0}
    src_to_dst : Map[tvm.tir.Var, tvm.ir.PrimExpr]
        mapping from variables in the src to the variables in the dst,
                e.g., {a -> m, b -> -n}
    dst_to_src : Map[tvm.tir.Var, tvm.ir.PrimExpr]
        mapping from variables in the dst to the variables in the src,
        e.g., {m -> a, n -> -b}
    """
    def __init__(self, src, dst, src_to_dst, dst_to_src):
        self.__init_handle_by_constructor__(
            _ffi_api.LinearSystemTransform, src, dst, src_to_dst, dst_to_src)


def solve_equations(equations, variables, ranges):
    """Solve linear equations.

    Parameters
    ----------
    equations: List[tvm.ir.PrimExpr] or LinearSystemTransform
        The linear relations between the variables (either equations or inequalities)
    variables : List[tvm.tir.Var]
        The variables in the system.
    ranges    : Map[tvm.tir.Var, tvm.ir.Range]
        The ranges of the variables.

    Returns
    -------
    linear_system_transform : LinearSystemTransform
        A new linear system, with less variables (if the problem is NOT of full rank),
        or no variable (if the problem is of full rank),
        or an empty linear system (if the problem is unsolvable).
        It also provides the ranges of the variables in the new system,
        as well as inequalities inferred from the problem.
        You can get the mapping from the original variables to the solution via
        linear_system_transform.src_to_dst.
    """
    if isinstance(equations, LinearSystemTransform):
        return _ffi_api.SolveEquations(equations)
    return _ffi_api.SolveEquations(variables, ranges, equations)
