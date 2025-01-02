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
# pylint: disable=invalid-name
"""Arithmetic data structure and utility"""
import enum
from typing import Union

import tvm._ffi
from tvm import tir, ir
from tvm.runtime import Object

from . import _ffi_api


class ProofStrength(enum.IntEnum):
    """Proof strength of the analysis"""

    DEFAULT = 0
    SYMBOLIC_BOUND = 1


class Extension(enum.Flag):
    """Extensions enabled for RewriteSimplifier

    Values should match `RewriteSimplifier::Extensions`
    """

    NoExtensions = 0
    TransitivelyProveInequalities = 1 << 0
    ConvertBooleanToAndOfOrs = 1 << 1
    ApplyConstraintsToBooleanBranches = 1 << 2
    ComparisonOfProductAndSum = 1 << 3


@tvm._ffi.register_object("arith.ModularSet")
class ModularSet(Object):
    """Represent range of (coeff * x + base) for x in Z"""

    def __init__(self, coeff, base):
        self.__init_handle_by_constructor__(_ffi_api.ModularSet, coeff, base)


@tvm._ffi.register_object("arith.ConstIntBound")
class ConstIntBound(Object):
    """Represent constant integer bound

    Parameters
    ----------
    min_value : int
        The minimum value of the bound.

    max_value : int
        The maximum value of the bound.
    """

    POS_INF = (1 << 63) - 1
    NEG_INF = -POS_INF

    def __init__(self, min_value, max_value):
        self.__init_handle_by_constructor__(_ffi_api.ConstIntBound, min_value, max_value)


class ConstraintScope:
    """Constraint scope.

    Parameters
    ----------
    fenter : function
        A function that will be called to create an enter context.

    Note
    ----
    Do not create object directly, use Analyzer.constraint_scope
    """

    def __init__(self, fenter):
        self._fenter = fenter
        self._fexit = None

    def __enter__(self):
        self._fexit = self._fenter()

    def __exit__(self, ptype, value, trace):
        self._fexit()


class Analyzer:
    """Integer arithmetic analyzer

    This is a stateful analyzer class that can
    be used to perform various symbolic integer analysis.
    """

    def __init__(self):
        _mod = _ffi_api.CreateAnalyzer()
        self._const_int_bound = _mod("const_int_bound")
        self._const_int_bound_update = _mod("const_int_bound_update")
        self._bind = _mod("bind")
        self._modular_set = _mod("modular_set")
        self._simplify = _mod("Simplify")
        self._rewrite_simplify = _mod("rewrite_simplify")
        self._get_rewrite_simplify_stats = _mod("get_rewrite_simplify_stats")
        self._reset_rewrite_simplify_stats = _mod("reset_rewrite_simplify_stats")
        self._canonical_simplify = _mod("canonical_simplify")
        self._int_set = _mod("int_set")
        self._enter_constraint_context = _mod("enter_constraint_context")
        self._can_prove_equal = _mod("can_prove_equal")
        self._can_prove = _mod("can_prove")
        self._get_enabled_extensions = _mod("get_enabled_extensions")
        self._set_enabled_extensions = _mod("set_enabled_extensions")

    def const_int_bound(self, expr):
        """Find constant integer bound for expr.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        Returns
        -------
        bound : ConstIntBound
            The result bound
        """
        return self._const_int_bound(expr)

    def modular_set(self, expr):
        """Find a modular set that expr belongs to.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        Returns
        -------
        result : ModularSet
            The result.
        """
        return self._modular_set(expr)

    def simplify(self, expr, steps=2):
        """Simplify expression via both rewrite and canonicalization.

        Parameters
        ----------
        expr : PrimExpr
            The expression.
        steps : The simplification runs in the order of
                rewrite_simplify (step 1) -> canonical_simplify (step 2) ->
                rewrite_simplify (step 3) -> canonical_simplify (step 4) -> ...
                param steps controls how many steps to run.
                Default is 2, i.e., rewrite_simplify + canonical_simplify.

        Returns
        -------
        result : Expr
            The result.
        """
        return self._simplify(expr, steps)

    def rewrite_simplify(self, expr):
        """Simplify expression via rewriting rules.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return self._rewrite_simplify(expr)

    @property
    def rewrite_simplify_stats(self):
        return self._get_rewrite_simplify_stats()

    def reset_rewrite_simplify_stats(self):
        self._reset_rewrite_simplify_stats()

    def canonical_simplify(self, expr):
        """Simplify expression via canonicalization.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return self._canonical_simplify(expr)

    def int_set(self, expr, dom_map):
        """Compute a symbolic IntSet that covers expr for all values in dom_map.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        dom_map : Dict[tvm.tir.Var, tvm.arith.IntSet]
            The domain for variables to be relaxed.

        Returns
        -------
        result : IntSet
            The result.
        """
        return self._int_set(expr, dom_map)

    def can_prove(self, expr, strength=ProofStrength.DEFAULT):
        """Check whether we can prove expr to be true.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        strength: ProofStrength
            The proof strength

        Returns
        -------
        result : Expr
            The result.
        """
        return self._can_prove(expr, strength)

    def bind(self, var: tir.Var, expr: Union[tir.PrimExpr, ir.Range]):
        """Bind a variable to the expression.

        Parameters
        ----------
        var : tvm.tir.Var
            The variable.

        expr : Union[tir.PrimExpr, ir.Range]
            The expression or the range to bind to.
        """
        return self._bind(var, expr)

    def constraint_scope(self, constraint):
        """Create a constraint scope.

        Parameters
        ----------
        constraint : PrimExpr
            The constraint expression.

        returns
        -------
        scope : ConstraintScope
            The constraint scope

        Examples
        --------
        .. code-block:: python

          x = te.var("x")
          analyzer = tvm.arith.Analyzer()
          with analzyer.constraint_scope(x % 3 == 0):
              # constraint in effect
              assert analyzer.modular_set(x).coeff == 3
          # constraint no longer in effect
          assert analyzer.modular_set(x).coeff != 3
        """

        def _fenter():
            return self._enter_constraint_context(constraint)

        return ConstraintScope(_fenter)

    def update(self, var, info, override=False):
        """Update infomation about var

        Parameters
        ----------
        var : tvm.tir.Var
            The variable.

        info : tvm.Object
            Related information.

        override : bool
            Whether allow override.
        """
        if isinstance(info, ConstIntBound):
            self._const_int_bound_update(var, info, override)
        else:
            raise TypeError("Do not know how to handle type {}".format(type(info)))

    def can_prove_equal(self, lhs: "PrimExpr", rhs: "PrimExpr"):
        """Whether we can prove that lhs == rhs

        Parameters
        ----------
        lhs: PrimExpr
            The left-hand side of the comparison

        rhs: PrimExpr
            The right-hand side of the comparison

        Returns
        -------
        result: bool
            Whether we can prove that lhs == rhs
        """
        return self._can_prove_equal(lhs, rhs)

    @property
    def enabled_extensions(self) -> Extension:
        """Return the currently enabled extensions"""
        value = self._get_enabled_extensions()
        return Extension(value)

    @enabled_extensions.setter
    def enabled_extensions(self, flags: Union[int, Extension]):
        """Enable extensions for the analyzer

        Parameters
        ----------
        flags: Union[int,Extension]

            The extensions to enable.
        """
        flags = Extension(flags).value
        self._set_enabled_extensions(flags)
