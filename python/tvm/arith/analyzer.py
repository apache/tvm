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

import tvm_ffi

from tvm import ir, tirx
from tvm.arith import IntSet
from tvm.runtime import Object

from . import _ffi_api


class ProofStrength(enum.IntEnum):
    """Proof strength of the analysis"""

    DEFAULT = 0
    SYMBOLIC_BOUND = 1


class CompareResult(enum.IntEnum):
    """Result of a transitive comparison.

    Values must match the C++ ``arith::CompareResult`` enum.
    """

    INCONSISTENT = 0
    EQ = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5
    NE = 6
    UNKNOWN = 7


class Extension(enum.Flag):
    """Extensions enabled for RewriteSimplifier

    Values should match `RewriteSimplifier::Extensions`
    """

    NoExtensions = 0
    TransitivelyProveInequalities = 1 << 0
    ConvertBooleanToAndOfOrs = 1 << 1
    ApplyConstraintsToBooleanBranches = 1 << 2
    ComparisonOfProductAndSum = 1 << 3


@tvm_ffi.register_object("arith.ModularSet")
class ModularSet(Object):
    """Represent range of (coeff * x + base) for x in Z"""

    def __init__(self, coeff, base):
        self.__init_handle_by_constructor__(_ffi_api.ModularSet, coeff, base)


@tvm_ffi.register_object("arith.ConstIntBound")
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


@tvm_ffi.register_object("arith.Analyzer")
class Analyzer(Object):
    """Integer arithmetic analyzer

    This is a stateful analyzer class that can be used to perform
    various symbolic integer analysis. The same analyzer instance can
    be passed to FFI APIs to share accumulated facts across calls.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Analyzer)

    def const_int_bound(self, expr: tirx.PrimExpr) -> ConstIntBound:
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
        return _ffi_api.AnalyzerConstIntBound(self, expr)

    def const_int_bound_is_bound(self, var: tirx.Var) -> bool:
        """Check if a variable is bound to a range.

        Parameters
        ----------
        var : tvm.tirx.Var
            The variable.

        Returns
        -------
        result : bool
            Whether the variable is bound to a range.
        """
        return _ffi_api.AnalyzerConstIntBoundIsBound(self, var)

    def modular_set(self, expr: tirx.PrimExpr) -> ModularSet:
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
        return _ffi_api.AnalyzerModularSet(self, expr)

    def simplify(self, expr: tirx.PrimExpr, steps: int = 2) -> tirx.PrimExpr:
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
        return _ffi_api.AnalyzerSimplify(self, expr, steps)

    def clone(self) -> "Analyzer":
        """Return a deep copy of this analyzer with independent state.

        The returned analyzer carries the same accumulated facts (variable
        bounds, modular sets, bindings, integer-set domains, literal
        constraints and transitive comparisons) as this one, but owns its own
        state: binding or simplifying on either analyzer afterwards does not
        affect the other. Unlike copying the handle, this is a true deep copy.

        Do not call this while a constraint scope is active on this analyzer.

        Returns
        -------
        result : Analyzer
            A new analyzer holding an independent copy of the facts.
        """
        return _ffi_api.AnalyzerClone(self)

    def rewrite_simplify(self, expr: tirx.PrimExpr) -> tirx.PrimExpr:
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
        return _ffi_api.AnalyzerRewriteSimplify(self, expr)

    @property
    def rewrite_simplify_stats(self):
        return _ffi_api.AnalyzerGetRewriteSimplifyStats(self)

    def reset_rewrite_simplify_stats(self):
        _ffi_api.AnalyzerResetRewriteSimplifyStats(self)

    def canonical_simplify(self, expr: tirx.PrimExpr) -> tirx.PrimExpr:
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
        return _ffi_api.AnalyzerCanonicalSimplify(self, expr)

    def int_set(self, expr: tirx.PrimExpr, dom_map: dict[tirx.Var, IntSet] | None = None) -> IntSet:
        """Compute a symbolic IntSet that covers expr for all values in dom_map.

        Parameters
        ----------
        expr : PrimExpr
            The expression.

        dom_map : Optional[Dict[tvm.tirx.Var, tvm.arith.IntSet]]
            The domain for variables to be relaxed.  When omitted, the analyzer
            uses the domains of the variables already bound to it.

        Returns
        -------
        result : IntSet
            The result.
        """
        return _ffi_api.AnalyzerIntSet(self, expr, dom_map)

    def can_prove(
        self, expr: tirx.PrimExpr, strength: ProofStrength = ProofStrength.DEFAULT
    ) -> bool:
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
        return _ffi_api.AnalyzerCanProve(self, expr, strength)

    def set_maximum_rewrite_steps(self, maximum: int) -> None:
        """Set the maximum allowed number of rewrite-simplify steps.

        When a positive limit is set, the simplifier raises an exception once
        it exceeds that number of rewrite steps.  This is useful for guarding
        against performance regressions in tests.

        Parameters
        ----------
        maximum : int
            The maximum number of rewrite steps, or a non-positive value to
            allow an unlimited number of steps.
        """
        _ffi_api.AnalyzerSetMaximumRewriteSteps(self, maximum)

    def bind(
        self,
        var: tirx.Var,
        expr: tirx.PrimExpr | ir.Range,
        allow_override: bool = False,
    ) -> None:
        """Bind a variable to the expression.

        Parameters
        ----------
        var : tvm.tirx.Var
            The variable.

        expr : Union[tirx.PrimExpr, ir.Range]
            The expression or the range to bind to.

        allow_override : bool
            Whether to allow overriding an existing binding for the variable.
        """
        return _ffi_api.AnalyzerBind(self, var, expr, allow_override)

    def constraint_scope(self, constraint: tirx.PrimExpr) -> ConstraintScope:
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
          with analyzer.constraint_scope(x % 3 == 0):
              # constraint in effect
              assert analyzer.modular_set(x).coeff == 3
          # constraint no longer in effect
          assert analyzer.modular_set(x).coeff != 3
        """

        def _fenter():
            return _ffi_api.AnalyzerEnterConstraintContext(self, constraint)

        return ConstraintScope(_fenter)

    def update(
        self, var: tirx.Var, info: ConstIntBound | ModularSet | IntSet, override: bool = False
    ) -> None:
        """Update information about var.

        Parameters
        ----------
        var : tvm.tirx.Var
            The variable.

        info : Union[ConstIntBound, ModularSet, IntSet]
            Related information.  A ``ConstIntBound`` updates the constant
            integer bound, a ``ModularSet`` updates the modular set, and an
            ``IntSet`` updates the integer-set domain of ``var``.

        override : bool
            Whether allow override.
        """
        if isinstance(info, ConstIntBound):
            _ffi_api.AnalyzerConstIntBoundUpdate(self, var, info, override)
        elif isinstance(info, ModularSet):
            _ffi_api.AnalyzerModularSetUpdate(self, var, info, override)
        elif isinstance(info, IntSet):
            _ffi_api.AnalyzerIntSetUpdate(self, var, info, override)
        else:
            raise TypeError(f"Do not know how to handle type {type(info)}")

    def can_prove_equal(self, lhs: tirx.PrimExpr, rhs: tirx.PrimExpr) -> bool:
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
        return _ffi_api.AnalyzerCanProveEqual(self, lhs, rhs)

    def try_compare(
        self, lhs: tirx.PrimExpr, rhs: tirx.PrimExpr, propagate_inequalities: bool = True
    ) -> CompareResult:
        """Compare lhs and rhs using previously provided known comparisons.

        Parameters
        ----------
        lhs : PrimExpr
            The left-hand side of the comparison.

        rhs : PrimExpr
            The right-hand side of the comparison.

        propagate_inequalities : bool
            If true, attempt to find a sequence of transitive inequalities that
            allow lhs and rhs to be compared.

        Returns
        -------
        result : CompareResult
            The most specific result that can be proven about the comparison.
            Returns ``CompareResult.UNKNOWN`` when nothing can be proven.
        """
        return CompareResult(_ffi_api.AnalyzerTryCompare(self, lhs, rhs, propagate_inequalities))

    @property
    def enabled_extensions(self) -> Extension:
        """Return the currently enabled extensions"""
        value = _ffi_api.AnalyzerGetEnabledExtensions(self)
        return Extension(value)

    @enabled_extensions.setter
    def enabled_extensions(self, flags: int | Extension):
        """Enable extensions for the analyzer

        Parameters
        ----------
        flags: Union[int,Extension]

            The extensions to enable.
        """
        flags = Extension(flags).value
        _ffi_api.AnalyzerSetEnabledExtensions(self, flags)
