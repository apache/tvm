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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-import
"""Algebraic data types in Relay."""
from tvm.ir import Constructor, TypeData

from .base import RelayNode, register_relay_node, Object
from . import _make
from .ty import Type
from .expr import ExprWithOp, RelayExpr, Call


class Pattern(RelayNode):
    """Base type for pattern matching constructs."""


@register_relay_node
class PatternWildcard(Pattern):
    """Wildcard pattern in Relay: Matches any ADT and binds nothing."""

    def __init__(self):
        """Constructs a wildcard pattern.

        Parameters
        ----------
        None

        Returns
        -------
        wildcard: PatternWildcard
            a wildcard pattern.
        """
        self.__init_handle_by_constructor__(_make.PatternWildcard)


@register_relay_node
class PatternVar(Pattern):
    """Variable pattern in Relay: Matches anything and binds it to the variable."""

    def __init__(self, var):
        """Construct a variable pattern.

        Parameters
        ----------
        var: tvm.relay.Var

        Returns
        -------
        pv: PatternVar
            A variable pattern.
        """
        self.__init_handle_by_constructor__(_make.PatternVar, var)


@register_relay_node
class PatternConstructor(Pattern):
    """Constructor pattern in Relay: Matches an ADT of the given constructor, binds recursively."""

    def __init__(self, constructor, patterns=None):
        """Construct a constructor pattern.

        Parameters
        ----------
        constructor: Constructor
            The constructor.
        patterns: Optional[List[Pattern]]
            Optional subpatterns: for each field of the constructor,
            match to the given subpattern (treated as a variable pattern by default).

        Returns
        -------
        wildcard: PatternWildcard
            a wildcard pattern.
        """
        if patterns is None:
            patterns = []
        self.__init_handle_by_constructor__(_make.PatternConstructor, constructor, patterns)


@register_relay_node
class PatternTuple(Pattern):
    """Constructor pattern in Relay: Matches a tuple, binds recursively."""

    def __init__(self, patterns=None):
        """Construct a tuple pattern.

        Parameters
        ----------
        patterns: Optional[List[Pattern]]
            Optional subpatterns: for each field of the constructor,
            match to the given subpattern (treated as a variable pattern by default).

        Returns
        -------
        wildcard: PatternWildcard
            a wildcard pattern.
        """
        if patterns is None:
            patterns = []
        self.__init_handle_by_constructor__(_make.PatternTuple, patterns)


@register_relay_node
class Clause(Object):
    """Clause for pattern matching in Relay."""

    def __init__(self, lhs, rhs):
        """Construct a clause.

        Parameters
        ----------
        lhs: tvm.relay.Pattern
            Left-hand side of match clause.
        rhs: tvm.relay.Expr
            Right-hand side of match clause.

        Returns
        -------
        clause: Clause
            The Clause.
        """
        self.__init_handle_by_constructor__(_make.Clause, lhs, rhs)


@register_relay_node
class Match(ExprWithOp):
    """Pattern matching expression in Relay."""

    def __init__(self, data, clauses, complete=True):
        """Construct a Match.

        Parameters
        ----------
        data: tvm.relay.Expr
            The value being deconstructed and matched.

        clauses: List[tvm.relay.Clause]
            The pattern match clauses.

        complete: Optional[Bool]
            Should the match be complete (cover all cases)?
            If yes, the type checker will generate an error if there are any missing cases.

        Returns
        -------
        match: tvm.relay.Expr
            The match expression.
        """
        self.__init_handle_by_constructor__(_make.Match, data, clauses, complete)
