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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""Algebraic data types in Relay."""
from .base import RelayNode, register_relay_node, NodeBase
from . import _make
from .ty import Type
from .expr import Expr, Call


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
class Constructor(Expr):
    """Relay ADT constructor."""

    def __init__(self, name_hint, inputs, belong_to):
        """Defines an ADT constructor.

        Parameters
        ----------
        name_hint : str
            Name of constructor (only a hint).
        inputs : List[Type]
            Input types.
        belong_to : tvm.relay.GlobalTypeVar
            Denotes which ADT the constructor belongs to.

        Returns
        -------
        con: Constructor
            A constructor.
        """
        self.__init_handle_by_constructor__(_make.Constructor, name_hint, inputs, belong_to)

    def __call__(self, *args):
        """Call the constructor.

        Parameters
        ----------
        args: List[relay.Expr]
            The arguments to the constructor.

        Returns
        -------
        call: relay.Call
            A call to the constructor.
        """
        return Call(self, args)


@register_relay_node
class TypeData(Type):
    """Stores the definition for an Algebraic Data Type (ADT) in Relay.

    Note that ADT definitions are treated as type-level functions because
    the type parameters need to be given for an instance of the ADT. Thus,
    any global type var that is an ADT header needs to be wrapped in a
    type call that passes in the type params.
    """

    def __init__(self, header, type_vars, constructors):
        """Defines a TypeData object.

        Parameters
        ----------
        header: tvm.relay.GlobalTypeVar
            The name of the ADT.
            ADTs with the same constructors but different names are
            treated as different types.
        type_vars: List[TypeVar]
            Type variables that appear in constructors.
        constructors: List[tvm.relay.Constructor]
            The constructors for the ADT.

        Returns
        -------
        type_data: TypeData
            The adt declaration.
        """
        self.__init_handle_by_constructor__(_make.TypeData, header, type_vars, constructors)


@register_relay_node
class Clause(NodeBase):
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
class Match(Expr):
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
