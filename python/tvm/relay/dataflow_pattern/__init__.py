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
"""The Relay Pattern Language and tooling."""
from tvm.relay import Expr
import tvm._ffi
from ...ir.base import Node
from ...ir import make_node
from ...runtime import Object
from ... import _ffi as tvm_ffi
from ..op import get
from . import _ffi as ffi


def register_df_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm._ffi.register_object(
            "relay.dataflow_pattern." + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)


class DFPattern(Node):
    """Base class of all Patterns.
    """

    def __call__(self, *args):
        return CallPattern(self, list(args))

    def __or__(self, other):
        return AltPattern(self, other)

    def __add__(self, other):
        return is_op("add")(self, other)

    def __sub__(self, other):
        return is_op("subtract")(self, other)

    def __mul__(self, other):
        return is_op("multiply")(self, other)

    def __truediv__(self, other):
        return is_op("divide")(self, other)

    def has_attr(self, attr_name: str, attr_value):
        """
        Add an attribute constraint to this pattern

        Parameters
        ----------
        attr_name: str
            The name of the attribute to match
        attr_value: Any
            The value of the attribute to match

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting AttrPattern
        """
        attrs = make_node("DictAttrs", **{attr_name: attr_value})
        return AttrPattern(self, attrs)

    def has_type(self, ttype):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        ttype: tvm.relay.Type
            The type to match

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting TypePattern
        """
        return has_type(ttype, self)

    def match(self, expr: Expr) -> bool:
        """
        Match this pattern to an expression

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression to match.

        Returns
        -------
        result: bool
            Whether or not the expression matches the pattern
        """
        return match(self, expr)

    def partition(self, expr: Expr) -> bool:
        """
        Parition the expression into functions defined by this pattern

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression to match.

        Returns
        -------
        result : tvm.relay.Expr
            The Expression with matched subgraphs replaced by function calls to that subgraph
        """
        return partition(self, expr)

    def dominates(self, parent, path=None):
        """
        Create a dominator for this pattern

        Parameters
        ----------
        parent: tvm.relay.dataflow_pattern.DFPattern
            The parent pattern this pattern dominates.
        path: tvm.relay.dataflow_pattern.DFPattern
            The fuzzy path pattern.

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting DominatorPattern
        """
        if path is None:
            path = wildcard()
        return DominatorPattern(parent, path, self)

    def optional(self, option_constructor):
        """
        Create a optional user of this pattern

        Parameters
        ----------
        option_constructor: function
            A function that takes a single Pattern parameter and returns
            a constructed pattern matching the option

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting Pattern
        """
        return self | option_constructor(self)


def is_input(name: str = "") -> DFPattern:
    """
    Syntatic sugar for creating an optionally named VarPattern

    Parameters
    ----------
    name: str
        The name of the input pattern to match

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting InputPattern
    """
    return VarPattern(name)


def is_op(op_name: str) -> DFPattern:
    """
    Syntatic sugar for creating an operator ExprPattern

    Parameters
    ----------
    op_name: String
        The name of the relay op

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting ExprPattern
    """
    op = get(op_name)
    return ExprPattern(op)


def wildcard() -> DFPattern:
    """
    Syntatic sugar for creating a WildcardPattern

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting WildcardPattern
    """
    return WildcardPattern()


def has_type(ttype, pattern: DFPattern = None) -> DFPattern:
    """
    Syntatic sugar for creating a TypePattern

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    ttype: tvm.relay.Type
        The type to match

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting TypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return TypePattern(pattern, ttype)


def has_attr(attr_name: DFPattern, attr_value, pattern=None) -> DFPattern:
    """
    Syntatic sugar for creating an AttrPattern

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern.

    attrs: tvm.Attrs
        The attributes to match

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting AttrPattern
    """
    if pattern is None:
        pattern = wildcard()
    return pattern.has_attr(attr_name, attr_value)


def dominates(parent: DFPattern, path: DFPattern, child: DFPattern) -> DFPattern:
    """
    Syntatic sugar for creating an Dominator pattern

    Parameters
    ----------
    parent: tvm.relay.dataflow_pattern.DFPattern
        The parent pattern.
    path: tvm.relay.dataflow_pattern.DFPattern
        The fuzzy path pattern.
    child: tvm.relay.dataflow_pattern.DFPattern
        The child pattern.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting DominatorPattern
    """
    return DominatorPattern(parent, path, child)


def match(pattern: DFPattern, expr: Expr) -> bool:
    """
    Match a pattern to an expression

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern.
    expr : tvm.relay.Expr
        The expression to match.
    """
    return ffi.match(pattern, expr)


@register_df_node
class ExprPattern(DFPattern):
    """A pattern which matches a constant expression.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expression to match.
    """

    def __init__(self, expr: Expr):
        self.__init_handle_by_constructor__(ffi.ExprPattern, expr)


@register_df_node
class VarPattern(DFPattern):
    """A local variable in Relay.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: tvm.relay.Type, optional
        The type annotation on the variable.
    """

    def __init__(self, name_hint: str, type_annotation=None):
        self.__init_handle_by_constructor__(
            ffi.VarPattern, name_hint, type_annotation)


@register_df_node
class CallPattern(DFPattern):
    """A pattern matching a function call node in Relay.

    Parameters
    ----------
    op: realy.dataflow_pattern.DFPattern
        The operation to be called.

    args: List[realy.dataflow_pattern.DFPattern]
        The arguments to the call.

    attrs: Optional[tvm.Attrs]
        Attributes to the call, can be None

    type_args: Optional[List[tvm.relay.Type]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.
    """

    def __init__(self, op, args, attrs=None, type_args=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(
            ffi.CallPattern, op, args, attrs, type_args)


@register_df_node
class TuplePattern(DFPattern):
    """A patern matching a Relay Tuple.

    Parameters
    ----------
    fields : List[tvm.relay.dataflow_pattern.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("TuplePattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on TuplePattern")


@register_df_node
class TupleGetItemPattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    tuple_value: tvm.relay.dataflow_pattern.DFPattern
        The input tuple expression.

    index: int
        The index.
    """

    def __init__(self, tuple_value: DFPattern, index):
        self.__init_handle_by_constructor__(
            ffi.TupleGetItemPattern, tuple_value, index)


@register_df_node
class AltPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relay.dataflow_pattern.DFPattern
        One possible matching Pattern
    right: tvm.relay.dataflow_pattern.DFPattern
        One possible matching Pattern
    """

    def __init__(self, left: DFPattern, right: DFPattern):
        self.__init_handle_by_constructor__(
            ffi.AltPattern, left, right)


@register_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)


@register_df_node
class TypePattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern that needs type annotation

    ttype: tvm.relay.Type
        The type to match
    """

    def __init__(self, pattern: DFPattern, ttype):
        self.__init_handle_by_constructor__(
            ffi.TypePattern, pattern, ttype)


@register_df_node
class AttrPattern(DFPattern):
    """Get match an expression with a certain attributes.
    Currently only supports Op Attributes, not call Attributes

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern.

    attrs: tvm.Attrs
        The attributes to match
    """

    def __init__(self, pattern: DFPattern, attrs):
        self.__init_handle_by_constructor__(
            ffi.AttrPattern, pattern, attrs)


@register_df_node
class DominatorPattern(DFPattern):
    """Match a domination graph.

    Parameters
    ----------
    parent: tvm.relay.dataflow_pattern.DFPattern
        The parent, i.e., the single node which produces something,
        later aggregated by the child
    path: tvm.relay.dataflow_pattern.DFPattern
        The fuzzy path pattern between parent and child,
        typically matches elementwise ops
    child: tvm.relay.dataflow_pattern.DFPattern
        The last node in the domination which is the end user
        for all nodes in the path and the parent
    """

    def __init__(self, parent: DFPattern, path: DFPattern, child: DFPattern):
        self.__init_handle_by_constructor__(
            ffi.DominatorPattern, parent, path, child)


class DFPatternCallback:
    """A Callback for Pattern Rewriting

    When rewrite is called on this DFPatternCallback, the backend will find matches for the
    pattern, call the callback function, and replace the matched expression with whatever
    the callback returns.

    Users are expect to inherit from this class and provide a "self.pattern" to match
    """

    def rewrite(self, expr: Expr) -> Expr:
        """
        Rewrite expression with this callback

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression to rewrite.

        Returns
        -------
        result : tvm.relay.Expr
            The Expression with matched subgraphs rewritten by the callbacks
        """
        return rewrite(self, expr)

    def callback(self, pre, post, node_map):
        """
        Callback function to use when we found a match to the pattern

        Parameters
        ----------
        pre : tvm.relay.Expr
            The matching expression from the original graph.
        post : tvm.relay.Expr
            The matching expression with rewritten inputs
        node_map : Map(DFPattern, List(Expr))
            The map between patterns and matched expressions

        Returns
        -------
        result : tvm.relay.Expr
            The Expression with matched subgraph rewritten by the callback
        """
        raise "Unimplemented"

class _DFPatternCallback(Object):
    """C++ implemenation"""
    def __init__(self, pattern, callback):
        self.__init_handle_by_constructor__(
            ffi.DFPatternCallback, pattern, callback)


def rewrite(callbacks, expr: Expr) -> Expr:
    """
    Rewrite expression with the given callbacks

    Parameters
    ----------
    callbacks: tvm.relay.dataflow_pattern.DFPatternCallback
        The input callback or list of callbacks.
    expr : tvm.relay.Expr
        The expression to rewrite.

    Returns
    -------
    result : tvm.relay.Expr
        The Expression with matched subgraphs rewritten by the callbacks
    """
    if isinstance(callbacks, DFPatternCallback):
        tmp = [_DFPatternCallback(callbacks.pattern, callbacks.callback)]
    else:
        tmp = []
        for callback in callbacks:
            tmp.append(_DFPatternCallback(callback.pattern, callback.callback))

    return ffi.rewrite(tmp, expr)

def partition(pattern: DFPattern, expr: Expr) -> Expr:
    """
    Parition the expression into a series of functions that match the pattern

    Parameters
    ----------
    partion: tvm.relay.dataflow_pattern.DFPattern
        The pattern to match
    expr : tvm.relay.Expr
        The expression to split into functions

    Returns
    -------
    result : tvm.relay.Expr
        The Expression with matched subgraphs replaced by function calls to that subgraph
    """
    return ffi.partition(pattern, expr)
