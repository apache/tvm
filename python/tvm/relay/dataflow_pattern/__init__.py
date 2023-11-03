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
# pylint: disable=no-member
from typing import Callable, Dict, List, Optional

import tvm._ffi
from tvm.relay.expr import RelayExpr as Expr

from ... import _ffi as tvm_ffi
from ... import ir as _ir
from ...ir import make_node
from ...ir.base import Node
from ...runtime import Object
from ..base import astext, pretty_print
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
        return tvm._ffi.register_object("relay.dataflow_pattern." + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)


class DFPattern(Node):
    """Base class of all Patterns."""

    def __str__(self):
        return pretty_print(self)

    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[Object->str]
            Optionally annotate function to provide additional
            information in the comment block.

        Returns
        -------
        text : str
            The text format of the expression.

        Notes
        -----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.
        """
        return astext(self, show_meta_data, annotate)

    def __call__(self, *args):
        args = list(args)
        if len(args) == 1 and args[0] is None:
            args = None
        return CallPattern(self, args)

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

    def has_attr(self, attrs: Dict[str, Object]):
        """
        Add an attribute constraint to this pattern

        Parameters
        ----------
        attrs: Dict[str, Object]

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting AttrPattern
        """
        attrs = make_node("DictAttrs", **attrs)
        return AttrPattern(self, attrs)

    def has_type(self, ttype: tvm.ir.type.Type):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        ttype: tvm.ir.type.Type
            The type to match

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting TypePattern
        """
        return has_type(ttype, self)

    def has_dtype(self, dtype: str):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        dtype: str
            The dtype to match

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting DataTypePattern
        """
        return has_dtype(dtype, self)

    def has_shape(self, shape: List[tvm.ir.PrimExpr]):
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        shape: List[tvm.ir.PrimExpr]
            The shape to match

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting ShapePattern
        """
        return has_shape(shape, self)

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

    def partition(
        self,
        expr: Expr,
        attrs: Optional[Dict[str, Object]] = None,
        check: Callable[[Expr], bool] = lambda x: True,
    ) -> Expr:
        """
        Partition the expression into functions defined by this pattern

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression to match.
        attrs : Optional[Dict[str, Object]]
            A dictionary of Attribute name/values to add to the paritioned function
        check : Callable[[Expr], bool]
            A function to perform more complicated checks on the matched expression.
            Returns true if partitioning should proceed, false otherwise.

        Returns
        -------
        result : tvm.relay.Expr
            The Expression with matched subgraphs replaced by function calls to that subgraph
        """
        return partition(self, expr, attrs, check)

    def dominates(self, parent: "DFPattern", path: "DFPattern" = None):
        """
        Create a dominator for this pattern.

        Parameters
        ----------
        parent: tvm.relay.dataflow_pattern.DFPattern
            The parent pattern this pattern dominates.
        path: tvm.relay.dataflow_pattern.DFPattern
            The fuzzy path pattern.

        Returns
        -------
        result: tvm.relay.dataflow_pattern.DFPattern
            The resulting DominatorPattern.
        """
        if path is None:
            path = wildcard()
        return DominatorPattern(parent, path, self)

    def optional(self, option_constructor: Callable[["DFPattern"], "DFPattern"]):
        """
        Create a optional user of this pattern.

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


def is_var(name: str = "") -> "DFPattern":
    """
    Syntatic sugar for creating an optionally named VarPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return VarPattern(name)


def is_constant() -> "DFPattern":
    """
    Syntatic sugar for creating a ConstantPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return ConstantPattern()


def is_expr(expr: Expr) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    expr: Expr
        The Relay expression to match.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return ExprPattern(expr)


def is_op(op_name: str) -> "DFPattern":
    """
    Syntatic sugar for creating an operator ExprPattern.

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


def is_tuple(fields: tvm.ir.container.Array) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    fields : Array[tvm.relay.dataflow_pattern.DFPattern]
        The fields in the tuple.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return TuplePattern(fields)


def is_tuple_get_item(tuple_value: "DFPattern", index: Optional[int] = None) -> "DFPattern":
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    tuple_value: tvm.relay.dataflow_pattern.DFPattern
        The input tuple expression.

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return TupleGetItemPattern(tuple_value, index)


def is_if(cond, true_branch, false_branch):
    """
    Syntatic sugar for creating an IfPattern.

    Parameters
    ----------
    cond: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the condition of If.

    true_branch: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the true branch of If.

    false_branch: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the false branch of If.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return IfPattern(cond, true_branch, false_branch)


def is_let(var, value, body):
    """
    Syntatic sugar for creating a LetPattern.

    Parameters
    ----------
    var: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the variable of Let.

    value: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the value of Let.

    body: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the body where the binding is in effect.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return LetPattern(var, value, body)


def wildcard() -> "DFPattern":
    """
    Syntatic sugar for creating a WildcardPattern.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting pattern.
    """
    return WildcardPattern()


def has_type(ttype: tvm.ir.type.Type, pattern: "DFPattern" = None) -> "DFPattern":
    """
    Syntatic sugar for creating a TypePattern

    Parameters
    ----------
    ttype: tvm.ir.type.Type
        The type to match

    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting TypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return TypePattern(pattern, ttype)


def has_dtype(dtype: str, pattern: "DFPattern" = None) -> "DFPattern":
    """
    Syntatic sugar for creating a DataTypePattern

    Parameters
    ----------
    dtype: str
        The dtype to match

    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting DataTypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return DataTypePattern(pattern, dtype)


def has_shape(shape: List[tvm.ir.PrimExpr], pattern: "DFPattern" = None) -> "DFPattern":
    """
    Syntatic sugar for creating a ShapePattern

    Parameters
    ----------
    shape: List[tvm.ir.PrimExpr]
        The shape to match

    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting ShapePattern
    """
    if pattern is None:
        pattern = wildcard()
    return ShapePattern(pattern, shape)


def has_attr(attrs, pattern=None) -> "DFPattern":
    """
    Syntatic sugar for creating an AttrPattern

    Parameters
    ----------
    attrs: Dict[str, Object]
        The attributes to match

    pattern: Optional[tvm.relay.dataflow_pattern.DFPattern]
        The input pattern.

    Returns
    -------
    result: tvm.relay.dataflow_pattern.DFPattern
        The resulting AttrPattern
    """
    if pattern is None:
        pattern = wildcard()
    return pattern.has_attr(attrs)


def dominates(parent: "DFPattern", path: "DFPattern", child: "DFPattern") -> "DFPattern":
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
        The resulting DominatorPattern.
    """
    return DominatorPattern(parent, path, child)


def match(pattern: "DFPattern", expr: Expr) -> bool:
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
        The name of the variable. Optional, if not provided,
        the pattern will match any VarNode.

    type_annotation: tvm.ir.type.Type, optional
        The type annotation on the variable.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.VarPattern, name_hint)


@register_df_node
class ConstantPattern(DFPattern):
    """A pattern matching a Relay Constant."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.ConstantPattern)


@register_df_node
class CallPattern(DFPattern):
    """A pattern matching a function call node in Relay.

    Parameters
    ----------
    op: relay.dataflow_pattern.DFPattern
        The operation to be called.

    args: List[relay.dataflow_pattern.DFPattern]
        The arguments to the call or None to match any arguments.

    """

    def __init__(
        self,
        op: "DFPattern",
        args: List["DFPattern"],
    ):
        self.__init_handle_by_constructor__(ffi.CallPattern, op, args)


@register_df_node
class FunctionPattern(DFPattern):
    """A pattern matching a function node in Relay.

    Parameters
    ----------
    params: List[relay.dataflow_pattern.DFPattern]
        The parameters to the Function or None to match any parameters.

    body: relay.dataflow_pattern.DFPattern
        The body fo the Function

    """

    def __init__(
        self,
        params: List["DFPattern"],
        body: "DFPattern",
    ):
        self.__init_handle_by_constructor__(ffi.FunctionPattern, params, body)


@register_df_node
class IfPattern(DFPattern):
    """A patern matching a Relay If.

    Parameters
    ----------
    cond: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the condition of If.

    true_branch: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the true branch of If.

    false_branch: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the false branch of If.
    """

    def __init__(self, cond: "DFPattern", true_branch: "DFPattern", false_branch: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.IfPattern, cond, true_branch, false_branch)


@register_df_node
class LetPattern(DFPattern):
    """A patern matching a Relay Let.

    Parameters
    ----------
    var: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the variable of Let.

    value: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the value of Let.

    body: tvm.relay.dataflow_pattern.DFPattern
        The pattern describing the body where the binding is in effect.

    """

    def __init__(self, var: "DFPattern", value: "DFPattern", body: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.LetPattern, var, value, body)


@register_df_node
class TuplePattern(DFPattern):
    """A patern matching a Relay Tuple.

    Parameters
    ----------
    fields : Array[tvm.relay.dataflow_pattern.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: tvm.ir.container.Array):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)

    def __getitem__(self, index: int):
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

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.
    """

    def __init__(self, tuple_value: "DFPattern", index: Optional[int] = None):
        match_index = index if index is not None else -1
        self.__init_handle_by_constructor__(ffi.TupleGetItemPattern, tuple_value, match_index)


@register_df_node
class AltPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relay.dataflow_pattern.DFPattern
        One possible matching pattern.
    right: tvm.relay.dataflow_pattern.DFPattern
        One possible matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.AltPattern, left, right)


@register_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)

    def redirect_to(
        self,
        pat: "DFPattern",
    ):
        """Redirect the WildcardPattern to another pattern

        Parameters
        ----------
        pat: relay.dataflow_pattern.DFPattern
            The pattern that wildcard is redirected to.
        """
        ffi.WildcardPattern_redirect_to(self, pat)


@register_df_node
class TypePattern(DFPattern):
    """A pattern that matches another pattern with a certain type annotation.

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    ttype: tvm.ir.type.Type
        The type to match.
    """

    def __init__(self, pattern: "DFPattern", ttype: tvm.ir.type.Type):
        self.__init_handle_by_constructor__(ffi.TypePattern, pattern, ttype)


@register_df_node
class DataTypePattern(DFPattern):
    """A pattern that matches another pattern with certain data type

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    dtype: str
        The dtype to match.
    """

    def __init__(self, pattern: "DFPattern", dtype: str):
        self.__init_handle_by_constructor__(ffi.DataTypePattern, pattern, dtype)


@register_df_node
class ShapePattern(DFPattern):
    """A pattern that matches another pattern with a certain tensor shape

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern that needs type annotation.

    shape: List[tvm.ir.PrimExpr]
        The shape to match.
    """

    def __init__(self, pattern: "DFPattern", shape: List[tvm.ir.PrimExpr]):
        self.__init_handle_by_constructor__(ffi.ShapePattern, pattern, shape)


@register_df_node
class AttrPattern(DFPattern):
    """Get match an expression with a certain attributes.
    Currently only supports Op Attributes, not call Attributes.

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The input pattern.

    attrs: tvm.ir.attrs.Attrs
        The attributes to match.
    """

    def __init__(self, pattern: "DFPattern", attrs: tvm.ir.attrs.Attrs):
        self.__init_handle_by_constructor__(ffi.AttrPattern, pattern, attrs)


@register_df_node
class DominatorPattern(DFPattern):
    """Match a domination graph.

    Parameters
    ----------
    parent: tvm.relay.dataflow_pattern.DFPattern
        The parent, i.e., the single node which produces something,
        later aggregated by the child.
    path: tvm.relay.dataflow_pattern.DFPattern
        The fuzzy path pattern between parent and child,
        typically matches elementwise ops.
    child: tvm.relay.dataflow_pattern.DFPattern
        The last node in the domination which is the end user
        for all nodes in the path and the parent.
    """

    def __init__(self, parent: "DFPattern", path: "DFPattern", child: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.DominatorPattern, parent, path, child)


class DFPatternCallback:
    """A Callback for Pattern Rewriting.

    When rewrite is called on this DFPatternCallback, the backend will find matches for the
    pattern, call the callback function, and replace the matched expression with whatever
    the callback returns.

    Users are expect to inherit from this class and provide a "self.pattern" to match

    Parameters
    ----------
    require_type: bool
        Whether InferType is required to be run before the callback.
    rewrite_once: bool
        If True, run the callback only once.
    """

    def __init__(self, require_type=False, rewrite_once=False):
        self.pattern = None
        self.require_type = require_type
        self.rewrite_once = rewrite_once

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
            The Expression with matched subgraphs rewritten by the callbacks.
        """
        return rewrite(self, expr)

    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        """
        Callback function to use when we found a match to the pattern

        Parameters
        ----------
        pre : tvm.relay.Expr
            The matching expression from the original graph.
        post : tvm.relay.Expr
            The matching expression with rewritten inputs
        node_map : tvm.ir.container.Map[DFPattern, List[Expr]]
            The map between patterns and matched expressions

        Returns
        -------
        result : tvm.relay.Expr
            The Expression with matched subgraph rewritten by the callback
        """
        raise NotImplementedError()


class _DFPatternCallback(Object):
    """C++ implemenation"""

    def __init__(self, pattern, callback, require_type, rewrite_once):
        self.__init_handle_by_constructor__(
            ffi.DFPatternCallback, pattern, callback, require_type, rewrite_once
        )


def rewrite(callbacks, expr: Expr, mod: Optional[_ir.IRModule] = None) -> Expr:
    """
    Rewrite expression with the given callbacks.

    Parameters
    ----------
    callbacks: tvm.relay.dataflow_pattern.DFPatternCallback
        The input callback or list of callbacks.
    expr : tvm.relay.Expr
        The expression to rewrite.
    mod : Optional[tvm.ir.IRModule]
        The module that associates with the expression.

    Returns
    -------
    result : tvm.relay.Expr
        The Expression with matched subgraphs rewritten by the callbacks.
    """
    if mod is None:
        mod = _ir.IRModule()
    callbacks = [callbacks] if isinstance(callbacks, DFPatternCallback) else callbacks
    tmp = []
    for callback in callbacks:
        assert callback.pattern is not None
        tmp.append(
            _DFPatternCallback(
                callback.pattern, callback.callback, callback.require_type, callback.rewrite_once
            )
        )

    return ffi.rewrite(tmp, expr, mod)


def partition(
    pattern: "DFPattern",
    expr: Expr,
    attrs: Optional[Dict[str, Object]] = None,
    check: Callable[[Expr], bool] = lambda x: True,
) -> Expr:
    """
    Parition the expression into a series of functions that match the pattern

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern to match
    expr : tvm.relay.Expr
        The expression to split into functions
    attrs : Optional[Dict[str, Object]]
        A dict of attributes to apply to the partitioned function
    check : Callable[[Expr], bool]
        A function to perform more complicated checks on the matched expression.
        Returns true if partitioning should proceed, false otherwise.

    Returns
    -------
    result : tvm.relay.Expr
        The Expression with matched subgraphs replaced by function calls to that subgraph
    """
    return ffi.partition(pattern, expr, attrs, check)
