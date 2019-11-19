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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains the set of passes for Relay, which exposes an interface for
configuring the passes and scripting them in Python.
"""
from . import _analysis
from . import _make
from .expr import Expr, Function, Var, Call, TupleGetItem
from .op.op import Op
from .ty import Type
from .module import Module
from .feature import Feature

import json


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _analysis.post_order_visit(expr, fvisit)


def well_formed(expr):
    """Check that each Var is only bound once (well formed).

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    well_form : bool
        Whether the input expression is well formed
    """
    return _analysis.well_formed(expr)


def check_kind(t, mod=None):
    """Check that the type is well kinded and return the kind.
    For example, this mean type cannot has tensor of tensor, or is a tuple type
    of 2 shapes.

    Parameters
    ----------
    t : tvm.relay.Type
        The type to check

    mod : Optional[tvm.relay.Module]
        The global module.

    Returns
    -------
    kind : Kind
        the kind of t

    Examples
    --------
    .. code:: python

        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Shape)])) == Shape
        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Type)])) == Type
    """
    if mod is not None:
        return _analysis.check_kind(t, mod)
    else:
        return _analysis.check_kind(t)


def check_constant(expr):
    """Check whether an expression is constant

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    result : bool
        Whether the expression is constant.
    """
    return _analysis.check_constant(expr)


def free_vars(expr):
    """Get free Vars from expression expr in Post DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of free variables in post DFS order.

    Note
    ----
    The fact that Vars are post-DFS ordred are useful in
    neural networks: usually this means weights of previous
    are ordered first.
    """
    return _analysis.free_vars(expr)


def bound_vars(expr):
    """Get bound vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of bound variables in post-DFS order.
    """
    return _analysis.bound_vars(expr)


def all_vars(expr):
    """Get all vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of all variables in post-DFS order.
    """
    return _analysis.all_vars(expr)


def free_type_vars(expr, mod=None):
    """Get free type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.relay.Module]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of free type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _analysis.free_type_vars(expr, use_mod)


def bound_type_vars(expr, mod=None):
    """Get bound type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.relay.Module]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of bound type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _analysis.bound_type_vars(expr, use_mod)


def all_type_vars(expr, mod=None):
    """Get all type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.relay.Module]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of all type variables in post-DFS order
    """
    use_mod = mod if mod is not None else Module()
    return _analysis.all_type_vars(expr, use_mod)


def alpha_equal(lhs, rhs):
    """Compare two Relay expr for structural equivalence (alpha equivalence).

    Parameters
    ----------
    lhs : tvm.relay.Expr
        One of the input Expression.

    rhs : tvm.relay.Expr
        One of the input Expression.

    Returns
    -------
    result : bool
        True iff lhs is alpha equal to rhs.
    """
    return bool(_make._alpha_equal(lhs, rhs))


def assert_alpha_equal(lhs, rhs):
    """Assert that two Relay expr is structurally equivalent. (alpha equivalence).

    Parameters
    ----------
    lhs : tvm.relay.Expr
        One of the input Expression.

    rhs : tvm.relay.Expr
        One of the input Expression.
    """
    _make._assert_alpha_equal(lhs, rhs)


def graph_equal(lhs, rhs):
    """Compare two Relay expr for data-flow equivalence.
    The difference between this and alpha-equality is that
    variables are not expected to match between lhs and rhs;
    they are treated as sources and are mapped between each other.

    Parameters
    ----------
    lhs : tvm.relay.Expr
      One of the input Expression.

    rhs : tvm.relay.Expr
      One of the input Expression.

    Returns
    -------
    result : bool
      True iff lhs is data-flow equivalent to rhs.
    """
    return bool(_make._graph_equal(lhs, rhs))


def assert_graph_equal(lhs, rhs):
    """Compare two Relay expr for data-flow equivalence.
    The difference between this and alpha-equality is that
    variables are not expected to match between lhs and rhs;
    they are treated as sources and are mapped between each other.

    Parameters
    ----------
    lhs : tvm.relay.Expr
      One of the input Expression.

    rhs : tvm.relay.Expr
      One of the input Expression.
    """
    _make._assert_graph_equal(lhs, rhs)


def collect_device_info(expr):
    """Collect the device allocation map for the given expression. The device
    ids are propagated from the `device_copy` operators.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.expr, int]
        A dictionary mapping tvm.relay.Expr to device type.
    """
    return _analysis.CollectDeviceInfo(expr)


def collect_device_annotation_ops(expr):
    """Collect the device annotation ops for the given expression.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.expr, int]
        A dictionary mapping tvm.relay.Expr to device type where the keys are
        annotation expressions.
    """
    return _analysis.CollectDeviceAnnotationOps(expr)


def get_total_mac_number(expr):
    """
    Count the number of MACs (multiply-accumulate) of a model

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    result : int64
      The number of MACs (multiply-accumulate) of a model
    """
    return _analysis.GetTotalMacNumber(expr)


def unmatched_cases(match, mod=None):
    """
    Finds cases that the match expression does not catch, if any.

    Parameters
    ----------
    match : tvm.relay.Match
        The match expression

    mod : Optional[tvm.relay.Module]
        The module (defaults to an empty module)

    Returns
    -------
    missing_patterns : [tvm.relay.Pattern]
        Patterns that the match expression does not catch.
    """
    return _analysis.unmatched_cases(match, mod)


def detect_feature(a, b=None):
    """
    Detect the feature used in a relay program.

    Parameters
    ----------
    a : Union[tvm.relay.Expr, tvm.relay.Module]
      The input expression or module.

    b : Optional[Union[tvm.relay.Expr, tvm.relay.Module]]
      The input expression or module.
      The two arguments cannot both be expression or module.

    Returns
    -------
    features : Set[Feature]
      Features used in the program.
    """
    if isinstance(a, Module):
        a, b = b, a
    return set([Feature(int(x)) for x in _analysis.detect_feature(a, b)])


def structural_hash(value):
    """Hash a Relay expression structurally.

    Parameters
    ----------
    expr : Union[tvm.relay.Expr, tvm.relay.Type]
      The expression to hash.

    Returns
    -------
    result : int
      The hash value
    """
    if isinstance(value, Expr):
        return int(_analysis._expr_hash(value))
    elif isinstance(value, Type):
        return int(_analysis._type_hash(value))
    else:
        msg = ("found value of type {0} expected" +
               "relay.Expr or relay.Type").format(type(value))
        raise TypeError(msg)

def _export_as_relayviz(expr):
    """Export a Relay function as a nested dictionary, following the RelayViz spec
    (https://discuss.tvm.ai/t/rfc-visualizing-relay-program-as-graph/4825/10). The dictionary will
    contain all information useful for visualizing the Relay program and is meant to be consumed
    by other visualizers.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    viz : dict
        Nested dictionary
    """

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    relayviz_nodes = []

    # Sort by node ID
    for node, node_idx in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, Function):
            relayviz_nodes.append({
                'node_kind': 'Function',
                'body': node_dict[node.body],
                'params': [node_dict[x] for x in node.params],
                'ret_type': {
                    'dtype': node.ret_type.dtype,
                    'shape': [int(x) for x in node.ret_type.shape]
                }
            })
        elif isinstance(node, Var):
            relayviz_nodes.append({
                'node_kind': 'Var',
                'name': node.name_hint,
                'dtype': node.type_annotation.dtype,
                'shape': [int(x) for x in node.type_annotation.shape]
            })
        elif isinstance(node, Call):
            relayviz_nodes.append({
                'node_kind': 'Call',
                'op': node_dict[node.op],
                'args': [node_dict[arg] for arg in node.args]
            })
        elif isinstance(node, Op):
            relayviz_nodes.append({
                'node_kind': 'Op',
                'name': node.name,
                'attrs': {}
            })
        elif isinstance(node, TupleGetItem):
            relayviz_nodes.append({
                'node_kind': 'TupleGetItem',
                'tuple_value': node_dict[node.tuple_value],
                'index': node.index
            })
        else:
            raise RuntimeError(
                    'Unknown node type. node_idx: {}, node: {}'.format(node_idx, type(node)))

    obj = {}
    obj['format'] = 'relayviz'
    obj['version'] = [1, 0]
    obj['nodes'] = relayviz_nodes
    return obj

def _export_as_graphviz(expr):
    from graphviz import Digraph
    obj = _export_as_relayviz(expr)
    dot = Digraph(format='svg')
    dot.attr(rankdir='BT')
    dot.attr('node', shape='box')
    for node_id, node in enumerate(obj['nodes']):
        if node['node_kind'] == 'Var':
            dot.node(str(node_id),
                     '{}:\nTensor[{}, {}])'.format(
                         node['name'], tuple(node['shape']), node['dtype']
                     ))
        elif node['node_kind'] == 'Call':
            dot.node(str(node_id), 'Call(op={})'.format(obj['nodes'][ node['op'] ]['name']))
            for arg in node['args']:
                dot.edge(str(arg), str(node_id))
        elif node['node_kind'] == 'Function':
            dot.node(str(node_id), 'Function')
            dot.edge(str(node['body']), str(node_id))
        elif node['node_kind'] == 'TupleGetItem':
            dot.node(str(node_id), 'TupleGetItem(idx={})'.format(node['index']))
            dot.edge(str(node['tuple_value']), str(node_id))
        elif node['node_kind'] == 'Op':
            pass
        else:
            raise RuntimeError(
                    'Node type {} not supported by GraphViz visualizer.'.format(node['node_kind']))
    return dot


def visualize(expr, output_format='graphviz'):
    possible_format = ['graphviz']
    if output_format not in possible_format:
        raise RuntimeError('output_format should be one of {}'.format(possible_format))

    if output_format == 'graphviz':
        return _export_as_graphviz(expr)
