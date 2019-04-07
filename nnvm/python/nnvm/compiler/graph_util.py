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
"""Utility function to get information from graph."""
from __future__ import absolute_import as _abs

import tvm
from . import graph_attr

from ..graph import create
from ..symbol import Group, ones_like

def infer_shape(graph, **shape):
    """Infer the shape given the shape of inputs.

    Parameters
    ----------
    graph : Graph
        The graph to perform shape inference from

    shape : dict of str to tuple
        The specific input shape.

    Returns
    -------
    in_shape : list of tuple
         Shape of inputs

    out_shape: list of tuple
         Shape of outputs
    """
    graph = graph_attr.set_shape_inputs(graph, shape)
    graph = graph.apply("InferShape")
    shape = graph.json_attr("shape")
    index = graph.index
    input_shape = [shape[index.entry_id(x)] for x in index.input_names]
    output_shape = [shape[index.entry_id(x)] for x in index.output_entries]
    return input_shape, output_shape


def infer_dtype(graph, **dtype):
    """Infer the type given the typeS of inputs.

    Parameters
    ----------
    graph : Graph
        The graph to perform type inference from

    dtype : dict of str to dtype
        The specific input data type.

    Returns
    -------
    in_dtype : list of tuple
         Dtype of inputs

    out_dtype: list of tuple
         Dtype of outputs
    """
    graph = graph_attr.set_dtype_inputs(graph, dtype)
    graph = graph.apply("InferType")
    dtype = graph.json_attr("dtype")
    index = graph.index
    input_dtype = [graph_attr.TCODE_TO_DTYPE[dtype[index.entry_id(x)]]
                   for x in index.input_names]
    output_dtype = [graph_attr.TCODE_TO_DTYPE[dtype[index.entry_id(x)]]
                    for x in index.output_entries]
    return input_dtype, output_dtype


_deep_compare = tvm.get_global_func("nnvm.graph.DeepCompare")

def check_graph_equal(grapha, graphb, compare_variable_attrs=False):
    """Check if two graphs have equal structure.

    Parameters
    ----------
    grapha : Graph
        The first graph

    graphb : Graph
        The second graph

    compare_variable_attrs : bool, optional
        Whether we want to compare attributes(names) on variables.
        Usually it is safe to skip it unless we want input name
        to exactly match

    Raises
    ------
    ValueError
        ValueError is raised with error message when graph not equal
    """
    err = _deep_compare(grapha, graphb, compare_variable_attrs)
    if err:
        raise ValueError("Graph compare error: " + err)

def get_gradient_graph(ys, xs, grad_ys=None):
    """Create gradient graph of ys with respect to xs.

    Parameters
    ----------
    ys : Symbol or list of Symbol
        Symbols from which the gradient is calculated.
    xs : Symbol or list of Symbol
        Symbols the gradient respect to.
        For group symbol, gradients for all outputs will be calculated.
    grad_ys : Symbol or list of Symbol
        Head gradients for ys.

    Returns
    -------
    ret : Graph
        Generated gradient graph.
    """
    if isinstance(ys, list):
        ys = Group(ys)
    g = create(ys)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    ny = len(ys.list_output_names())
    if grad_ys is None:
        grad_ys = [ones_like(ys[i]) for i in range(ny)]
    g._set_symbol_list_attr('grad_ys_out_grad', grad_ys)
    return g.apply('Gradient')

def gradients(ys, xs, grad_ys=None):
    """Create gradient symbol of ys respect to xs.

    Parameters
    ----------
    ys : Symbol or list of Symbol
        Symbols from which the gradient is calculated.
    xs : Symbol or list of Symbol
        Symbols the gradient respect to.
        For group symbol, gradients for all outputs will be calculated.
    grad_ys : Symbol or list of Symbol
        Head gradients for ys.

    Returns
    -------
    ret : list of Symbol
        Generated gradient symbol. For each xs,
        all gradients from ys are merged into a single symbol.
    """
    grad_g = get_gradient_graph(ys, xs, grad_ys)
    nx = len(Group(xs).list_output_names()) \
        if isinstance(xs, list) else len(xs.list_output_names())
    ret = [grad_g.symbol[i] for i in range(nx)]
    return ret
