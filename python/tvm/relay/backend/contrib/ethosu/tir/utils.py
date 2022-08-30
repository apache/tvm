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
"""Helper utility functions used by the NPU TIR compiler"""
import tvm
from tvm import arith


def get_op_attrs(stmt):
    """Iterate through nested attribute statements accumulating their values
    in an attribute dictionary.

    The "pragma_" prefix is removed as a convenience.

    Parameters
    ----------
    stmt : tvm.tir.AttrStmt
        The outermost attribute statement to begin from.

    Returns
    -------
    attrs : dict of str to object
        The attribute dictionary.
    stmt : tvm.tir.Stmt
        The body after having collected the final attribute statement.

    """
    attrs = {}
    while isinstance(stmt, tvm.tir.AttrStmt):
        # The pragma scheduler inserts "pragma_" before all the
        # attr names, this is annoying so we get rid of it
        attr = stmt.attr_key.replace("pragma_", "")
        attrs[attr] = stmt.value
        stmt = stmt.body

    return attrs, stmt


def get_strides(index, stride_vars):
    """Get the striding of given vars in an indexing expression.

    Parameters
    ----------
    index : tvm.tir.PrimExpr
        The index expression where the stride vars are present.
    stride_vars : list of tvm.tir.Var
        The vars to determine the striding of.

    Returns
    -------
    strides : list of int
        The striding of each stride var in the index expression
        in the same order as the stride vars were given.

    """
    strides = [1] * len(stride_vars)
    dmap = {}

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Var):
            dmap[stmt] = arith.IntervalSet(0, 0)

    tvm.tir.stmt_functor.post_order_visit(index, _visit)
    min_value = int(arith.Analyzer().int_set(index, dmap).min_value)
    for var in dmap:
        if var in stride_vars:
            # NOTE: Doing this using a [0, 1] interval doesn't work reliably
            # Seems to be a bug
            dmap[var] = arith.IntervalSet(1, 1)
            max_value = int(arith.Analyzer().int_set(index, dmap).max_value)
            stride = int(max_value - min_value)
            i = stride_vars.index(var)
            strides[i] = stride
            dmap[var] = arith.IntervalSet(0, 0)

    return strides


def get_base_address(index):
    """Determine the first (base) address accessed by an index expression.

    Parameters
    ----------
    index : tvm.tir.PrimExpr
        The index expression to determine the base address of.

    Returns
    -------
    base_address:
        The first address accessed by the index expression.

    """
    dmap = {}

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.Var):
            dmap[stmt] = arith.IntervalSet(0, 0)

    tvm.tir.stmt_functor.post_order_visit(index, _visit)
    base_address = int(arith.Analyzer().int_set(index, dmap).min_value)
    return base_address


def get_outer_loops(stmt, layout):
    """Get the outer loops of an operator.

    Parameters
    ----------
    stmt : tvm.tir.For
        The outermost loop.
    layout : str
        The output tensor layout (NHWC or NHCWB16).

    Returns
    -------
    n : tvm.tir.For
        The batch loop.
    h : tvm.tir.For
        The height loop.
    w : tvm.tir.For
        The width loop.
    c : tvm.tir.For
        The channels loop.
    b : tvm.tir.For
        The brick loop. None for NHWC
    body : tvm.tir.Stmt
        The inner body of the loops.

    """
    if layout == "NHWC":
        n = stmt
        h = n.body
        w = h.body
        c = w.body
        b = tvm.tir.For(tvm.tir.Var("b", "int32"), 0, 0, 0, tvm.tir.Evaluate(0))
        return n, h, w, c, b, c.body
    if layout == "NHCWB16":
        n = stmt
        h = n.body
        cb = h.body
        w = cb.body
        b = w.body
        return n, h, w, cb, b, b.body
    return None


def get_loads(stmt):
    """Get the BufferLoad statements.

    Parameters
    ----------
    stmt : tvm.tir.Stmt
        The statement to get the BufferLoads from.

    Returns
    -------
    loads : list of tvm.tir.BufferLoad
        The BufferLoads found.

    """
    loads = []

    def _visit(s):
        if isinstance(s, tvm.tir.BufferLoad):
            loads.append(s)

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    return loads


def get_stores(stmt):
    """Get the BufferStore statements.

    Parameters
    ----------
    stmt : tvm.tir.Stmt
        The statement to get the BufferStores from.

    Returns
    -------
    stores : list of tvm.tir.BufferStore
        The BufferStores found.

    """
    stores = []

    def _visit(s):
        if isinstance(s, tvm.tir.BufferStore):
            stores.append(s)

    tvm.tir.stmt_functor.post_order_visit(stmt, _visit)
    return stores
