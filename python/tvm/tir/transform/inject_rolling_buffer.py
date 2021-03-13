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
import tvm
from tvm import arith

from collections import defaultdict


def InjectRollingBuffer():
    """Inject rolling buffer statements.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    buffer_to_attrs = defaultdict(list)
    rolling_buffers = set()
    rolling_buffer_to_info = dict()
    iter_vars = list()
    hoist_buffer_to_for = defaultdict(list)

    class RollingBufferInfo:
        def __init__(self, rolling_axis, rolling_extent, axis_overlaps, axis_iter_vars):
            self.rolling_axis = rolling_axis
            self.rolling_extent = rolling_extent
            self.axis_overlaps = axis_overlaps
            self.axis_iter_vars = axis_iter_vars

    def _pre_visit(stmt):
        if isinstance(stmt, tvm.tir.For):
            # Manage the stack of iter_vars
            iter_vars.append(stmt)

        elif isinstance(stmt, tvm.tir.AttrStmt):
            if isinstance(stmt.node, tvm.tir.Buffer):
                if stmt.attr_key == "rolling_buffer" and stmt.value.value == True:
                    # If the attribute is indicating that a buffer should be a rolling
                    # buffer, then update the rolling_buffers set to include the bufffer
                    rolling_buffers.add(stmt.node)
                # Keep a dictionary associating attribute statements with the buffers
                # they reference. We'll need this if the buffer gets hoisted and we
                # need to hoist all of its attributes at the same time.
                buffer_to_attrs[stmt.node].append(stmt)

        elif isinstance(stmt, tvm.tir.BufferRealize):
            if stmt.buffer in rolling_buffers:
                # If a BufferRealize has been identified as needing to be made into
                # a rolling buffer, begin the analysis...
                bound_iter_vars = []
                bound_strides = []
                bound_overlaps = []
                # We use the bound information of the BufferRealize to calculate
                # how we can legally roll
                for bound in stmt.bounds:
                    # If the bound is an int, we can't roll over it
                    if isinstance(bound.min, tvm.tir.IntImm):
                        iter_var = None
                        stride = 0
                    # If the bound is just a Var, that implies the stride is 1
                    elif isinstance(bound.min, tvm.tir.Var):
                        iter_var = bound.min
                        stride = 1
                    # Otherwise, it's the iter var multiplied by the stride
                    # If not we're in unknown behaviour, so assert
                    else:
                        assert isinstance(bound.min, tvm.tir.Mul)
                        assert isinstance(bound.min.a, tvm.tir.Var)
                        assert isinstance(bound.min.b, tvm.tir.IntImm)
                        iter_var = bound.min.a
                        stride = bound.min.b.value
                    bound_iter_vars.append(iter_var)
                    bound_strides.append(stride)
                    if iter_var is not None:
                        bound_overlaps.append(bound.extent.value - stride)
                    else:
                        bound_overlaps.append(0)

                # Pick the outermost iter_var that's mentioned in the bounds
                # to be the rolling axis
                roll_iter_var = None
                roll_axis = -1
                for loop in iter_vars:
                    iter_var = loop.loop_var
                    if iter_var in bound_iter_vars:
                        roll_iter_var = iter_var
                        roll_axis = bound_iter_vars.index(iter_var)
                        break

                # We must have found an axis to roll over
                assert roll_iter_var is not None
                assert roll_axis != -1
                rolling_buffer_info = RollingBufferInfo(
                    roll_axis, stmt.bounds[roll_axis].extent.value, bound_overlaps, bound_iter_vars
                )
                rolling_buffer_to_info[stmt.buffer] = rolling_buffer_info
                new_bounds = []
                for i, extent in enumerate(stmt.buffer.shape):
                    if i == rolling_buffer_info.rolling_axis:
                        new_bounds.append(tvm.ir.Range(rolling_buffer_info.rolling_extent))
                    else:
                        new_bounds.append(tvm.ir.Range(extent))
                new_realize = tvm.tir.BufferRealize(
                    stmt.buffer, new_bounds, stmt.condition, stmt.body, stmt.span
                )
                hoist_buffer_to_for[iter_var].append(new_realize)

    def _post_visit(stmt):
        if isinstance(stmt, tvm.tir.For):
            # Manage the stack of iter_vars
            iter_vars.pop()
            # If the loop corresponds to an iter_var that needs a BufferRealize
            # hoisting to its scope, perform the hoisting
            if stmt.loop_var in hoist_buffer_to_for:
                body = stmt
                for realize in hoist_buffer_to_for[stmt.loop_var]:
                    attrs = buffer_to_attrs[realize.buffer]
                    new_realize = tvm.tir.BufferRealize(
                        realize.buffer, realize.bounds, realize.condition, body, realize.span
                    )
                    # The attributes attached to the BufferRealize need hoisting too
                    for attr in attrs:
                        if attr.attr_key == "rolling_buffer":
                            continue
                        new_realize = tvm.tir.AttrStmt(
                            attr.node, attr.attr_key, attr.value, new_realize, attr.span
                        )
                    body = new_realize
                return body
        elif isinstance(stmt, tvm.tir.AttrStmt):
            if stmt.node in rolling_buffers:
                # Remove the attribute statements attached to rolling buffers
                # because they will have been hoisted to the relevant rolling
                # scope
                return stmt.body
        elif isinstance(stmt, tvm.tir.BufferRealize):
            if stmt.buffer in rolling_buffers:
                # Remove the original BufferRealize for rolling buffers
                # because they will have been hoisted to the relevant rolling
                # scope
                return stmt.body
        elif isinstance(stmt, tvm.tir.BufferStore):
            if stmt.buffer in rolling_buffer_to_info:
                rolling_buffer_info = rolling_buffer_to_info[stmt.buffer]
                indices = []
                # First modify the access indices to use modulo arithmetic
                # for the rolling axis
                for i, index in enumerate(stmt.indices):
                    if i == rolling_buffer_info.rolling_axis:
                        indices.append(tvm.tir.FloorMod(index, rolling_buffer_info.rolling_extent))
                    else:
                        indices.append(index)
                buffer_store = tvm.tir.BufferStore(stmt.buffer, stmt.value, indices, stmt.span)
                # Then wrap the BufferStores in some Ifs to avoid recomputing elements
                for i, iter_var in enumerate(rolling_buffer_info.axis_iter_vars):
                    if iter_var is not None and rolling_buffer_info.axis_overlaps[i] > 0:
                        dmap = {iter_var: arith.IntervalSet(0, 0)}
                        term_2 = arith.Analyzer().int_set(stmt.indices[i], dmap).min_value
                        buffer_store = tvm.tir.IfThenElse(
                            tvm.tir.Or(
                                iter_var < 1, term_2 >= rolling_buffer_info.axis_overlaps[i]
                            ),
                            buffer_store,
                            None,
                        )
                return buffer_store
        elif isinstance(stmt, tvm.tir.BufferLoad):
            if stmt.buffer in rolling_buffer_to_info:
                rolling_buffer_info = rolling_buffer_to_info[stmt.buffer]
                indices = []
                # Modify the access indices to use modulo arithmetic
                # for the rolling axis
                for i, index in enumerate(stmt.indices):
                    if i == rolling_buffer_info.rolling_axis:
                        indices.append(tvm.tir.FloorMod(index, rolling_buffer_info.rolling_extent))
                    else:
                        indices.append(index)
                return tvm.tir.BufferLoad(stmt.buffer, indices, stmt.span)

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(
                f.body,
                _pre_visit,
                _post_visit,
                [
                    "tir.AttrStmt",
                    "tir.BufferRealize",
                    "tir.For",
                    "tir.BufferStore",
                    "tir.BufferLoad",
                ],
            )
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.InjectRollingBuffer"
    )
