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
"""Statement functor utilities for IR transformations"""

from typing import TypeVar

import tvm
from tvm.ir import PrimExpr, Range

from . import _ffi_api
from .expr_functor import ExprMutator, ExprVisitor, _visit_array
from .function import PrimFunc

T = TypeVar("T")


class StmtFunctor:
    """An abstract visitor over Statement, with visiting functions defined for each Stmt type."""

    def __init__(self):
        self._dispatch_map = {
            "tirx.Bind": self.visit_bind_,
            "tirx.AttrStmt": self.visit_attr_,
            "tirx.IfThenElse": self.visit_if_then_else_,
            "tirx.For": self.visit_for_,
            "tirx.While": self.visit_while_,
            "tirx.Break": self.visit_break_,
            "tirx.Continue": self.visit_continue_,
            "tirx.Allocate": self.visit_allocate_,
            "tirx.AllocateConst": self.visit_allocate_const_,
            "tirx.DeclBuffer": self.visit_decl_buffer_,
            "tirx.BufferStore": self.visit_buffer_store_,
            "tirx.BufferRealize": self.visit_buffer_realize_,
            "tirx.AssertStmt": self.visit_assert_,
            "tirx.ProducerStore": self.visit_producer_store_,
            "tirx.ProducerRealize": self.visit_producer_realize_,
            "tirx.Prefetch": self.visit_prefetch_,
            "tirx.SeqStmt": self.visit_seqstmt_,
            "tirx.Evaluate": self.visit_evaluate_,
            "tirx.SBlock": self.visit_block_,
            "tirx.SBlockRealize": self.visit_block_realize_,
            "tirx.ExecScopeStmt": self.visit_exec_scope_stmt_,
            "tirx.TilePrimitiveCall": self.visit_op_call_,
            "tirx.AllocBuffer": self.visit_alloc_buffer_,
        }

    def visit_stmt(self, stmt):
        """Apply the visitor to a statement.

        Parameters
        ----------
        stmt : tvm.tirx.Stmt
            The statement to be visited.

        Returns
        -------
        result : Any
            The result of the visit.
        """
        if stmt is None:
            return None
        if isinstance(stmt, tvm.tirx.TilePrimitiveCall):
            # subclass of TilePrimitiveCall only exists in python side
            # and are not handled by dispatch map
            key = "TilePrimitiveCall"
        else:
            key = stmt.__class__.__name__
        if key.endswith("Node"):
            key = key[:-4]  # Remove the "Node" suffix

        key = "tirx." + key
        if key in self._dispatch_map:
            return self._dispatch_map[key](stmt)

        return self.visit_stmt_default_(stmt)

    def visit_stmt_default_(self, op):
        """Default visitor implementation for statements."""
        raise NotImplementedError(f"Do not have a default for {op.__class__.__name__}")

    def visit_bind_(self, op):
        """Visitor for Bind nodes."""
        return self.visit_stmt_default_(op)

    def visit_attr_(self, op):
        """Visitor for AttrStmt nodes."""
        return self.visit_stmt_default_(op)

    def visit_if_then_else_(self, op):
        """Visitor for IfThenElse nodes."""
        return self.visit_stmt_default_(op)

    def visit_for_(self, op):
        """Visitor for For nodes."""
        return self.visit_stmt_default_(op)

    def visit_while_(self, op):
        """Visitor for While nodes."""
        return self.visit_stmt_default_(op)

    def visit_break_(self, op):
        """Visitor for Break nodes."""
        return self.visit_stmt_default_(op)

    def visit_continue_(self, op):
        """Visitor for Continue nodes."""
        return self.visit_stmt_default_(op)

    def visit_allocate_(self, op):
        """Visitor for Allocate nodes."""
        return self.visit_stmt_default_(op)

    def visit_allocate_const_(self, op):
        """Visitor for AllocateConst nodes."""
        return self.visit_stmt_default_(op)

    def visit_decl_buffer_(self, op):
        """Visitor for DeclBuffer nodes."""
        return self.visit_stmt_default_(op)

    def visit_buffer_store_(self, op):
        """Visitor for BufferStore nodes."""
        return self.visit_stmt_default_(op)

    def visit_buffer_realize_(self, op):
        """Visitor for BufferRealize nodes."""
        raise ValueError("BufferRealize is not allowed")

    def visit_assert_(self, op):
        """Visitor for AssertStmt nodes."""
        return self.visit_stmt_default_(op)

    def visit_producer_store_(self, op):
        """Visitor for ProducerStore nodes."""
        raise ValueError("ProducerStore is not allowed")

    def visit_producer_realize_(self, op):
        """Visitor for ProducerRealize nodes."""
        raise ValueError("ProducerRealize is not allowed")

    def visit_prefetch_(self, op):
        """Visitor for Prefetch nodes."""
        raise ValueError("Prefetch is not allowed")

    def visit_seqstmt_(self, op):
        """Visitor for SeqStmt nodes."""
        return self.visit_stmt_default_(op)

    def visit_evaluate_(self, op):
        """Visitor for Evaluate nodes."""
        return self.visit_stmt_default_(op)

    def visit_block_(self, op):
        """Visitor for Block nodes."""
        return self.visit_stmt_default_(op)

    def visit_block_realize_(self, op):
        """Visitor for BlockRealize nodes."""
        return self.visit_stmt_default_(op)

    def visit_exec_scope_stmt_(self, op):
        """Visitor for ExecScopeStmt nodes."""
        return self.visit_stmt_default_(op)

    def visit_op_call_(self, op):
        """Visitor for TilePrimitiveCall nodes."""
        return self.visit_stmt_default_(op)

    def visit_buffer_region_(self, op):
        """Visitor for BufferRegion nodes."""
        return self.visit_stmt_default_(op)

    def visit_alloc_buffer_(self, op):
        """Visitor for AllocBuffer nodes."""
        return self.visit_stmt_default_(op)

    def __call__(self, stmt):
        """Call visitor on statement.

        Parameters
        ----------
        stmt : tvm.tirx.Stmt
            The statement.

        Returns
        -------
        result : Any
            The result of visiting.
        """
        return self.visit_stmt(stmt)


class StmtVisitor(StmtFunctor):
    """A visitor over Stmt.

    This is a visitor that recursively traverses a statement. Subclasses can
    override the visit methods to customize the behavior.
    """

    def visit_expr(self, expr):
        """Visit expressions that occur in a statement.

        This method can be overridden to implement expression
        traversal in a statement visitor.

        Parameters
        ----------
        expr : PrimExpr
            The expression to be visited.
        """
        pass

    def visit_bind_(self, op):
        """Visitor implementation for Bind."""
        self.visit_expr(op.value)

    def visit_attr_(self, op):
        """Visitor implementation for AttrStmt."""
        self.visit_expr(op.value)
        self.visit_stmt(op.body)

    def visit_if_then_else_(self, op):
        """Visitor implementation for IfThenElse."""
        self.visit_expr(op.condition)
        self.visit_stmt(op.then_case)
        if op.else_case:
            self.visit_stmt(op.else_case)

    def visit_for_(self, op):
        """Visitor implementation for For."""
        self.visit_expr(op.min)
        self.visit_expr(op.extent)
        if op.step is not None:
            self.visit_expr(op.step)
        self.visit_stmt(op.body)

    def visit_while_(self, op):
        """Visitor implementation for While."""
        self.visit_expr(op.condition)
        self.visit_stmt(op.body)

    def visit_break_(self, op):
        """Visitor implementation for Break."""
        pass

    def visit_continue_(self, op):
        """Visitor implementation for Continue."""
        pass

    def visit_allocate_(self, op):
        """Visitor implementation for Allocate."""
        _visit_array(op.extents, lambda x: self.visit_expr(x))
        self.visit_stmt(op.body)
        self.visit_expr(op.condition)

    def visit_allocate_const_(self, op):
        """Visitor implementation for AllocateConst."""
        _visit_array(op.extents, lambda x: self.visit_expr(x))
        self.visit_stmt(op.body)

    def visit_decl_buffer_(self, op):
        """Visitor implementation for DeclBuffer."""
        if hasattr(op, "body"):
            self.visit_stmt(op.body)
            return
        return

    def visit_buffer_store_(self, op):
        """Visitor implementation for BufferStore."""
        self.visit_expr(op.value)
        _visit_array(op.indices, lambda x: self.visit_expr(x))
        if op.predicate is not None:
            self.visit_expr(op.predicate)

    def visit_assert_(self, op):
        """Visitor implementation for AssertStmt."""
        self.visit_expr(op.condition)
        for message_part in op.message_parts:
            if isinstance(message_part, PrimExpr):
                self.visit_expr(message_part)

    def visit_seqstmt_(self, op):
        """Visitor implementation for SeqStmt."""
        _visit_array(op.seq, lambda s: self.visit_stmt(s))

    def visit_evaluate_(self, op):
        """Visitor implementation for Evaluate."""
        self.visit_expr(op.value)

    def visit_block_(self, op):
        """Visitor implementation for Block."""
        # Visit IterVars
        for iter_var in op.iter_vars:
            self.visit_expr(iter_var.dom.min)
            self.visit_expr(iter_var.dom.extent)

        # Visit buffer regions (reads and writes)
        def _visit_buffer_region(buffer_region):
            for r in buffer_region.region:
                self.visit_expr(r.min)
                self.visit_expr(r.extent)

        _visit_array(op.reads, _visit_buffer_region)
        _visit_array(op.writes, _visit_buffer_region)

        # Visit match buffers
        for match_buffer in op.match_buffers:
            _visit_buffer_region(match_buffer.source)

        # Visit init statement
        if op.init is not None:
            self.visit_stmt(op.init)

        # Visit body
        self.visit_stmt(op.body)

    def visit_block_realize_(self, op):
        """Visitor implementation for BlockRealize."""
        _visit_array(op.iter_values, lambda x: self.visit_expr(x))
        self.visit_expr(op.predicate)
        self.visit_stmt(op.block)

    def visit_exec_scope_stmt_(self, op):
        """Visitor implementation for ExecScopeStmt."""
        self.visit_stmt(op.body)

    def visit_op_call_(self, op):
        """Visitor implementation for TilePrimitiveCall."""
        for arg in op.args:
            if isinstance(arg, PrimExpr):
                self.visit_expr(arg)
            elif isinstance(arg, tvm.tirx.Stmt):
                self.visit_stmt(arg)
            elif isinstance(arg, tvm.tirx.BufferRegion):
                self.visit_buffer_region_(arg)
        for value in op.config.values():
            if isinstance(value, PrimExpr):
                self.visit_expr(value)
            elif isinstance(value, tvm.tirx.Stmt):
                self.visit_stmt(value)

    def visit_buffer_region_(self, op):
        """Visitor implementation for BufferRegion."""

        def _visit_range(range):
            self.visit_expr(range.min)
            self.visit_expr(range.extent)

        _visit_array(op.region, _visit_range)

    def visit_alloc_buffer_(self, op):
        """Visitor implementation for AllocBuffer."""
        if hasattr(op, "body"):
            self.visit_stmt(op.body)
            return
        return


class StmtMutator(StmtFunctor):
    """A mutator over Stmt.

    This is a mutator that recursively transforms a statement. Subclasses can
    override the visit methods to customize the behavior.
    """

    def visit_expr(self, expr):
        """Visit and mutate expressions that occur in a statement.

        This method can be overridden to implement expression
        mutation in a statement mutator.

        Parameters
        ----------
        expr : PrimExpr
            The expression to be visited.

        Returns
        -------
        result : PrimExpr
            The mutated expression.
        """
        return expr

    def visit_bind_(self, op):
        """Mutator implementation for Bind."""
        value = self.visit_expr(op.value)

        if value is op.value:
            return op

        return tvm.tirx.Bind(op.var, value, op.span)

    def visit_attr_(self, op):
        """Mutator implementation for AttrStmt."""
        value = self.visit_expr(op.value)
        body = self.visit_stmt(op.body)

        if value is op.value and body is op.body:
            return op

        return tvm.tirx.AttrStmt(op.node, op.attr_key, value, body, op.span)

    def visit_if_then_else_(self, op):
        """Mutator implementation for IfThenElse."""
        condition = self.visit_expr(op.condition)
        then_case = self.visit_stmt(op.then_case)
        else_case = self.visit_stmt(op.else_case) if op.else_case else None

        if condition is op.condition and then_case is op.then_case and else_case is op.else_case:
            return op

        return tvm.tirx.IfThenElse(condition, then_case, else_case, op.span)

    def visit_for_(self, op):
        """Mutator implementation for For."""
        min_val = self.visit_expr(op.min)
        extent = self.visit_expr(op.extent)
        step = self.visit_expr(op.step) if op.step is not None else None
        body = self.visit_stmt(op.body)

        if min_val is op.min and extent is op.extent and step is op.step and body is op.body:
            return op

        return tvm.tirx.For(
            op.loop_var,
            min_val,
            extent,
            op.kind,
            body,
            op.thread_binding,
            op.annotations,
            step,
            op.span,
        )

    def visit_while_(self, op):
        """Mutator implementation for While."""
        condition = self.visit_expr(op.condition)
        body = self.visit_stmt(op.body)

        if condition is op.condition and body is op.body:
            return op

        return tvm.tirx.While(condition, body, op.span)

    def visit_break_(self, op):
        """Mutator implementation for Break."""
        return op

    def visit_continue_(self, op):
        """Mutator implementation for Continue."""
        return op

    def visit_allocate_(self, op):
        """Mutator implementation for Allocate."""
        extents = [self.visit_expr(extent) for extent in op.extents]
        body = self.visit_stmt(op.body)
        condition = self.visit_expr(op.condition)

        extents_changed = any(old is not new for old, new in zip(op.extents, extents))

        if not extents_changed and body is op.body and condition is op.condition:
            return op

        return tvm.tirx.Allocate(
            op.buffer_var, op.dtype, extents, condition, body, op.annotations, op.span
        )

    def visit_allocate_const_(self, op):
        """Mutator implementation for AllocateConst."""
        extents = [self.visit_expr(extent) for extent in op.extents]
        body = self.visit_stmt(op.body)

        extents_changed = any(old is not new for old, new in zip(op.extents, extents))

        if not extents_changed and body is op.body:
            return op

        # Create the data_or_idx parameter based on what's available
        if op.data is not None:
            data_or_idx = op.data
        elif op.irmod_storage_idx is not None:
            data_or_idx = op.irmod_storage_idx
        else:
            data_or_idx = None

        return tvm.tirx.AllocateConst(
            op.buffer_var, op.dtype, extents, data_or_idx, body, op.annotations, op.span
        )

    def visit_decl_buffer_(self, op):
        """Mutator implementation for DeclBuffer."""
        if hasattr(op, "body"):
            body = self.visit_stmt(op.body)
            if body is op.body:
                return op
            return tvm.tirx.DeclBuffer(op.buffer, body, op.span)
        return op

    def visit_buffer_store_(self, op):
        """Mutator implementation for BufferStore."""
        value = self.visit_expr(op.value)
        indices = [self.visit_expr(idx) for idx in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None

        indices_changed = any(old is not new for old, new in zip(op.indices, indices))

        if value is op.value and not indices_changed and predicate is op.predicate:
            return op

        return tvm.tirx.BufferStore(op.buffer, value, indices, predicate, op.span)

    def visit_buffer_realize_(self, op):
        """Mutator implementation for BufferRealize."""
        bounds = []
        bounds_changed = False

        for r in op.bounds:
            new_min = self.visit_expr(r.min)
            new_extent = self.visit_expr(r.extent)

            if new_min is not r.min or new_extent is not r.extent:
                bounds_changed = True
                bounds.append(tvm.ir.Range(new_min, new_extent))
            else:
                bounds.append(r)

        condition = self.visit_expr(op.condition)
        body = self.visit_stmt(op.body)

        if not bounds_changed and condition is op.condition and body is op.body:
            return op

        return tvm.tirx.BufferRealize(op.buffer, bounds, condition, body, op.span)

    def visit_assert_(self, op):
        """Mutator implementation for AssertStmt."""
        condition = self.visit_expr(op.condition)
        message_parts = []
        message_parts_changed = False
        for message_part in op.message_parts:
            if isinstance(message_part, PrimExpr):
                new_message_part = self.visit_expr(message_part)
                if new_message_part is not message_part:
                    message_parts_changed = True
                message_parts.append(new_message_part)
            else:
                message_parts.append(message_part)

        if condition is op.condition and not message_parts_changed:
            return op

        return tvm.tirx.AssertStmt(op.kind, condition, message_parts, op.span)

    def visit_producer_store_(self, op):
        """Mutator implementation for ProducerStore."""
        value = self.visit_expr(op.value)
        indices = [self.visit_expr(idx) for idx in op.indices]

        indices_changed = any(old is not new for old, new in zip(op.indices, indices))

        if value is op.value and not indices_changed:
            return op

        return tvm.tirx.ProducerStore(op.producer, value, indices, op.span)

    def visit_producer_realize_(self, op):
        """Mutator implementation for ProducerRealize."""
        bounds = []
        bounds_changed = False

        for r in op.bounds:
            new_min = self.visit_expr(r.min)
            new_extent = self.visit_expr(r.extent)

            if new_min is not r.min or new_extent is not r.extent:
                bounds_changed = True
                bounds.append(tvm.ir.Range(new_min, new_extent))
            else:
                bounds.append(r)

        condition = self.visit_expr(op.condition)
        body = self.visit_stmt(op.body)

        if not bounds_changed and condition is op.condition and body is op.body:
            return op

        return tvm.tirx.ProducerRealize(
            op.producer, bounds, condition, body, op.storage_scope, op.span
        )

    def visit_prefetch_(self, op):
        """Mutator implementation for Prefetch."""
        bounds = []
        bounds_changed = False

        for r in op.bounds:
            new_min = self.visit_expr(r.min)
            new_extent = self.visit_expr(r.extent)

            if new_min is not r.min or new_extent is not r.extent:
                bounds_changed = True
                bounds.append(tvm.ir.Range(new_min, new_extent))
            else:
                bounds.append(r)

        if not bounds_changed:
            return op

        return tvm.tirx.Prefetch(op.buffer, bounds, op.span)

    def visit_seqstmt_(self, op):
        """Mutator implementation for SeqStmt."""
        new_seq = []
        changed = False

        for stmt in op.seq:
            new_stmt = self.visit_stmt(stmt)
            if new_stmt is not stmt:
                changed = True
            if isinstance(new_stmt, tvm.tirx.SeqStmt):
                # Flatten nested SeqStmt
                new_seq.extend(new_stmt.seq)
                changed = True
            else:
                new_seq.append(new_stmt)

        if not changed:
            return op

        if len(new_seq) == 1:
            return new_seq[0]

        return tvm.tirx.SeqStmt(new_seq, op.span)

    def visit_evaluate_(self, op):
        """Mutator implementation for Evaluate."""
        value = self.visit_expr(op.value)

        if value is op.value:
            return op

        return tvm.tirx.Evaluate(value, op.span)

    def visit_block_(self, op):
        """Mutator implementation for Block."""
        # Process iter_vars
        iter_vars = []
        iter_vars_changed = False

        for iv in op.iter_vars:
            old_dom = iv.dom
            new_min = self.visit_expr(old_dom.min)
            new_extent = self.visit_expr(old_dom.extent)

            if new_min is not old_dom.min or new_extent is not old_dom.extent:
                iter_vars_changed = True
                new_dom = tvm.ir.Range(new_min, new_extent)
                iter_vars.append(tvm.tirx.IterVar(new_dom, iv.var, iv.iter_type, iv.thread_tag))
            else:
                iter_vars.append(iv)

        # Process reads/writes buffer regions
        def _mutate_buffer_regions(regions):
            new_regions = []
            regions_changed = False

            for region in regions:
                new_ranges = []
                ranges_changed = False

                for r in region.region:
                    new_min = self.visit_expr(r.min)
                    new_extent = self.visit_expr(r.extent)

                    if new_min is not r.min or new_extent is not r.extent:
                        ranges_changed = True
                        new_ranges.append(tvm.ir.Range(new_min, new_extent))
                    else:
                        new_ranges.append(r)

                if ranges_changed:
                    regions_changed = True
                    new_regions.append(tvm.tirx.BufferRegion(region.buffer, new_ranges))
                else:
                    new_regions.append(region)

            return new_regions, regions_changed

        reads, reads_changed = _mutate_buffer_regions(op.reads)
        writes, writes_changed = _mutate_buffer_regions(op.writes)

        # Process match buffers
        match_buffers = []
        match_buffers_changed = False

        for match_buffer in op.match_buffers:
            source_region = match_buffer.source
            new_ranges = []
            ranges_changed = False

            for r in source_region.region:
                new_min = self.visit_expr(r.min)
                new_extent = self.visit_expr(r.extent)

                if new_min is not r.min or new_extent is not r.extent:
                    ranges_changed = True
                    new_ranges.append(tvm.ir.Range(new_min, new_extent))
                else:
                    new_ranges.append(r)

            if ranges_changed:
                match_buffers_changed = True
                new_source = tvm.tirx.BufferRegion(source_region.buffer, new_ranges)
                match_buffers.append(tvm.tirx.MatchBufferRegion(match_buffer.buffer, new_source))
            else:
                match_buffers.append(match_buffer)

        # Process init and body
        init = self.visit_stmt(op.init) if op.init is not None else None
        body = self.visit_stmt(op.body)

        # Check if anything changed
        if (
            not iter_vars_changed
            and not reads_changed
            and not writes_changed
            and not match_buffers_changed
            and (init is op.init or (init is None and op.init is None))
            and body is op.body
        ):
            return op
        return tvm.tirx.SBlock(
            iter_vars,
            reads,
            writes,
            op.name_hint,
            body,
            init,
            op.alloc_buffers,
            match_buffers,
            op.annotations,
        )

    def visit_block_realize_(self, op):
        """Mutator implementation for BlockRealize."""
        iter_values = [self.visit_expr(val) for val in op.iter_values]
        predicate = self.visit_expr(op.predicate)
        block = self.visit_stmt(op.block)

        iter_values_changed = any(old is not new for old, new in zip(op.iter_values, iter_values))

        if not iter_values_changed and predicate is op.predicate and block is op.block:
            return op

        if not isinstance(block, tvm.tirx.SBlock):
            raise TypeError(f"Expected SBlock, but got {type(block)}")

        return tvm.tirx.SBlockRealize(iter_values, predicate, block)

    def visit_exec_scope_stmt_(self, op):
        """Mutator implementation for ExecScopeStmt."""
        body = self.visit_stmt(op.body)

        if body is op.body:
            return op

        return tvm.tirx.ExecScopeStmt(op.exec_scope, body, op.span)

    def visit_op_call_(self, op):
        """Mutator implementation for TilePrimitiveCall."""
        new_args = []
        args_changed = False

        for arg in op.args:
            if isinstance(arg, PrimExpr):
                new_arg = self.visit_expr(arg)
            elif isinstance(arg, tvm.tirx.Stmt):
                new_arg = self.visit_stmt(arg)
            elif isinstance(arg, tvm.tirx.BufferRegion):
                new_arg = self.visit_buffer_region_(arg)
            else:
                new_arg = arg

            if new_arg is not arg:
                args_changed = True
            new_args.append(new_arg)

        # Also mutate PrimExpr values in the config map
        new_config = {}
        config_changed = False
        for key, value in op.config.items():
            if isinstance(value, PrimExpr):
                new_value = self.visit_expr(value)
            elif isinstance(value, tvm.tirx.Stmt):
                new_value = self.visit_stmt(value)
            else:
                new_value = value
            if new_value is not value:
                config_changed = True
            new_config[key] = new_value

        if not args_changed and not config_changed:
            return op

        return tvm.tirx.TilePrimitiveCall(
            *new_args, op=op.op, workspace=op.workspace, config=new_config, dispatch=op.dispatch
        )

    def visit_buffer_region_(self, op):
        """Mutator implementation for BufferRegion."""

        def _mutate_range(range):
            new_min = self.visit_expr(range.min)
            new_extent = self.visit_expr(range.extent)

            if new_min is range.min and new_extent is range.extent:
                return range
            else:
                return Range.from_min_extent(new_min, new_extent)

        region = [_mutate_range(r) for r in op.region]

        if all(old_r is new_r for old_r, new_r in zip(op.region, region)):
            return op
        else:
            return tvm.tirx.BufferRegion(op.buffer, region)

    def visit_alloc_buffer_(self, op):
        """Mutator implementation for AllocBuffer."""
        if hasattr(op, "body"):
            body = self.visit_stmt(op.body)
            if body is op.body:
                return op
            return tvm.tirx.AllocBuffer(op.buffer, body, op.annotations, op.span)
        return op

    def __call__(self, stmt):
        """Call mutator on statement.

        Parameters
        ----------
        stmt : tvm.tirx.Stmt
            The statement to be mutated.

        Returns
        -------
        result : tvm.tirx.Stmt
            The mutated statement
        """
        return self.visit_stmt(stmt)


class StmtExprVisitor(StmtVisitor, ExprVisitor):
    """A visitor over both statements and expressions.

    This class inherits from both StmtVisitor and ExprVisitor to recursively visit
    both statements and expressions.
    """

    def __init__(self):
        StmtVisitor.__init__(self)
        self._stmt_dispatch_map = self._dispatch_map.copy()
        ExprVisitor.__init__(self)
        self._expr_dispatch_map = self._dispatch_map.copy()
        self._dispatch_map = {}
        self._dispatch_map.update(self._stmt_dispatch_map)
        self._dispatch_map.update(self._expr_dispatch_map)

    def visit_expr(self, expr):
        """Visit an expression used in a statement.

        Parameters
        ----------
        expr : PrimExpr
            The expression to be visited.
        """
        return ExprVisitor.visit_expr(self, expr)


class StmtExprMutator(StmtMutator, ExprMutator):
    """A mutator over both statements and expressions.

    This class inherits from both StmtMutator and ExprMutator to recursively transform
    both statements and expressions.
    """

    def __init__(self):
        StmtMutator.__init__(self)
        self._stmt_dispatch_map = self._dispatch_map.copy()
        ExprMutator.__init__(self)
        self._expr_dispatch_map = self._dispatch_map.copy()
        self._dispatch_map = {}
        self._dispatch_map.update(self._stmt_dispatch_map)
        self._dispatch_map.update(self._expr_dispatch_map)

    def visit_expr(self, expr):
        """Mutate an expression used in a statement.

        Parameters
        ----------
        expr : PrimExpr
            The expression to be mutated.

        Returns
        -------
        result : PrimExpr
            The mutated expression.
        """
        return ExprMutator.visit_expr(self, expr)


def ir_transform(stmt, preorder, postorder, only_enable=None):
    """Recursively visit and transform ir nodes in post DFS order.

    Parameters
    ----------
    stmt : tvm.tirx.Stmt
        The input to be transformed.

    preorder: function
        The function called in before recursive mutation
        If preorder returns None, then the transform will proceed to recursive call.
        If preorder returns a not None tvm.tirx.Stmt/Expr, the transformer will simply return it and
        won't do further recursion.

    postorder : function
        The function called after recursive mutation.

    only_enable : Optional[List[str]]
        List of types that we only enable.

    Returns
    -------
    result : tvm.tirx.Stmt
        The result.
    """
    return _ffi_api.IRTransform(stmt, preorder, postorder, only_enable)  # type: ignore


def post_order_visit(stmt, fvisit):
    """Recursively visit the ir in post DFS order node, apply fvisit
       Each node is guaranteed to be visited only once.

    Parameters
    ----------
    fvisit: function
        The visitor function.
    """
    return _ffi_api.PostOrderVisit(stmt, fvisit)  # type: ignore


def pre_order_visit(stmt, fvisit):
    """Recursive pre-order visit on stmt AST, applying fvisit on each node.
       If fvisit returns False, it won't visit the children of the node.

    Parameters
    ----------
    fvisit: function of the signature Object -> bool
        The visitor function.
    """
    return _ffi_api.PreOrderVisit(stmt, fvisit)  # type: ignore


def substitute(node, vmap):
    """Substitute the var specified by vmap.

    Parameters
    ----------
    node: ObjectRef
        The input.

    vmap : Dict[Var, PrimExpr]
        The variable mapping.

    Returns
    -------
    result : tvm.tirx.Stmt
        The result.
    """
    return _ffi_api.Substitute(node, vmap)  # type: ignore


def renew_defs(func: PrimFunc):
    """Re-generate the definition nodes for a TIR, including VarDef, BufferDef.
    This pass works as a simple DeepCopy to duplicate a function with different Vars and
    Buffers but the same behavior

    Parameters
    ----------
    func: PrimFunc
        The input function

    Returns
    -------
    result : PrimFunc
        The new generated func.
    """
    return _ffi_api.RenewDefs(func)  # type: ignore
