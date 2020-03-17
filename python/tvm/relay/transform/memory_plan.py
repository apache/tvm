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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks
"""
A pass for manifesting explicit memory allocations.
"""
from typing import Optional, Dict
import attr

from ..expr_functor import ExprMutator
from .. import op, expr
from ..function import Function
from ... import register_func, ir, cpu
from ..._ffi.runtime_ctypes import TVMContext
from . import function_pass


def is_primitive(call):
    return (
        hasattr(call, "op")
        and hasattr(call.op, "attrs")
        and hasattr(call.op.attrs, "Primitive")
        and int(call.op.attrs.Primitive) == 1
    )


@attr.s(auto_attribs=True)
class Region:
    """
    Represents a control-free allocation region.

    The below pass groups sets of allocations into regions,
    then replaces the region with a single allocation.
    """
    var: expr.Var
    size: expr.Expr
    alignment: Optional[expr.Expr]
    dtype: Optional[str]
    ctx: TVMContext
    offsets: Dict[expr.Var, expr.Expr] = {}

    def grow(
            self, old_storage: expr.Var,
            size: expr.Expr, alignment: expr.Expr,
            ctx: TVMContext,
            dtype: str) -> None:
        """Grow the region by a given allocation as well as track the old storage
           for later rewriting the program to use the allocated region.
        """
        if self.dtype:
            assert self.dtype == dtype, "must have matching dtypes in a region"
        else:
            self.dtype = dtype

        if self.alignment:
            assert ir.structural_equal(
                self.alignment, alignment
            ), "must have matching alignments in a region"
        else:
            self.alignment = alignment

        if self.ctx:
            assert (self.ctx.device_type == ctx.device_type and
                    self.ctx.device_id == ctx.device_id), "must have matching context"
        else:
            assert ctx
            self.ctx = ctx

        # Record the offset at which we allocate the storage.
        self.offsets[old_storage] = self.size

        self.size = self.size + size

    def to_expr(self) -> expr.Expr:
        if self.ctx is None:
            self.ctx = cpu(0)

        return op.memory.alloc_storage(self.size, self.alignment, self.ctx, self.dtype)


def iterative_let(let, each_binding, kont):
    bindings = []
    while isinstance(let, expr.Let):
        lhs = let.var
        rhs = let.value
        bindings.append(each_binding(lhs, rhs))
        let = let.body

    return kont(bindings, let)


def mk_let(bindings, body):
    for var, value in reversed(bindings):
        assert var
        assert value
        assert body
        body = expr.Let(var, value, body)
    return body


class StorageCoalesce(ExprMutator):
    """
    A pass for coalescing allocations into region/arena allocations.

    After this pass each allocation comes from the same backing storage,
    but will never overlap even in time, i.e. the allocations are just
    packed into a contiguous block of memory.

    A secondary part of memory planning will perform liveness analysis to
    overlap these in time, i.e when an early tensor dies we will attempt
    to reuse its slot.
    """

    def __init__(self):
        super().__init__()
        self.regions = []

    def enter_scope(self):
        zero = expr.const(0, dtype="int64")
        region_var = expr.var(f"region{len(self.regions)}")
        region = Region(region_var, zero, None, None, None)
        self.regions.append(region)

    def exit_scope(self, body: expr.Expr) -> expr.Expr:
        """When leaving a scope build a region allocation for the scope."""
        region = self.regions.pop()
        if len(region.offsets) == 0:
            return body
        else:
            storage_expr = region.to_expr()
            assert storage_expr, "can not be None"
            assert region.var
            assert storage_expr
            assert body
            return expr.Let(region.var, storage_expr, body)

    def current_region(self) -> Region:
        return self.regions[-1]

    def visit_function(self, fn):
        """Transform the function body to use region allocation scheme."""
        func = fn
        if func.attrs and getattr(func.attrs, "Primitive", 0) == 1:
            return func
        else:
            self.enter_scope()
            body = self.visit(func.body)
            body = self.exit_scope(body)
            return Function(
                func.params,
                body,
                func.ret_type,
                func.type_params,
                func.attrs,
            )

    def visit_if(self, ite):
        self.enter_scope()
        true_branch = self.visit(ite.true_branch)
        true_branch = self.exit_scope(true_branch)

        self.enter_scope()
        false_branch = self.visit(ite.false_branch)
        false_branch = self.exit_scope(false_branch)

        return expr.If(ite.cond, true_branch, false_branch)

    def visit_let(self, let):
        def _each_binding(lhs, rhs):
            if isinstance(rhs, expr.Call) and rhs.op == op.op.get(
                    "memory.alloc_storage"
            ):
                return self.process_alloc_storage(lhs, rhs)
            elif isinstance(rhs, expr.Call) and rhs.op == op.op.get(
                    "memory.alloc_tensor"
            ):
                return self.process_alloc_tensor(lhs, rhs)
            else:
                return lhs, rhs

        result = iterative_let(let, _each_binding, mk_let)
        assert result
        return result

    def process_alloc_storage(self, lhs, call):
        size, alignment = call.args
        dtype = call.attrs.dtype
        ctx = TVMContext(call.attrs.device_type, call.attrs.device_id)
        region = self.current_region()
        region.grow(lhs, size, alignment, ctx, dtype)
        return lhs, region.var

    def process_alloc_tensor(self, lhs, call):
        region = self.current_region()
        storage, old_offset, shape = call.args
        offset = region.offsets[storage]
        assert (
            old_offset.data.asnumpy().item() == 0
        ), "no offsets should yet be allocated"
        return (
            lhs,
            expr.Call(call.op, [region.var, offset, shape], call.attrs),
        )

@function_pass(opt_level=0)
class MemoryPlan:
    """An explicit pass wrapper around ManifestAlloc."""

    def transform_function(self, func, mod, _):
        mod.import_from_std("core.rly")
        sc = StorageCoalesce()
        before = func
        func = sc.visit(func)
        return before


register_func("relay.transform.MemoryPlan", MemoryPlan)
