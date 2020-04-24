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
from typing import Optional, Dict, List, Tuple
import attr

from ..expr_functor import ExprMutator
from .. import op, expr
from ..function import Function
from ... import register_func, ir, cpu
from ..._ffi.runtime_ctypes import TVMContext
from ... import IRModule
from .. import transform
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
    offsets: Dict[expr.Var, Tuple[expr.Expr, expr.Expr]]

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
        offset_var: expr.RelayExpr = expr.var(f"offset{len(self.offsets)}")
        self.offsets[old_storage] = (offset_var, self.size)

        self.size = self.size + size

    def offset_for(self, alloc: expr.Expr) -> expr.Expr:
        return self.offsets[alloc][0]

    def to_expr(self, body: expr.Expr) -> expr.Expr:
        """
        Generate the prelude code for a region, wrapping the body in it.

        The prelude contains the single allocation for a region, and
        all offset computations.
        """

        if self.ctx is None:
            self.ctx = cpu(0)

        # Generate bindings for each and every size computation
        # we must do this to maintain ANF.
        bindings: List[Tuple[expr.Expr, expr.Expr]] = []

        # First compute the total size.
        total_size = expr.var("total_size")
        bindings.append((total_size, const_eval(self.size)))

        # Allocate the entire region with a single call.
        alloc = op.memory.alloc_storage(total_size, self.alignment, self.ctx, self.dtype)
        bindings.append((self.var, alloc))

        # Generate variables which contain all of the offset math.
        # Ensure we constant evaluate away all the math here.
        #
        # In theory we can support dynamic offsets but this
        # requires another round of memory planning and
        # potentially colaescing.
        for alloc in self.offsets:
            (var, offset) = self.offsets[alloc]
            offset = const_eval(offset)
            bindings.append((var, offset))

        return mk_let(bindings, body)


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

def const_eval(exp):
    mod = IRModule.from_expr(exp)
    mod = transform.FoldConstant()(mod)
    return mod["main"].body

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

    def enter_scope(self) -> None:
        zero = expr.const(0, dtype="int64")
        region_var = expr.var(f"region{len(self.regions)}")
        region = Region(region_var, zero, None, None, None, {})
        self.regions.append(region)

    def exit_scope(self, body: expr.Expr) -> expr.Expr:
        """When leaving a scope build a region allocation for the scope."""
        region = self.regions.pop()
        if len(region.offsets) == 0:
            return body
        else:
            return region.to_expr(body)

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
        offset = region.offset_for(storage)
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
        print(func)
        return before


register_func("relay.transform.MemoryPlan", MemoryPlan)
