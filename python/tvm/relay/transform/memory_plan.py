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
from collections import defaultdict
import attr

from ..expr_functor import ExprMutator
from .. import op, expr
from ..function import Function
from ... import register_func, ir, cpu
from ..._ffi.runtime_ctypes import Device
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
    device: Device
    offsets: Dict[expr.Var, Tuple[expr.Expr, expr.Expr]]

    @staticmethod
    def empty(region_no):
        zero = expr.const(0, dtype="int64")
        assert len(zero.data.shape) == 0
        region_var = expr.var(f"region{region_no}")
        return Region(region_var, zero, None, None, None, {})

    def grow(
        self,
        old_storage: expr.Var,
        size: expr.Expr,
        alignment: expr.Expr,
        dev: Device,
        dtype: str,
    ) -> None:
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

        if self.device:
            assert (
                self.device.device_type == dev.device_type
                and self.device.device_id == dev.device_id
            ), "must have matching device"
        else:
            assert dev
            self.device = dev

        new_size = (
            (size + self.alignment - expr.const(1, "int64")) / self.alignment * self.alignment
        )

        # Record the offset at which we allocate the storage.
        offset_var: expr.RelayExpr = expr.var(f"offset{len(self.offsets)}")
        self.offsets[old_storage] = (offset_var, self.size)

        self.size = self.size + new_size

    def offset_for(self, alloc: expr.Expr) -> expr.Expr:
        return self.offsets.get(alloc, [None])[0]

    def to_expr(self, body: expr.Expr) -> expr.Expr:
        """
        Generate the prelude code for a region, wrapping the body in it.

        The prelude contains the single allocation for a region, and
        all offset computations.
        """

        if self.device is None:
            self.device = cpu(0)

        # Generate bindings for each and every size computation
        # we must do this to maintain ANF.
        bindings: List[Tuple[expr.Expr, expr.Expr]] = []

        # First compute the total size.
        total_size = expr.var(f"total_size{hash(body)}")
        bindings.append((total_size, self.size))

        # Allocate the entire region with a single call.
        alloc = op.memory.alloc_storage(total_size, self.alignment, self.device, self.dtype)
        bindings.append((self.var, alloc))

        # Generate variables which contain all of the offset math.
        # Ensure we constant evaluate away all the math here.
        #
        # In theory we can support dynamic offsets but this
        # requires another round of memory planning and
        # potentially colaescing.
        for alloc in self.offsets:
            (var, offset) = self.offsets[alloc]
            bindings.append((var, offset))

        body = mk_let(bindings, body)
        return body


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


def const_eval(mod, exp):
    mod = IRModule.from_expr(exp, type_defs=mod.type_definitions)
    mod = transform.FoldConstant()(mod)
    return mod["main"]


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
        region_no = len(self.regions)
        self.regions.append(defaultdict(lambda: Region.empty(region_no)))

    def exit_scope(self, body: expr.Expr) -> expr.Expr:
        """When leaving a scope build a region allocation for the scope."""
        dtype_region = self.regions.pop()
        for _, region in reversed(list(dtype_region.items())):
            if len(region.offsets) != 0:
                body = region.to_expr(body)

        return body

    def current_region(self, dtype) -> Region:
        current_scope = self.regions[-1]
        return current_scope[dtype]

    def new_region_and_offset(self, old_storage):
        for dtype_region in reversed(self.regions):
            for dtype in dtype_region:
                region = dtype_region[dtype]
                offset = region.offset_for(old_storage)
                if offset:
                    return region, offset

        raise Exception("could not find offset in any valid region")

    def visit_function(self, fn):
        """Transform the function body to use region allocation scheme."""
        func = fn
        if getattr(func.attrs, "Primitive", 0) == 1:
            return super().visit_function(func)
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

    def mk_let(self, dynamic_regions):
        """Let bind the dynamic regions"""

        def _mk_let(bindings, body):
            for var, value in reversed(bindings):
                assert var
                assert value is not None
                assert body
                body = expr.Let(var, value, body)
                if var in dynamic_regions:
                    body = self.exit_scope(body)

            return body

        return _mk_let

    def visit_let(self, let):
        dynamic_regions = []

        def _each_binding(lhs, rhs):
            if isinstance(rhs, expr.Call) and rhs.op == op.op.get("memory.alloc_storage"):
                return self.process_alloc_storage(dynamic_regions, lhs, rhs)
            elif isinstance(rhs, expr.Call) and rhs.op == op.op.get("memory.alloc_tensor"):
                return self.process_alloc_tensor(lhs, rhs)
            else:
                return lhs, rhs

        result = iterative_let(let, _each_binding, self.mk_let(dynamic_regions))
        assert result
        return result

    def process_alloc_storage(self, dynamic_regions, lhs, call):
        """Process alloc_storage"""
        size, alignment = call.args
        dtype = call.attrs.dtype
        dev = Device(call.attrs.device_type, call.attrs.device_id)

        if not isinstance(size, expr.Constant):
            self.enter_scope()
            dynamic_regions.append(lhs)
        else:
            # A new scope is created when entering a new region with different
            # device device.
            region = self.current_region(dtype)
            if region.device and region.device.device_type != dev.device_type:
                self.enter_scope()
                dynamic_regions.append(lhs)

        region = self.current_region(dtype)
        region.grow(lhs, size, alignment, dev, dtype)
        return lhs, region.var

    def process_alloc_tensor(self, lhs, call):
        """Process alloc tensor. Region and offset are computed"""
        storage, old_offset, shape = call.args
        region, offset = self.new_region_and_offset(storage)

        assert old_offset.data.numpy().item() == 0, "no offsets should yet be allocated"
        return (
            lhs,
            expr.Call(call.op, [region.var, offset, shape], call.attrs),
        )


class LiftConst(ExprMutator):
    """An internal pass to lift constants to the top level of function."""

    def __init__(self):
        self.i = 0
        self.constants = []
        self.top_level = True
        super().__init__()

    def visit_constant(self, const):
        var = expr.var(f"const{self.i}")
        self.i += 1
        self.constants.append((var, const))
        return var

    def visit_function(self, fn):
        if int(getattr(fn.attrs, "Primitive", 0)) == 1:
            return fn

        outer_constant = self.constants
        self.constants = []
        # Populates self.constants.
        body = self.visit(fn.body)
        body = mk_let(self.constants, body)
        self.constants = outer_constant

        return Function(fn.params, body, fn.ret_type, fn.type_params, fn.attrs)

    def visit_let(self, let):
        bindings = []
        while isinstance(let, expr.Let):
            new_var = self.visit(let.var)
            new_val = self.visit(let.value)
            bindings.append((new_var, new_val))
            let = let.body

        new_body = self.visit(let)
        return mk_let(bindings, new_body)


@function_pass(opt_level=0)
class MemoryPlan:
    """An explicit pass wrapper around StorageCoalesce."""

    def transform_function(self, func, mod, _):
        mod.import_from_std("core.rly")
        sc = StorageCoalesce()
        func = sc.visit(func)
        return func


register_func("relay.transform.MemoryPlan", MemoryPlan)


@function_pass(opt_level=0)
class LiftConstants:
    """An explicit pass wrapper around LiftConst."""

    def transform_function(self, func, mod, _):
        mod.import_from_std("core.rly")
        func = LiftConst().visit(func)
        return func


register_func("relay.transform.LiftConstants", LiftConstants)
