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
from __future__ import annotations
import attr
import numpy as np
from ..expr_functor import ExprMutator, ExprVisitor
from ..function import Function
from ..scope_builder import ScopeBuilder
from . import transform
from .. import op
from ... import DataType, register_func
from .. import ty, expr
from ..backend import compile_engine
from ..._ffi.runtime_ctypes import TVMContext
from ...import cpu
from typing import Optional
from collections import defaultdict

def is_primitive(call):
    return hasattr(call, 'op') and hasattr(call.op, 'attrs') and \
           hasattr(call.op.attrs, 'Primitive') and int(call.op.attrs.Primitive) == 1

def iterative_let(let, each_binding, kont):
    bindings = []
    while isinstance(let, expr.Let):
        lhs = let.var
        rhs = let.value
        bindings.append(each_binding(lhs, rhs))
        let = let.body

    return kont(bindings, let)


@attr.s(auto_attribs=True, hash=False, eq=False)
class DeviceDomain:
    domain: Optional[TVMContext]

    def join(self, other: DeviceDomain) -> DeviceDomain:
        if self.domain is None and other.domain is None:
            return self
        elif self.domain is None:
            return other
        elif other.domain is None:
            return self
        elif (self.domain.device_type == other.domain.device_type and
              self.domain.device_id == other.domain.device_id):
            return self
        else:
            import pdb; pdb.set_trace()
            raise Exception("all expressions must have a singular device")

    def __hash__(self):
        if self.domain is None:
            return id(self)
        else:
            return hash((self.domain.device_type, self.domain.device_id))

    def __eq__(self, other):
        if self.domain is None and other.domain is None:
            return id(self) == id(other)
        else:
            return self.domain == other.domain

def bottom():
    return DeviceDomain(None)

def device_type(ctx):
    return DeviceDomain(ctx)

class ContextAnalysis(ExprVisitor):
    """Compute on which device each sub-expression will execute."""
    def __init__(self, default_device):
        super().__init__()
        self.expr_to_device = defaultdict(bottom)
        self.device_uf = {}
        self.default_device = default_device

    def lookup(self, device):
        while device in self.device_uf:
            device = self.device_uf[device]
        return device

    def unify(self, device_one, device_two):
        device_one = self.lookup(device_one)
        device_two = self.lookup(device_two)
        unified_device = device_one.join(device_two)
        if not device_one == unified_device:
            self.device_uf[device_one] = unified_device
        if not device_two == unified_device:
            self.device_uf[device_two] = unified_device
        return unified_device

    def unify_expr(self, expr1, expr2):
        """Compute the device type of both expressions and unify them."""
        return self.unify(self.device_for(expr1), self.device_for(expr2))

    def device_for(self, expr):
        return self.lookup(self.expr_to_device[expr])

    def visit_let(self, let):
        self.unify(self.device_for(let.var), self.device_for(let.value))
        self.unify_expr(let, let.body)
        super().visit_let(let)

    def visit_function(self, func):
        self.unify(self.device_for(func), self.device_for(func.body))
        super().visit_function(func)

    def visit_var(self, var):
        self.device_for(var)

    def device_copy(self, inp, output, src_dev_type, dst_dev_type):
        src_dev_type = device_type(TVMContext(src_dev_type, 0))
        self.unify(self.device_for(inp), src_dev_type)
        dst_dev_type = device_type(TVMContext(dst_dev_type, 0))
        self.unify(self.device_for(output), dst_dev_type)

    def unify_call(self, func, inputs, outputs):
        # if func == op.op.get("memory.alloc_tensor"):
        #     import pdb; pdb.set_trace()
        device = bottom()
        for arg in inputs:
            device = self.unify(device, self.device_for(arg))

        device = self.unify(device, self.device_for(func))

        for out in outputs:
            device = self.unify(device, self.device_for(out))

        return device

    def visit_call(self, call):
        if call.op == op.op.get("device_copy"):
            (input_tensor,) = call.args
            self.device_copy(input_tensor, call, call.attrs.src_dev_type, call.attrs.dst_dev_type)
        elif call.op == op.op.get("memory.alloc_storage"):
            self.unify(self.device_for(call), device_type(TVMContext(call.attrs.device_type, call.attrs.device_id)))
        elif call.op == op.op.get("memory.alloc_tensor"):
            storage = call.args[0]
            self.unify(self.device_for(storage), self.device_for(call))
        elif call.op == op.op.get("memory.invoke_tvm_op"):
            if call.args[0].body.op == op.op.get("device_copy"):
                input_tensor = call.args[1][0]
                output_tensor = call.args[2][0]
                self.device_copy(input_tensor, output_tensor, call.attrs.src_dev_type, call.attrs.dst_dev_type)
            else:
                self.unify_call(call.args[0], call.args[1].fields, call.args[2].fields)
                super().visit_call(call)
        elif isinstance(call.op, Function):
            device = bottom()
            for arg in call.args:
                self.visit(arg)
                device = self.unify(device, self.device_for(arg))

            for param in call.op.params:
                self.visit(param)
                device = self.unify(device, self.device_for(param))

            out_device = self.device_for(call.op)
            self.unify(self.device_for(call), out_device)
            super().visit_call(call)
        else:
            self.unify_call(call.op, call.args, [call])
            super().visit_call(call)

    def results(self):
        results = {}
        for exp in self.expr_to_device:
            device = self.lookup(self.expr_to_device[exp])
            if device.domain is None:
                results[exp] = self.default_device
            else:
                results[exp] = device.domain

        return results

def mk_analysis_annotator(results):
    def _annotator(exp):
        if exp in results:
            return f"<{results[exp]}>"
        else:
            return ""

    return _annotator
