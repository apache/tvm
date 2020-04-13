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

    def visit_call(self, call):
        if call.op == op.op.get("device_copy"):
            (input_tensor,) = call.args
            src_dev_type = device_type(TVMContext(call.attrs.src_dev_type, 0))
            self.unify(self.device_for(input_tensor), src_dev_type)
            dst_dev_type = device_type(TVMContext(call.attrs.dst_dev_type, 0))
            self.unify(self.device_for(call), dst_dev_type)
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
            device = bottom()
            for arg in call.args:
                self.visit(arg)
                device = self.unify(device, self.device_for(arg))

            device = self.unify(device, self.device_for(call.op))
            self.unify(device, self.device_for(call))
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

# TODO(@jroesch): port to c++ and unify with existing code
class LinearizeRetType:
    """A linear view of a Relay type, handles a linear order
       for nested tuples, and tensor types.
    """

    def __init__(self, typ):
        """Initialize the linearizer."""
        self.typ = typ

    def unpack(self):
        """Return the linear representation of the type."""
        def _unpack(typ, out):
            # TODO(@jroesch): replace with new flattening pass
            if isinstance(typ, ty.TensorType):
                out.append(typ)
            elif isinstance(typ, ty.TupleType):
                for field_ty in typ.fields:
                    _unpack(field_ty, out)
            else:
                raise Exception("unsupported Relay type: {0}".format(typ))

        output = []
        _unpack(self.typ, output)
        return output

    def pack(self, seq):
        """Repack a linear type as a nested type."""
        def _pack(value, typ, out):
            if isinstance(typ, ty.TensorType):
                out.append(value)
            elif isinstance(typ, ty.TupleType):
                tuple_out = []
                for i, field_ty in enumerate(typ.fields):
                    _pack(value[i], field_ty, tuple_out)
                out.append(expr.Tuple(tuple_out))
            else:
                raise Exception("unsupported Relay type: {0}".format(typ))

        if len(seq) == 1:
            return seq[0]
        else:
            out = []
            _pack(seq, self.typ, out)
            assert len(out) == 1, "must return fully packed type"
            return out[0]


class ManifestAllocPass(ExprMutator):
    """A pass for explicitly manifesting all memory allocations in Relay."""

    def __init__(self, target_host, context_analysis):
        self.invoke_tvm = op.memory.invoke_tvm_op
        self.alloc_storage = op.memory.alloc_storage
        self.alloc_tensor = op.memory.alloc_tensor
        self.shape_func = op.memory.shape_func
        self.scopes = [ScopeBuilder()]
        self.target_host = target_host
        self.compute_dtype = "int64"
        self.context_analysis = context_analysis
        super().__init__()

    def get_context(self, expr):
        return self.context_analysis[expr]

    def current_scope(self):
        return self.scopes[-1]

    def shape_of(self, e):
        return op.shape_of(e, self.compute_dtype)

    def visit_tuple(self, tup):
        scope = self.current_scope()
        new_fields = []
        for field in tup.fields:
            field = self.visit(field)
            if isinstance(field, expr.Constant):
                field = scope.let('const', field)
            new_fields.append(field)
        return expr.Tuple(new_fields)

    def compute_alignment(self, dtype):
        dtype = DataType(dtype)
        align = (dtype.bits // 8) * dtype.lanes
        # MAGIC CONSTANT FROM device_api.h
        if align < 64:
            align = 64

        return expr.const(align, dtype="int64")

    def compute_storage_in_relay(self, shape, dtype):
        dtype = DataType(dtype)
        els = op.prod(shape)
        num = expr.const(dtype.bits * dtype.lanes, self.compute_dtype)
        num = num + expr.const(7, self.compute_dtype)
        div = expr.const(8, self.compute_dtype)
        return els * (num / div)

    def compute_storage(self, tensor_type):
        dtype = DataType(tensor_type.dtype)
        shape = [int(sh) for sh in tensor_type.shape]
        size = 1
        for sh in shape:
            size *= sh
        size *= (dtype.bits * dtype.lanes + 7) // 8
        return expr.const(size, dtype=self.compute_dtype)

    def make_static_allocation(self, scope, tensor_type, ctx, name_hint):
        """Allocate a tensor with a statically known shape."""
        shape = [int(sh) for sh in tensor_type.shape]
        if len(shape) == 0:
            shape = expr.const(np.array([]).astype(
                self.compute_dtype), dtype=self.compute_dtype)
        else:
            shape = expr.const(np.array(shape), dtype=self.compute_dtype)
        size = self.compute_storage(tensor_type)
        alignment = self.compute_alignment(tensor_type.dtype)
        dtype = tensor_type.dtype

        # Just need to pass the context here !
        print(ctx)
        sto = scope.let("storage_{0}".format(name_hint), self.alloc_storage(
            size, alignment, dtype))
        # TODO(@jroesch): There is a bug with typing based on the constant shape.
        tensor = self.alloc_tensor(sto, shape, dtype, tensor_type.shape)
        return scope.let("tensor_{0}".format(name_hint), tensor)

    def visit_let(self, let):
        scope = ScopeBuilder()

        self.scopes.append(scope)
        while isinstance(let, expr.Let):
            new_val = self.visit(let.value)
            scope.let(let.var, new_val)
            let = let.body

        new_body = self.visit(let)
        scope.ret(new_body)
        self.scopes.pop()

        return scope.get()

    def visit_call(self, call):
        if is_primitive(call):
            # Because we are in ANF we do not need to visit the arguments.
            scope = self.current_scope()
            new_args = [self.visit(arg) for arg in call.args]
            ins = expr.Tuple(new_args)
            ret_type = call.checked_type
            view = LinearizeRetType(ret_type)
            out_types = view.unpack()

            is_dynamic = ty.type_has_any(ret_type)
            # TODO(@jroesch): restore this code, more complex then it seems
            # for arg in call.args:
            #     is_dynamic = is_dynamic or arg.checked_type.is_dynamic()

            if is_dynamic:
                shape_func_ins = []
                engine = compile_engine.get()
                cfunc = engine.lower_shape_func(call.op, self.target_host)
                input_states = cfunc.shape_func_param_states

                is_inputs = []
                input_pos = 0
                for i, (arg, state) in enumerate(zip(new_args, input_states)):
                    state = int(state)
                    # Pass Shapes
                    if state == 2:
                        if isinstance(arg.type_annotation, ty.TupleType):
                            for j in range(len(arg.type_annotation.fields)):
                                let_in_arg = scope.let("in_arg_{0}".format(input_pos + j),
                                                       expr.TupleGetItem(arg, j))
                                sh_of = self.visit(self.shape_of(let_in_arg))
                                shape_func_ins.append(
                                    scope.let("in_shape_{0}".format(input_pos + j), sh_of))
                            input_pos += len(arg.type_annotation.fields)
                        else:
                            sh_of = self.visit(self.shape_of(arg))
                            shape_func_ins.append(
                                scope.let("in_shape_{0}".format(input_pos), sh_of))
                            input_pos += 1
                        is_inputs.append(0)
                    # Pass Inputs
                    elif state == 1:
                        new_arg = self.visit(arg)
                        shape_func_ins.append(
                            scope.let("in_shape_{0}".format(input_pos), new_arg))
                        input_pos += 1
                        is_inputs.append(1)
                    # TODO(@jroesch): handle 3rd case
                    else:
                        raise Exception("unsupported shape function input state")

                out_shapes = []
                for i, out in enumerate(cfunc.outputs):
                    tt = ty.TensorType(out.shape, out.dtype)
                    alloc = self.make_static_allocation(scope, tt, i)
                    alloc = scope.let("shape_func_out_{0}".format(i), alloc)
                    out_shapes.append(alloc)

                shape_call = self.shape_func(
                    call.op,
                    expr.Tuple(shape_func_ins),
                    expr.Tuple(out_shapes), is_inputs)

                scope.let("shape_func", shape_call)

                storages = []
                for out_shape, out_type in zip(out_shapes, out_types):
                    size = self.compute_storage_in_relay(
                        out_shape, out_type.dtype)
                    alignment = self.compute_alignment(out_type.dtype)
                    sto = scope.let("storage_{i}".format(i=i), self.alloc_storage(
                        size, alignment, out_type.dtype))
                    storages.append(sto)

                outs = []
                sh_ty_storage = zip(out_shapes, out_types, storages)
                for i, (out_shape, out_type, storage) in enumerate(sh_ty_storage):
                    alloc = self.alloc_tensor(
                        storage,
                        out_shape,
                        out_type.dtype,
                        out_type.shape)
                    alloc = scope.let("out_{i}".format(i=i), alloc)
                    outs.append(alloc)

                tuple_outs = expr.Tuple(outs)
                invoke = self.invoke_tvm(call.op, ins, tuple_outs)
                scope.let("", invoke)
                return outs[0] if len(outs) == 1 else tuple_outs
            else:
                outs = []
                for i, out_ty in enumerate(out_types):
                    import pdb; pdb.set_trace()
                    out = self.make_static_allocation(scope, out_ty, i)
                    outs.append(out)

                output = expr.Tuple(outs)
                invoke = self.invoke_tvm(call.op, ins, output)
                scope.let("", invoke)
                return view.pack(output)
        else:
            return super().visit_call(call)


@transform.function_pass(opt_level=0)
class ManifestAlloc:
    """The explicit pass wrapper around ManifestAlloc."""
    def __init__(self, target_host):
        self.target_host = target_host
        self.default_device = 0 # kCPU

    def transform_function(self, func, mod, _):
        # TODO(@jroesch): Is there a way to do one shot initilization?
        # can we have def pass_init?
        mod.import_from_std("core.rly")
        ca = ContextAnalysis(cpu(0))
        ca.visit(func)
        print(func.astext(annotate=mk_analysis_annotator(ca.results())))
        ea = ManifestAllocPass(self.target_host, ca.results())
        func = ea.visit(func)
        return func


register_func("relay.transform.ManifestAlloc", ManifestAlloc)
