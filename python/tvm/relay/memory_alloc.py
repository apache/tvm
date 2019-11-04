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
# pylint: disable=no-else-return,invalid-name,len-as-condition
"""
A pass for manifesting explicit memory allocations.
"""
import numpy as np
from .expr_functor import ExprMutator
from .scope_builder import ScopeBuilder
from . import transform
from . import op, ty, expr
from .. import TVMType, register_func
from .backend import compile_engine


def is_primitive(call):
    return hasattr(call.op, 'attrs') and int(call.op.attrs.Primitive) == 1

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
    """A pass for explictly manifesting all memory allocations in Relay."""

    def __init__(self, target_host):
        self.invoke_tvm = op.memory.invoke_tvm_op
        self.alloc_storage = op.memory.alloc_storage
        self.alloc_tensor = op.memory.alloc_tensor
        self.shape_func = op.memory.shape_func
        self.scopes = [ScopeBuilder()]
        self.target_host = target_host
        self.compute_dtype = "int64"
        super().__init__()

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
        dtype = TVMType(dtype)
        align = (dtype.bits // 8) * dtype.lanes
        # MAGIC CONSTANT FROM device_api.h
        if align < 64:
            align = 64

        return expr.const(align, dtype="int64")

    def compute_storage_in_relay(self, shape, dtype):
        dtype = TVMType(dtype)
        els = op.prod(shape)
        num = expr.const(dtype.bits * dtype.lanes, self.compute_dtype)
        num = num + expr.const(7, self.compute_dtype)
        div = expr.const(8, self.compute_dtype)
        return els * (num / div)

    def compute_storage(self, tensor_type):
        dtype = TVMType(tensor_type.dtype)
        shape = [int(sh) for sh in tensor_type.shape]
        size = 1
        for sh in shape:
            size *= sh
        size *= (dtype.bits * dtype.lanes + 7) // 8
        return expr.const(size, dtype=self.compute_dtype)

    def make_static_allocation(self, scope, tensor_type, i):
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
        sto = scope.let("storage_{0}".format(i), self.alloc_storage(
            size, alignment, dtype))
        # TODO(@jroesch): There is a bug with typing based on the constant shape.
        tensor = self.alloc_tensor(sto, shape, dtype, tensor_type.shape)
        return scope.let("tensor_{0}".format(i), tensor)

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

            is_dynamic = ret_type.is_dynamic()
            # TODO(@jroesch): restore this code, more complex then it seems
            # for arg in call.args:
            #     is_dynamic = is_dynamic or arg.checked_type.is_dynamic()

            if is_dynamic:
                assert isinstance(ret_type, ty.TensorType)
                shape_func_ins = []
                engine = compile_engine.get()
                cfunc = engine.lower_shape_func(call.op, self.target_host)
                input_states = cfunc.shape_func_param_states

                is_inputs = []
                for i, (arg, state) in enumerate(zip(new_args, input_states)):
                    state = int(state)
                    # Pass Shapes
                    if state == 2:
                        sh_of = self.visit(self.shape_of(arg))
                        shape_func_ins.append(
                            scope.let("in_shape_{0}".format(i), sh_of))
                        is_inputs.append(0)
                    # Pass Inputs
                    elif state == 1:
                        new_arg = self.visit(arg)
                        shape_func_ins.append(
                            scope.let("in_shape_{0}".format(i), new_arg))
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

                out_types = []
                out_types.append(call.checked_type)

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

                invoke = self.invoke_tvm(call.op, ins, expr.Tuple(outs))
                scope.let("", invoke)
                return outs[0]
            else:
                view = LinearizeRetType(ret_type)
                out_tys = view.unpack()

                outs = []
                for i, out_ty in enumerate(out_tys):
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

    def transform_function(self, func, mod, _):
        # TODO(@jroesch): Is there a way to do one shot initilization?
        # can we have def pass_init?
        mod.import_from_std("core.rly")
        ea = ManifestAllocPass(self.target_host)
        func = ea.visit(func)
        return func


register_func("relay.transform.ManifestAlloc", ManifestAlloc)
