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
"""The type functor of Relay."""
from .ty import (
    TypeVar,
    IncompleteType,
    TensorType,
    FuncType,
    TupleType,
    TypeRelation,
    RefType,
    GlobalTypeVar,
    TypeCall,
)
from .adt import TypeData


class TypeFunctor:
    """
    An abstract visitor defined over Type.

    Defines the default dispatch over types.
    """

    def __init__(self):
        # TODO(weberlo): make type vars hashable, so we can memoize
        pass

    # pylint: disable=no-else-return
    def visit(self, typ):
        """Apply the visitor to a type."""
        if isinstance(typ, TypeVar):
            return self.visit_type_var(typ)
        elif isinstance(typ, IncompleteType):
            return self.visit_incomplete_type(typ)
        elif isinstance(typ, TensorType):
            return self.visit_tensor_type(typ)
        elif isinstance(typ, FuncType):
            return self.visit_func_type(typ)
        elif isinstance(typ, TupleType):
            return self.visit_tuple_type(typ)
        elif isinstance(typ, TypeRelation):
            return self.visit_type_relation(typ)
        elif isinstance(typ, RefType):
            return self.visit_ref_type(typ)
        elif isinstance(typ, GlobalTypeVar):
            return self.visit_global_type_var(typ)
        elif isinstance(typ, TypeCall):
            return self.visit_type_call(typ)
        elif isinstance(typ, TypeData):
            return self.visit_type_data(typ)
        else:
            raise Exception(f"unhandled case: {type(typ)}")

    def visit_type_var(self, _):
        raise NotImplementedError()

    def visit_incomplete_type(self, _):
        raise NotImplementedError()

    def visit_tensor_type(self, _):
        raise NotImplementedError()

    def visit_func_type(self, _):
        raise NotImplementedError()

    def visit_tuple_type(self, _):
        raise NotImplementedError()

    def visit_type_relation(self, _):
        raise NotImplementedError()

    def visit_ref_type(self, _):
        raise NotImplementedError()

    def visit_global_type_var(self, _):
        raise NotImplementedError()

    def visit_type_call(self, _):
        raise NotImplementedError()

    def visit_type_data(self, _):
        raise NotImplementedError()


class TypeVisitor(TypeFunctor):
    """
    A visitor over Type.

    The default behavior recursively traverses the AST.
    """

    def visit_type_var(self, tv):
        pass

    def visit_incomplete_type(self, it):
        pass

    def visit_tensor_type(self, tt):
        pass

    def visit_func_type(self, ft):
        for arg_type in ft.arg_types:
            self.visit(arg_type)
        self.visit(ft.ret_type)
        for type_param in getattr(ft, "type_params", []):
            self.visit(type_param)
        for type_constraint in getattr(ft, "type_constraints", []):
            self.visit(type_constraint)

    def visit_tuple_type(self, tt):
        for field in tt.fields:
            self.visit(field)

    def visit_type_relation(self, tr):
        for arg in tr.args:
            self.visit(arg)

    def visit_ref_type(self, rt):
        self.visit(rt.value)

    def visit_global_type_var(self, gtv):
        pass

    def visit_type_call(self, tc):
        self.visit(tc.func)
        for arg in tc.args:
            self.visit(arg)

    def visit_type_data(self, td):
        self.visit(td.header)
        for type_var in td.type_vars:
            self.visit(type_var)


class TypeMutator(TypeFunctor):
    """
    A functional visitor over Type.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """

    def visit_type_var(self, tv):
        return TypeVar(tv.name_hint, tv.kind)

    def visit_incomplete_type(self, it):
        return IncompleteType(it.kind)

    def visit_tensor_type(self, tt):
        return TensorType(tt.shape, tt.dtype)

    def visit_func_type(self, ft):
        new_arg_types = [self.visit(arg_type) for arg_type in ft.arg_types]
        new_ret_type = self.visit(ft.ret_type)
        new_type_params = [self.visit(type_param) for type_param in getattr(ft, "type_params", [])]
        new_type_constraints = [
            self.visit(type_constraint) for type_constraint in getattr(ft, "type_constraints", [])
        ]
        return FuncType(new_arg_types, new_ret_type, new_type_params, new_type_constraints)

    def visit_tuple_type(self, tt):
        return TupleType([self.visit(field) for field in tt.fields])

    def visit_type_relation(self, tr):
        return TypeRelation(tr.func, [self.visit(arg) for arg in tr.args], tr.num_inputs, tr.attrs)

    def visit_ref_type(self, rt):
        return RefType(self.visit(rt.value))

    def visit_global_type_var(self, gtv):
        return GlobalTypeVar(gtv.name_hint, gtv.kind)

    def visit_type_call(self, tc):
        return TypeCall(self.visit(tc.func), [self.visit(arg) for arg in tc.args])

    def visit_type_data(self, td):
        return TypeData(
            self.visit(td.header),
            [self.visit(type_var) for type_var in td.type_vars],
            td.constructors,
        )
