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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""A prelude containing useful global functions and ADT definitions."""
from .ty import GlobalTypeVar, TensorType, Any, scalar_type
from .expr import Var, Function, GlobalVar, If, const
from .op.tensor import add, subtract, equal
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard
from . import op
from .module import Module

class TensorArrayOps(object):
    """Contains tensor array related ops"""

    def __init__(self, prelude, dtype):
        """Create tensor array ops registry"""
        self.prelude = prelude
        self.dtype = dtype

    def get_name(self, canonical):
        """Get name corresponding to the caninical name"""
        return self.prelude.get_name(canonical, self.dtype)

    def get_var(self, canonical):
        """Get var corresponding to the caninical name"""
        return self.prelude.get_var(canonical, self.dtype)

    def define_tensor_adt(self):
        """Defines the dynamic tensor ADT, which is the container for tensors
        with variable shapes."""
        tensor_type_name = self.get_name('tensor_t')
        tensor_type_var = GlobalTypeVar(tensor_type_name)
        setattr(self.prelude, tensor_type_name, tensor_type_var)
        tensor0_type = TensorType([], self.dtype)
        tensor1_type = TensorType([Any()], self.dtype)
        tensor2_type = TensorType([Any(), Any()], self.dtype)
        tensor3_type = TensorType([Any(), Any(), Any()], self.dtype)
        tensor4_type = TensorType([Any(), Any(), Any(), Any()], self.dtype)
        tensor5_type = TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype)
        tensor6_type = TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype)
        tensor_nil_name = self.get_name('tensor_nil')
        tensor0_name = self.get_name('tensor0')
        tensor1_name = self.get_name('tensor1')
        tensor2_name = self.get_name('tensor2')
        tensor3_name = self.get_name('tensor3')
        tensor4_name = self.get_name('tensor4')
        tensor5_name = self.get_name('tensor5')
        tensor6_name = self.get_name('tensor6')
        tensor_nil_case = Constructor(tensor_nil_name, [], tensor_type_var)
        tensor0_case = Constructor(tensor0_name, [tensor0_type], tensor_type_var)
        tensor1_case = Constructor(tensor1_name, [tensor1_type], tensor_type_var)
        tensor2_case = Constructor(tensor2_name, [tensor2_type], tensor_type_var)
        tensor3_case = Constructor(tensor3_name, [tensor3_type], tensor_type_var)
        tensor4_case = Constructor(tensor4_name, [tensor4_type], tensor_type_var)
        tensor5_case = Constructor(tensor5_name, [tensor5_type], tensor_type_var)
        tensor6_case = Constructor(tensor6_name, [tensor6_type], tensor_type_var)
        setattr(self.prelude, tensor_nil_name, tensor_nil_case)
        setattr(self.prelude, tensor0_name, tensor0_case)
        setattr(self.prelude, tensor1_name, tensor1_case)
        setattr(self.prelude, tensor2_name, tensor2_case)
        setattr(self.prelude, tensor3_name, tensor3_case)
        setattr(self.prelude, tensor4_name, tensor4_case)
        setattr(self.prelude, tensor5_name, tensor5_case)
        setattr(self.prelude, tensor6_name, tensor6_case)
        self.prelude.mod[tensor_type_var] = TypeData(tensor_type_var, [], [tensor_nil_case,
                                                                           tensor0_case,
                                                                           tensor1_case,
                                                                           tensor2_case,
                                                                           tensor3_case,
                                                                           tensor4_case,
                                                                           tensor5_case,
                                                                           tensor6_case])

    def define_tensor_take(self):
        """Defines a function to return a range of tensor_t on axis 0.
            tensor_take(t, lower, upper) :
            tensor_t -> Tensor[(), int32] -> Tensor[(), int32] -> tensor_t
        """
        take_name = self.get_name("tensor_take")
        take_var = GlobalVar(take_name)
        setattr(self.prelude, take_name, take_var)
        tensor_t = self.get_var('tensor_t')
        tensor1_var = self.get_var('tensor1')
        tensor2_var = self.get_var('tensor2')
        tensor3_var = self.get_var('tensor3')
        tensor4_var = self.get_var('tensor4')
        tensor5_var = self.get_var('tensor5')
        tensor6_var = self.get_var('tensor6')
        t = Var('tensor', tensor_t())
        lower = Var('lower', scalar_type('int32'))
        upper = Var('upper', scalar_type('int32'))
        t1 = Var('t1')
        t2 = Var('t2')
        t3 = Var('t3')
        t4 = Var('t4')
        t5 = Var('t5')
        t6 = Var('t6')
        tensor1_case =\
            Clause(PatternConstructor(tensor1_var, [PatternVar(t1)]),
                   tensor1_var(op.take(t1, op.arange(lower, upper, dtype='int32'))))
        tensor2_case =\
            Clause(PatternConstructor(tensor2_var, [PatternVar(t2)]),
                   tensor2_var(op.take(t2, op.arange(lower, upper, dtype='int32'), axis=0)))
        tensor3_case =\
            Clause(PatternConstructor(tensor3_var, [PatternVar(t3)]),
                   tensor3_var(op.take(t3, op.arange(lower, upper, dtype='int32'), axis=0)))
        tensor4_case =\
            Clause(PatternConstructor(tensor4_var, [PatternVar(t4)]),
                   tensor4_var(op.take(t4, op.arange(lower, upper, dtype='int32'), axis=0)))
        tensor5_case =\
            Clause(PatternConstructor(tensor5_var, [PatternVar(t5)]),
                   tensor5_var(op.take(t5, op.arange(lower, upper, dtype='int32'), axis=0)))
        tensor6_case =\
            Clause(PatternConstructor(tensor6_var, [PatternVar(t6)]),
                   tensor6_var(op.take(t6, op.arange(lower, upper, dtype='int32'), axis=0)))
        self.prelude.mod[take_var] =\
            Function([t, lower, upper],
                     Match(t, [tensor1_case,
                               tensor2_case,
                               tensor3_case,
                               tensor4_case,
                               tensor5_case,
                               tensor6_case], False),
                     tensor_t(), [])

    def define_tensor_expand_dims(self):
        """Defines a function to grow a tensor_t's rank by adding one dimension in front
        of the original tensor_t.
        tensor_expand_dims(t) : tensor_t -> tensor_t
        """
        expand_dims_name = self.get_name("tensor_expand_dims")
        expand_dims_var = GlobalVar(expand_dims_name)
        setattr(self.prelude, expand_dims_name, expand_dims_var)
        tensor_type_var = self.get_var('tensor_t')
        x = Var("x", tensor_type_var())
        t0 = Var("t0")
        t1 = Var("t1")
        t2 = Var("t2")
        t3 = Var("t3")
        t4 = Var("t4")
        t5 = Var("t5")
        tensor0_var = self.get_var('tensor0')
        tensor1_var = self.get_var('tensor1')
        tensor2_var = self.get_var('tensor2')
        tensor3_var = self.get_var('tensor3')
        tensor4_var = self.get_var('tensor4')
        tensor5_var = self.get_var('tensor5')
        tensor6_var = self.get_var('tensor6')
        tensor0_case = Clause(PatternConstructor(tensor0_var, [PatternVar(t0)]),
                              tensor1_var(op.expand_dims(t0, 0, 1)))
        tensor1_case = Clause(PatternConstructor(tensor1_var, [PatternVar(t1)]),
                              tensor2_var(op.expand_dims(t1, 0, 1)))
        tensor2_case = Clause(PatternConstructor(tensor2_var, [PatternVar(t2)]),
                              tensor3_var(op.expand_dims(t2, 0, 1)))
        tensor3_case = Clause(PatternConstructor(tensor3_var, [PatternVar(t3)]),
                              tensor4_var(op.expand_dims(t3, 0, 1)))
        tensor4_case = Clause(PatternConstructor(tensor4_var, [PatternVar(t4)]),
                              tensor5_var(op.expand_dims(t4, 0, 1)))
        tensor5_case = Clause(PatternConstructor(tensor5_var, [PatternVar(t5)]),
                              tensor6_var(op.expand_dims(t5, 0, 1)))
        self.prelude.mod[expand_dims_var] =\
            Function([x],
                     Match(x, [tensor0_case,
                               tensor1_case,
                               tensor2_case,
                               tensor3_case,
                               tensor4_case,
                               tensor5_case], False))

    def define_tensor_concat(self):
        """Defines a function to concatenate two tensor_t on the first axis

        tensor_concatenate(t) : tensor_t -> tensor_t -> tensor_t
        """
        concat_name = self.get_name("tensor_concatenate")
        concat_var = GlobalVar(concat_name)
        setattr(self.prelude, concat_name, concat_var)
        tensor_type_var = self.get_var('tensor_t')
        x = Var("x", tensor_type_var())
        y = Var("y", tensor_type_var())

        tensor1_var = self.get_var('tensor1')
        tensor2_var = self.get_var('tensor2')
        tensor3_var = self.get_var('tensor3')
        tensor4_var = self.get_var('tensor4')
        t11 = Var("t11")
        t12 = Var("t12")
        t21 = Var("t21")
        t22 = Var("t22")
        t31 = Var("t31")
        t32 = Var("t32")
        t41 = Var("t41")
        t42 = Var("t42")
        tensor1_case = Clause(PatternConstructor(tensor1_var, [PatternVar(t11)]),
                              Match(y, [Clause(PatternConstructor(tensor1_var, [PatternVar(t12)]),
                                               tensor1_var(op.concatenate([t11, t12], axis=0)))],
                                    False))
        tensor2_case = Clause(PatternConstructor(tensor2_var, [PatternVar(t21)]),
                              Match(y, [Clause(PatternConstructor(tensor2_var, [PatternVar(t22)]),
                                               tensor2_var(op.concatenate([t21, t22], axis=0)))],
                                    False))
        tensor3_case = Clause(PatternConstructor(tensor3_var, [PatternVar(t31)]),
                              Match(y, [Clause(PatternConstructor(tensor3_var, [PatternVar(t32)]),
                                               tensor3_var(op.concatenate([t31, t32], axis=0)))],
                                    False))
        tensor4_case = Clause(PatternConstructor(tensor4_var, [PatternVar(t41)]),
                              Match(y, [Clause(PatternConstructor(tensor4_var, [PatternVar(t42)]),
                                               tensor4_var(op.concatenate([t41, t42], axis=0)))],
                                    False))
        # op.concatenate does not support tensor with rank higher than 4
        self.prelude.mod[concat_var] =\
            Function([x, y], Match(x, [tensor1_case,
                                       tensor2_case,
                                       tensor3_case,
                                       tensor4_case], False))

    def define_tensor_array(self):
        """Defines a function to create a tensor array with size n.
        tensor_array(n) : Tensor[(), int32] -> list[tensor_t]
        """
        tensor_array_constructor_name = self.get_name("tensor_array")
        tensor_array_constructor_var = GlobalVar(tensor_array_constructor_name)
        setattr(self.prelude, tensor_array_constructor_name, tensor_array_constructor_var)
        tensor_nil_var = self.get_var('tensor_nil')
        tensor_type_var = self.get_var('tensor_t')
        n = Var("x", scalar_type('int32'))
        body = If(equal(n, const(0)),
                  self.prelude.nil(),
                  self.prelude.cons(tensor_nil_var(),
                                    tensor_array_constructor_var(subtract(n, const(1)))))
        self.prelude.mod[tensor_array_constructor_var] = \
            Function([n], body, self.prelude.l(tensor_type_var()), [])

    def define_tensor_array_read(self):
        """Defines a function to get the head of a list. Assume the list has at least one
        element.

        tensor_array_read(ta, n) : list[tensor_t] -> Tensor[(), int32] -> tensor_t
        """
        read_name = self.get_name("tensor_array_read")
        read_var = GlobalVar(read_name)
        setattr(self.prelude, read_name, read_var)
        tensor_type_var = self.get_var('tensor_t')

        tensor_array = Var("tensor_array", self.prelude.l(tensor_type_var()))
        n = Var("x", scalar_type('int32'))
        self.prelude.mod[read_var] =\
            Function([tensor_array, n], self.prelude.nth(tensor_array, n), tensor_type_var(), [])

    def define_tensor_array_write(self):
        """Defines a function to update a tensor array at index n with value v.
        tensor_array_write(ta, n, v) :
            list[tensor_t] -> Tensor[(), int32] -> tensor_t -> list[tensor_t]
        """
        write_name = self.get_name("tensor_array_write")
        write_var = GlobalVar(write_name)
        setattr(self.prelude, write_name, write_var)
        tensor_type_var = self.get_var('tensor_t')
        tensor_array = Var("tensor_array", self.prelude.l(tensor_type_var()))
        n = Var("x", scalar_type('int32'))
        v = Var("v", tensor_type_var())
        self.prelude.mod[write_var] =\
            Function([tensor_array, n, v], self.prelude.update(tensor_array, n, v),
                     self.prelude.l(tensor_type_var()), [])

    def define_tensor_array_unstack_tensor1(self):
        """Defines a function to unstack the values of a tensor_t with rank 1 in a tensor array.
        tensor_array_unstack_tensor1(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor1_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))
        tensor_type_var = self.get_var('tensor_t')
        tensor0_var = self.get_var('tensor0')
        helper_body =\
            If(equal(i, up),
               self.prelude.nil(),
               self.prelude.cons(tensor0_var(op.take(tensor, i)),
                                 helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(tensor_type_var()), [])
        unstack_name = self.get_name("tensor_array_unstack_tensor1")
        unstack_var = GlobalVar(unstack_name)
        setattr(self.prelude, unstack_name, unstack_var)
        tensor1 = Var("tensor", TensorType([Any()], self.dtype))
        shape = op.shape_of(tensor1)
        ndim = op.take(shape, const(0))
        self.prelude.mod[unstack_var] =\
            Function([tensor1], helper_var(const(0), ndim, tensor1),
                     self.prelude.l(tensor_type_var()), [])

    def define_tensor_array_unstack_tensor2(self):
        """Defines a function to unstack the values of a tensor_t with rank 2 in a tensor array.

        tensor_array_unstack_tensor2(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor2_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.prelude.nil(),
                         self.prelude.cons(self.get_var('tensor1')(op.take(tensor, i, axis=0)),
                                           helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(self.get_var('tensor_t')()), [])

        tensor_array_unstack_tensor2_name = self.get_name("tensor_array_unstack_tensor2")
        tensor_array_unstack_tensor2_var = GlobalVar(tensor_array_unstack_tensor2_name)
        setattr(self.prelude, tensor_array_unstack_tensor2_name, tensor_array_unstack_tensor2_var)
        tensor2 = Var("tensor", TensorType([Any(), Any()], self.dtype))
        shape = op.shape_of(tensor2)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor2_var] =\
            Function([tensor2], helper_var(const(0), ndim, tensor2),
                     self.prelude.l(self.get_var('tensor_t')()), [])

    def define_tensor_array_unstack_tensor3(self):
        """Defines a function to unstack the values of a tensor_t with rank 3 in a tensor array.

        tensor_array_unstack_tensor3(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor3_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.prelude.nil(),
                         self.prelude.cons(self.get_var('tensor2')(op.take(tensor, i, axis=0)),
                                           helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(self.get_var('tensor_t')()), [])

        tensor_array_unstack_tensor3_name = self.get_name("tensor_array_unstack_tensor3")
        tensor_array_unstack_tensor3_var = GlobalVar(tensor_array_unstack_tensor3_name)
        setattr(self.prelude, tensor_array_unstack_tensor3_name, tensor_array_unstack_tensor3_var)
        tensor3 = Var("tensor", TensorType([Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor3)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor3_var] =\
            Function([tensor3], helper_var(const(0), ndim, tensor3),
                     self.prelude.l(self.get_var('tensor_t')()), [])

    def define_tensor_array_unstack_tensor4(self):
        """Defines a function to unstack the values of a tensor_t with rank 4 in a tensor array.

        tensor_array_unstack_tensor4(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor4_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.prelude.nil(),
                         self.prelude.cons(self.get_var('tensor3')(op.take(tensor, i, axis=0)),
                                           helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(self.get_var('tensor_t')()), [])

        tensor_array_unstack_tensor4_name = self.get_name("tensor_array_unstack_tensor4")
        tensor_array_unstack_tensor4_var = GlobalVar(tensor_array_unstack_tensor4_name)
        setattr(self.prelude, tensor_array_unstack_tensor4_name, tensor_array_unstack_tensor4_var)
        tensor4 = Var("tensor", TensorType([Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor4)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor4_var] =\
            Function([tensor4], helper_var(const(0), ndim, tensor4),
                     self.prelude.l(self.get_var('tensor_t')()), [])

    def define_tensor_array_unstack_tensor5(self):
        """Defines a function to unstack the values of a tensor_t with rank 5 in a tensor array.

        tensor_array_unstack_tensor5(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor5_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.prelude.nil(),
                         self.prelude.cons(self.get_var('tensor4')(op.take(tensor, i, axis=0)),
                                           helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(self.get_var('tensor_t')()), [])

        tensor_array_unstack_tensor5_name = self.get_name("tensor_array_unstack_tensor5")
        tensor_array_unstack_tensor5_var = GlobalVar(tensor_array_unstack_tensor5_name)
        setattr(self.prelude, tensor_array_unstack_tensor5_name, tensor_array_unstack_tensor5_var)
        tensor5 = Var("tensor", TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor5)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor5_var] =\
            Function([tensor5], helper_var(const(0), ndim, tensor5),
                     self.prelude.l(self.get_var('tensor_t')()), [])

    def define_tensor_array_unstack_tensor6(self):
        """Defines a function to unstack the values of a tensor_t with rank 6 in a tensor array.

        tensor_array_unstack_tensor6(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor6_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type('int32'))
        i = Var("i", scalar_type('int32'))

        helper_body = If(equal(i, up),
                         self.prelude.nil(),
                         self.prelude.cons(self.get_var('tensor5')(op.take(tensor, i, axis=0)),
                                           helper_var(add(i, const(1)), up, tensor)))
        self.prelude.mod[helper_var] =\
            Function([i, up, tensor], helper_body, self.prelude.l(self.get_var('tensor_t')()), [])

        tensor_array_unstack_tensor6_name = self.get_name("tensor_array_unstack_tensor6")
        tensor_array_unstack_tensor6_var = GlobalVar(tensor_array_unstack_tensor6_name)
        setattr(self.prelude, tensor_array_unstack_tensor6_name, tensor_array_unstack_tensor6_var)
        tensor6 = Var("tensor", TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor6)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor6_var] =\
            Function([tensor6], helper_var(const(0), ndim, tensor6),
                     self.prelude.l(self.get_var('tensor_t')()), [])

    def define_tensor_array_scatter(self):
        """Defines a function to scatter the values of a tensor_t in indices of a tensor array.
        tensor_array_scatter(ta, indices, value) :
            list[tensor_t] -> Tensor[(Any), int32] -> tensor_t -> list[tensor_t]
        """
        tensor_array_scatter_helper_name = self.get_name("tensor_array_scatter_helper")
        tensor_array_scatter_helper_var = GlobalVar(tensor_array_scatter_helper_name)
        tensor_t = self.get_var('tensor_t')
        ta = Var("ta", self.prelude.l(tensor_t()))
        current = Var("current", scalar_type('int32'))
        limit = Var("limit", scalar_type('int32'))
        indices_ = Var('indices_', TensorType([Any()], 'int32'))
        values_ = Var('values_', self.prelude.l(tensor_t()))
        write_var = self.get_var('tensor_array_write')
        read_var = self.get_var('tensor_array_read')
        helper_body = If(equal(current, limit),
                         ta,
                         tensor_array_scatter_helper_var(
                             write_var(ta, op.take(indices_, current),
                                       read_var(values_, current)),
                             add(current, const(1)),
                             limit, indices_, values_))
        self.prelude.mod[tensor_array_scatter_helper_var] =\
            Function([ta, current, limit, indices_, values_],
                     helper_body, self.prelude.l(tensor_t()), [])
        tensor_array_scatter_name = self.get_name("tensor_array_scatter")
        tensor_array_scatter_var = GlobalVar(tensor_array_scatter_name)
        setattr(self.prelude, tensor_array_scatter_name, tensor_array_scatter_var)
        tensor_array = Var("tensor_array", self.prelude.l(tensor_t()))
        indices = Var('indices', TensorType([Any()], 'int32'))
        values = Var('values', self.prelude.l(tensor_t()))
        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = tensor_array_scatter_helper_var(tensor_array, const(0), limit, indices, values)
        self.prelude.mod[tensor_array_scatter_var] =\
            Function([tensor_array, indices, values], body, self.prelude.l(tensor_t()), [])

    def define_tensor_array_split(self):
        """Defines a function to split the values of a tensor_t into a tensor array.
        tensor_array_split(ta, value, lengths) :
            list[tensor_t] -> tensor_t -> Tensor[(Any), int32] -> list[tensor_t]
        """
        tensor_t = self.get_var('tensor_t')
        tensor_array_split_helper_name = self.get_name("ta_split_helper")
        tensor_array_split_helper_var = GlobalVar(tensor_array_split_helper_name)
        setattr(self.prelude, tensor_array_split_helper_name, tensor_array_split_helper_var)
        ta1 = Var("tensor_array", self.prelude.l(tensor_t()))
        value1 = Var('value1', tensor_t())
        offset1 = Var('offset1', scalar_type('int32'))
        current1 = Var('current1', scalar_type('int32'))
        limit1 = Var('limit1', scalar_type('int32'))
        lengths1 = Var('lengths', TensorType([Any()], 'int32'))
        write_var = self.get_var('tensor_array_write')
        take_var = self.get_var('tensor_take')
        helper1_body = If(equal(current1, limit1),
                          ta1,
                          write_var(
                              tensor_array_split_helper_var(
                                  ta1,
                                  value1,
                                  add(offset1, op.take(lengths1, current1)),
                                  add(current1, const(1)),
                                  limit1,
                                  lengths1
                              ),
                              current1,
                              take_var(value1,
                                       offset1,
                                       add(op.take(lengths1, current1), offset1))))
        self.prelude.mod[tensor_array_split_helper_var] = \
            Function([ta1, value1, offset1, current1, limit1, lengths1],
                     helper1_body, self.prelude.l(tensor_t()), [])
        split_name = self.get_name("tensor_array_split")
        split_var = GlobalVar(split_name)
        setattr(self.prelude, split_name, split_var)
        tensor_array = Var("tensor_array", self.prelude.l(tensor_t()))
        value = Var('value', tensor_t())
        lengths = Var('lengths', TensorType([Any()], 'int32'))
        lengths_shape = op.shape_of(lengths)
        lengths_limit = op.take(lengths_shape, const(0))
        body = tensor_array_split_helper_var(
            tensor_array,
            value,
            const(0),
            const(0),
            lengths_limit,
            lengths)
        self.prelude.mod[split_var] =\
            Function([tensor_array, value, lengths], body, self.prelude.l(tensor_t()), [])

    def define_tensor_array_concat(self):
        """Defines a function to return the values in the tensor array as concatenated tensor_t.
        tensor_array_concat(ta) : list[tensor_t] -> tensor_t
        """
        concat_name = self.get_name("tensor_array_concat")
        concat_var = GlobalVar(concat_name)
        setattr(self.prelude, concat_name, concat_var)
        tensor_concat_var = self.get_var('tensor_concatenate')
        tensor_t = self.get_var('tensor_t')
        tensor_nil_var = self.get_var('tensor_nil')
        tensor_array = Var("tensor_array", self.prelude.l(tensor_t()))
        hd = Var("hd")
        tl = Var("tl")
        nil_case = Clause(PatternConstructor(self.prelude.nil), tensor_nil_var())
        cons_case = Clause(PatternConstructor(self.prelude.cons, [PatternVar(hd), PatternVar(tl)]),
                           Match(tl, [
                               Clause(PatternConstructor(self.prelude.nil), hd),
                               Clause(PatternWildcard(),
                                      tensor_concat_var(hd, concat_var(tl)))
                           ], False))
        self.prelude.mod[concat_var] =\
            Function([tensor_array],
                     Match(tensor_array, [nil_case, cons_case], False), tensor_t(), [])

    def define_tensor_array_gather(self):
        """Defines a function to return the selected values in a tensor array as tensor_t.
        tensor_array_gather(ta, indices) : list[tensor_t] -> Tensor[(Any), int32] -> tensor_t
        """
        helper_name = self.get_name("tensor_array_gather_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor_type_var = self.get_var('tensor_t')
        stack_var = self.get_var('tensor_array_stack')
        read_var = self.get_var('tensor_array_read')
        ta = Var("ta", self.prelude.l(tensor_type_var()))
        accu = Var("accu", self.prelude.l(tensor_type_var()))
        current = Var("current", scalar_type('int32'))
        limit = Var("limit", scalar_type('int32'))
        indices_ = Var('indices_', TensorType([Any()], 'int32'))
        helper_body =\
            If(equal(current, const(0)),
               stack_var(accu),
               helper_var(
                   ta,
                   self.prelude.cons(
                       read_var(
                           ta, op.take(indices_, subtract(current, const(1)))), accu),
                   subtract(current, const(1)),
                   limit, indices_))
        self.prelude.mod[helper_var] = \
            Function([ta, accu, current, limit, indices_], helper_body, tensor_type_var(), [])
        gather_name = self.get_name("tensor_array_gather")
        gather_var = GlobalVar(gather_name)
        setattr(self.prelude, gather_name, gather_var)
        tensor_array = Var("tensor_array", self.prelude.l(tensor_type_var()))
        indices = Var('indices', TensorType([Any()], 'int32'))
        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = helper_var(tensor_array, self.prelude.nil(), limit, limit, indices)
        self.prelude.mod[gather_var] =\
            Function([tensor_array, indices], body, tensor_type_var(), [])

    def define_tensor_array_stack(self):
        """Defines a function to get the values in the tensor array as a stack tensor_t.
        tensor_array_stack(l) : list[tensor_t] -> tensor_t
        """
        stack_name = self.get_name("tensor_array_stack")
        stack_var = GlobalVar(stack_name)
        setattr(self.prelude, stack_name, stack_var)
        tensor_type_var = self.get_var('tensor_t')
        tensor_array = Var("tensor_array", self.prelude.l(tensor_type_var()))
        expand_dims_var = self.get_var('tensor_expand_dims')
        concat_var = self.get_var('tensor_concatenate')
        tensor_array_expand_dims = self.prelude.map(expand_dims_var, tensor_array)
        tensors = self.prelude.foldl(concat_var,
                                     self.prelude.hd(tensor_array_expand_dims),
                                     self.prelude.tl(tensor_array_expand_dims))
        self.prelude.mod[stack_var] = Function([tensor_array], tensors, tensor_type_var(), [])

    def register(self):
        """Register all tensor array ops in Prelude"""
        self.define_tensor_adt()
        self.define_tensor_take()
        self.define_tensor_expand_dims()
        self.define_tensor_concat()
        self.define_tensor_array()
        self.define_tensor_array_read()
        self.define_tensor_array_write()
        self.define_tensor_array_unstack_tensor1()
        self.define_tensor_array_unstack_tensor2()
        self.define_tensor_array_unstack_tensor3()
        self.define_tensor_array_unstack_tensor4()
        self.define_tensor_array_unstack_tensor5()
        self.define_tensor_array_unstack_tensor6()
        self.define_tensor_array_scatter()
        self.define_tensor_array_split()
        self.define_tensor_array_concat()
        self.define_tensor_array_stack()
        # TODO(wweic): Gather fails in PartialEvaluate
        # self.define_tensor_array_gather()

class Prelude:
    """Contains standard definitions."""

    def __init__(self, mod=None):
        if mod is None:
            mod = Module()
        self.mod = mod
        self.load_prelude()

    def get_name(self, canonical, dtype):
        """Get name corresponding to the canonical name"""
        if canonical == 'tensor_t':
            return 'tensor_{}_t'.format(dtype)
        return "{}_{}".format(canonical, dtype)

    def get_var(self, canonical, dtype):
        """Get var corresponding to the canonical name"""
        name = self.get_name(canonical, dtype)
        return getattr(self, name)

    def load_prelude(self):
        """Parses the Prelude from Relay's text format into a module."""
        # TODO(@jroesch): we should remove this helper when we port over prelude
        self.mod.import_from_std("prelude.rly")

        self.l = self.mod.get_global_type_var("List")
        list_adt = self.mod[self.l]
        self.cons = list_adt.constructors[0]
        self.nil = list_adt.constructors[1]

        self.optional = self.mod.get_global_type_var("Option")
        optional_adt = self.mod[self.optional]
        self.some = optional_adt.constructors[0]
        self.none = optional_adt.constructors[1]

        self.tree = self.mod.get_global_type_var("Tree")
        tree_adt = self.mod[self.tree]
        self.rose = tree_adt.constructors[0]

        GLOBAL_DEFS = [
            "id",
            "compose",
            "flip",
            "hd",
            "tl",
            "nth",
            "update",
            "map",
            "foldl",
            "foldr",
            "foldr1",
            "concat",
            "filter",
            "zip",
            "rev",
            "map_accuml",
            "map_accumr",
            "unfoldl",
            "unfoldr",
            "sum",
            "length",
            "tmap",
            "size",
            "iterate",
        ]
        for global_def in GLOBAL_DEFS:
            setattr(self, global_def, self.mod.get_global_var(global_def))

        for dtype in ['float32',
                      'float16',
                      'float64',
                      'int32',
                      'uint8',
                      'int8',
                      'int16',
                      'uint16',
                      'int64']:
            tensor_array_ops = TensorArrayOps(self, dtype)
            tensor_array_ops.register()
