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
from tvm.ir import IRModule, TypeCall
from tvm.tir import Any
from tvm.relay.transform import ToANormalFormExpr

from .ty import GlobalTypeVar, TensorType, scalar_type
from .expr import Var, GlobalVar, If, const
from .function import Function
from .op.tensor import add, subtract, equal
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard
from . import op, transform
from .analysis import free_vars


def get_tensor_array_shape(expr, dtype, prelude):
    """Get the static shape of a tensor array if it has fixed rank shape.

    By design, static ADT tensor in TVM has type name in the format
    of static_tensor_dim0_dim1_..._dimN_t.

    Parameters
    ----------
    expr : Relay Expr
        Input expression.

    dtype : str
        Data type.

    prelude : Prelude
        Tensor array prelude

    Returns
    -------
    shape : tuple of (int, Any) or None
        The output shape. None if input tensor array
        has dynamic shape.
    """
    mod = prelude.mod
    mod["main"] = Function(free_vars(expr), expr)
    mod = transform.InferType()(mod)
    checked_type = mod["main"].body.checked_type
    assert isinstance(checked_type, TypeCall), "Input must be a tensor array."
    ta_type_str = checked_type.args[0].func.name_hint
    static_ta_ty_start = "static_tensor_{}".format(dtype)
    if ta_type_str.startswith(static_ta_ty_start):
        shape_str = ta_type_str.replace("{}_".format(static_ta_ty_start), "").replace("_t", "")
        shape = []
        if "scalar" not in shape_str:
            for dim_str in shape_str.split("_"):
                if dim_str in ["?", "any"]:
                    shape.append(Any())
                else:
                    shape.append(int(dim_str))
        return tuple(shape)
    return None


def _get_name_static(canonical, dtype, shape, batch_dim=None, extra_shapes=None):
    """Get name for static shape tensor array op

    By design, static ADT tensor in TVM has type name in the format
    of static_tensor_dim0_dim1_..._dimN_t
    or static_tensor_batch1_dim0_dim1_..._dimN_t if tensorlist stack only have one item.

    Parameters
    ----------
    canonical : String
        Tensor array op name

    dtype : str
        Data type.

    shape : tuple of (int, Any) or None
        Tensor array shape

    batch_dim: None or int
        1 if tensorlist stack only have one item.
        None by default

    Returns
    -------
    name : String
        The tensor array op name
    """
    shape_str = _to_str(shape)

    if extra_shapes is not None:
        for n, s in extra_shapes.items():
            extra_shape_str = "_{}_{}".format(n, _to_str(s))
            shape_str += extra_shape_str

    if len(shape_str) == 0:
        shape_str = "scalar"
    if canonical == "tensor_t":
        return "static_tensor_{}_{}_t".format(dtype, shape_str)
    if batch_dim is None or canonical in ["tensor_constructor", "tensor_nil"]:
        return "{}_{}_{}".format(canonical, dtype, shape_str)
    if batch_dim != 1:
        return "{}_{}_{}".format(canonical, dtype, shape_str)
    return "{}_{}_batch{}_{}".format(canonical, dtype, str(batch_dim), shape_str)


def _to_str(shape):
    dim_names = []
    for dim in shape:
        if isinstance(dim, Any):
            dim_names.append("any")
        else:
            dim_names.append(str(dim))
    return "_".join(dim_names)


class StaticTensorArrayOps(object):
    """Contains tensor array related ops for fixed rank tensor array"""

    def __init__(self, prelude, dtype, shape, batch_dim=None):
        """Create tensor array ops registry"""
        self.prelude = prelude
        self.dtype = dtype
        self.shape = shape
        self.batch_dim = batch_dim
        self.list, self.cons, self.nil = self.prelude.mod.get_type("List")

    def get_name(self, canonical, extra_shapes=None):
        """Get name corresponding to the canonical name"""
        return _get_name_static(canonical, self.dtype, self.shape, self.batch_dim, extra_shapes)

    def get_global_var(self, canonical):
        """Get global corresponding to the canonical name"""
        return self.prelude.get_global_var_static(canonical, self.dtype, self.shape, self.batch_dim)

    def get_type(self, canonical):
        """Get type corresponding to the canonical name"""
        return self.prelude.get_type_static(canonical, self.dtype, self.shape)

    def get_ctor(self, canonical):
        """Get ctor corresponding to the canonical name"""
        return self.prelude.get_ctor_static("tensor_t", canonical, self.dtype, self.shape)

    def define_tensor_adt(self):
        """Defines the static tensor ADT, which is the container for tensors
        with fixed shapes."""
        tensor_type_name = self.get_name("tensor_t")

        # This is effectively functioning as a monomorphizer.
        # TODO(@jroesch): we should add full shape polymoprhism
        # and do monomorphization.
        #
        # Skip register if tensor type is already registered.
        global_type_names = set()
        for g_ty_var in self.prelude.mod.get_global_type_vars():
            global_type_names.add(g_ty_var.name_hint)

        if tensor_type_name in global_type_names:
            self.tensor_type_var = self.get_type("tensor_t")
            return

        self.tensor_type_var = GlobalTypeVar(tensor_type_name)

        tensor_type = TensorType(self.shape, self.dtype)
        tensor_constructor_name = self.get_name("tensor_constructor")

        tensor_nil_name = self.get_name("tensor_nil")
        tensor_nil_case = Constructor(tensor_nil_name, [], self.tensor_type_var)
        tensor_case = Constructor(tensor_constructor_name, [tensor_type], self.tensor_type_var)

        self.prelude.mod[self.tensor_type_var] = TypeData(
            self.tensor_type_var, [], [tensor_nil_case, tensor_case]
        )

    def define_tensor_array(self):
        """Defines a function to create a tensor array with size n.
        tensor_array(n) : Tensor[(), int32] -> list[tensor_t]
        """
        tensor_array_constructor_name = self.get_name("tensor_array")
        tensor_array_constructor_var = self._create_global_var(tensor_array_constructor_name)

        tensor_nil_var = self.get_ctor("tensor_nil")
        tensor_type_var = self.get_ctor("tensor_t")
        n = Var("x", scalar_type("int32"))
        body = If(
            equal(n, const(0)),
            self.nil(),
            self.cons(tensor_nil_var(), tensor_array_constructor_var(subtract(n, const(1)))),
        )
        self.prelude.mod[tensor_array_constructor_var] = Function(
            [n], body, self.list(tensor_type_var()), []
        )

    def define_tensor_take(self):
        """Defines a function to return a range of tensor_t on axis 0.
        tensor_take(t, lower, upper) :
        tensor_t -> Tensor[(), int32] -> Tensor[(), int32] -> tensor_t
        """
        # We don't register take for scalar tensor.
        ndim = len(self.shape)
        if ndim == 0:
            return

        take_name = self.get_name("tensor_take")

        if self.is_cached(take_name):
            return

        take_var = GlobalVar(take_name)

        origin_tensor_constructor = self.get_ctor("tensor_constructor")

        output_shape = [
            Any(),
        ] + list(self.shape[1:])
        tensor_type_var, tensor_constructor, _ = self._get_adt_by_shape(output_shape)

        t = Var("tensor", self.tensor_type_var())
        lower = Var("lower", scalar_type("int32"))
        upper = Var("upper", scalar_type("int32"))
        tvar = Var("t")
        case = Clause(
            PatternConstructor(origin_tensor_constructor, [PatternVar(tvar)]),
            tensor_constructor(op.take(tvar, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        self.prelude.mod[take_var] = Function(
            [t, lower, upper], Match(t, [case], False), tensor_type_var(), []
        )

    def define_tensor_concatenate(self):
        """Defines a function to concatenate two tensor_t on axis 0.
        tensor_concatenate(t) : tensor_t -> tensor_t -> tensor_t
        """
        # We don't register concatenate for scalar tensor.
        ndim = len(self.shape)
        if ndim == 0:
            return

        concat_name = self.get_name("tensor_concatenate")
        concat_var = GlobalVar(concat_name)
        if self.is_cached(concat_name):
            return

        output_shape = [
            Any(),
        ] + list(self.shape[1:])
        tensor_type_var, tensor_constructor, _ = self._get_adt_by_shape(output_shape)

        origin_tensor_constructor = self.get_ctor("tensor_constructor")
        origin_tensor_type_var = self.tensor_type_var
        x = Var("x", origin_tensor_type_var())
        y = Var("y", origin_tensor_type_var())
        t1 = Var("t1")
        t2 = Var("t2")

        case = Clause(
            PatternConstructor(origin_tensor_constructor, [PatternVar(t1)]),
            Match(
                y,
                [
                    Clause(
                        PatternConstructor(origin_tensor_constructor, [PatternVar(t2)]),
                        tensor_constructor(op.concatenate([t1, t2], axis=0)),
                    )
                ],
                False,
            ),
        )

        self.prelude.mod[concat_var] = Function(
            [x, y], Match(x, [case], False), tensor_type_var(), []
        )

    def define_tensor_expand_dims(self):
        """Defines a function to grow a tensor_t's rank by adding one dimension in front
        of the original tensor_t.
        tensor_expand_dims(t) : tensor_t -> tensor_t
        """
        expand_dims_name = self.get_name("tensor_expand_dims")
        expand_dims_var = self._create_global_var(expand_dims_name)
        setattr(self.prelude, expand_dims_name, expand_dims_var)
        origin_tensor_type_var = self.tensor_type_var
        origin_tensor_constructor = self.get_ctor("tensor_constructor")
        x = Var("x", origin_tensor_type_var())

        # Note: we set the added axis to be Any() instead of 1 due to
        # in stack op, we need to recursively concatenate.
        new_axis = Any() if self.batch_dim is None or self.batch_dim != 1 else self.batch_dim
        tensor_type_var, tensor_constructor, _ = self._get_adt_by_shape(
            [
                new_axis,
            ]
            + list(self.shape)
        )
        t = Var("t")
        case = Clause(
            PatternConstructor(origin_tensor_constructor, [PatternVar(t)]),
            tensor_constructor(op.expand_dims(t, 0, 1)),
        )

        self.prelude.mod[expand_dims_var] = Function(
            [x], Match(x, [case], False), tensor_type_var(), []
        )

    def define_tensor_array_read(self):
        """Defines a function to get the nth element of a list. Assume the list has at least one
        element.
        tensor_array_read(ta, n) : list[static_tensor_t] -> Tensor[(), int32] ->
        Tensor[self.shape, self.dtype]
        """
        read_name = self.get_name("tensor_array_read")

        if self.is_cached(read_name):
            return

        read_var = GlobalVar(read_name)

        tensor_array = Var("tensor_array", self.list(self.tensor_type_var()))
        n = Var("x", scalar_type("int32"))
        self.prelude.mod[read_var] = Function(
            [tensor_array, n], self.prelude.nth(tensor_array, n), self.tensor_type_var(), []
        )

    def is_cached(self, name):
        try:
            self.prelude.mod.get_global_var(name)
            return True
        except ValueError:
            return False

    def define_tensor_array_write(self):
        """Defines a function to update a tensor array at index n with value v.
        tensor_array_write(ta, n, v) :
            list[static_tensor_t] -> Tensor[(), int32] -> Tensor[self.shape, self.dtype] ->
            list[static_tensor_t]
        """
        write_name = self.get_name("tensor_array_write")
        if self.is_cached(write_name):
            return

        write_var = GlobalVar(write_name)
        tensor_array = Var("tensor_array", self.list(self.tensor_type_var()))
        n = Var("x", scalar_type("int32"))
        v = Var("v", self.tensor_type_var())
        self.prelude.mod[write_var] = Function(
            [tensor_array, n, v],
            self.prelude.update(tensor_array, n, v),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack(self):
        """Defines a function to unstack the values of a tensor_t in a tensor array.
        tensor_array_unstack_tensor(t) : tensor_t -> list[tensor_t]
        """
        ndim = len(self.shape)
        # We don't register unstack for scalar tensor array
        if ndim == 0:
            return

        helper_name = self.get_name("tensor_array_unstack_helper")
        helper_var = self._create_global_var(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType(self.shape, self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))
        tensor_var = Var("tensor", TensorType(self.shape, self.dtype))

        reduced_tensor_type_var, tensor_constructor, _ = self._get_adt_by_shape(self.shape[1:])
        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                tensor_constructor(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(reduced_tensor_type_var()), []
        )

        unstack_name = self.get_name("tensor_array_unstack")
        unstack_var = self._create_global_var(unstack_name)
        setattr(self.prelude, unstack_name, unstack_var)
        shape = op.shape_of(tensor_var)
        unstack_length = op.take(shape, const(0))
        self.prelude.mod[unstack_var] = Function(
            [tensor_var],
            helper_var(const(0), unstack_length, tensor_var),
            self.list(reduced_tensor_type_var()),
            [],
        )

    def define_tensor_array_scatter(self, indices_shape=None, force_update=False):
        """Defines a function to scatter the values of a tensor_t in indices of a tensor array.
        tensor_array_scatter(ta, indices, value) :
            list[tensor_t] -> Tensor[(Any), int32] -> tensor_t -> list[tensor_t]

        Set static indices shape by specifying indices_shape.
        Set force_update to get static indices shape operator.
        """
        # When this operator has already been registered, only update
        # when force_update is set. This should be used only when we need to
        # redefine this op for static indices shape.

        extra_shapes = {"indices": indices_shape} if indices_shape is not None else None
        tensor_array_scatter_name = self.get_name("tensor_array_scatter", extra_shapes)
        if hasattr(self.prelude, tensor_array_scatter_name) and not force_update:
            return

        tensor_array_scatter_helper_name = self.get_name(
            "tensor_array_scatter_helper", extra_shapes
        )

        tensor_array_scatter_helper_var = self._create_global_var(tensor_array_scatter_helper_name)
        ta = Var("ta", self.list(self.tensor_type_var()))
        current = Var("current", scalar_type("int32"))
        limit = Var("limit", scalar_type("int32"))
        indices_ = Var("indices_", TensorType(indices_shape or [Any()], "int32"))
        values_ = Var("values_", self.list(self.tensor_type_var()))
        write_var = self.get_global_var("tensor_array_write")
        read_var = self.get_global_var("tensor_array_read")
        helper_body = If(
            equal(current, limit),
            ta,
            tensor_array_scatter_helper_var(
                write_var(ta, op.take(indices_, current), read_var(values_, current)),
                add(current, const(1)),
                limit,
                indices_,
                values_,
            ),
        )
        self.prelude.mod[tensor_array_scatter_helper_var] = Function(
            [ta, current, limit, indices_, values_],
            helper_body,
            self.list(self.tensor_type_var()),
            [],
        )

        tensor_array_scatter_var = self._create_global_var(tensor_array_scatter_name)
        setattr(self.prelude, tensor_array_scatter_name, tensor_array_scatter_var)
        tensor_array = Var("tensor_array", self.list(self.tensor_type_var()))

        indices = Var("indices", TensorType(indices_shape or [Any()], "int32"))
        values = Var("values", self.list(self.tensor_type_var()))
        if indices_shape is None:
            indices_shape = op.shape_of(indices)
            limit = op.take(indices_shape, const(0))
        else:
            limit = const(indices_shape[0])

        body = tensor_array_scatter_helper_var(tensor_array, const(0), limit, indices, values)
        self.prelude.mod[tensor_array_scatter_var] = Function(
            [tensor_array, indices, values], body, self.list(self.tensor_type_var()), []
        )

    def define_tensor_array_split(self, value_shape=None, lengths_shape=None, force_update=False):
        """Defines a function to split the values of a tensor_t into a tensor array.
        tensor_array_split(ta, value, lengths) :
            list[tensor_t] -> tensor_t -> Tensor[(Any), int32] -> list[tensor_t]

        Set static value and lengths shapes by specifying value_shape and lengths_shape.
        Set force_update to get static value and lengths shape operator.
        """
        # Skip scalar case
        ndim = len(self.shape)
        if ndim == 0:
            return

        # When this operator has already been registered, only update
        # when force_update is set. This should be used only when we need to
        # redefine this op for static value/indices shape.
        split_name = self.get_name("tensor_array_split")

        if self.is_cached(split_name):
            if not force_update:
                return
            tensor_array_split_helper_var = self.get_global_var("ta_split_helper")
            split_var = self.get_global_var("tensor_array_split")
        else:
            tensor_array_split_helper_name = self.get_name("ta_split_helper")
            tensor_array_split_helper_var = GlobalVar(tensor_array_split_helper_name)
            split_var = GlobalVar(split_name)

        output_shape = [
            Any(),
        ] + list(self.shape[1:])
        output_tensor_type_var, _, output_ops = self._get_adt_by_shape(output_shape)
        output_ops.define_tensor_array_write()
        write_var = output_ops.get_global_var("tensor_array_write")

        if value_shape is None:
            value_type_var = self.tensor_type_var
            take_var = self.get_global_var("tensor_take")
        else:
            value_type_var, _, value_adts = self._get_adt_by_shape(value_shape)
            value_adts.define_tensor_take()
            take_var = value_adts.get_global_var("tensor_take")

        ta1 = Var("tensor_array", self.list(output_tensor_type_var()))
        value1 = Var("value1", value_type_var())
        offset1 = Var("offset1", scalar_type("int32"))
        current1 = Var("current1", scalar_type("int32"))
        limit1 = Var("limit1", scalar_type("int32"))
        lengths1 = Var("lengths", TensorType(lengths_shape or [Any()], "int32"))

        helper1_body = If(
            equal(current1, limit1),
            ta1,
            write_var(
                tensor_array_split_helper_var(
                    ta1,
                    value1,
                    add(offset1, op.take(lengths1, current1)),
                    add(current1, const(1)),
                    limit1,
                    lengths1,
                ),
                current1,
                take_var(value1, offset1, add(op.take(lengths1, current1), offset1)),
            ),
        )

        self.prelude.mod[tensor_array_split_helper_var] = Function(
            [ta1, value1, offset1, current1, limit1, lengths1],
            helper1_body,
            self.list(output_tensor_type_var()),
            [],
        )
        tensor_array = Var("tensor_array", self.list(output_tensor_type_var()))

        value = Var("value", value_type_var())
        lengths = Var("lengths", TensorType(lengths_shape or [Any()], "int32"))
        if lengths_shape is None:
            lengths_shape = op.shape_of(lengths)
            lengths_limit = op.take(lengths_shape, const(0))
        else:
            lengths_limit = const(lengths_shape[0])
        body = tensor_array_split_helper_var(
            tensor_array, value, const(0), const(0), lengths_limit, lengths
        )

        self.prelude.mod[split_var] = Function(
            [tensor_array, value, lengths], body, self.list(output_tensor_type_var()), []
        )

    def define_tensor_array_concat(self):
        """Defines a function to return the values in the tensor array as concatenated tensor_t.
        tensor_array_concat(ta) : list[tensor_t] -> tensor_t
        """
        # We don't register concat for scalar tensor array.
        ndim = len(self.shape)
        if ndim == 0:
            return

        concat_name = self.get_name("tensor_array_concat")

        if self.is_cached(concat_name):
            return

        concat_var = GlobalVar(concat_name)

        output_shape = [
            Any(),
        ] + list(self.shape[1:])

        tensor_type_var, _, output_ops = self._get_adt_by_shape(output_shape)

        # Register tensor concatenate and get tensor_nil var for output shape
        output_ops.define_tensor_concatenate()
        tensor_concat_var = output_ops.get_global_var("tensor_concatenate")
        tensor_nil_var = output_ops.get_ctor("tensor_nil")

        tensor_array = Var("tensor_array", self.list(tensor_type_var()))
        hd = Var("hd")
        tl = Var("tl")
        nil_case = Clause(PatternConstructor(self.nil), tensor_nil_var())
        cons_case = Clause(
            PatternConstructor(self.cons, [PatternVar(hd), PatternVar(tl)]),
            Match(
                tl,
                [
                    Clause(PatternConstructor(self.nil), hd),
                    Clause(PatternWildcard(), tensor_concat_var(hd, concat_var(tl))),
                ],
                False,
            ),
        )
        self.prelude.mod[concat_var] = Function(
            [tensor_array], Match(tensor_array, [nil_case, cons_case], False), tensor_type_var(), []
        )

    def define_tensor_array_stack(self):
        """Defines a function to get the values in the tensor array as a stack tensor_t.
        tensor_array_stack(l) : list[tensor_t] -> tensor_t
        """
        stack_name = self.get_name("tensor_array_stack")
        stack_var = self._create_global_var(stack_name)
        setattr(self.prelude, stack_name, stack_var)
        tensor_array = Var("tensor_array", self.list(self.tensor_type_var()))
        expand_dims_var = self.get_global_var("tensor_expand_dims")

        # Register tensor_concatenate for output_shape
        new_axis = Any() if not self.batch_dim or self.batch_dim != 1 else self.batch_dim
        output_shape = [
            new_axis,
        ] + list(self.shape)
        _, _, output_ops = self._get_adt_by_shape(output_shape)
        output_ops.define_tensor_concatenate()
        concat_var = output_ops.get_global_var("tensor_concatenate")

        tensor_array_expand_dims = self.prelude.map(expand_dims_var, tensor_array)
        if self.batch_dim is not None and self.batch_dim == 1:
            # only one element
            tensors = self.prelude.id(
                self.prelude.hd(tensor_array_expand_dims),
            )
        else:
            tensors = self.prelude.foldl(
                concat_var,
                self.prelude.hd(tensor_array_expand_dims),
                self.prelude.tl(tensor_array_expand_dims),
            )

        output_tensor_type_var, _, _ = self._get_adt_by_shape(output_shape)
        self.prelude.mod[stack_var] = Function(
            [tensor_array], tensors, output_tensor_type_var(), []
        )

    def define_tensor_array_gather(self):
        """Defines a function to return the selected values in a tensor array as tensor_t.
        tensor_array_gather(ta, indices) : list[tensor_t] -> Tensor[(Any), int32] -> tensor_t
        """
        helper_name = self.get_name("tensor_array_gather_helper")
        helper_var = self._create_global_var(helper_name)

        new_axis = Any() if self.batch_dim is None or self.batch_dim != 1 else self.batch_dim
        output_shape = [
            new_axis,
        ] + list(self.shape)
        output_tensor_type_var, _, _ = self._get_adt_by_shape(output_shape)
        stack_var = self.get_global_var("tensor_array_stack")
        read_var = self.get_global_var("tensor_array_read")
        ta = Var("ta", self.list(self.tensor_type_var()))
        accu = Var("accu", self.list(self.tensor_type_var()))
        current = Var("current", scalar_type("int32"))
        limit = Var("limit", scalar_type("int32"))
        indices_ = Var("indices_", TensorType([Any()], "int32"))
        helper_body = If(
            equal(current, const(0)),
            stack_var(accu),
            helper_var(
                ta,
                self.cons(read_var(ta, op.take(indices_, subtract(current, const(1)))), accu),
                subtract(current, const(1)),
                limit,
                indices_,
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [ta, accu, current, limit, indices_], helper_body, output_tensor_type_var(), []
        )
        gather_name = self.get_name("tensor_array_gather")
        gather_var = self._create_global_var(gather_name)

        tensor_array = Var("tensor_array", self.list(self.tensor_type_var()))
        indices = Var("indices", TensorType([Any()], "int32"))
        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = helper_var(tensor_array, self.nil(), limit, limit, indices)
        self.prelude.mod[gather_var] = Function(
            [tensor_array, indices], body, output_tensor_type_var(), []
        )

    def define_tensor_get_data(self):
        """Defines a function to get a Tensor from tensor_t with given shape."""
        tensor_get_data_name = self.get_name("tensor_get_data")
        tensor_get_data_var = self._create_global_var(tensor_get_data_name)

        tensor_constructor = self.get_ctor("tensor_constructor")
        t = Var("tensor", self.tensor_type_var())
        tvar = Var("t")
        case = Clause(PatternConstructor(tensor_constructor, [PatternVar(tvar)]), tvar)
        self.prelude.mod[tensor_get_data_var] = Function(
            [t], Match(t, [case], False), TensorType(self.shape, self.dtype), []
        )

    def register(self):
        """Register all tensor array ops in Prelude"""
        self.define_tensor_adt()
        self.define_tensor_take()
        self.define_tensor_concatenate()
        self.define_tensor_expand_dims()
        self.define_tensor_array()
        self.define_tensor_array_read()
        self.define_tensor_array_write()
        self.define_tensor_array_unstack()
        self.define_tensor_array_scatter()
        self.define_tensor_array_split()
        self.define_tensor_array_concat()
        self.define_tensor_array_stack()
        self.define_tensor_array_gather()
        self.define_tensor_get_data()

    def _get_adt_by_shape(self, shape):
        """Get ADT type and constructor with given shape."""
        adt_ops = StaticTensorArrayOps(self.prelude, self.dtype, shape, self.batch_dim)
        adt_ops.define_tensor_adt()
        tensor_type_var = adt_ops.get_type("tensor_t")
        tensor_constructor = adt_ops.get_ctor("tensor_constructor")
        return tensor_type_var, tensor_constructor, adt_ops

    def _create_global_var(self, name):
        """Create a GlobalVar if doesn't exist in prelude."""
        global_var_name_set = set()
        for g_var_name in self.prelude.mod.get_global_vars():
            global_var_name_set.add(g_var_name.name_hint)
        if name not in global_var_name_set:
            gvar = GlobalVar(name)
        else:
            gvar = self.prelude.mod.get_global_var(name)

        return gvar


class TensorArrayOps(object):
    """Contains tensor array related ops"""

    def __init__(self, prelude, dtype):
        """Create tensor array ops registry"""
        self.prelude = prelude
        self.dtype = dtype
        self.list, self.cons, self.nil = self.prelude.mod.get_type("List")

    def get_name(self, canonical):
        """Get name corresponding to the canonical name"""
        return self.prelude.get_name(canonical, self.dtype)

    def get_global_var(self, canonical):
        """Get global corresponding to the canonical name"""
        return self.prelude.get_global_var(canonical, self.dtype)

    def get_type(self, canonical):
        """Get type corresponding to the canonical name"""
        return self.prelude.get_type(canonical, self.dtype)

    def get_ctor(self, canonical):
        """Get ctor corresponding to the canonical name"""
        return self.prelude.get_ctor(self.tensor_type_var.name_hint, canonical, self.dtype)

    def define_tensor_adt(self):
        """Defines the dynamic tensor ADT, which is the container for tensors
        with variable shapes."""
        tensor_type_name = self.get_name("tensor_t")
        self.tensor_type_var = tensor_type_var = GlobalTypeVar(tensor_type_name)

        tensor0_type = TensorType([], self.dtype)
        tensor1_type = TensorType([Any()], self.dtype)
        tensor2_type = TensorType([Any(), Any()], self.dtype)
        tensor3_type = TensorType([Any(), Any(), Any()], self.dtype)
        tensor4_type = TensorType([Any(), Any(), Any(), Any()], self.dtype)
        tensor5_type = TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype)
        tensor6_type = TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype)
        tensor_nil_name = self.get_name("tensor_nil")
        tensor0_name = self.get_name("tensor0")
        tensor1_name = self.get_name("tensor1")
        tensor2_name = self.get_name("tensor2")
        tensor3_name = self.get_name("tensor3")
        tensor4_name = self.get_name("tensor4")
        tensor5_name = self.get_name("tensor5")
        tensor6_name = self.get_name("tensor6")
        tensor_nil_case = Constructor(tensor_nil_name, [], tensor_type_var)
        tensor0_case = Constructor(tensor0_name, [tensor0_type], tensor_type_var)
        tensor1_case = Constructor(tensor1_name, [tensor1_type], tensor_type_var)
        tensor2_case = Constructor(tensor2_name, [tensor2_type], tensor_type_var)
        tensor3_case = Constructor(tensor3_name, [tensor3_type], tensor_type_var)
        tensor4_case = Constructor(tensor4_name, [tensor4_type], tensor_type_var)
        tensor5_case = Constructor(tensor5_name, [tensor5_type], tensor_type_var)
        tensor6_case = Constructor(tensor6_name, [tensor6_type], tensor_type_var)

        self.prelude.mod[tensor_type_var] = TypeData(
            tensor_type_var,
            [],
            [
                tensor_nil_case,
                tensor0_case,
                tensor1_case,
                tensor2_case,
                tensor3_case,
                tensor4_case,
                tensor5_case,
                tensor6_case,
            ],
        )

    def define_tensor_take(self):
        """Defines a function to return a range of tensor_t on axis 0.
        tensor_take(t, lower, upper) :
        tensor_t -> Tensor[(), int32] -> Tensor[(), int32] -> tensor_t
        """
        take_name = self.get_name("tensor_take")
        take_var = GlobalVar(take_name)

        tensor_t = self.tensor_type_var
        tensor1_var = self.get_ctor("tensor1")
        tensor2_var = self.get_ctor("tensor2")
        tensor3_var = self.get_ctor("tensor3")
        tensor4_var = self.get_ctor("tensor4")
        tensor5_var = self.get_ctor("tensor5")
        tensor6_var = self.get_ctor("tensor6")

        t = Var("tensor", tensor_t())
        lower = Var("lower", scalar_type("int32"))
        upper = Var("upper", scalar_type("int32"))
        t1 = Var("t1")
        t2 = Var("t2")
        t3 = Var("t3")
        t4 = Var("t4")
        t5 = Var("t5")
        t6 = Var("t6")
        tensor1_case = Clause(
            PatternConstructor(tensor1_var, [PatternVar(t1)]),
            tensor1_var(op.take(t1, op.arange(lower, upper, dtype="int32"))),
        )
        tensor2_case = Clause(
            PatternConstructor(tensor2_var, [PatternVar(t2)]),
            tensor2_var(op.take(t2, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        tensor3_case = Clause(
            PatternConstructor(tensor3_var, [PatternVar(t3)]),
            tensor3_var(op.take(t3, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        tensor4_case = Clause(
            PatternConstructor(tensor4_var, [PatternVar(t4)]),
            tensor4_var(op.take(t4, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        tensor5_case = Clause(
            PatternConstructor(tensor5_var, [PatternVar(t5)]),
            tensor5_var(op.take(t5, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        tensor6_case = Clause(
            PatternConstructor(tensor6_var, [PatternVar(t6)]),
            tensor6_var(op.take(t6, op.arange(lower, upper, dtype="int32"), axis=0)),
        )
        self.prelude.mod[take_var] = Function(
            [t, lower, upper],
            Match(
                t,
                [
                    tensor1_case,
                    tensor2_case,
                    tensor3_case,
                    tensor4_case,
                    tensor5_case,
                    tensor6_case,
                ],
                False,
            ),
            tensor_t(),
            [],
        )

    def define_tensor_expand_dims(self):
        """Defines a function to grow a tensor_t's rank by adding one dimension in front
        of the original tensor_t.
        tensor_expand_dims(t) : tensor_t -> tensor_t
        """
        expand_dims_name = self.get_name("tensor_expand_dims")
        expand_dims_var = GlobalVar(expand_dims_name)
        tensor_type_var = self.tensor_type_var

        x = Var("x", tensor_type_var())
        t0 = Var("t0")
        t1 = Var("t1")
        t2 = Var("t2")
        t3 = Var("t3")
        t4 = Var("t4")
        t5 = Var("t5")
        tensor0_var = self.get_ctor("tensor0")
        tensor1_var = self.get_ctor("tensor1")
        tensor2_var = self.get_ctor("tensor2")
        tensor3_var = self.get_ctor("tensor3")
        tensor4_var = self.get_ctor("tensor4")
        tensor5_var = self.get_ctor("tensor5")
        tensor6_var = self.get_ctor("tensor6")
        tensor0_case = Clause(
            PatternConstructor(tensor0_var, [PatternVar(t0)]), tensor1_var(op.expand_dims(t0, 0, 1))
        )
        tensor1_case = Clause(
            PatternConstructor(tensor1_var, [PatternVar(t1)]), tensor2_var(op.expand_dims(t1, 0, 1))
        )
        tensor2_case = Clause(
            PatternConstructor(tensor2_var, [PatternVar(t2)]), tensor3_var(op.expand_dims(t2, 0, 1))
        )
        tensor3_case = Clause(
            PatternConstructor(tensor3_var, [PatternVar(t3)]), tensor4_var(op.expand_dims(t3, 0, 1))
        )
        tensor4_case = Clause(
            PatternConstructor(tensor4_var, [PatternVar(t4)]), tensor5_var(op.expand_dims(t4, 0, 1))
        )
        tensor5_case = Clause(
            PatternConstructor(tensor5_var, [PatternVar(t5)]), tensor6_var(op.expand_dims(t5, 0, 1))
        )
        self.prelude.mod[expand_dims_var] = Function(
            [x],
            Match(
                x,
                [
                    tensor0_case,
                    tensor1_case,
                    tensor2_case,
                    tensor3_case,
                    tensor4_case,
                    tensor5_case,
                ],
                False,
            ),
            tensor_type_var(),
        )

    def define_tensor_concat(self):
        """Defines a function to concatenate two tensor_t on the first axis

        tensor_concatenate(t) : tensor_t -> tensor_t -> tensor_t
        """
        concat_name = self.get_name("tensor_concatenate")
        concat_var = GlobalVar(concat_name)

        tensor_type_var = self.tensor_type_var
        x = Var("x", tensor_type_var())
        y = Var("y", tensor_type_var())

        tensor1_var = self.get_ctor("tensor1")
        tensor2_var = self.get_ctor("tensor2")
        tensor3_var = self.get_ctor("tensor3")
        tensor4_var = self.get_ctor("tensor4")
        t11 = Var("t11")
        t12 = Var("t12")
        t21 = Var("t21")
        t22 = Var("t22")
        t31 = Var("t31")
        t32 = Var("t32")
        t41 = Var("t41")
        t42 = Var("t42")
        tensor1_case = Clause(
            PatternConstructor(tensor1_var, [PatternVar(t11)]),
            Match(
                y,
                [
                    Clause(
                        PatternConstructor(tensor1_var, [PatternVar(t12)]),
                        tensor1_var(op.concatenate([t11, t12], axis=0)),
                    )
                ],
                False,
            ),
        )
        tensor2_case = Clause(
            PatternConstructor(tensor2_var, [PatternVar(t21)]),
            Match(
                y,
                [
                    Clause(
                        PatternConstructor(tensor2_var, [PatternVar(t22)]),
                        tensor2_var(op.concatenate([t21, t22], axis=0)),
                    )
                ],
                False,
            ),
        )
        tensor3_case = Clause(
            PatternConstructor(tensor3_var, [PatternVar(t31)]),
            Match(
                y,
                [
                    Clause(
                        PatternConstructor(tensor3_var, [PatternVar(t32)]),
                        tensor3_var(op.concatenate([t31, t32], axis=0)),
                    )
                ],
                False,
            ),
        )
        tensor4_case = Clause(
            PatternConstructor(tensor4_var, [PatternVar(t41)]),
            Match(
                y,
                [
                    Clause(
                        PatternConstructor(tensor4_var, [PatternVar(t42)]),
                        tensor4_var(op.concatenate([t41, t42], axis=0)),
                    )
                ],
                False,
            ),
        )
        # op.concatenate does not support tensor with rank higher than 4
        self.prelude.mod[concat_var] = Function(
            [x, y],
            Match(x, [tensor1_case, tensor2_case, tensor3_case, tensor4_case], False),
            tensor_type_var(),
        )

    def define_tensor_array(self):
        """Defines a function to create a tensor array with size n.
        tensor_array(n) : Tensor[(), int32] -> list[tensor_t]
        """
        tensor_array_constructor_name = self.get_name("tensor_array")
        tensor_array_constructor_var = GlobalVar(tensor_array_constructor_name)
        setattr(self.prelude, tensor_array_constructor_name, tensor_array_constructor_var)
        tensor_nil_var = self.get_ctor("tensor_nil")
        tensor_type_var = self.get_ctor("tensor_t")
        n = Var("x", scalar_type("int32"))
        body = If(
            equal(n, const(0)),
            self.nil(),
            self.cons(tensor_nil_var(), tensor_array_constructor_var(subtract(n, const(1)))),
        )
        self.prelude.mod[tensor_array_constructor_var] = Function(
            [n], body, self.list(tensor_type_var()), []
        )

    def define_tensor_array_read(self):
        """Defines a function to get the head of a list. Assume the list has at least one
        element.

        tensor_array_read(ta, n) : list[tensor_t] -> Tensor[(), int32] -> tensor_t
        """
        read_name = self.get_name("tensor_array_read")
        read_var = GlobalVar(read_name)
        setattr(self.prelude, read_name, read_var)
        tensor_type_var = self.tensor_type_var

        tensor_array = Var("tensor_array", self.list(tensor_type_var()))
        n = Var("x", scalar_type("int32"))
        self.prelude.mod[read_var] = Function(
            [tensor_array, n], self.prelude.nth(tensor_array, n), tensor_type_var(), []
        )

    def define_tensor_array_write(self):
        """Defines a function to update a tensor array at index n with value v.
        tensor_array_write(ta, n, v) :
            list[tensor_t] -> Tensor[(), int32] -> tensor_t -> list[tensor_t]
        """
        write_name = self.get_name("tensor_array_write")
        write_var = GlobalVar(write_name)

        tensor_type_var = self.tensor_type_var
        tensor_array = Var("tensor_array", self.list(tensor_type_var()))
        n = Var("x", scalar_type("int32"))
        v = Var("v", tensor_type_var())
        self.prelude.mod[write_var] = Function(
            [tensor_array, n, v],
            self.prelude.update(tensor_array, n, v),
            self.list(tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack_tensor1(self):
        """Defines a function to unstack the values of a tensor_t with rank 1 in a tensor array.
        tensor_array_unstack_tensor1(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor1_helper")
        helper_var = GlobalVar(helper_name)
        tensor = Var("t", TensorType([Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))
        tensor_type_var = self.tensor_type_var
        tensor0_var = self.get_ctor("tensor0")
        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(tensor0_var(op.take(tensor, i)), helper_var(add(i, const(1)), up, tensor)),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(tensor_type_var()), []
        )
        unstack_name = self.get_name("tensor_array_unstack_tensor1")
        unstack_var = GlobalVar(unstack_name)
        tensor1 = Var("tensor", TensorType([Any()], self.dtype))
        shape = op.shape_of(tensor1)
        ndim = op.take(shape, const(0))
        self.prelude.mod[unstack_var] = Function(
            [tensor1], helper_var(const(0), ndim, tensor1), self.list(tensor_type_var()), []
        )

    def define_tensor_array_unstack_tensor2(self):
        """Defines a function to unstack the values of a tensor_t with rank 2 in a tensor array.

        tensor_array_unstack_tensor2(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor2_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))

        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                self.get_ctor("tensor1")(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(self.tensor_type_var()), []
        )

        tensor_array_unstack_tensor2_name = self.get_name("tensor_array_unstack_tensor2")
        tensor_array_unstack_tensor2_var = GlobalVar(tensor_array_unstack_tensor2_name)
        setattr(self.prelude, tensor_array_unstack_tensor2_name, tensor_array_unstack_tensor2_var)
        tensor2 = Var("tensor", TensorType([Any(), Any()], self.dtype))
        shape = op.shape_of(tensor2)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor2_var] = Function(
            [tensor2],
            helper_var(const(0), ndim, tensor2),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack_tensor3(self):
        """Defines a function to unstack the values of a tensor_t with rank 3 in a tensor array.

        tensor_array_unstack_tensor3(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor3_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))

        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                self.get_ctor("tensor2")(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(self.tensor_type_var()), []
        )

        tensor_array_unstack_tensor3_name = self.get_name("tensor_array_unstack_tensor3")
        tensor_array_unstack_tensor3_var = GlobalVar(tensor_array_unstack_tensor3_name)
        setattr(self.prelude, tensor_array_unstack_tensor3_name, tensor_array_unstack_tensor3_var)
        tensor3 = Var("tensor", TensorType([Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor3)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor3_var] = Function(
            [tensor3],
            helper_var(const(0), ndim, tensor3),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack_tensor4(self):
        """Defines a function to unstack the values of a tensor_t with rank 4 in a tensor array.

        tensor_array_unstack_tensor4(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor4_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))

        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                self.get_ctor("tensor3")(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(self.tensor_type_var()), []
        )

        tensor_array_unstack_tensor4_name = self.get_name("tensor_array_unstack_tensor4")
        tensor_array_unstack_tensor4_var = GlobalVar(tensor_array_unstack_tensor4_name)
        setattr(self.prelude, tensor_array_unstack_tensor4_name, tensor_array_unstack_tensor4_var)
        tensor4 = Var("tensor", TensorType([Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor4)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor4_var] = Function(
            [tensor4],
            helper_var(const(0), ndim, tensor4),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack_tensor5(self):
        """Defines a function to unstack the values of a tensor_t with rank 5 in a tensor array.

        tensor_array_unstack_tensor5(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor5_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))

        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                self.get_ctor("tensor4")(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(self.tensor_type_var()), []
        )

        tensor_array_unstack_tensor5_name = self.get_name("tensor_array_unstack_tensor5")
        tensor_array_unstack_tensor5_var = GlobalVar(tensor_array_unstack_tensor5_name)
        setattr(self.prelude, tensor_array_unstack_tensor5_name, tensor_array_unstack_tensor5_var)
        tensor5 = Var("tensor", TensorType([Any(), Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor5)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor5_var] = Function(
            [tensor5],
            helper_var(const(0), ndim, tensor5),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_unstack_tensor6(self):
        """Defines a function to unstack the values of a tensor_t with rank 6 in a tensor array.

        tensor_array_unstack_tensor6(t) : tensor_t -> list[tensor_t]
        """
        helper_name = self.get_name("tensor_array_unstack_tensor6_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor = Var("t", TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype))
        up = Var("up", scalar_type("int32"))
        i = Var("i", scalar_type("int32"))

        helper_body = If(
            equal(i, up),
            self.nil(),
            self.cons(
                self.get_ctor("tensor5")(op.take(tensor, i, axis=0)),
                helper_var(add(i, const(1)), up, tensor),
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [i, up, tensor], helper_body, self.list(self.tensor_type_var()), []
        )

        tensor_array_unstack_tensor6_name = self.get_name("tensor_array_unstack_tensor6")
        tensor_array_unstack_tensor6_var = GlobalVar(tensor_array_unstack_tensor6_name)
        setattr(self.prelude, tensor_array_unstack_tensor6_name, tensor_array_unstack_tensor6_var)
        tensor6 = Var("tensor", TensorType([Any(), Any(), Any(), Any(), Any(), Any()], self.dtype))
        shape = op.shape_of(tensor6)
        ndim = op.take(shape, const(0))
        self.prelude.mod[tensor_array_unstack_tensor6_var] = Function(
            [tensor6],
            helper_var(const(0), ndim, tensor6),
            self.list(self.tensor_type_var()),
            [],
        )

    def define_tensor_array_scatter(self):
        """Defines a function to scatter the values of a tensor_t in indices of a tensor array.
        tensor_array_scatter(ta, indices, value) :
            list[tensor_t] -> Tensor[(Any), int32] -> tensor_t -> list[tensor_t]
        """
        tensor_array_scatter_helper_name = self.get_name("tensor_array_scatter_helper")
        tensor_array_scatter_helper_var = GlobalVar(tensor_array_scatter_helper_name)
        tensor_t = self.tensor_type_var
        ta = Var("ta", self.list(tensor_t()))
        current = Var("current", scalar_type("int32"))
        limit = Var("limit", scalar_type("int32"))
        indices_ = Var("indices_", TensorType([Any()], "int32"))
        values_ = Var("values_", self.list(tensor_t()))
        write_var = self.get_global_var("tensor_array_write")
        read_var = self.get_global_var("tensor_array_read")
        helper_body = If(
            equal(current, limit),
            ta,
            tensor_array_scatter_helper_var(
                write_var(ta, op.take(indices_, current), read_var(values_, current)),
                add(current, const(1)),
                limit,
                indices_,
                values_,
            ),
        )
        self.prelude.mod[tensor_array_scatter_helper_var] = Function(
            [ta, current, limit, indices_, values_], helper_body, self.list(tensor_t()), []
        )
        tensor_array_scatter_name = self.get_name("tensor_array_scatter")
        tensor_array_scatter_var = GlobalVar(tensor_array_scatter_name)
        setattr(self.prelude, tensor_array_scatter_name, tensor_array_scatter_var)
        tensor_array = Var("tensor_array", self.list(tensor_t()))
        indices = Var("indices", TensorType([Any()], "int32"))
        values = Var("values", self.list(tensor_t()))
        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = tensor_array_scatter_helper_var(tensor_array, const(0), limit, indices, values)
        self.prelude.mod[tensor_array_scatter_var] = Function(
            [tensor_array, indices, values], body, self.list(tensor_t()), []
        )

    def define_tensor_array_split(self):
        """Defines a function to split the values of a tensor_t into a tensor array.
        tensor_array_split(ta, value, lengths) :
            list[tensor_t] -> tensor_t -> Tensor[(Any), int32] -> list[tensor_t]
        """
        tensor_t = self.tensor_type_var
        tensor_array_split_helper_name = self.get_name("ta_split_helper")
        tensor_array_split_helper_var = GlobalVar(tensor_array_split_helper_name)
        setattr(self.prelude, tensor_array_split_helper_name, tensor_array_split_helper_var)
        ta1 = Var("tensor_array", self.list(tensor_t()))
        value1 = Var("value1", tensor_t())
        offset1 = Var("offset1", scalar_type("int32"))
        current1 = Var("current1", scalar_type("int32"))
        limit1 = Var("limit1", scalar_type("int32"))
        lengths1 = Var("lengths", TensorType([Any()], "int32"))
        write_var = self.get_global_var("tensor_array_write")
        take_var = self.get_global_var("tensor_take")
        helper1_body = If(
            equal(current1, limit1),
            ta1,
            write_var(
                tensor_array_split_helper_var(
                    ta1,
                    value1,
                    add(offset1, op.take(lengths1, current1)),
                    add(current1, const(1)),
                    limit1,
                    lengths1,
                ),
                current1,
                take_var(value1, offset1, add(op.take(lengths1, current1), offset1)),
            ),
        )
        self.prelude.mod[tensor_array_split_helper_var] = Function(
            [ta1, value1, offset1, current1, limit1, lengths1],
            helper1_body,
            self.list(tensor_t()),
            [],
        )
        split_name = self.get_name("tensor_array_split")
        split_var = GlobalVar(split_name)
        setattr(self.prelude, split_name, split_var)
        tensor_array = Var("tensor_array", self.list(tensor_t()))
        value = Var("value", tensor_t())
        lengths = Var("lengths", TensorType([Any()], "int32"))
        lengths_shape = op.shape_of(lengths)
        lengths_limit = op.take(lengths_shape, const(0))
        body = tensor_array_split_helper_var(
            tensor_array, value, const(0), const(0), lengths_limit, lengths
        )
        self.prelude.mod[split_var] = Function(
            [tensor_array, value, lengths], body, self.list(tensor_t()), []
        )

    def define_tensor_array_concat(self):
        """Defines a function to return the values in the tensor array as concatenated tensor_t.
        tensor_array_concat(ta) : list[tensor_t] -> tensor_t
        """
        concat_name = self.get_name("tensor_array_concat")
        concat_var = GlobalVar(concat_name)
        setattr(self.prelude, concat_name, concat_var)
        tensor_concat_var = self.get_global_var("tensor_concatenate")
        tensor_t = self.tensor_type_var
        tensor_nil_var = self.get_ctor("tensor_nil")
        tensor_array = Var("tensor_array", self.list(tensor_t()))
        hd = Var("hd")
        tl = Var("tl")
        nil_case = Clause(PatternConstructor(self.nil), tensor_nil_var())
        cons_case = Clause(
            PatternConstructor(self.cons, [PatternVar(hd), PatternVar(tl)]),
            Match(
                tl,
                [
                    Clause(PatternConstructor(self.nil), hd),
                    Clause(PatternWildcard(), tensor_concat_var(hd, concat_var(tl))),
                ],
                False,
            ),
        )
        self.prelude.mod[concat_var] = Function(
            [tensor_array], Match(tensor_array, [nil_case, cons_case], False), tensor_t(), []
        )

    def define_tensor_array_gather(self):
        """Defines a function to return the selected values in a tensor array as tensor_t.
        tensor_array_gather(ta, indices) : list[tensor_t] -> Tensor[(Any), int32] -> tensor_t
        """
        helper_name = self.get_name("tensor_array_gather_helper")
        helper_var = GlobalVar(helper_name)
        setattr(self.prelude, helper_name, helper_var)
        tensor_type_var = self.tensor_type_var
        stack_var = self.get_var("tensor_array_stack")
        read_var = self.get_var("tensor_array_read")
        ta = Var("ta", self.list(tensor_type_var()))
        accu = Var("accu", self.list(tensor_type_var()))
        current = Var("current", scalar_type("int32"))
        limit = Var("limit", scalar_type("int32"))
        indices_ = Var("indices_", TensorType([Any()], "int32"))
        helper_body = If(
            equal(current, const(0)),
            stack_var(accu),
            helper_var(
                ta,
                self.cons(read_var(ta, op.take(indices_, subtract(current, const(1)))), accu),
                subtract(current, const(1)),
                limit,
                indices_,
            ),
        )
        self.prelude.mod[helper_var] = Function(
            [ta, accu, current, limit, indices_], helper_body, tensor_type_var(), []
        )
        gather_name = self.get_name("tensor_array_gather")
        gather_var = GlobalVar(gather_name)
        setattr(self.prelude, gather_name, gather_var)
        tensor_array = Var("tensor_array", self.list(tensor_type_var()))
        indices = Var("indices", TensorType([Any()], "int32"))
        indices_shape = op.shape_of(indices)
        limit = op.take(indices_shape, const(0))
        body = helper_var(tensor_array, self.nil(), limit, limit, indices)
        self.prelude.mod[gather_var] = Function(
            [tensor_array, indices], body, tensor_type_var(), []
        )

    def define_tensor_array_stack(self):
        """Defines a function to get the values in the tensor array as a stack tensor_t.
        tensor_array_stack(l) : list[tensor_t] -> tensor_t
        """
        stack_name = self.get_name("tensor_array_stack")
        stack_var = GlobalVar(stack_name)
        setattr(self.prelude, stack_name, stack_var)
        tensor_type_var = self.tensor_type_var
        tensor_array = Var("tensor_array", self.list(tensor_type_var()))
        expand_dims_var = self.get_global_var("tensor_expand_dims")
        concat_var = self.get_global_var("tensor_concatenate")

        tensor_array_expand_dims = self.prelude.map(expand_dims_var, tensor_array)
        tensors = self.prelude.foldl(
            concat_var,
            self.prelude.hd(tensor_array_expand_dims),
            self.prelude.tl(tensor_array_expand_dims),
        )
        self.prelude.mod[stack_var] = Function(
            [tensor_array], ToANormalFormExpr(tensors), tensor_type_var(), []
        )

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
            mod = IRModule()
        self.mod = mod
        self.load_prelude()

    def get_name(self, canonical, dtype):
        """Get name corresponding to the canonical name"""
        if canonical == "tensor_t":
            return "tensor_{}_t".format(dtype)
        return "{}_{}".format(canonical, dtype)

    def get_global_var(self, canonical, dtype):
        """Get global var corresponding to the canonical name"""
        name = self.get_name(canonical, dtype)
        return self.mod.get_global_var(name)

    def get_type(self, canonical, dtype):
        """Get type corresponding to the canonical name"""
        name = self.get_name(canonical, dtype)
        return self.mod.get_global_type_var(name)

    def get_ctor(self, ty_name, canonical, dtype):
        """Get constructor corresponding to the canonical name"""
        name = self.get_name(canonical, dtype)
        ctors = self.mod.get_type(ty_name)
        for ctor in ctors:
            if ctor.name_hint == name:
                return ctor
        raise Exception(f"could not find {name}")

    def get_tensor_ctor(self, canonical, dtype):
        ty = self.get_type("tensor_t", dtype)
        return self.get_ctor(ty.name_hint, canonical, dtype)

    def get_name_static(self, canonical, dtype, shape, batch_dim=None):
        """Get name corresponding to the canonical name"""
        return _get_name_static(canonical, dtype, shape, batch_dim)

    def get_global_var_static(self, canonical, dtype, shape, batch_dim=None):
        """Get var corresponding to the canonical name"""
        name = self.get_name_static(canonical, dtype, shape, batch_dim)
        return self.mod.get_global_var(name)

    def get_type_static(self, canonical, dtype, shape):
        """Get type corresponding to the canonical name"""
        name = self.get_name_static(canonical, dtype, shape)
        return self.mod.get_global_type_var(name)

    def get_ctor_static(self, ty_name, name, dtype, shape):
        """Get constructor corresponding to the canonical name"""
        ty_name = self.get_name_static(ty_name, dtype, shape)
        name = self.get_name_static(name, dtype, shape)
        ctors = self.mod.get_type(ty_name)
        for ctor in ctors:
            if ctor.name_hint == name:
                return ctor
        raise Exception(f"could not find {name}")

    def get_tensor_ctor_static(self, name, dtype, shape):
        """Get constructor corresponding to the canonical name"""
        return self.get_ctor_static("tensor_t", name, dtype, shape)

    def load_prelude(self):
        """Parses the Prelude from Relay's text format into a module."""
        # TODO(@jroesch): we should remove this helper when we port over prelude
        self.mod.import_from_std("prelude.rly")

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

        for dtype in [
            "float32",
            "float16",
            "float64",
            "int32",
            "uint8",
            "int8",
            "int16",
            "uint16",
            "int64",
        ]:
            tensor_array_ops = TensorArrayOps(self, dtype)
            tensor_array_ops.register()

        # Renamer doesn't properly deal with constructors, etc
        # self.mod = AnnotateSpans()(self.mod)
