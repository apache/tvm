# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression nodes of Relay."""
from __future__ import absolute_import
from numbers import Number as _Number

import numpy as _np
from .base import RelayNode, register_relay_node
from . import _make
from . import _expr
from . import ty as _ty
from .._ffi import base as _base
from .. import nd as _nd
from .. import convert

# will be registered afterwards
_op_make = None

class Expr(RelayNode):
    """The base type for all Relay expressions."""
    @property
    def checked_type(self):
        """Get the checked type of tvm.relay.Expr.

        Returns
        -------
        checked_type : tvm.relay.Type
            The checked type.
        """
        ret = self._checked_type_
        if ret is None:
            raise ValueError("The type checker has not populated"
                             " the checked_type for this node")
        return ret

    def astype(self, dtype):
        """Cast the content type of the current data to dtype.

        Parameters
        ----------
        dtype : str
            The target data type.

        Note
        ----
        This function only works for TensorType Exprs.

        Returns
        -------
        result : tvm.relay.Expr
            The result expression.
        """
        return _make.dtype_cast(self, dtype)

    def __add__(self, other):
        if isinstance(other, Expr):
            return _op_make.add(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expr):
            return _op_make.subtract(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rsub__(self, other):
        if isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, Expr):
            return _op_make.multiply(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Expr):
            return _op_make.divide(self, other)
        elif isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __rdiv__(self, other):
        if isinstance(other, _Number):
            raise TypeError('convert "%s" with `const` first' % str(other))
        else:
            raise TypeError("type %s not supported" % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)


@register_relay_node
class Constant(Expr):
    """A constant expression in Relay.

    Parameters
    ----------
    data : tvm.nd.NDArray
        The data content of the constant expression.
    """
    def __init__(self, data):
        self.__init_handle_by_constructor__(_make.Constant, data)


@register_relay_node
class Tuple(Expr):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : List[tvm.relay.Expr]
        The fields in the tuple.
    """
    def __init__(self, fields):
        self.__init_handle_by_constructor__(_make.Tuple, fields)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


@register_relay_node
class Var(Expr):
    """A local variable in Relay.

    Local variable can be used to declare input
    arguments to a function, or intermediate variables.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: tvm.relay.Type, optional
        The type annotation on the variable.
    """
    def __init__(self, name_hint, type_annotation=None):
        self.__init_handle_by_constructor__(
            _make.Var, name_hint, type_annotation)

    @property
    def name_hint(self):
        """Get name hint of the current var."""
        name = self.vid.name_hint
        return name


@register_relay_node
class GlobalVar(Expr):
    """A global variable in Tvm.Relay.

    GlobalVar is used to refer to the global functions
    stored in the module.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """
    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.GlobalVar, name_hint)

    def __call__(self, *args):
        """Invoke the gobal function.

        Parameters
        ----------
        args: List[relay.Expr]
            Arguments.
        """
        return Call(self, args, None, None)


@register_relay_node
class Function(Expr):
    """A function declaration expression.

    Parameters
    ----------
    params: List[tvm.relay.Var]
        List of input parameters to the function.

    body: tvm.relay.Expr
        The body of the function.

    ret_type: Optional[tvm.relay.Type]
        The return type annotation of the function.

    type_params: Optional[List[tvm.relay.TypeParam]]
        The additional type parameters, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self,
                 params,
                 body,
                 ret_type=None,
                 type_params=None):
        if type_params is None:
            type_params = convert([])

        self.__init_handle_by_constructor__(
            _make.Function, params, body, ret_type, type_params)

    def __call__(self, *args):
        """Invoke the gobal function.

        Parameters
        ----------
        args: List[relay.Expr]
            Arguments.
        """
        return Call(self, args, None, None)


@register_relay_node
class Call(Expr):
    """Function call node in Relay.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    op: tvm.relay.Op or any tvm.relay.Expr with function type.
        The operation to be called.

    args: List[tvm.relay.Expr]
        The arguments to the call.

    attrs: Optional[tvm.Attrs]
        Attributes to the call, can be None

    type_args: Optional[List[tvm.relay.Type]]
        The additional type arguments, this is only
        used in advanced usecase of template functions.
    """
    def __init__(self, op, args, attrs=None, type_args=None):
        if not type_args:
            type_args = []
        self.__init_handle_by_constructor__(
            _make.Call, op, args, attrs, type_args)


@register_relay_node
class Let(Expr):
    """Let variable binding expression.

    Parameters
    ----------
    variable: tvm.relay.Var
        The local variable to be bound.

    value: tvm.relay.Expr
        The value to be bound.

    body: tvm.relay.Expr
        The body of the let binding.
    """
    def __init__(self, variable, value, body):
        self.__init_handle_by_constructor__(
            _make.Let, variable, value, body)


@register_relay_node
class If(Expr):
    """A conditional expression in Relay.

    Parameters
    ----------
    cond: tvm.relay.Expr
        The condition.

    true_branch: tvm.relay.Expr
        The expression evaluated when condition is true.

    false_branch: tvm.relay.Expr
        The expression evaluated when condition is false.
    """
    def __init__(self, cond, true_branch, false_branch):
        self.__init_handle_by_constructor__(
            _make.If, cond, true_branch, false_branch)


@register_relay_node
class TupleGetItem(Expr):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple expression.

    index: int
        The index.
    """
    def __init__(self, tuple_value, index):
        self.__init_handle_by_constructor__(
            _make.TupleGetItem, tuple_value, index)


class TempExpr(Expr):
    """Baseclass of all TempExpr.

    TempExprs are pass specific expression that can be
    useful to define intermediate result in the
    rewriting pass such as layout or type transformation.
    """
    def realize(self):
        """Convert the expression to a normal(non-temp) Expr.

        Returns
        -------
        The corresponding normal expression.
        """
        return _expr.TempExprRealize(self)


class ExprFunctor(object):
    """
    An abstract visitor defined over Expr.

    Defines the default dispatch over expressions, and
    implements memoization.
    """
    def __init__(self):
        self.memo_map = {}

    # pylint: disable=no-else-return
    def visit(self, expr):
        """Apply the visitor to an expression."""
        found = self.memo_map.get(expr)
        if found:
            return found

        if isinstance(expr, Function):
            res = self.visit_function(expr)
        elif isinstance(expr, Call):
            res = self.visit_call(expr)
        elif isinstance(expr, Let):
            res = self.visit_let(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        self.memo_map[expr] = res
        return res

    def visit_function(self, _):
        raise NotImplementedError()

    def visit_let(self, _):
        raise NotImplementedError()

    def visit_call(self, _):
        raise NotImplementedError()

    def visit_var(self, _):
        raise NotImplementedError()

    def visit_type(self, typ):
        return typ

    def visit_if(self, _):
        raise NotImplementedError()

    def visit_tuple(self, _):
        raise NotImplementedError()

    def visit_tuple_getitem(self, _):
        raise NotImplementedError()

    def visit_constant(self, _):
        raise NotImplementedError()

    def visit_global_var(self, _):
        raise NotImplementedError()


class ExprMutator(ExprFunctor):
    """
    A functional visitor over Expr.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """
    def visit_function(self, fn):
        new_body = self.visit(fn.body)
        return Function(
            list(fn.params),
            fn.ret_type, new_body,
            fn.type_params)

    def visit_let(self, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return Let(new_var, new_val, new_body)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_fn, new_args, call.attrs)

    def visit_var(self, rvar):
        return rvar

    def visit_global_id(self, global_var):
        return global_var

    def visit_if(self, ite):
        return If(
            self.visit(ite.guard),
            self.visit(ite.true_b),
            self.visit(ite.false_b))

    def visit_tuple(self, tup):
        return Tuple([self.visit(field) for field in tup.fields])

    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return TupleGetItem(tuple_value, op.index)
        return op

    def visit_global_var(self, gvar):
        return gvar

    def visit_constant(self, rconst):
        return rconst


class TupleWrapper(object):
    """TupleWrapper.

    This class is a Python wrapper for a Relay tuple of known size.
    It allows for accessing the fields of the Relay tuple as though
    it were a Python tuple.

    Parameters
    ----------
    tuple_value: tvm.relay.Expr
        The input tuple

    size: int
        The size of the tuple.
    """
    def __init__(self, tuple_value, size):
        self.tuple_value = tuple_value
        self.size = size

    def astuple(self):
        """Returns the underlying Relay tuple if this wrapper is passed
        as an argument to an FFI function."""
        return self.tuple_value

    def astext(self):
        """Get the text format of the tuple expression.

        Returns
        -------
        text : str
            The text format of the tuple expression.
        """
        return self.tuple_value.astext()

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Tuple index out of range")
        return TupleGetItem(self.tuple_value, index)

    def __len__(self):
        return self.size

    def __repr__(self):
        return ("TupleWrapper(" + self.tuple_value.__repr__() +
                ", " + str(self.size) + ")")

    def astype(self, _):
        raise TypeError("astype cannot be used on tuple")


def var(name_hint,
        type_annotation=None,
        shape=None,
        dtype="float32"):
    """Create a new tvm.relay.Var.

    This is a simple wrapper function that allows specify
    shape and dtype directly.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
        This name only acts as a hint, and is not used
        for equality.

    type_annotation: Optional[tvm.relay.Type, str]
        The type annotation on the variable.
        When type_annotation is a str, we will create a scalar variable.

    shape: Optional[List[tvm.Expr]]
        The shape of the tensor type.

    dtype: str, optional
        The data type of the tensor.

    Examples
    --------
    .. code-block:: python

      # The following 4 lines are equivalent to each other
      x = tvm.relay.Var("x", tvm.relay.TensorType([1, 2]))
      x = tvm.relay.var("x", tvm.relay.TensorType([1, 2]))
      x = tvm.relay.var("x", shape=[1, 2])
      x = tvm.relay.var("x", shape=[1, 2], dtype="float32")

      # The following 2 lines are equivalent to each other.
      y = tvm.relay.var("x", "float32")
      y = tvm.relay.var("x", shape=(), dtype="float32")
    """
    if type_annotation is not None and shape is not None:
        raise ValueError("Can only specify either type_annotation or shape.")
    if shape is not None:
        type_annotation = _ty.TensorType(shape, dtype)
    elif isinstance(type_annotation, str):
        type_annotation = _ty.TensorType((), type_annotation)
    return Var(name_hint, type_annotation)


def const(value, dtype=None):
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: str, optional
        The data type of the value.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    if isinstance(value, (_base.numeric_types, (bool, list))):
        value = _np.array(value, dtype=dtype)
        # convert default to int32 and float32
        if dtype is None:
            if value.dtype == "float64":
                value = value.astype("float32")
            elif value.dtype == "int64":
                value = value.astype("int32")
    if isinstance(value, (_np.ndarray, _np.generic)):
        value = _nd.array(value)

    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")
    return Constant(value)


def bind(expr, binds):
    """Bind an free variables in expr or function arguments.

    We can bind parameters expr if it is a function.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    binds : Union[Map[tvm.relay.Var, tvm.relay.Expr], Map[str, tvm.relay.Expr]]
        The specific bindings.

    Returns
    -------
    result : tvm.relay.Expr
        The expression or function after binding.
    """
    return _expr.Bind(expr, binds)
