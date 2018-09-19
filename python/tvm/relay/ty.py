# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The type nodes of the Relay language."""
from typing import List
from enum import IntEnum
from tvm import expr
from .base import Span, NodeBase, register_relay_node
from . import _make


class Type(NodeBase):
    """The base type for all Relay types."""

    def __eq__(self, other): # type: (Type) -> bool
        """Compare two Relay types for structural equivalence using
           alpha equivalence.
        """
        return bool(_make._type_alpha_eq(self, other))

    def __ne__(self, other): # (Type) -> bool
        return not self.__eq__(other)

    def same_as(self, other): # (Type) -> bool
        """Compares two Relay types by referential equality."""
        return super().__eq__(other)


@register_relay_node
class TensorType(Type):
    """A concrete TensorType in Relay, see tvm/relay/type.h for more details.

    This is the type assigned to tensor's with a known dype and shape. For
    example a tensor of `float32` and `(5, 5)`.
    """
    shape = None # type: List[expr.Expr]
    dtype = None # type: str
    span = None # type: Span

    def __init__(self, shape, dtype):
        # type: (List[expr.Expr], str) -> None
        """Construct a tensor type.

        Parameters
        ----------
        shape: list of tvm.Expr
        dtype: str

        Returns
        -------
        tensor_type: The TensorType
        """
        self.__init_handle_by_constructor__(_make.TensorType, shape, dtype)


class Kind(IntEnum):
    """The kind of a type parameter, represents a variable shape,
       base type, type, or dimension.

       This controls what a type parameter is allowed to be instantiated
       with. For example one's of kind BaseType can only be `float32`, `int32`,
       and so on.
    """
    ShapeVar = 0
    Shape = 1
    BaseType = 2
    Type = 3


@register_relay_node
class TypeParam(Type):
    """A type parameter used for generic types in Relay,
    see tvm/relay/type.h for more details.

    A type parameter represents a type placeholder which will
    be filled in later on. This allows the user to write
    functions which are generic over types.
    """
    var = None # type: expr.Var
    kind = None # type: Kind
    span = None # type: Span

    def __init__(self, var, kind):
        # type: (expr.Var, Kind) -> None
        """Construct a TypeParam.

        Parameters
        ----------
        var: tvm.expr.Var
            The tvm.Var which backs the type parameter.

        kind: Kind
            The kind of the type parameter.

        Returns
        -------
        type_param: TypeParam
            The type parameter.
        """
        self.__init_handle_by_constructor__(_make.TypeParam, var, kind)


@register_relay_node
class TypeConstraint(Type):
    """Abstract class representing a type constraint."""
    pass


@register_relay_node
class FuncType(Type):
    """A function type in Relay, see tvm/relay/type.h for more details.

    This is the type assigned to functions in Relay. They consist of
    a list of type parameters which enable the definition of generic
    fucntions, a set of type constraints which we omit for the time
    being, a sequence of argument types, and a return type.

    We informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`
    """
    type_params = None # type: List[TypeParam]
    type_constraints = None # type: List[TypeConstraint]
    arg_types = None # type: List[Type]
    ret_type = None # type: Type
    span = None # type: Span

    def __init__(self,
                 arg_types, # type: List[Type],
                 ret_type, # type: Type,
                 type_params, # type: List[TypeParam],
                 type_constraints, # type: List[TypeConstraint]
                 ):
        # type: (...) -> None
        """Construct a function type.

        Parameters
        ----------
        arg_types:  list of Type
        ret_type: Type
        type_params: list of TypeParam
        type_constraints: list of TypeConstraint

        Returns
        -------
        func_type: FuncType
            The function type.
        """
        self.__init_handle_by_constructor__(
            _make.FuncType, arg_types, ret_type, type_params, type_constraints)


@register_relay_node
class IncompleteType(Type):
    """An incomplete type."""

    def __init__(self, kind):
        # type: (Kind) -> None
        self.__init_handle_by_constructor__(_make.IncompleteType, kind)
