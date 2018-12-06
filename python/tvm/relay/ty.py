# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The type nodes of the Relay language."""
from enum import IntEnum
from .base import RelayNode, register_relay_node
from . import _make


class Type(RelayNode):
    """The base type for all Relay types."""

    def __eq__(self, other):
        """Compare two Relay types for structural equivalence using
           alpha equivalence.
        """
        return bool(_make._type_alpha_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Compares two Relay types by referential equality."""
        return super().__eq__(other)


@register_relay_node
class TensorType(Type):
    """A concrete TensorType in Relay.

    This is the type assigned to tensor's with a known dype and shape. For
    example a tensor of `float32` and `(5, 5)`.

    Parameters
    ----------
    shape : List[tvm.Expr]
        The shape of the Tensor

    dtype : Optional[str]
        The content data type.
        Default to "float32".

    Returns
    -------
    tensor_type : tvm.relay.TensorType
        The tensor type.
    """
    def __init__(self, shape, dtype="float32"):
        self.__init_handle_by_constructor__(
            _make.TensorType, shape, dtype)

    @property
    def concrete_shape(self):
        """Get shape of the type as concrete tuple of int.

        Returns
        -------
        shape : List[int]
            The concrete shape of the Type.

        Raises
        ------
        TypeError : If the shape is symbolic
        """
        return tuple(int(x) for x in self.shape)


class Kind(IntEnum):
    """The kind of a type parameter, represents a variable shape,
       base type, type, or dimension.

       This controls what a type parameter is allowed to be instantiated
       with. For example one's of kind BaseType can only be `float32`, `int32`,
       and so on.
    """
    Type = 0
    ShapeVar = 1
    BaseType = 2
    Shape = 3

@register_relay_node
class TypeVar(Type):
    """A type variable used for generic types in Relay,
    see tvm/relay/type.h for more details.

    A type variable represents a type placeholder which will
    be filled in later on. This allows the user to write
    functions which are generic over types.
    """

    def __init__(self, var, kind=Kind.Type):
        """Construct a TypeVar.

        Parameters
        ----------
        var : tvm.expr.Var
            The tvm.Var which backs the type parameter.

        kind : Optional[Kind]
            The kind of the type parameter.
            Default to Kind.Type.

        Returns
        -------
        type_var : tvm.relay.TypeVar
            The type variable.
        """
        self.__init_handle_by_constructor__(_make.TypeVar, var, kind)


@register_relay_node
class TypeConstraint(Type):
    """Abstract class representing a type constraint."""
    pass


@register_relay_node
class TupleType(Type):
    """A tuple type in Relay, see tvm/relay/type.h for more details.

    Lists the type of each field in the tuple.
    """

    def __init__(self, fields):
        """Constructs a tuple type

        Parameters
        ----------
        fields : List[tvm.relay.Type]
            The fields in the tuple

        Returns
        -------
        tuple_type : tvm.relay.TupleType
            the tuple type
        """
        self.__init_handle_by_constructor__(_make.TupleType, fields)


@register_relay_node
class FuncType(Type):
    """A function type in Relay, see tvm/relay/type.h for more details.

    This is the type assigned to functions in Relay. They consist of
    a list of type parameters which enable the definition of generic
    functions, a set of type constraints which we omit for the time
    being, a sequence of argument types, and a return type.

    We informally write them as:
    `forall (type_params), (arg_types) -> ret_type where type_constraints`

    Parameters
    ----------
    arg_types : List[tvm.relay.Type]
        The argument types

    ret_type : tvm.relay.Type
        The return type.

    type_params : Optional[List[tvm.relay.TypeVar]]
        The type parameters

    type_constraints : Optional[List[tvm.relay.TypeConstraint]]
        The type constraints.
    """
    def __init__(self,
                 arg_types,
                 ret_type,
                 type_params=None,
                 type_constraints=None):
        if type_params is None:
            type_params = []
        if type_constraints is None:
            type_constraints = []
        self.__init_handle_by_constructor__(
            _make.FuncType, arg_types, ret_type, type_params, type_constraints)


@register_relay_node
class IncompleteType(Type):
    """An incomplete type."""
    def __init__(self, kind=Kind.Type):
        self.__init_handle_by_constructor__(_make.IncompleteType, kind)


@register_relay_node
class TypeRelation(TypeConstraint):
    """Type relation in relay.

    Parameters
    ----------
    func : EnvFunc
        User defined relation function.

    args : [tvm.relay.Type]
        List of types to the func.

    num_inputs : int
        Number of input arguments in args,
        this act as a hint for type inference.

    attrs : Attrs
        The attribute attached to the relation information

    Returns
    -------
    type_relation : tvm.relay.TypeRelation
        The type relation.
    """
    def __init__(self, func, args, num_inputs, attrs):
        self.__init_handle_by_constructor__(_make.TypeRelation,
                                            func, args, num_inputs, attrs)


def scalar_type(dtype):
    """Creates a scalar type.

    This function returns TensorType((), dtype)

    Parameters
    ----------
    dtype : str
        The content data type.

    Returns
    -------
    s_type : tvm.relay.TensorType
        The result type.
    """
    return TensorType((), dtype)
